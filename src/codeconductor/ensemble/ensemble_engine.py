"""
Main Ensemble Engine for CodeConductor

Orchestrates multiple LLMs for consensus-based code generation with RLHF integration.
"""

import asyncio
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from codeconductor.context.rag_system import RAGSystem
from codeconductor.feedback.rlhf_agent import RLHFAgent
from codeconductor.monitoring.memory_watchdog import (
    start_memory_watchdog,
    stop_memory_watchdog,
)
from codeconductor.runners.test_runner import PytestRunner
from codeconductor.utils.kpi import TestSummary, build_kpi, write_json

from .code_reviewer import CodeReviewer
from .consensus_calculator import ConsensusCalculator, ConsensusResult
from .model_manager import ModelManager
from .model_selector import ModelSelector, SelectionInput
from .query_dispatcher import QueryDispatcher

# Try to import RLHF components
try:
    from feedback.rlhf_agent import RLHFAgent

    RLHF_AVAILABLE = True
except ImportError:
    RLHF_AVAILABLE = False
    print("⚠️ RLHF not available. Install with: pip install stable-baselines3 gymnasium")

# Try to import RAG components
try:
    from codeconductor.context.rag_system import RAGSystem

    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("Warning: RAG not available. Install with: pip install langchain chromadb")

# Try to import CodeReviewer components
try:
    from .code_reviewer import CodeReviewer

    CODE_REVIEWER_AVAILABLE = True
except ImportError:
    CODE_REVIEWER_AVAILABLE = False
    print("⚠️ CodeReviewer not available. Install with: pip install stable-baselines3")

logger = logging.getLogger(__name__)


@dataclass
class EnsembleRequest:
    task_description: str
    context: dict[str, Any] | None = None
    min_models: int = 2
    timeout: float = 30.0
    test_results: list[dict[str, Any]] | None = None
    code_quality: float = 0.5
    user_feedback: float = 0.0


@dataclass
class EnsembleResponse:
    consensus: dict[str, Any]
    confidence: float
    disagreements: list[str]
    model_responses: list[dict[str, Any]]
    execution_time: float
    rlhf_action: int | None = None
    rlhf_action_description: str | None = None
    selected_model: str | None = None


class EnsembleEngine:
    """Main ensemble engine that coordinates multiple LLMs with RLHF integration."""

    def __init__(
        self,
        min_confidence: float = 0.7,
        use_rlhf: bool = True,
        use_rag: bool = True,
        use_code_reviewer: bool = True,
    ):
        self.model_manager = ModelManager()
        self.consensus_calculator = ConsensusCalculator()
        self.model_selector = ModelSelector()
        # LRU cache with TTL (env-configurable)
        try:
            from codeconductor.utils.lru_cache import LRUCacheTTL

            cache_size = int(os.getenv("CACHE_SIZE", "100"))
            cache_ttl = int(os.getenv("CACHE_TTL_SECONDS", "1800"))
            cache_ns = os.getenv("CACHE_NAMESPACE", "default")
            self.response_cache = LRUCacheTTL(
                max_entries=cache_size, ttl_seconds=cache_ttl, namespace=cache_ns
            )
        except Exception:  # pragma: no cover
            self.response_cache = None
        self.min_confidence = min_confidence
        # Allow disabling RLHF via environment (RLHF_DISABLE=1)
        rlhf_disabled_env = os.getenv("RLHF_DISABLE", "0").strip() == "1"
        self.use_rlhf = (use_rlhf and RLHF_AVAILABLE) and (not rlhf_disabled_env)
        # Respect ALLOW_NET: when set to '0', disable RAG completely
        self.use_rag = (
            use_rag and RAG_AVAILABLE and (os.getenv("ALLOW_NET", "0") != "0")
        )
        self.use_code_reviewer = use_code_reviewer and CODE_REVIEWER_AVAILABLE
        self.rlhf_agent = None
        self.rag_system = None
        self.code_reviewer = None
        # Telemetry snapshots
        self.last_selector_decision: dict[str, Any] = {}
        self.last_cache_hit: bool = False
        self.last_artifacts_dir: str | None = None
        self.last_export_path: str | None = None

        if self.use_rlhf:
            self._initialize_rlhf()

        if self.use_rag:
            self._initialize_rag()

        if self.use_code_reviewer:
            self._initialize_code_reviewer()

    def _get_selector_policy(self) -> str:
        return (os.getenv("SELECTOR_POLICY", "latency") or "latency").lower()

    def _build_cache_key(
        self,
        *,
        prompt: str,
        persona: str,
        policy: str,
        model: str,
        sampling: dict,
    ) -> str:
        if not self.response_cache:
            return ""

    def _save_run_artifacts(
        self,
        prompt: str,
        policy: str,
        selector: dict[str, Any],
        consensus_meta: dict[str, Any],
        consensus_obj: ConsensusResult | None = None,
    ) -> None:
        base = os.getenv("ARTIFACTS_DIR", "artifacts")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(base, "runs", ts)
        os.makedirs(run_dir, exist_ok=True)
        self.last_artifacts_dir = run_dir

        def _write(name: str, obj: Any) -> None:
            try:
                with open(os.path.join(run_dir, name), "w", encoding="utf-8") as f:
                    json.dump(obj, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

        # run_config
        run_config = {
            "policy": policy,
            "env": {
                "SELECTOR_POLICY": os.getenv("SELECTOR_POLICY", ""),
                "CACHE_SIZE": os.getenv("CACHE_SIZE", ""),
                "CACHE_TTL_SECONDS": os.getenv("CACHE_TTL_SECONDS", ""),
                "CACHE_NAMESPACE": os.getenv("CACHE_NAMESPACE", ""),
            },
            "prompt_len": len(prompt or ""),
        }
        _write("run_config.json", run_config)

        # selector decision
        _write("selector_decision.json", selector)

        # consensus (robust guards against non-dict types)
        consensus_dump = {}
        try:
            if isinstance(consensus_meta, dict):
                consensus_dump.update(
                    {
                        "method": consensus_meta.get("method"),
                        "winner": consensus_meta.get("winner"),
                        "cached": bool(consensus_meta.get("cached", False)),
                    }
                )
                if isinstance(consensus_meta.get("candidates"), list):
                    consensus_dump["candidates"] = consensus_meta.get("candidates", [])
            else:
                consensus_dump.update({"method": None, "winner": None, "cached": False})
        except Exception:
            consensus_dump.update({"method": None, "winner": None, "cached": False})

        try:
            if consensus_obj is not None:
                consensus_dump.update(
                    {
                        "confidence": float(
                            getattr(consensus_obj, "confidence", 0.0) or 0.0
                        ),
                        "code_quality": float(
                            getattr(consensus_obj, "code_quality_score", 0.0) or 0.0
                        ),
                        "syntax_valid": bool(
                            getattr(consensus_obj, "syntax_valid", False)
                        ),
                        # Include generated code content for downstream exporters/materialization
                        "consensus": getattr(consensus_obj, "consensus", "") or "",
                    }
                )
        except Exception:
            pass

        _write("consensus.json", consensus_dump)

    def _maybe_attach_export(self, policy: str) -> None:
        try:
            if os.getenv("ATTACH_EXPORT", "0") != "1":
                return
            # Lazy import to avoid hard dependency during tests where package path differs
            from codeconductor.utils.exporter import export_latest_run, verify_manifest

            zip_path, manifest = export_latest_run(
                artifacts_dir=os.getenv("ARTIFACTS_DIR", "artifacts"),
                include_raw=os.getenv("EXPORT_INCLUDE_RAW", "0") == "1",
                redact_env=os.getenv("EXPORT_REDACT_ENV", "1") != "0",
                size_limit_mb=int(os.getenv("EXPORT_SIZE_LIMIT_MB", "50")),
                retention=int(os.getenv("EXPORT_RETENTION", "20")),
                policy=policy,
                selected_model=self.last_selector_decision.get("chosen"),
                cache_hit=self.last_cache_hit,
                app_version=os.getenv("APP_VERSION", None),
                git_commit=os.getenv("GIT_COMMIT", None),
            )
            self.last_export_path = zip_path
            ver = verify_manifest(zip_path)
            size_mb = 0.0
            try:
                size_mb = round(os.path.getsize(zip_path) / (1024 * 1024), 2)
            except Exception:
                pass
            logger.info(
                json.dumps(
                    {
                        "event": "export_bundle",
                        "path": zip_path,
                        "size_mb": size_mb,
                        "policy": policy,
                        "hit": self.last_cache_hit,
                        "verified": bool(ver.get("verified")),
                    }
                )
            )
        except Exception as e:
            logger.warning(f"Export attach failed: {e}")

    def _initialize_rlhf(self):
        """Initialize RLHF agent if available."""
        try:
            self.rlhf_agent = RLHFAgent()
            if not self.rlhf_agent.load_model():
                logger.warning("RLHF model not found. Training new model...")
                # Try to train a new model if none exists
                if self.rlhf_agent.train(total_timesteps=1000):
                    logger.info("RLHF model trained successfully")
                else:
                    logger.warning("Failed to train RLHF model. Disabling RLHF.")
                    self.use_rlhf = False
            else:
                logger.info("RLHF agent initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize RLHF agent: {e}. Disabling RLHF.")
            self.use_rlhf = False

    def _initialize_rag(self):
        """Initialize RAG system if available."""
        try:
            self.rag_system = RAGSystem()
            logger.info("RAG system initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize RAG system: {e}. Disabling RAG.")
            self.use_rag = False

    def _initialize_code_reviewer(self):
        """Initialize CodeReviewer if available."""
        try:
            # Get available models for the reviewer
            models = ["phi3", "codellama", "mistral"]  # Default models
            self.code_reviewer = CodeReviewer(models)
            logger.info("CodeReviewer initialized successfully")
        except Exception as e:
            logger.warning(
                f"Failed to initialize CodeReviewer: {e}. Disabling CodeReviewer."
            )
            self.use_code_reviewer = False

    def _select_models_with_rlhf(
        self, request: EnsembleRequest, available_models: list[str]
    ) -> tuple[list[str], int | None, str | None]:
        """Use RLHF agent to select optimal models based on task and context."""
        if not self.use_rlhf or not self.rlhf_agent:
            # Fallback to default selection
            return available_models[: request.min_models], None, None

        try:
            # Calculate test reward from test results
            test_reward = self._calculate_test_reward(request.test_results or [])

            # Estimate task complexity
            task_complexity = self._estimate_task_complexity(request.task_description)

            # Create observation for RLHF agent
            observation = np.array(
                [
                    test_reward,
                    request.code_quality,
                    request.user_feedback,
                    task_complexity,
                ],
                dtype=np.float32,
            )

            # Get action from RLHF agent
            action, _ = self.rlhf_agent.predict_action(observation)
            action_description = self.rlhf_agent.get_action_description(action)

            logger.info(f"RLHF selected action {action}: {action_description}")

            # Map action to model selection strategy
            if action == 0:  # use_model_A (default)
                selected_models = available_models[: request.min_models]
            elif action == 1:  # use_model_B (alternative)
                # Use different models if available
                if len(available_models) >= 2:
                    selected_models = [available_models[1]] + available_models[
                        2 : request.min_models
                    ]
                else:
                    selected_models = available_models[: request.min_models]
            elif action == 2:  # retry_with_fix
                # Use all available models for consensus
                selected_models = available_models[: min(len(available_models), 4)]
            else:  # action == 3, escalate_to_gpt4
                # Use more models for complex tasks
                selected_models = available_models[: min(len(available_models), 6)]

            # Ensure we have at least min_models
            if len(selected_models) < request.min_models:
                selected_models = available_models[: request.min_models]

            return selected_models, action, action_description

        except Exception as e:
            logger.error(f"RLHF model selection failed: {e}. Using fallback.")
            return available_models[: request.min_models], None, None

    def _calculate_test_reward(self, test_results: list[dict[str, Any]]) -> float:
        """Calculate reward based on test results."""
        if not test_results:
            return 0.0

        total_tests = len(test_results)
        passed_tests = sum(1 for test in test_results if test.get("passed", False))
        return passed_tests / total_tests if total_tests > 0 else 0.0

    def _estimate_task_complexity(self, task_description: str) -> float:
        """Estimate task complexity based on keywords."""
        if not task_description:
            return 0.5

        # Keywords that indicate complexity
        complex_keywords = [
            "api",
            "database",
            "authentication",
            "security",
            "async",
            "threading",
            "machine learning",
            "algorithm",
            "optimization",
            "performance",
            "testing",
            "deployment",
            "microservice",
            "distributed",
            "complex",
        ]

        simple_keywords = [
            "print",
            "hello world",
            "simple",
            "basic",
            "function",
            "variable",
        ]

        task_lower = task_description.lower()

        # Count complexity indicators
        complex_count = sum(1 for keyword in complex_keywords if keyword in task_lower)
        simple_count = sum(1 for keyword in simple_keywords if keyword in task_lower)

        # Calculate complexity score
        if complex_count > 0:
            complexity = min(complex_count / 3.0, 1.0)
        elif simple_count > 0:
            complexity = max(0.1, 1.0 - simple_count / 2.0)
        else:
            complexity = 0.5  # Default medium complexity

        return complexity

    def apply_fixes(self, code: str, fixes: list[str]) -> str:
        """
        Apply suggested fixes to code (simplified implementation).

        Args:
            code: Original code
            fixes: List of suggested fixes

        Returns:
            Code with applied fixes
        """
        if not fixes:
            return code

        # Add comments about applied fixes
        fix_comments = "\n".join([f"# Applied fix: {fix}" for fix in fixes])
        return f"{code}\n\n{fix_comments}"

    async def initialize(self) -> bool:
        """Initialize the ensemble engine and all components."""
        try:
            logger.info("🚀 Initializing Ensemble Engine...")

            # Initialize model manager
            self.model_manager = ModelManager()
            logger.info("✅ Model manager initialized")

            # Initialize consensus calculator
            self.consensus_calculator = ConsensusCalculator()
            logger.info("✅ Consensus calculator initialized")

            # Initialize query dispatcher
            self.query_dispatcher = QueryDispatcher()
            logger.info("✅ Query dispatcher initialized")

            # Initialize RLHF agent if enabled
            if self.use_rlhf:
                self._initialize_rlhf()

            # Initialize RAG system if enabled
            if self.use_rag:
                self._initialize_rag()

            # Initialize code reviewer if enabled
            if self.use_code_reviewer:
                self._initialize_code_reviewer()

            # Start memory watchdog
            try:
                await start_memory_watchdog(self.model_manager, check_interval=30.0)
                logger.info("✅ Memory watchdog started")
            except Exception as e:
                logger.warning(f"⚠️ Failed to start memory watchdog: {e}")

            logger.info("✅ Ensemble Engine initialization completed")
            return True

        except Exception as e:
            logger.error(f"❌ Ensemble Engine initialization failed: {e}")
            return False

    async def cleanup(self):
        """Clean up resources and stop background tasks."""
        try:
            logger.info("🧹 Cleaning up Ensemble Engine...")

            # Stop memory watchdog
            try:
                await stop_memory_watchdog()
                logger.info("✅ Memory watchdog stopped")
            except Exception as e:
                logger.warning(f"⚠️ Failed to stop memory watchdog: {e}")

            # Emergency unload all models
            try:
                if hasattr(self, "model_manager") and self.model_manager:
                    unloaded_count = await self.model_manager.emergency_unload_all()
                    logger.info(f"✅ Emergency unloaded {unloaded_count} models")
            except Exception as e:
                logger.warning(f"⚠️ Failed to unload models: {e}")

            logger.info("✅ Ensemble Engine cleanup completed")

        except Exception as e:
            logger.error(f"❌ Ensemble Engine cleanup failed: {e}")

    async def process_request(
        self,
        task_description: str,
        timeout: float = 30.0,
        prefer_fast_models: bool = False,
        enable_fallback: bool = False,
    ) -> dict[str, Any]:
        """
        Process a task request with optional fast model priority and fallback.

        Args:
            task_description: The task to process
            timeout: Request timeout in seconds
            prefer_fast_models: If True, prioritize fast models (Ollama)
            enable_fallback: If True, enable fallback strategies

        Returns:
            Dict with results including generated_code, confidence, etc.
        """
        # Kontrollera GPU_DISABLED först
        if os.getenv("CC_GPU_DISABLED", "0") == "1":
            logger.info(
                "[MOCK] CC_GPU_DISABLED=1 active — returning mock content (EnsembleEngine.process_request)"
            )
            return {
                "generated_code": f"[MOCKED] {task_description}",
                "confidence": 0.95,
                "model_used": "ensemble-mock",
                "execution_time": 0.01,
                "cached": False,
                "method": "mock",
            }
        
        # Create EnsembleRequest from parameters
        request = EnsembleRequest(task_description=task_description, timeout=timeout)

        # Process with fast model priority if requested
        if prefer_fast_models:
            return await self._process_request_fast_priority(request)
        else:
            response = await self._process_request_internal(request)
            return self._convert_response_to_dict(response)

    async def _process_request_internal(
        self, request: EnsembleRequest
    ) -> EnsembleResponse:
        """Process a task request through the ensemble."""
        start_time = asyncio.get_event_loop().time()
        wall_start_iso = datetime.utcnow().replace(tzinfo=None).isoformat() + "Z"
        t_first_green_iso: str | None = None

        logger.info(f"Processing ensemble request: {request.task_description[:50]}...")

        # Augment task description with RAG context if available
        augmented_task = request.task_description
        if self.use_rag and self.rag_system:
            try:
                augmented_task = self.rag_system.augment_prompt(
                    request.task_description
                )
                logger.info("Task description augmented with RAG context")
            except Exception as e:
                logger.warning(f"Failed to augment task with RAG: {e}")

        try:
            # Estimate task complexity
            task_complexity = self._estimate_task_complexity(request.task_description)
            logger.info(f"🎯 Task complexity: {task_complexity:.2f}")

            # Get available models
            available_model_ids = await self.model_manager.get_available_model_ids()
            if not available_model_ids:
                raise Exception("❌ No models available")

            # Complexity-based loading for RTX 5090
            from .model_manager import COMPLEXITY_BASED_LOADING

            # Determine loading config based on complexity
            loading_config = "medium_load"  # Default
            for (
                min_complexity,
                max_complexity,
            ), config in COMPLEXITY_BASED_LOADING.items():
                if min_complexity <= task_complexity < max_complexity:
                    loading_config = config
                    break

            logger.info(
                f"🎯 Using {loading_config} config for complexity {task_complexity:.2f}"
            )

            # Use memory-safe loading for RTX 5090 and ensure 3-5 models when possible
            loaded_model_ids = (
                await self.model_manager.ensure_models_loaded_with_memory_check(
                    loading_config
                )
            )

            # Prefer 3 models (up to 5 if aggressive load), else fallback to available
            if len(loaded_model_ids) >= 3:
                max_models = 5 if loading_config == "aggressive_load" else 3
                # Env override to cap parallelism
                try:
                    max_env = int(os.getenv("MAX_PARALLEL_MODELS", "0") or "0")
                    if max_env > 0:
                        max_models = max_env
                except Exception:
                    pass
                selected_model_ids = loaded_model_ids[: max(1, max_models)]
                logger.info(
                    f"✅ Using {len(selected_model_ids)} memory-safe loaded models (voting): {selected_model_ids}"
                )
            else:
                # Fallback to available models — honor MAX_PARALLEL_MODELS if provided
                try:
                    max_env = int(os.getenv("MAX_PARALLEL_MODELS", "0") or "0")
                except Exception:
                    max_env = 0
                fallback_cap = max_env if max_env > 0 else 3
                fallback_k = min(fallback_cap, len(available_model_ids))
                selected_model_ids = available_model_ids[:fallback_k]
                logger.info(
                    f"⚠️ Using fallback models (cap={fallback_cap}): {selected_model_ids}"
                )

            # Get model objects from model manager
            all_models = await self.model_manager.list_models()
            models = [model for model in all_models if model.id in selected_model_ids]

            # Model selection policy ordering + sampling
            policy = self._get_selector_policy()
            prompt_text = (
                augmented_task
                if isinstance(augmented_task, str)
                else request.task_description
            )
            prompt_len = len(prompt_text or "")
            selection = self.model_selector.select(
                SelectionInput(models=models, prompt_len=prompt_len, policy=policy)
            )
            # Expose selector decision for UI/telemetry
            try:
                self.last_selector_decision = {
                    "policy": policy,
                    "scores": selection.scores,
                    "chosen": selection.selected_model,
                    "fallbacks": selection.fallbacks,
                    "sampling": selection.sampling,
                    "why": getattr(selection, "why", {}),
                }
            except Exception:
                self.last_selector_decision = {"policy": policy}
            # Reorder models by selector decision (primary first)
            id_to_model = {m.id: m for m in models}
            ordered_ids = [
                mid
                for mid, _ in sorted(
                    selection.scores.items(), key=lambda kv: kv[1], reverse=True
                )
            ]
            models = [id_to_model[mid] for mid in ordered_ids if mid in id_to_model]

            # Persist selector decision artifact for UI
            try:
                import json as _json

                base = os.getenv("ARTIFACTS_DIR", "artifacts")
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_dir = os.path.join(base, "runs", ts)
                os.makedirs(run_dir, exist_ok=True)
                self.last_artifacts_dir = run_dir
                with open(
                    os.path.join(run_dir, "selector_decision.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    _json.dump(
                        self.last_selector_decision, f, ensure_ascii=False, indent=2
                    )
            except Exception:
                pass

            # Cache short-circuit (best-effort)
            cached_code = None
            if self.response_cache and models:
                persona = (
                    (request.context or {}).get("persona", "default")
                    if hasattr(request, "context")
                    else "default"
                )
                primary_id = models[0].id
                cache_key = self._build_cache_key(
                    prompt=prompt_text or "",
                    persona=persona,
                    policy=policy,
                    model=primary_id,
                    sampling=selection.sampling,
                )
                if cache_key:
                    cached_code = self.response_cache.get(cache_key)
            if cached_code:
                # Build a minimal response using cached consensus
                logger.info("🔥 Cache hit – skipping inference")
                self.last_cache_hit = True
                # Save minimal artifacts for cache hit
                try:
                    self._save_run_artifacts(
                        prompt_text or "",
                        policy,
                        self.last_selector_decision,
                        {
                            "method": "codebleu+heuristic",
                            "winner": {"model": models[0].id, "score": 1.0},
                            "candidates": [],
                            "cached": True,
                        },
                    )
                except Exception:
                    pass
                return EnsembleResponse(
                    consensus={"content": cached_code},
                    confidence=1.0,
                    disagreements=[],
                    model_responses=[],
                    execution_time=0.0,
                    selected_model=models[0].id if models else None,
                )
            else:
                self.last_cache_hit = False

            # Dispatch queries in parallel with timeout
            async with QueryDispatcher(timeout=request.timeout) as dispatcher:
                raw_results = await asyncio.wait_for(
                    dispatcher.dispatch_parallel(
                        models, augmented_task, request.context
                    ),
                    timeout=request.timeout + 10.0,  # Add extra buffer
                )
                logger.info(f"🔍 Raw results from dispatch_parallel: {raw_results}")

            # Convert raw results to format expected by consensus calculator
            formatted_results = self._format_results_for_consensus(raw_results)

            # Handle empty results path with optional single-consensus fallback
            if not formatted_results:
                if os.getenv("CONSENSUS_ALLOW_SINGLE", "0") == "1":
                    logger.warning(
                        "CONSENSUS_ALLOW_SINGLE=1 — writing degraded consensus artifact"
                    )
                    try:
                        self._save_run_artifacts(
                            prompt_text or "",
                            policy,
                            self.last_selector_decision,
                            {
                                "method": "codebleu_fast",
                                "winner": {
                                    "model": models[0].id if models else None,
                                    "score": 1.0,
                                },
                                "candidates": [],
                                "cached": False,
                                "degraded": True,
                            },
                        )
                    except Exception:
                        pass
                    return EnsembleResponse(
                        consensus={"content": ""},
                        confidence=0.0,
                        disagreements=["degraded"],
                        model_responses=[],
                        execution_time=asyncio.get_event_loop().time() - start_time,
                        selected_model=models[0].id if models else None,
                    )
                else:
                    raise Exception("No model results available for consensus")

            # Calculate consensus (majority/best-score handled inside calculator)
            consensus_result = self.consensus_calculator.calculate_consensus(
                formatted_results
            )
            # Make consensus content available for downstream steps (tests, review)
            generated_code = consensus_result.consensus

            # Save successful result to cache (best-effort)
            try:
                if (
                    self.response_cache
                    and models
                    and consensus_result
                    and consensus_result.consensus
                ):
                    persona = (
                        (request.context or {}).get("persona", "default")
                        if hasattr(request, "context")
                        else "default"
                    )
                    primary_id = models[0].id
                    cache_key = self._build_cache_key(
                        prompt=prompt_text or "",
                        persona=persona,
                        policy=policy,
                        model=primary_id,
                        sampling=selection.sampling,
                    )
                    if cache_key:
                        self.response_cache.set(cache_key, consensus_result.consensus)
            except Exception:
                pass

            # Save artifacts for this run
            try:
                # Build candidates list with CodeBLEU-like scores and output SHA
                candidates: list[dict[str, Any]] = []
                try:
                    model_scores = getattr(consensus_result, "model_scores", {}) or {}
                    for item in formatted_results:
                        model_id = item.get("model")
                        content = item.get("content", "") or ""
                        score = float(model_scores.get(model_id, 0.0))
                        output_sha = (
                            hashlib.sha256(content.encode("utf-8")).hexdigest()
                            if content
                            else None
                        )
                        candidates.append(
                            {
                                "model": model_id,
                                "score": score,
                                "output_sha": output_sha,
                            }
                        )
                    # Sort by score desc, keep all (UI may show top-3)
                    candidates.sort(key=lambda d: d.get("score", 0.0), reverse=True)
                except Exception:
                    candidates = []

                # Determine winner model (highest score)
                try:
                    best_model = None
                    best_score = float(consensus_result.confidence)
                    if candidates:
                        best_model = candidates[0].get("model")
                        best_score = float(candidates[0].get("score", best_score))
                    elif formatted_results:
                        best_model = formatted_results[0].get("model")
                    winner_obj = {"model": best_model, "score": best_score}
                except Exception:
                    winner_obj = {
                        "model": models[0].id if models else None,
                        "score": float(consensus_result.confidence),
                    }

                self._save_run_artifacts(
                    prompt_text or "",
                    policy,
                    self.last_selector_decision,
                    {
                        "method": "codebleu+heuristic",
                        "winner": winner_obj,
                        "candidates": candidates,
                        "cached": False,
                    },
                    consensus_result,
                )
                # --- KPI writing (after consensus) ---
                try:
                    run_dir = Path(
                        self.last_artifacts_dir
                        or os.getenv("ARTIFACTS_DIR", "artifacts")
                    )
                    tests_dir = run_dir / "tests"
                    tests_dir.mkdir(parents=True, exist_ok=True)

                    # Run tests before/after (best effort). If before not known, treat as 0.
                    # For MVP, we execute pytest only once (after) to reduce overhead, and set before=0.
                    # Future PR can persist a pre-run snapshot.
                    before = TestSummary(
                        suite_name="pytest", total=0, passed=0, failed=0, skipped=0
                    )

                    pr = PytestRunner(
                        prompt=request.task_description, code=generated_code or ""
                    )
                    pr_res = pr.run()
                    total = int(pr_res.get("total_tests", 0))
                    passed = int(pr_res.get("passed_tests", 0))
                    failed = max(0, total - passed)
                    after = TestSummary(
                        suite_name="pytest",
                        total=total,
                        passed=passed,
                        failed=failed,
                        skipped=0,
                    )
                    # Persist raw reports (best effort)
                    write_json(
                        tests_dir / "before_report.json",
                        {
                            "total": 0,
                            "passed": 0,
                            "failed": 0,
                            "skipped": 0,
                            "suite": "pytest",
                        },
                    )
                    write_json(tests_dir / "after_report.json", pr_res)

                    # Mark first green
                    if after.total > 0 and after.failed == 0:
                        t_first_green_iso = (
                            datetime.utcnow().replace(tzinfo=None).isoformat() + "Z"
                        )

                    # KPI fields
                    sampling = (
                        self.last_selector_decision.get("sampling", {})
                        if isinstance(self.last_selector_decision, dict)
                        else {}
                    )
                    codebleu_weights_env = os.getenv("CODEBLEU_WEIGHTS")
                    codebleu_lang_env = (
                        os.getenv("CODEBLEU_LANG")
                        or os.getenv("CODEBLEU_LANGUAGE")
                        or "python"
                    )

                    # Exit status
                    tests_missing = after.total == 0
                    exit_status = {
                        "patched": bool(bool(generated_code)),
                        "tests_passed": bool(after.total > 0 and after.failed == 0),
                        "tests_missing": bool(tests_missing),
                        "error": None,
                    }

                    kpi = build_kpi(
                        run_id=(
                            self.last_selector_decision.get("run_id")
                            or os.path.basename(str(run_dir))
                            or "unknown"
                        ),
                        artifacts_dir=run_dir,
                        t_start_iso=wall_start_iso,
                        t_first_green_iso=t_first_green_iso,
                        ttft_ms=int(
                            (asyncio.get_event_loop().time() - start_time) * 1000.0
                        ),
                        tests_before=before,
                        tests_after=after,
                        winner_model=winner_obj.get("model"),
                        winner_score=float(winner_obj.get("score", 0.0)),
                        consensus_method="codebleu+heuristic",
                        sampling=sampling,
                        codebleu_weights_env=codebleu_weights_env,
                        codebleu_lang_env=codebleu_lang_env,
                        exit_status=exit_status,
                    )
                    write_json(run_dir / "kpi.json", kpi)
                    logger.info(
                        f"KPI written: {run_dir / 'kpi.json'} run_id={kpi.get('run_id')}"
                    )
                except Exception as e:  # pragma: no cover
                    logger.warning(f"KPI writing failed: {e}")
            except Exception:
                pass

            # Review code if CodeReviewer is available
            if self.use_code_reviewer and self.code_reviewer and generated_code:
                try:
                    logger.info("🔍 Starting code review...")
                    review_results = self.code_reviewer.review_code(
                        request.task_description, generated_code, request.test_results
                    )
                    logger.info(f"🔍 Code review completed: {review_results}")
                except Exception as e:
                    logger.warning(f"⚠️ Code review failed: {e}")

            # Post-request memory cleanup
            try:
                logger.info("🧹 Performing post-request memory cleanup...")
                cleanup_performed = await self.model_manager.check_and_cleanup_memory(
                    loading_config
                )
                if cleanup_performed:
                    logger.info("🧹 Memory cleanup completed")
                else:
                    logger.info("🧹 No memory cleanup needed")
            except Exception as e:
                logger.warning(f"⚠️ Post-request memory cleanup failed: {e}")

            # Calculate execution time
            execution_time = asyncio.get_event_loop().time() - start_time

            # Compute a simple agreement heuristic from model scores
            try:
                scores_list = sorted(
                    (consensus_result.model_scores or {}).values(), reverse=True
                )
                if len(scores_list) >= 2:
                    # High agreement when top two scores are close
                    model_agreement = max(
                        0.0, min(1.0, 1.0 - (scores_list[0] - scores_list[1]))
                    )
                elif len(scores_list) == 1:
                    model_agreement = 1.0
                else:
                    model_agreement = 0.0
            except Exception:
                model_agreement = 0.0

            # Create response
            response = EnsembleResponse(
                consensus={
                    "code": generated_code,
                    "explanation": getattr(consensus_result, "reasoning", ""),
                    "confidence": consensus_result.confidence,
                    "model_agreement": model_agreement,
                },
                confidence=consensus_result.confidence,
                disagreements=[],
                model_responses=formatted_results,
                execution_time=execution_time,
            )

            logger.info(f"✅ Ensemble request completed in {execution_time:.2f}s")
            return response

        except Exception as e:
            logger.error(f"Ensemble request failed: {e}")
            execution_time = asyncio.get_event_loop().time() - start_time
            # Best-effort: write a minimal KPI so exporters/benchmarks can proceed
            try:
                base = os.getenv("ARTIFACTS_DIR", "artifacts")
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_dir = Path(
                    self.last_artifacts_dir or os.path.join(base, "runs", ts)
                )
                run_dir.mkdir(parents=True, exist_ok=True)
                self.last_artifacts_dir = str(run_dir)
                before = TestSummary(
                    suite_name="pytest", total=0, passed=0, failed=0, skipped=0
                )
                after = TestSummary(
                    suite_name="pytest", total=0, passed=0, failed=0, skipped=0
                )
                sampling = (
                    self.last_selector_decision.get("sampling", {})
                    if isinstance(self.last_selector_decision, dict)
                    else {}
                )
                kpi = build_kpi(
                    run_id=os.path.basename(str(run_dir)),
                    artifacts_dir=str(run_dir),
                    t_start_iso=wall_start_iso,
                    t_first_green_iso=None,
                    ttft_ms=int(execution_time * 1000.0),
                    tests_before=before,
                    tests_after=after,
                    winner_model=None,
                    winner_score=0.0,
                    consensus_method="failed",
                    sampling=sampling,
                    codebleu_weights_env=os.getenv("CODEBLEU_WEIGHTS"),
                    codebleu_lang_env=(
                        os.getenv("CODEBLEU_LANG")
                        or os.getenv("CODEBLEU_LANGUAGE")
                        or "python"
                    ),
                    exit_status={
                        "patched": False,
                        "tests_passed": False,
                        "tests_missing": True,
                        "error": str(e),
                    },
                )
                write_json(run_dir / "kpi.json", kpi)
                logger.info(f"KPI written (failure): {run_dir / 'kpi.json'}")
            except Exception:
                pass

            return EnsembleResponse(
                consensus={},
                confidence=0.0,
                disagreements=[f"Request failed: {str(e)}"],
                model_responses=[],
                execution_time=execution_time,
                rlhf_action=None,
                rlhf_action_description=None,
                selected_model=None,
            )

    def _format_results_for_consensus(
        self, raw_results: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Convert raw dispatch results to format expected by consensus calculator."""
        formatted_results: list[dict[str, Any]] = []
        logger.info(f"🔍 Formatting raw_results: {raw_results}")

        # Support either {model_id: response} or dispatcher summary {successful:{}, failed:{}}
        source_mapping: dict[str, Any] = {}
        if isinstance(raw_results, dict) and isinstance(
            raw_results.get("successful"), dict
        ):
            source_mapping = raw_results.get("successful", {})  # type: ignore[assignment]
        elif isinstance(raw_results, dict):
            source_mapping = raw_results
        else:
            source_mapping = {}

        for model_id, response_data in source_mapping.items():
            try:
                content = (
                    self._extract_response_content(response_data)
                    if isinstance(response_data, dict)
                    else ""
                )
                success = isinstance(response_data, dict) and (
                    "error" not in response_data
                )
                # Explicitly mark empty content cases from dispatcher
                if (
                    isinstance(response_data, dict)
                    and response_data.get("empty_content") is True
                ):
                    success = False
                    content = ""
                result_dict = {
                    "model": model_id,
                    "success": success,
                    "content": content,
                    "response_time": 0.0,
                }
                formatted_results.append(result_dict)
                logger.info(
                    f"🔍 Formatted result for {model_id}: success={result_dict['success']}, response_length={len(result_dict['content'])}"
                )
            except Exception as e:
                logger.warning(f"Formatting failed for {model_id}: {e}")

        return formatted_results

    def _extract_response_content(self, response_data: dict) -> str:
        """Extract the actual response content from model response data."""
        try:
            if "error" in response_data:
                return f"Error: {response_data['error']}"

            # Handle LM Studio format
            if "choices" in response_data:
                return response_data["choices"][0]["message"]["content"]

            # Handle Ollama format
            if "response" in response_data:
                return response_data["response"]

            # Fallback: return as JSON string
            import json

            return json.dumps(response_data)

        except Exception as e:
            return f"Failed to extract response: {e}"

    async def get_model_status(self) -> dict[str, Any]:
        """Get status of all models."""
        # Get all discovered models
        models = await self.model_manager.list_models()

        status = {
            "total_models": len(models),
            "online_models": 0,
            "models": {},
        }

        # Check health for each model
        for model_info in models:
            model_status = {
                "name": model_info.name,
                "provider": model_info.provider,
                "endpoint": model_info.endpoint,
                "is_available": model_info.is_available,
            }

            status["models"][model_info.id] = model_status

            if model_info.is_available:
                status["online_models"] += 1

        return status

    async def health_check_all(self) -> dict[str, bool]:
        """Perform health check on all models."""
        health_tasks = []
        model_ids = list(self.model_manager.models.keys())

        for model_id in model_ids:
            task = self.model_manager.health_check(model_id)
            health_tasks.append(task)

        results = await asyncio.gather(*health_tasks, return_exceptions=True)

        health_status = {}
        for model_id, result in zip(model_ids, results, strict=False):
            if isinstance(result, bool):
                health_status[model_id] = result
            else:
                health_status[model_id] = False

        return health_status

    async def _process_request_fast_priority(
        self, request: EnsembleRequest
    ) -> dict[str, Any]:
        """Process request prioritizing fast models (Ollama)."""
        logger.info("⚡ Using fast model priority strategy")

        try:
            # Get available models
            all_models = await self.model_manager.list_models()

            # Only use Ollama models for fast priority
            ollama_models = [m for m in all_models if m.provider == "ollama"]

            # Use available Ollama models (up to min_models)
            selected_models = ollama_models[: request.min_models]

            # Fail fast if we don't have enough Ollama models
            if len(selected_models) < request.min_models:
                raise Exception(
                    f"Insufficient fast models: {len(selected_models)} Ollama models < {request.min_models} required"
                )

            logger.info(
                f"🎯 Fast priority selected (Ollama only): {[m.id for m in selected_models]}"
            )

            # Process with selected models
            async with QueryDispatcher() as dispatcher:
                raw_results = await dispatcher.dispatch_parallel(
                    selected_models, request.task_description, request.context
                )

            # Format and return results
            formatted_results = self._format_results_for_consensus(raw_results)
            consensus_result = self.consensus_calculator.calculate_consensus(
                formatted_results
            )

            return {
                "generated_code": consensus_result.consensus,
                "confidence": consensus_result.confidence,
                "selected_model": (
                    selected_models[0].id if selected_models else "unknown"
                ),
                "strategy": "fast_priority",
                "models_used": [m.id for m in selected_models],
                "success": True,
            }

        except Exception as e:
            logger.error(f"❌ Fast priority failed: {e}")
            return {
                "generated_code": f"Error: {str(e)}",
                "confidence": 0.0,
                "selected_model": "unknown",
                "strategy": "fast_priority_failed",
                "success": False,
            }

    async def process_request_with_fallback(
        self, task_description: str
    ) -> dict[str, Any]:
        """Process request with intelligent fallback strategies - ALL MODELS."""
        logger.info("🔄 Using intelligent fallback strategy with all available models")

        # Try standard processing first (all models) with timeout
        try:
            result = await asyncio.wait_for(
                self.process_request(
                    task_description, timeout=30, prefer_fast_models=False
                ),
                timeout=45.0,  # Add extra timeout wrapper
            )
            if result.get("success"):
                result["strategy"] = "standard_ensemble"
                return result
        except TimeoutError:
            logger.warning("⏰ Standard processing timed out, trying fallback")
        except Exception as e:
            logger.warning(f"Standard processing failed: {e}")

        # Fallback: try with reduced model requirements
        try:
            logger.info("🔄 Trying fallback with reduced requirements")

            # Get available models with timeout
            all_models = await asyncio.wait_for(
                self.model_manager.list_models(), timeout=10.0
            )
            healthy_models = await asyncio.wait_for(
                self.model_manager.list_healthy_models(), timeout=10.0
            )
            available_models = [m for m in all_models if m.id in healthy_models]

            if not available_models:
                raise Exception("No healthy models available")

            # Create a custom request with reduced requirements
            request = EnsembleRequest(
                task_description=task_description,
                timeout=60,
                min_models=1,  # Reduce to 1 model if needed
            )

            # Process with available models with timeout
            async with QueryDispatcher() as dispatcher:
                raw_results = await asyncio.wait_for(
                    dispatcher.dispatch_parallel(
                        available_models, request.task_description, request.context
                    ),
                    timeout=60.0,
                )

            # Format and return results
            formatted_results = self._format_results_for_consensus(raw_results)
            consensus_result = self.consensus_calculator.calculate_consensus(
                formatted_results
            )

            return {
                "generated_code": consensus_result.consensus,
                "confidence": consensus_result.confidence,
                "selected_model": (
                    available_models[0].id if available_models else "unknown"
                ),
                "strategy": "fallback_reduced_requirements",
                "models_used": [m.id for m in available_models],
                "success": True,
            }

        except TimeoutError:
            logger.warning("⏰ Fallback with reduced requirements timed out")
        except Exception as e:
            logger.warning(f"Fallback with reduced requirements failed: {e}")

        # Final fallback: try with just one model
        try:
            logger.info("🔄 Trying single model fallback")

            # Get any available model with timeout
            all_models = await asyncio.wait_for(
                self.model_manager.list_models(), timeout=10.0
            )
            if not all_models:
                raise Exception("No models available")

            # Use the first available model
            single_model = all_models[0]

            request = EnsembleRequest(
                task_description=task_description,
                timeout=60,
                min_models=1,
            )

            async with QueryDispatcher() as dispatcher:
                raw_results = await asyncio.wait_for(
                    dispatcher.dispatch_parallel(
                        [single_model], request.task_description, request.context
                    ),
                    timeout=60.0,
                )

            # Extract response from single model
            if raw_results and single_model.id in raw_results:
                response_data = raw_results[single_model.id]
                generated_code = self._extract_response_content(response_data)

                return {
                    "generated_code": generated_code,
                    "confidence": 0.5,  # Medium confidence for single model
                    "selected_model": single_model.id,
                    "strategy": "single_model_fallback",
                    "models_used": [single_model.id],
                    "success": True,
                }
            else:
                raise Exception("No response from single model")

        except TimeoutError:
            logger.warning("⏰ Single model fallback timed out")
        except Exception as e:
            logger.warning(f"Single model fallback failed: {e}")

        # Final fallback: return error
        return {
            "generated_code": "All fallback strategies failed",
            "confidence": 0.0,
            "selected_model": "none",
            "strategy": "all_failed",
            "success": False,
        }

    def _convert_response_to_dict(self, response: EnsembleResponse) -> dict[str, Any]:
        """Convert EnsembleResponse to dictionary format."""
        # Extract the actual content from consensus
        if isinstance(response.consensus, dict) and "content" in response.consensus:
            generated_code = response.consensus["content"]
        else:
            generated_code = str(response.consensus)

        return {
            "generated_code": generated_code,
            "confidence": response.confidence,
            "selected_model": response.selected_model,
            "execution_time": response.execution_time,
            "strategy": "standard",
            "models_used": len(response.model_responses),  # Count of models used
            "disagreements": response.disagreements,  # Include disagreements list
            "success": True,
        }
