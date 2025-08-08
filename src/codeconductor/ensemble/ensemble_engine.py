"""
Main Ensemble Engine for CodeConductor

Orchestrates multiple LLMs for consensus-based code generation with RLHF integration.
"""

import asyncio
import logging
import numpy as np
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

from .model_manager import ModelManager
from .model_selector import ModelSelector, SelectionInput
from .query_dispatcher import QueryDispatcher
from .consensus_calculator import ConsensusCalculator, ConsensusResult
from .code_reviewer import CodeReviewer
from codeconductor.feedback.rlhf_agent import RLHFAgent
from codeconductor.context.rag_system import RAGSystem
from codeconductor.monitoring.memory_watchdog import (
    start_memory_watchdog,
    stop_memory_watchdog,
    get_memory_watchdog,
)

# Try to import RLHF components
try:
    from feedback.rlhf_agent import RLHFAgent

    RLHF_AVAILABLE = True
except ImportError:
    RLHF_AVAILABLE = False
    print("⚠️ RLHF not available. Install with: pip install stable-baselines3 gymnasium")

# Try to import RAG components
try:
    from context.rag_system import RAGSystem

    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("⚠️ RAG not available. Install with: pip install langchain chromadb")

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
    context: Optional[Dict[str, Any]] = None
    min_models: int = 2
    timeout: float = 30.0
    test_results: Optional[List[Dict[str, Any]]] = None
    code_quality: float = 0.5
    user_feedback: float = 0.0


@dataclass
class EnsembleResponse:
    consensus: Dict[str, Any]
    confidence: float
    disagreements: List[str]
    model_responses: List[Dict[str, Any]]
    execution_time: float
    rlhf_action: Optional[int] = None
    rlhf_action_description: Optional[str] = None
    selected_model: Optional[str] = None


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
        self.use_rlhf = use_rlhf and RLHF_AVAILABLE
        self.use_rag = use_rag and RAG_AVAILABLE
        self.use_code_reviewer = use_code_reviewer and CODE_REVIEWER_AVAILABLE
        self.rlhf_agent = None
        self.rag_system = None
        self.code_reviewer = None
        # Telemetry snapshots
        self.last_selector_decision: Dict[str, Any] = {}
        self.last_cache_hit: bool = False
        self.last_artifacts_dir: Optional[str] = None
        self.last_export_path: Optional[str] = None

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
        selector: Dict[str, Any],
        consensus_meta: Dict[str, Any],
        consensus_obj: Optional[ConsensusResult] = None,
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

        # consensus
        consensus_dump = {
            "method": consensus_meta.get("method"),
            "winner": consensus_meta.get("winner"),
            "cached": consensus_meta.get("cached", False),
        }
        if consensus_obj is not None:
            consensus_dump.update(
                {
                    "confidence": consensus_obj.confidence,
                    "code_quality": consensus_obj.code_quality_score,
                    "syntax_valid": consensus_obj.syntax_valid,
                }
            )
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
        try:
            return self.response_cache.make_key(
                prompt=prompt,
                persona=persona,
                policy=policy,
                model=model,
                params=sampling,
            )
        except Exception:
            return ""

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
        self, request: EnsembleRequest, available_models: List[str]
    ) -> tuple[List[str], Optional[int], Optional[str]]:
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

    def _calculate_test_reward(self, test_results: List[Dict[str, Any]]) -> float:
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

    def apply_fixes(self, code: str, fixes: List[str]) -> str:
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
    ) -> Dict[str, Any]:
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
                selected_model_ids = loaded_model_ids[:max_models]
                logger.info(
                    f"✅ Using {len(selected_model_ids)} memory-safe loaded models (voting): {selected_model_ids}"
                )
            else:
                # Fallback to available models
                fallback_k = min(3, len(available_model_ids))
                selected_model_ids = available_model_ids[:fallback_k]
                logger.info(f"⚠️ Using fallback models: {selected_model_ids}")

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

            # Calculate consensus (majority/best-score handled inside calculator)
            consensus_result = self.consensus_calculator.calculate_consensus(
                formatted_results
            )

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
                self._save_run_artifacts(
                    prompt_text or "",
                    policy,
                    self.last_selector_decision,
                    {
                        "method": "codebleu+heuristic",
                        "winner": {
                            "model": models[0].id if models else None,
                            "score": consensus_result.confidence,
                        },
                        "cached": False,
                    },
                    consensus_result,
                )
            except Exception:
                pass

            # Extract generated code from consensus
            generated_code = consensus_result.consensus

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

            # Create response
            response = EnsembleResponse(
                consensus={
                    "code": generated_code,
                    "explanation": consensus_result.explanation,
                    "confidence": consensus_result.confidence,
                    "model_agreement": consensus_result.model_agreement,
                },
                confidence=consensus_result.confidence,
                disagreements=consensus_result.disagreements,
                model_responses=formatted_results,
                execution_time=execution_time,
            )

            logger.info(f"✅ Ensemble request completed in {execution_time:.2f}s")
            return response

        except Exception as e:
            logger.error(f"Ensemble request failed: {e}")
            execution_time = asyncio.get_event_loop().time() - start_time

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
        self, raw_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Convert raw dispatch results to format expected by consensus calculator."""
        formatted_results = []
        logger.info(f"🔍 Formatting raw_results: {raw_results}")

        for model_id, response_data in raw_results.items():
            # Create a dictionary with content key as expected by consensus calculator
            result_dict = {
                "model": model_id,
                "success": "error" not in response_data,
                "content": self._extract_response_content(response_data),
                "response_time": 0.0,  # We don't track this in ensemble engine
            }
            formatted_results.append(result_dict)
            logger.info(
                f"🔍 Formatted result for {model_id}: success={result_dict['success']}, response_length={len(result_dict['content'])}"
            )

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

    async def get_model_status(self) -> Dict[str, Any]:
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

    async def health_check_all(self) -> Dict[str, bool]:
        """Perform health check on all models."""
        health_tasks = []
        model_ids = list(self.model_manager.models.keys())

        for model_id in model_ids:
            task = self.model_manager.health_check(model_id)
            health_tasks.append(task)

        results = await asyncio.gather(*health_tasks, return_exceptions=True)

        health_status = {}
        for model_id, result in zip(model_ids, results):
            if isinstance(result, bool):
                health_status[model_id] = result
            else:
                health_status[model_id] = False

        return health_status

    async def _process_request_fast_priority(
        self, request: EnsembleRequest
    ) -> Dict[str, Any]:
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
                "selected_model": selected_models[0].id
                if selected_models
                else "unknown",
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
    ) -> Dict[str, Any]:
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
        except asyncio.TimeoutError:
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
                "selected_model": available_models[0].id
                if available_models
                else "unknown",
                "strategy": "fallback_reduced_requirements",
                "models_used": [m.id for m in available_models],
                "success": True,
            }

        except asyncio.TimeoutError:
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

        except asyncio.TimeoutError:
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

    def _convert_response_to_dict(self, response: EnsembleResponse) -> Dict[str, Any]:
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
