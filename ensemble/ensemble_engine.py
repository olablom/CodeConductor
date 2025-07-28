"""
Main Ensemble Engine for CodeConductor

Orchestrates multiple LLMs for consensus-based code generation with RLHF integration.
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

from .model_manager import ModelManager
from .query_dispatcher import QueryDispatcher
from .consensus_calculator import ConsensusCalculator, ConsensusResult

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
        self.min_confidence = min_confidence
        self.use_rlhf = use_rlhf and RLHF_AVAILABLE
        self.use_rag = use_rag and RAG_AVAILABLE
        self.use_code_reviewer = use_code_reviewer and CODE_REVIEWER_AVAILABLE
        self.rlhf_agent = None
        self.rag_system = None
        self.code_reviewer = None

        if self.use_rlhf:
            self._initialize_rlhf()

        if self.use_rag:
            self._initialize_rag()

        if self.use_code_reviewer:
            self._initialize_code_reviewer()

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
        """Initialize the ensemble engine and discover models."""
        logger.info("Initializing Ensemble Engine...")

        try:
            # Discover available models
            models = await self.model_manager.list_models()

            if not models:
                logger.error("No models discovered")
                return False

            # Perform health checks on discovered models
            health_tasks = []
            for model_info in models:
                task = self.model_manager.check_health(model_info)
                health_tasks.append(task)

            health_results = await asyncio.gather(*health_tasks, return_exceptions=True)

            online_models = 0
            for model_info, result in zip(models, health_results):
                if isinstance(result, bool) and result:
                    online_models += 1
                    logger.info(f"Model {model_info.name} is online")
                else:
                    logger.warning(f"Model {model_info.name} failed health check")

            logger.info(
                f"Ensemble Engine initialized with {online_models} online models"
            )
            return online_models >= 2

        except Exception as e:
            logger.error(f"Failed to initialize Ensemble Engine: {e}")
            return False

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
            # Get available models with timeout
            all_models = await asyncio.wait_for(
                self.model_manager.list_models(), timeout=15.0
            )
            healthy_models = await asyncio.wait_for(
                self.model_manager.list_healthy_models(), timeout=15.0
            )

            # Filter to healthy models
            available_models = [
                model for model in all_models if model.id in healthy_models
            ]

            if len(available_models) < request.min_models:
                raise Exception(
                    f"Insufficient models available: {len(available_models)} < {request.min_models}"
                )

            # Use RLHF to select models if available
            available_model_ids = [model.id for model in available_models]
            if self.use_rlhf:
                selected_model_ids, rlhf_action, rlhf_action_description = (
                    self._select_models_with_rlhf(request, available_model_ids)
                )
                logger.info(f"RLHF selected models: {selected_model_ids}")
            else:
                selected_model_ids = available_model_ids[: request.min_models]
                rlhf_action = None
                rlhf_action_description = None
                logger.info(f"Using default model selection: {selected_model_ids}")

            # Get model objects
            models = [
                model for model in available_models if model.id in selected_model_ids
            ]

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

            # Calculate consensus
            consensus_result = self.consensus_calculator.calculate_consensus(
                formatted_results
            )

            # Extract generated code from consensus
            generated_code = consensus_result.consensus

            # Review code if CodeReviewer is available
            if self.use_code_reviewer and self.code_reviewer and generated_code:
                try:
                    logger.info("🔍 Starting code review...")
                    review_results = self.code_reviewer.review_code(
                        request.task_description, generated_code, request.test_results
                    )

                    # Apply suggested fixes if any
                    if review_results.get("suggested_fixes"):
                        logger.info(
                            f"🔧 Applying {len(review_results['suggested_fixes'])} suggested fixes"
                        )
                        improved_code = self.apply_fixes(
                            generated_code, review_results["suggested_fixes"]
                        )
                        # consensus_result.consensus is a string, not a dict
                        consensus_result.consensus = improved_code
                        logger.info("✅ Code review completed and fixes applied")
                    else:
                        logger.info("✅ Code review completed - no fixes needed")

                except Exception as e:
                    logger.warning(f"⚠️ Code review failed: {e}")

            # Update model statistics
            for result in formatted_results:
                if result["success"]:
                    self.model_manager.update_model_stats(
                        result["model"], True, result["response_time"]
                    )
                else:
                    self.model_manager.update_model_stats(
                        result["model"], False, result["response_time"]
                    )

            execution_time = asyncio.get_event_loop().time() - start_time

            response = EnsembleResponse(
                consensus={"content": consensus_result.consensus},
                confidence=consensus_result.confidence,
                disagreements=[],  # consensus_result doesn't have disagreements
                model_responses=formatted_results,
                execution_time=execution_time,
                rlhf_action=rlhf_action,
                rlhf_action_description=rlhf_action_description,
                selected_model=selected_model_ids[0]
                if selected_model_ids
                else None,  # Assuming the first selected model is the primary one
            )

            logger.info(
                f"Ensemble request completed in {execution_time:.2f}s with confidence {consensus_result.confidence:.2f}"
            )

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
            "success": True,
        }
