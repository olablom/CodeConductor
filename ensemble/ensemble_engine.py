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

    def __init__(self, min_confidence: float = 0.7, use_rlhf: bool = True):
        self.model_manager = ModelManager()
        self.consensus_calculator = ConsensusCalculator()
        self.min_confidence = min_confidence
        self.use_rlhf = use_rlhf and RLHF_AVAILABLE
        self.rlhf_agent = None

        if self.use_rlhf:
            self._initialize_rlhf()

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

    async def process_request(self, request: EnsembleRequest) -> EnsembleResponse:
        """Process a task request through the ensemble."""
        start_time = asyncio.get_event_loop().time()

        logger.info(f"Processing ensemble request: {request.task_description[:50]}...")

        try:
            # Get available models
            all_models = await self.model_manager.list_models()
            healthy_models = await self.model_manager.list_healthy_models()

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
            models = {
                model.id: model
                for model in available_models
                if model.id in selected_model_ids
            }

            # Dispatch queries in parallel
            async with QueryDispatcher(timeout=request.timeout) as dispatcher:
                raw_results = await dispatcher.dispatch_parallel(
                    models, request.task_description, request.context
                )

            # Convert raw results to format expected by consensus calculator
            formatted_results = self._format_results_for_consensus(raw_results)

            # Calculate consensus
            consensus_result = self.consensus_calculator.calculate_consensus(
                formatted_results
            )

            # Update model statistics
            for result in formatted_results:
                if result.success:
                    self.model_manager.update_model_stats(
                        result.model_id, True, result.response_time
                    )
                else:
                    self.model_manager.update_model_stats(
                        result.model_id, False, result.response_time
                    )

            execution_time = asyncio.get_event_loop().time() - start_time

            response = EnsembleResponse(
                consensus=consensus_result.consensus,
                confidence=consensus_result.confidence,
                disagreements=consensus_result.disagreements,
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

    def _format_results_for_consensus(self, raw_results: Dict[str, Any]) -> List[Any]:
        """Convert raw dispatch results to format expected by consensus calculator."""
        formatted_results = []

        for model_id, response_data in raw_results.items():
            # Create a result object with success/response attributes
            result_obj = type(
                "Result",
                (),
                {
                    "model_id": model_id,
                    "success": "error" not in response_data,
                    "response": self._extract_response_content(response_data),
                    "response_time": 0.0,  # We don't track this in ensemble engine
                },
            )()
            formatted_results.append(result_obj)

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

    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models."""
        status = {
            "total_models": len(self.model_manager.models),
            "online_models": 0,
            "models": {},
        }

        for model_id, model in self.model_manager.models.items():
            model_status = {
                "name": model.name,
                "status": model.status.value,
                "response_time": model.response_time,
                "success_rate": model.success_rate,
                "capabilities": model.capabilities,
            }

            status["models"][model_id] = model_status

            if model.status.value == "online":
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
