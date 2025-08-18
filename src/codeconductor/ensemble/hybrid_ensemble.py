#!/usr/bin/env python3
"""
Hybrid Ensemble for CodeConductor MVP
Intelligently combines local and cloud LLMs for optimal results.
"""

import asyncio
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure absolute imports resolve when executed in different contexts
root_src = Path(__file__).resolve().parents[1]
if str(root_src) not in sys.path:
    sys.path.insert(0, str(root_src))

from codeconductor.ensemble.complexity_analyzer import (
    ComplexityAnalyzer,
    ComplexityResult,
)
from codeconductor.ensemble.consensus_calculator import ConsensusCalculator
from codeconductor.ensemble.model_manager import ModelManager
from codeconductor.ensemble.query_dispatcher import QueryDispatcher
from codeconductor.integrations.cloud_escalator import CloudEscalator, CloudResponse

logger = logging.getLogger(__name__)


@dataclass
class HybridResult:
    """Result from hybrid ensemble."""

    task: str
    local_responses: dict[str, Any]
    cloud_responses: list[CloudResponse]
    final_consensus: Any
    complexity_analysis: ComplexityResult
    total_cost: float
    total_time: float
    escalation_used: bool
    escalation_reason: str
    local_confidence: float
    cloud_confidence: float


class HybridEnsemble:
    """Intelligently combines local and cloud LLMs."""

    def __init__(self):
        self.complexity_analyzer = ComplexityAnalyzer()
        self.model_manager = ModelManager()
        self.query_dispatcher = QueryDispatcher(self.model_manager)
        self.consensus_calculator = ConsensusCalculator()
        self.cloud_escalator = CloudEscalator()

        # Performance tuning - TEMPORARY FIX: Use only mistral
        self.local_timeout = 15  # Reduced timeout for faster response
        self.cloud_timeout = 30
        self.min_local_confidence = 0.6  # Lower threshold for escalation
        self.max_local_models = 1  # TEMPORARY: Use only 1 model (mistral)

    async def process_task(
        self, task: str, context: dict | None = None
    ) -> HybridResult:
        """
        Process task using hybrid local + cloud approach.

        Args:
            task: The development task
            context: Optional context information

        Returns:
            HybridResult with combined local and cloud responses
        """
        start_time = time.time()

        try:
            complexity = self.complexity_analyzer.analyze_complexity(task, context)
            local_responses = await self._try_local_models_optimized(task, complexity)
            local_confidence = self._calculate_local_confidence(local_responses)

            # Step 3: Enhanced escalation decision with safety checks
            escalation_decision = self._make_escalation_decision(
                complexity, local_confidence, task
            )

            cloud_responses = []
            total_cost = 0.0
            escalation_reason = ""

            if escalation_decision["should_escalate"]:
                escalation_reason = escalation_decision["reason"]
                logger.info(f"☁️ Escalating to cloud: {escalation_reason}")

                # Safety check: only escalate if cloud APIs are available
                if not self.cloud_escalator.is_available():
                    logger.warning("⚠️ Cloud APIs not available, skipping escalation")
                    escalation_reason = "Cloud APIs not available"
                    escalation_decision["should_escalate"] = False

                if self.cloud_escalator.is_available():
                    # Get suggested cloud models
                    complexity.suggested_models[:2]  # Use top 2

                    # Escalate to cloud with timeout
                    try:
                        cloud_responses = await asyncio.wait_for(
                            self.cloud_escalator.escalate_task(
                                task, local_confidence=local_confidence
                            ),
                            timeout=self.cloud_timeout,
                        )
                        total_cost = sum(response.cost for response in cloud_responses)
                        logger.info(
                            f"✅ Cloud escalation complete: {len(cloud_responses)} responses, cost: ${total_cost:.4f}"
                        )
                    except TimeoutError:
                        logger.warning("⏰ Cloud escalation timed out")
                        escalation_reason += " (timed out)"
                    except Exception as e:
                        logger.error(f"❌ Cloud escalation failed: {e}")
                        escalation_reason += f" (error: {str(e)[:50]})"
                        cloud_responses = []
                else:
                    escalation_reason += " (cloud APIs not available)"
                    logger.warning(
                        "⚠️ Cloud escalation requested but APIs not available"
                    )
            else:
                escalation_reason = "Local models sufficient"
                logger.info(f"🏠 Using local models only: {escalation_reason}")

            # Step 4: Combine results and calculate consensus
            final_consensus = await self._combine_results(
                task, local_responses, cloud_responses
            )

            total_time = time.time() - start_time
            cloud_confidence = self._calculate_cloud_confidence(cloud_responses)

            result = HybridResult(
                task=task,
                local_responses=local_responses,
                cloud_responses=cloud_responses,
                final_consensus=final_consensus,
                complexity_analysis=complexity,
                total_cost=total_cost,
                total_time=total_time,
                escalation_used=bool(cloud_responses),
                escalation_reason=escalation_reason,
                local_confidence=local_confidence,
                cloud_confidence=cloud_confidence,
            )

            logger.info(
                f"🎉 Hybrid processing complete: {total_time:.2f}s, cost: ${total_cost:.4f}"
            )
            return result
        except Exception as e:
            print(f"🚨 EXCEPTION in process_task: {e}", flush=True)
            raise

    async def _try_local_models_optimized(
        self, task: str, complexity: ComplexityResult
    ) -> dict[str, Any]:
        """Try local models with optimized settings."""
        try:
            # Adaptive model selection based on complexity
            if complexity.level.value == "simple":
                max_models = min(2, self.max_local_models)
            elif complexity.level.value == "moderate":
                max_models = min(
                    2, self.max_local_models
                )  # Reduced for better performance
            else:
                max_models = min(
                    1, self.max_local_models
                )  # Single model for complex tasks

            # Use faster timeout for local models
            responses = await asyncio.wait_for(
                self.query_dispatcher.dispatch(task, max_models=max_models),
                timeout=self.local_timeout,
            )

            logger.info(f"🏠 Local models responded: {len(responses)}")
            return responses

        except TimeoutError:
            logger.warning("⏰ Local models timed out - trying with single model")
            # Fallback: try with just one model
            try:
                responses = await asyncio.wait_for(
                    self.query_dispatcher.dispatch(task, max_models=1),
                    timeout=self.local_timeout,
                )
                logger.info(f"🏠 Fallback single model responded: {len(responses)}")
                return responses
            except TimeoutError:
                logger.error("⏰ Even single model timed out")
                return {}
        except Exception as e:
            logger.error(f"❌ Local models failed: {e}")
            return {}

    def _calculate_local_confidence(self, local_responses: dict[str, Any]) -> float:
        """Calculate confidence from local responses with enhanced logic."""
        if not local_responses:
            return 0.0

        # Enhanced confidence calculation
        valid_responses = 0
        total_responses = len(local_responses)
        response_quality = 0.0

        for _model_id, response in local_responses.items():
            if isinstance(response, dict) and "error" not in response:
                valid_responses += 1

                # Assess response quality
                content = ""
                if "choices" in response and response["choices"]:
                    content = response["choices"][0]["message"]["content"]
                elif "response" in response:
                    content = response["response"]
                else:
                    content = str(response)

                # Quality scoring based on content length and structure
                if len(content) > 100:  # Substantial response
                    response_quality += 0.3
                if "def " in content or "class " in content:  # Contains code
                    response_quality += 0.4
                if "import " in content or "from " in content:  # Proper imports
                    response_quality += 0.2
                if "test" in content.lower() or "assert" in content:  # Contains tests
                    response_quality += 0.1

        base_confidence = (
            valid_responses / total_responses if total_responses > 0 else 0.0
        )
        quality_bonus = min(response_quality / total_responses, 0.3)  # Max 30% bonus

        final_confidence = min(base_confidence + quality_bonus, 1.0)
        return final_confidence

    def _make_escalation_decision(
        self, complexity: ComplexityResult, local_confidence: float, task: str
    ) -> dict[str, Any]:
        """Make intelligent escalation decision."""

        # Base escalation logic
        should_escalate = False
        reason = ""

        # 1. Complexity-based escalation
        if complexity.requires_cloud:
            should_escalate = True
            reason = (
                f"Task requires cloud models (complexity: {complexity.level.value})"
            )

        # 2. Confidence-based escalation
        elif local_confidence < self.min_local_confidence:
            should_escalate = True
            reason = f"Low local confidence ({local_confidence:.2f} < {self.min_local_confidence})"

        # 3. Expert-level tasks
        elif complexity.level.value == "expert":
            should_escalate = True
            reason = "Expert-level task detected"

        # 4. Check if cloud is available
        if should_escalate and not self.cloud_escalator.is_available():
            should_escalate = False
            reason += " (but cloud APIs not available)"

        return {
            "should_escalate": should_escalate,
            "reason": reason,
            "complexity_level": complexity.level.value,
            "local_confidence": local_confidence,
            "cloud_available": self.cloud_escalator.is_available(),
        }

    def _calculate_cloud_confidence(
        self, cloud_responses: list[CloudResponse]
    ) -> float:
        """Calculate confidence from cloud responses."""
        if not cloud_responses:
            return 0.0

        # Cloud responses typically have high confidence
        total_confidence = sum(response.confidence for response in cloud_responses)
        return total_confidence / len(cloud_responses)

    async def _combine_results(
        self,
        task: str,
        local_responses: dict[str, Any],
        cloud_responses: list[CloudResponse],
    ) -> Any:
        """Combine local and cloud results for final consensus."""
        all_responses = []

        # Add local responses - PASS ORIGINAL FORMAT TO CONSENSUS CALCULATOR
        for _model_id, response in local_responses.items():
            if isinstance(response, dict) and "error" not in response:
                # Pass the original OpenAI format directly to consensus calculator
                all_responses.append(response)

        # Add cloud responses (convert to OpenAI format)
        for cloud_response in cloud_responses:
            # Convert cloud response to OpenAI format
            openai_format = {
                "choices": [{"message": {"content": cloud_response.content}}],
                "model": cloud_response.model,
                "confidence": cloud_response.confidence,
            }
            all_responses.append(openai_format)

        # Calculate consensus
        if all_responses:
            consensus = self.consensus_calculator.calculate_consensus(all_responses)
            return consensus
        else:
            # Fallback consensus
            return {
                "task": task,
                "approach": "Follow best practices",
                "requirements": ["Implement the requested functionality"],
                "language": "python",
            }

    async def estimate_cost(self, task: str) -> dict[str, float]:
        """Estimate cost for hybrid processing."""
        complexity = self.complexity_analyzer.analyze_complexity(task)

        # Local cost is always 0
        local_cost = 0.0

        # Cloud cost estimation
        cloud_cost = 0.0
        if complexity.requires_cloud and self.cloud_escalator.is_available():
            cloud_models = complexity.suggested_models[:2]
            cloud_costs = self.cloud_escalator.estimate_cost(task, cloud_models)
            cloud_cost = sum(cloud_costs.values())

        return {
            "local": local_cost,
            "cloud": cloud_cost,
            "total": local_cost + cloud_cost,
            "escalation_likely": complexity.requires_cloud,
        }

    async def get_status(self) -> dict[str, Any]:
        """Get status of hybrid ensemble components."""
        try:
            local_models = await self.model_manager.list_models()
            local_count = len(local_models) if local_models else 0
        except Exception:
            local_count = 0

        return {
            "local_models": local_count,
            "cloud_available": self.cloud_escalator.is_available(),
            "complexity_analyzer": "ready",
            "consensus_calculator": "ready",
            "local_timeout": self.local_timeout,
            "cloud_timeout": self.cloud_timeout,
            "min_local_confidence": self.min_local_confidence,
        }

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance tuning metrics."""
        return {
            "local_timeout": self.local_timeout,
            "cloud_timeout": self.cloud_timeout,
            "min_local_confidence": self.min_local_confidence,
            "max_local_models": self.max_local_models,
        }


# Convenience functions
async def process_with_hybrid_ensemble(
    task: str, context: dict | None = None
) -> HybridResult:
    """Process task with hybrid ensemble."""
    ensemble = HybridEnsemble()
    return await ensemble.process_task(task, context)


async def estimate_hybrid_cost(task: str) -> dict[str, float]:
    """Estimate cost for hybrid processing."""
    ensemble = HybridEnsemble()
    return await ensemble.estimate_cost(task)


if __name__ == "__main__":
    # Demo
    async def demo():
        ensemble = HybridEnsemble()

        # Test tasks of different complexity
        test_tasks = [
            "Create a simple calculator class with basic operations",
            "Implement a secure authentication system with JWT tokens",
            "Design a distributed microservices architecture for e-commerce",
            "Fix the bug in the login function",
        ]

        print("🎯 CodeConductor Hybrid Ensemble Demo")
        print("=" * 50)

        for task in test_tasks:
            print(f"\n📝 Task: {task}")

            # Estimate cost
            cost_estimate = await ensemble.estimate_cost(task)
            print(f"💰 Cost estimate: ${cost_estimate['total']:.4f}")
            print(f"☁️ Escalation likely: {cost_estimate['escalation_likely']}")

            # Process task
            try:
                result = await ensemble.process_task(task)

                print(f"🏷️  Complexity: {result.complexity_analysis.level.value}")
                print(f"🏠 Local responses: {len(result.local_responses)}")
                print(f"☁️ Cloud responses: {len(result.cloud_responses)}")
                print(f"⏱️  Total time: {result.total_time:.2f}s")
                print(f"💰 Total cost: ${result.total_cost:.4f}")
                print(f"🎯 Escalation used: {result.escalation_used}")
                print(f"📝 Escalation reason: {result.escalation_reason}")

            except Exception as e:
                print(f"❌ Error: {e}")

    asyncio.run(demo())
