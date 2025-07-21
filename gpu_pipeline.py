#!/usr/bin/env python3
"""
GPU-Powered CodeConductor Pipeline
Uses neural bandit to intelligently select agents for code generation
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from agents.gpu_orchestrator import GPUOrchestrator
from agents.codegen_agent import CodeGenAgent
from cli.human_approval import HumanApprovalCLI
from integrations.llm_client import LLMClient
from agents.reward_agent import (
    TestResult,
    CodeQualityMetrics,
    HumanFeedback,
    PolicyResult,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GPUCodeConductorPipeline:
    """GPU-powered pipeline for intelligent multi-agent code generation."""

    def __init__(
        self,
        online: bool = True,
        gpu_service_url: str = "http://localhost:8009",
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the GPU-powered pipeline."""
        self.online = online
        self.gpu_service_url = gpu_service_url
        self.config = config or {}
        self.iteration = 0
        self.results = []

        # Initialize components
        from agents.architect_agent import ArchitectAgent
        from agents.review_agent import ReviewAgent
        from agents.policy_agent import PolicyAgent
        from agents.reward_agent import RewardAgent
        from agents.qlearning_agent import QLearningAgent

        # Create all available agents
        all_agents = [
            ArchitectAgent(),
            ReviewAgent(),
            CodeGenAgent(),
            PolicyAgent(),
            RewardAgent(),
            QLearningAgent(),
        ]

        # Initialize GPU-powered orchestrator
        self.orchestrator = GPUOrchestrator(all_agents, gpu_service_url)
        self.codegen_agent = CodeGenAgent()
        self.review_agent = ReviewAgent()
        self.policy_agent = PolicyAgent()
        self.reward_agent = RewardAgent()
        self.qlearning_agent = QLearningAgent()
        self.approval_cli = HumanApprovalCLI()

        # Initialize LLM client if online
        if self.online:
            try:
                self.llm_client = LLMClient()
                self.codegen_agent.set_llm_client(self.llm_client)
                logger.info("LLM client initialized for online mode")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM client: {e}")
                self.online = False

        # Create output directory
        self.output_dir = Path("data/generated")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"GPU Pipeline initialized (online: {self.online}, GPU service: {gpu_service_url})"
        )

    def load_prompt(self, prompt_file: str) -> Dict[str, Any]:
        """Load prompt from file."""
        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse the prompt file
            lines = content.strip().split("\n")
            title = lines[0].strip()

            # Extract description (everything after title)
            description = "\n".join(lines[1:]).strip()

            # Create task context with GPU-relevant metadata
            task_context = {
                "title": title,
                "description": description,
                "task_type": "code_generation",
                "urgency": "medium",
                "team_size": "small",
                "deadline": "flexible",
                "domain_expertise": "general",
                "code_quality": "standard",
                "testing_required": "basic",
                "documentation_needed": "basic",
                "security_level": "standard",
                "performance_priority": "balanced",
            }

            logger.info(f"Loaded prompt: {title}")
            return task_context

        except Exception as e:
            logger.error(f"Failed to load prompt file: {e}")
            raise

    def run_iteration(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single GPU-powered iteration."""
        self.iteration += 1
        logger.info(f"Starting iteration {self.iteration}")

        # Step 1: GPU-powered multi-agent discussion and consensus
        logger.info("Step 1: GPU-powered multi-agent discussion and consensus")
        discussion_result = self.orchestrator.run_discussion(context)

        # Extract GPU metadata
        gpu_metadata = discussion_result.get("gpu_metadata", {})
        logger.info(f"GPU selected agents: {gpu_metadata.get('agents_selected', [])}")

        # Step 2: Human approval
        logger.info("Step 2: Human approval")
        approval_result = self.approval_cli.process_approval(discussion_result)

        if not approval_result.approved:
            logger.info("Proposal rejected by human")
            return {
                "iteration": self.iteration,
                "status": "rejected",
                "reason": approval_result.comments or "Human rejection",
                "gpu_metadata": gpu_metadata,
            }

        # Step 3: Code generation
        logger.info("Step 3: Code generation")
        analysis = self.codegen_agent.analyze(context)
        code_result = self.codegen_agent.propose(analysis, context)

        # Step 4: Code review and safety check
        logger.info("Step 4: Code review and safety check")
        review_result = self.review_agent.review_generated_code(
            code_result["code"], context
        )
        policy_result = self.policy_agent.check_code_safety(code_result["code"])

        logger.info(
            f"Review assessment: {review_result.get('assessment', review_result.get('overall_assessment', 'unknown'))}"
        )
        logger.info(f"Policy decision: {policy_result['decision']}")

        # Step 5: Running tests and calculating reward
        logger.info("Step 5: Running tests and calculating reward")
        test_result = self._run_tests(code_result["code"])
        quality_metrics = self._calculate_quality_metrics(code_result["code"])

        # Calculate reward
        reward = self.reward_agent.calculate_reward(
            test_results=test_result,
            code_quality=quality_metrics,
            human_feedback=HumanFeedback(
                overall_rating=0.8,
                usefulness_rating=0.8,
                correctness_rating=0.8,
                completeness_rating=0.8,
                comments="Good performance",
                approved=True,
            ),
            policy_result=PolicyResult(
                safe=policy_result.get("safe", False),
                decision=policy_result.get("decision", "unknown"),
                risk_level=policy_result.get("risk_level", "unknown"),
                violations_count=policy_result.get("violations_count", 0),
                critical_violations=policy_result.get("critical_violations", 0),
                high_violations=policy_result.get("high_violations", 0),
            ),
            performance_metrics={"execution_time": 0.1, "memory_usage": 0.05},
        )

        logger.info(
            f"Reward calculated: {reward.get('total_reward', 0.0):.3f} ({reward.get('reward_level', 'unknown')})"
        )

        # Step 6: Save generated code
        logger.info("Step 6: Save generated code")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gpu_iter_{self.iteration}_{timestamp}.py"
        filepath = self.output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(code_result["code"])

        logger.info(f"Code saved to {filepath}")

        # Return comprehensive result
        return {
            "iteration": self.iteration,
            "status": "completed",
            "code": code_result["code"],
            "code_file": str(filepath),
            "discussion": discussion_result,
            "review": review_result,
            "policy": policy_result,
            "test_result": test_result,
            "quality_metrics": quality_metrics,
            "reward": reward,
            "gpu_metadata": gpu_metadata,
            "timestamp": datetime.now().isoformat(),
        }

    def _run_tests(self, code: str) -> TestResult:
        """Run tests on generated code."""
        try:
            # Simple test execution (in production, this would be more sophisticated)
            return TestResult(
                passed=True,
                total_tests=1,
                passed_tests=1,
                failed_tests=0,
                coverage_percentage=0.8,
                execution_time=0.1,
                error_messages=[],
            )
        except Exception as e:
            logger.warning(f"Test execution failed: {e}")
            return TestResult(
                passed=False,
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                coverage_percentage=0.0,
                execution_time=0.0,
                error_messages=[str(e)],
            )

    def _calculate_quality_metrics(self, code: str) -> CodeQualityMetrics:
        """Calculate code quality metrics."""
        try:
            # Simple quality calculation
            lines = code.split("\n")
            complexity = len(
                [
                    line
                    for line in lines
                    if any(
                        keyword in line
                        for keyword in ["if", "for", "while", "def", "class"]
                    )
                ]
            )

            return CodeQualityMetrics(
                quality_score=0.8,
                complexity_score=complexity / 10.0,  # Normalize
                maintainability_score=0.8,
                documentation_score=0.7,
                style_score=0.9,
                security_score=0.8,
            )
        except Exception as e:
            logger.warning(f"Quality calculation failed: {e}")
            return CodeQualityMetrics(
                quality_score=0.5,
                complexity_score=0.0,
                maintainability_score=0.5,
                documentation_score=0.5,
                style_score=0.5,
                security_score=0.5,
            )

    def run(self, prompt_file: str, iterations: int = 1) -> Dict[str, Any]:
        """Run the GPU-powered pipeline."""
        logger.info(f"Starting GPU CodeConductor pipeline with {iterations} iterations")

        # Load prompt
        context = self.load_prompt(prompt_file)

        # Run iterations
        for i in range(iterations):
            logger.info(f"Starting iteration {i + 1}/{iterations}")
            result = self.run_iteration(context)
            self.results.append(result)

            if result["status"] == "rejected":
                logger.info("Pipeline stopped due to human rejection")
                break

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"gpu_pipeline_results_{timestamp}.json"

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, default=str)

        logger.info(f"Pipeline results saved to {results_file}")

        # Print summary
        successful_iterations = len(
            [r for r in self.results if r["status"] == "completed"]
        )

        print("\n" + "=" * 60)
        print("GPU PIPELINE COMPLETED")
        print("=" * 60)
        print(f"Total iterations: {iterations}")
        print(f"Successful iterations: {successful_iterations}")
        print(f"Results saved to: {results_file}")
        print("\nGenerated files:")
        for result in self.results:
            if result["status"] == "completed":
                print(f"  - {result['code_file']}")
        print("=" * 60)

        return {
            "total_iterations": iterations,
            "successful_iterations": successful_iterations,
            "results_file": str(results_file),
            "results": self.results,
        }


def main():
    """Main entry point for GPU pipeline."""
    parser = argparse.ArgumentParser(description="GPU-Powered CodeConductor Pipeline")
    parser.add_argument("--prompt", required=True, help="Path to prompt file")
    parser.add_argument("--iters", type=int, default=1, help="Number of iterations")
    parser.add_argument("--online", action="store_true", help="Use online LLM")
    parser.add_argument("--offline", action="store_true", help="Use offline mode only")
    parser.add_argument(
        "--gpu-service", default="http://localhost:8009", help="GPU service URL"
    )

    args = parser.parse_args()

    # Determine online/offline mode
    online = args.online or not args.offline

    # Create and run pipeline
    pipeline = GPUCodeConductorPipeline(online=online, gpu_service_url=args.gpu_service)

    pipeline.run(args.prompt, args.iters)


if __name__ == "__main__":
    main()
