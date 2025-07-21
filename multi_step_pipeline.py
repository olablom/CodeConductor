#!/usr/bin/env python3
"""
Multi-Step Reasoning Pipeline
Cutting-edge AI orchestration with chain-of-thought planning
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

from agents.multi_step_orchestrator import MultiStepOrchestrator
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


class MultiStepPipeline:
    """Cutting-edge multi-step reasoning pipeline for intelligent AI orchestration."""

    def __init__(
        self,
        online: bool = True,
        gpu_service_url: str = "http://localhost:8009",
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the multi-step pipeline."""
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

        # Initialize multi-step orchestrator
        self.orchestrator = MultiStepOrchestrator(all_agents, gpu_service_url)
        self.approval_cli = HumanApprovalCLI()

        # Initialize LLM client if online
        if self.online:
            try:
                self.llm_client = LLMClient()
                logger.info("LLM client initialized for online mode")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM client: {e}")
                self.online = False

        # Create output directory
        self.output_dir = Path("data/generated")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Multi-Step Pipeline initialized (online: {self.online}, GPU service: {gpu_service_url})"
        )

    def load_prompt(self, prompt_file: str) -> Dict[str, Any]:
        """Load prompt from file with enhanced context."""
        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse the prompt file
            lines = content.strip().split("\n")
            title = lines[0].strip()

            # Extract description (everything after title)
            description = "\n".join(lines[1:]).strip()

            # Create enhanced task context for multi-step planning
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
                "risk_tolerance": "medium",
                "complexity_level": "moderate",
                "workflow_preferences": {
                    "enable_parallel_execution": True,
                    "enable_adaptive_planning": True,
                    "enable_chain_of_thought": True,
                },
            }

            logger.info(f"Loaded prompt: {title}")
            return task_context

        except Exception as e:
            logger.error(f"Failed to load prompt file: {e}")
            raise

    def run_iteration(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single multi-step iteration with chain-of-thought planning."""
        self.iteration += 1
        logger.info(f"Starting multi-step iteration {self.iteration}")

        # Step 1: Multi-step workflow planning
        logger.info(
            "Step 1: Multi-step workflow planning with chain-of-thought reasoning"
        )
        workflow_plan = self.orchestrator.plan_workflow(context)

        logger.info(f"Workflow planned: {len(workflow_plan.steps)} steps")
        logger.info(
            f"Estimated duration: {workflow_plan.total_estimated_duration:.1f}s"
        )
        logger.info(f"Confidence: {workflow_plan.confidence_score:.2f}")
        logger.info(f"Critical path: {workflow_plan.critical_path}")

        # Step 2: Human approval of workflow plan
        logger.info("Step 2: Human approval of workflow plan")
        plan_summary = {
            "title": context.get("title", "Unknown Task"),
            "summary": f"Multi-step workflow with {len(workflow_plan.steps)} steps",
            "consensus_reached": True,
            "consensus": {
                "decision": "workflow_planned",
                "confidence": workflow_plan.confidence_score,
                "reasoning": f"Planned {len(workflow_plan.steps)} steps with {workflow_plan.confidence_score:.2f} confidence",
            },
            "discussion_summary": {
                "steps_planned": len(workflow_plan.steps),
                "estimated_duration": workflow_plan.total_estimated_duration,
                "risk_level": max(workflow_plan.risk_assessment.values())
                if workflow_plan.risk_assessment
                else 0.0,
            },
        }

        approval_result = self.approval_cli.process_approval(plan_summary)

        if not approval_result.approved:
            logger.info("Workflow plan rejected by human")
            return {
                "iteration": self.iteration,
                "status": "rejected",
                "reason": approval_result.comments or "Human rejection",
                "workflow_plan": workflow_plan,
            }

        # Step 3: Execute multi-step workflow
        logger.info("Step 3: Execute multi-step workflow with adaptive execution")
        execution_result = self.orchestrator.execute_workflow(workflow_plan, context)

        logger.info(
            f"Workflow execution completed: {execution_result['success_rate']:.1%} success rate"
        )
        logger.info(f"Total duration: {execution_result['total_duration']:.1f}s")
        logger.info(f"Adaptations made: {len(execution_result['adaptations_made'])}")

        # Step 4: Process final result
        logger.info("Step 4: Process final result")
        final_result = execution_result.get("final_result", {})

        if not final_result or not final_result.get("final_code"):
            logger.warning("No final code generated")
            return {
                "iteration": self.iteration,
                "status": "failed",
                "reason": "No code generated",
                "execution_result": execution_result,
            }

        # Step 5: Save generated code
        logger.info("Step 5: Save generated code")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"multistep_iter_{self.iteration}_{timestamp}.py"
        filepath = self.output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(final_result["final_code"])

        logger.info(f"Code saved to {filepath}")

        # Return comprehensive result
        return {
            "iteration": self.iteration,
            "status": "completed",
            "code": final_result["final_code"],
            "code_file": str(filepath),
            "workflow_plan": workflow_plan,
            "execution_result": execution_result,
            "final_result": final_result,
            "timestamp": datetime.now().isoformat(),
        }

    def run(self, prompt_file: str, iterations: int = 1) -> Dict[str, Any]:
        """Run the multi-step pipeline."""
        logger.info(f"Starting Multi-Step Pipeline with {iterations} iterations")

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
        results_file = self.output_dir / f"multistep_pipeline_results_{timestamp}.json"

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, default=str)

        logger.info(f"Pipeline results saved to {results_file}")

        # Print summary
        successful_iterations = len(
            [r for r in self.results if r["status"] == "completed"]
        )

        print("\n" + "=" * 60)
        print("MULTI-STEP PIPELINE COMPLETED")
        print("=" * 60)
        print(f"Total iterations: {iterations}")
        print(f"Successful iterations: {successful_iterations}")
        print(f"Results saved to: {results_file}")

        if successful_iterations > 0:
            print("\nWorkflow Statistics:")
            for result in self.results:
                if result["status"] == "completed":
                    execution = result.get("execution_result", {})
                    print(
                        f"  - Iteration {result['iteration']}: {execution.get('success_rate', 0):.1%} success rate"
                    )
                    print(f"    Duration: {execution.get('total_duration', 0):.1f}s")
                    print(
                        f"    Adaptations: {len(execution.get('adaptations_made', []))}"
                    )

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
    """Main entry point for multi-step pipeline."""
    parser = argparse.ArgumentParser(description="Multi-Step Reasoning Pipeline")
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
    pipeline = MultiStepPipeline(online=online, gpu_service_url=args.gpu_service)

    pipeline.run(args.prompt, args.iters)


if __name__ == "__main__":
    main()
