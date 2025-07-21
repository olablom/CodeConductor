#!/usr/bin/env python3
"""
CodeConductor Pipeline - Complete Multi-Agent Code Generation System

This pipeline integrates:
1. Multi-agent discussion and consensus
2. Human-in-the-loop approval
3. Code generation with CodeGenAgent
4. RL feedback loop

Usage:
    python pipeline.py --prompt prompts/simple_api.md --iters 1 --online
    python pipeline.py --prompt prompts/complex_app.md --iters 3 --offline
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

from agents.orchestrator import AgentOrchestrator
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


class CodeConductorPipeline:
    """Complete pipeline for multi-agent code generation with human approval."""

    def __init__(self, online: bool = True, config: Optional[Dict[str, Any]] = None):
        """Initialize the pipeline."""
        self.online = online
        self.config = config or {}
        self.iteration = 0
        self.results = []

        # Initialize components
        from agents.architect_agent import ArchitectAgent
        from agents.review_agent import ReviewAgent
        from agents.policy_agent import PolicyAgent
        from agents.reward_agent import (
            RewardAgent,
        )
        from agents.qlearning_agent import QLearningAgent

        # Create agents for orchestrator
        agents = [ArchitectAgent(), ReviewAgent(), CodeGenAgent()]

        self.orchestrator = AgentOrchestrator(agents)
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

        logger.info(f"Pipeline initialized (online: {self.online})")

    def load_prompt(self, prompt_file: str) -> Dict[str, Any]:
        """Load prompt from file."""
        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse the prompt file
            lines = content.strip().split("\n")
            title = lines[0].strip()
            description = "\n".join(lines[1:]).strip()

            return {
                "title": title,
                "summary": description,
                "task_type": "api"
                if "api" in title.lower() or "rest" in description.lower()
                else "application",
                "language": "python",
                "requirements": description,
                "complexity": "moderate",
            }
        except Exception as e:
            logger.error(f"Failed to load prompt from {prompt_file}: {e}")
            raise

    def run_iteration(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single iteration of the pipeline."""
        self.iteration += 1
        logger.info(f"Starting iteration {self.iteration}")

        # Step 1: Multi-agent discussion and consensus
        logger.info("Step 1: Multi-agent discussion and consensus")
        discussion_result = self.orchestrator.run_discussion(context)
        consensus = discussion_result.get("consensus")

        if not consensus:
            logger.warning("Failed to reach consensus, using first proposal")
            # Use the first proposal as consensus if no consensus reached
            if discussion_result.get("final_proposals"):
                consensus = discussion_result["final_proposals"][0]
            else:
                logger.error("No proposals available")
                return {"error": "No proposals available", "iteration": self.iteration}

        # Step 2: Human approval
        logger.info("Step 2: Human approval")
        approval_result = self.approval_cli.process_approval(consensus)

        if not approval_result.approved:
            logger.info("Proposal rejected by human")
            return {
                "status": "rejected",
                "iteration": self.iteration,
                "reason": approval_result.comments or "No reason provided",
            }

        # Step 3: Code generation
        logger.info("Step 3: Code generation")
        # Use the approved proposal (original or edited)
        approved_proposal = approval_result.proposal
        analysis = approved_proposal.get("analysis", {})
        code_result = self.codegen_agent.propose(analysis, context)

        # Step 4: Code review and safety check
        logger.info("Step 4: Code review and safety check")

        # Review the generated code
        review_result = self.review_agent.review_generated_code(
            code_result["code"], context
        )

        # Check code safety with policy agent
        policy_result = self.policy_agent.check_code_safety(code_result["code"])

        # Log review and policy results
        logger.info(f"Review assessment: {review_result['overall_assessment']}")
        logger.info(f"Policy decision: {policy_result['decision']}")

        # Check if code should be blocked
        if policy_result["decision"] == "block":
            logger.error("Code blocked by policy agent due to safety violations")
            return {
                "status": "blocked",
                "iteration": self.iteration,
                "reason": "Code contains safety violations",
                "policy_violations": policy_result["violations"],
                "review_issues": review_result["issues"],
            }

        # Add review and policy results to code_result
        code_result["review"] = review_result
        code_result["policy_check"] = policy_result

        # Step 5: Run tests and calculate reward (RL)
        if hasattr(self, "reward_agent") and hasattr(self, "qlearning_agent"):
            logger.info("Step 5: Running tests and calculating reward")

            # Run tests on generated code
            test_result = self._run_tests(code_result["code"])

            # Create reward inputs
            code_quality = CodeQualityMetrics(
                quality_score=review_result.get("quality_score", 0.5),
                complexity_score=1.0
                - review_result.get("complexity_metrics", {}).get(
                    "complexity_per_line", 0.5
                ),
                maintainability_score=0.8,
                documentation_score=1.0
                if review_result.get("compliance_status", {}).get(
                    "documentation", False
                )
                else 0.5,
                style_score=0.9
                if review_result.get("compliance_status", {}).get("pep8", True)
                else 0.6,
                security_score=1.0 if policy_result.get("safe", False) else 0.3,
            )

            human_feedback = HumanFeedback(
                overall_rating=0.8 if approval_result.approved else 0.3,
                usefulness_rating=0.8,
                correctness_rating=0.8,
                completeness_rating=0.7,
                comments=approval_result.comments or "",
                approved=approval_result.approved,
            )

            policy_result_obj = PolicyResult(
                safe=policy_result.get("safe", False),
                decision=policy_result.get("decision", "block"),
                risk_level=policy_result.get("risk_level", "high"),
                violations_count=policy_result.get("total_violations", 0),
                critical_violations=policy_result.get("critical_violations", 0),
                high_violations=policy_result.get("high_violations", 0),
            )

            # Calculate reward
            reward_result = self.reward_agent.calculate_reward(
                test_result, code_quality, human_feedback, policy_result_obj
            )

            # Update Q-learning
            current_state = self.qlearning_agent.get_state(context)
            next_state = self.qlearning_agent.get_state(
                {
                    **context,
                    "iteration": self.iteration + 1,
                    "reward": reward_result["total_reward"],
                }
            )

            # Create action from current iteration
            current_action = self.qlearning_agent.get_actions(current_state)[
                0
            ]  # Use first action for now

            # Update Q-value
            self.qlearning_agent.update_q_value(
                current_state, current_action, reward_result["total_reward"], next_state
            )

            # Add reward results to code_result
            code_result["reward"] = reward_result
            code_result["test_result"] = test_result

            logger.info(
                f"Reward calculated: {reward_result['total_reward']:.3f} ({reward_result['reward_level']})"
            )

            # Check if we should continue learning
            if (
                reward_result["total_reward"] < 0.7
                and self.iteration < self.max_iterations
            ):
                logger.info(
                    "Low reward detected, continuing to next iteration for learning"
                )
                return self._run_iteration(context, self.iteration + 1)

        # Step 6: Save generated code
        logger.info("Step 5: Save generated code")
        filename = f"iter_{self.iteration}.py"
        filepath = self.output_dir / filename

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(code_result["code"])
            logger.info(f"Code saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save code: {e}")
            code_result["save_error"] = str(e)

        # Step 6: Collect results
        result = {
            "iteration": self.iteration,
            "status": "completed",
            "consensus": consensus,
            "approval": {
                "approved": approval_result.approved,
                "decision": approval_result.user_decision,
                "comments": approval_result.comments,
                "timestamp": approval_result.timestamp.isoformat(),
            },
            "code_generation": code_result,
            "output_file": str(filepath),
            "timestamp": datetime.now().isoformat(),
        }

        self.results.append(result)
        logger.info(f"Iteration {self.iteration} completed successfully")

        return result

    def _run_tests(self, code: str) -> TestResult:
        """
        Run tests on generated code.

        Args:
            code: Generated code to test

        Returns:
            TestResult with test execution results
        """
        try:
            import tempfile
            import ast

            # Create temporary file with generated code
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                temp_file = f.name

            # Test 1: Syntax check
            syntax_errors = []
            try:
                ast.parse(code)
                syntax_passed = True
            except SyntaxError as e:
                syntax_passed = False
                syntax_errors.append(f"Syntax error: {e}")

            # Test 2: Basic execution test
            execution_errors = []
            execution_passed = False
            try:
                # Try to import the module (basic test)
                import importlib.util

                spec = importlib.util.spec_from_file_location("test_module", temp_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                execution_passed = True
            except Exception as e:
                execution_errors.append(f"Execution error: {e}")

            # Test 3: Check for basic functionality
            functionality_passed = False
            if "def " in code or "class " in code:
                functionality_passed = True

            # Calculate test results
            total_tests = 3
            passed_tests = sum([syntax_passed, execution_passed, functionality_passed])
            failed_tests = total_tests - passed_tests

            # Calculate coverage (simplified)
            lines = code.split("\n")
            non_empty_lines = [line for line in lines if line.strip()]
            coverage_percentage = min(
                100.0, len(non_empty_lines) / max(len(lines), 1) * 100
            )

            # Execution time (simplified)
            execution_time = 0.1  # Mock execution time

            # Combine error messages
            error_messages = syntax_errors + execution_errors

            # Overall test result
            test_passed = passed_tests >= 2  # At least 2 out of 3 tests must pass

            # Clean up
            import os

            os.unlink(temp_file)

            return TestResult(
                passed=test_passed,
                total_tests=total_tests,
                passed_tests=passed_tests,
                failed_tests=failed_tests,
                coverage_percentage=coverage_percentage,
                execution_time=execution_time,
                error_messages=error_messages,
            )

        except Exception as e:
            logger.error(f"Error running tests: {e}")
            return TestResult(
                passed=False,
                total_tests=1,
                passed_tests=0,
                failed_tests=1,
                coverage_percentage=0.0,
                execution_time=0.0,
                error_messages=[f"Test execution error: {e}"],
            )

    def run(self, prompt_file: str, iterations: int = 1) -> Dict[str, Any]:
        """Run the complete pipeline."""
        logger.info(f"Starting CodeConductor pipeline with {iterations} iterations")

        # Load prompt
        context = self.load_prompt(prompt_file)
        logger.info(f"Loaded prompt: {context['title']}")

        # Run iterations
        for i in range(iterations):
            try:
                result = self.run_iteration(context)
                if result.get("status") == "rejected":
                    logger.info("Pipeline stopped due to rejection")
                    break
            except Exception as e:
                logger.error(f"Error in iteration {i + 1}: {e}")
                result = {"error": str(e), "iteration": i + 1}
                self.results.append(result)
                break

        # Save pipeline results
        results_file = (
            self.output_dir
            / f"pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        try:
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "pipeline_config": {
                            "online": self.online,
                            "iterations": iterations,
                            "prompt_file": prompt_file,
                        },
                        "context": context,
                        "results": self.results,
                    },
                    f,
                    indent=2,
                    default=str,
                )
            logger.info(f"Pipeline results saved to {results_file}")
        except Exception as e:
            logger.error(f"Failed to save pipeline results: {e}")

        return {
            "total_iterations": len(self.results),
            "successful_iterations": len(
                [r for r in self.results if r.get("status") == "completed"]
            ),
            "results_file": str(results_file),
            "results": self.results,
        }


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(description="CodeConductor Pipeline")
    parser.add_argument("--prompt", required=True, help="Path to prompt file")
    parser.add_argument("--iters", type=int, default=1, help="Number of iterations")
    parser.add_argument("--online", action="store_true", help="Use online LLM")
    parser.add_argument("--offline", action="store_true", help="Use offline mode only")

    args = parser.parse_args()

    # Determine online/offline mode
    online = args.online or not args.offline

    # Validate prompt file
    if not os.path.exists(args.prompt):
        print(f"Error: Prompt file '{args.prompt}' not found")
        sys.exit(1)

    # Run pipeline
    try:
        pipeline = CodeConductorPipeline(online=online)
        result = pipeline.run(args.prompt, args.iters)

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED")
        print("=" * 60)
        print(f"Total iterations: {result['total_iterations']}")
        print(f"Successful iterations: {result['successful_iterations']}")
        print(f"Results saved to: {result['results_file']}")

        if result["successful_iterations"] > 0:
            print("\nGenerated files:")
            for r in result["results"]:
                if r.get("status") == "completed":
                    print(f"  - {r['output_file']}")

        print("=" * 60)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
