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

        # Create agents for orchestrator
        agents = [ArchitectAgent(), ReviewAgent(), CodeGenAgent()]

        self.orchestrator = AgentOrchestrator(agents)
        self.codegen_agent = CodeGenAgent()
        self.review_agent = ReviewAgent()
        self.policy_agent = PolicyAgent()
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

        # Step 5: Save generated code
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
            print(f"\nGenerated files:")
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
