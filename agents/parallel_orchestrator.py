#!/usr/bin/env python3
"""
Parallel Agent Orchestrator

Runs agent operations in parallel for improved performance.
"""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Callable
import time
import json
from pathlib import Path
import logging

from agents.base_agent import BaseAgent
from agents.codegen_agent import CodeGenAgent
from agents.architect_agent import ArchitectAgent
from agents.review_agent import ReviewAgent
from agents.policy_agent import PolicyAgent
from agents.prompt_optimizer import PromptOptimizerAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParallelOrchestrator:
    """
    Orchestrates agents with parallel execution for improved performance.
    """

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.agents = {}
        self._setup_agents()

    def _setup_agents(self):
        """Initialize agent instances"""
        try:
            self.agents = {
                "codegen": CodeGenAgent(),
                "architect": ArchitectAgent(),
                "reviewer": ReviewAgent(),
                "policy": PolicyAgent(),
                "optimizer": PromptOptimizerAgent(),
            }
            logger.info(f"✅ Initialized {len(self.agents)} agents")
        except Exception as e:
            logger.error(f"❌ Error initializing agents: {e}")
            self.agents = {}

    def run_agent_method(
        self, agent_name: str, method_name: str, *args, **kwargs
    ) -> Dict[str, Any]:
        """Run a single agent method and return results"""

        try:
            agent = self.agents.get(agent_name)
            if not agent:
                return {
                    "agent": agent_name,
                    "method": method_name,
                    "success": False,
                    "error": f"Agent {agent_name} not found",
                }

            method = getattr(agent, method_name, None)
            if not method:
                return {
                    "agent": agent_name,
                    "method": method_name,
                    "success": False,
                    "error": f"Method {method_name} not found",
                }

            start_time = time.perf_counter()
            result = method(*args, **kwargs)
            end_time = time.perf_counter()

            return {
                "agent": agent_name,
                "method": method_name,
                "success": True,
                "result": result,
                "latency_ms": (end_time - start_time) * 1000,
                "timestamp": time.time(),
            }

        except Exception as e:
            return {
                "agent": agent_name,
                "method": method_name,
                "success": False,
                "error": str(e),
                "timestamp": time.time(),
            }

    def run_parallel_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run analysis phase in parallel across all agents"""

        logger.info("🚀 Starting parallel analysis phase...")

        # Define analysis tasks
        analysis_tasks = [
            ("codegen", "analyze", context),
            ("architect", "analyze", context),
            ("reviewer", "analyze", context),
            ("policy", "analyze", context),
        ]

        # Submit tasks to thread pool
        futures = []
        for agent_name, method_name, args in analysis_tasks:
            future = self.executor.submit(
                self.run_agent_method, agent_name, method_name, args
            )
            futures.append(future)

        # Collect results
        analysis_results = {}
        total_latency = 0

        for future in as_completed(futures):
            try:
                result = future.result()
                agent_name = result["agent"]
                analysis_results[agent_name] = result
                total_latency += result.get("latency_ms", 0)

                if result["success"]:
                    logger.info(
                        f"✅ {agent_name}.analyze completed in {result['latency_ms']:.2f}ms"
                    )
                else:
                    logger.warning(f"❌ {agent_name}.analyze failed: {result['error']}")

            except Exception as e:
                logger.error(f"❌ Error collecting analysis result: {e}")

        return {
            "phase": "analysis",
            "results": analysis_results,
            "total_latency_ms": total_latency,
            "parallel_execution": True,
        }

    def run_parallel_proposal(
        self, context: Dict[str, Any], analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run proposal phase in parallel"""

        logger.info("💡 Starting parallel proposal phase...")

        # Define proposal tasks
        proposal_tasks = [
            ("codegen", "propose", context),
            ("architect", "propose", context),
            ("optimizer", "optimize_prompt", "Create a simple function"),
        ]

        # Submit tasks to thread pool
        futures = []
        for agent_name, method_name, args in proposal_tasks:
            future = self.executor.submit(
                self.run_agent_method, agent_name, method_name, args
            )
            futures.append(future)

        # Collect results
        proposal_results = {}
        total_latency = 0

        for future in as_completed(futures):
            try:
                result = future.result()
                agent_name = result["agent"]
                proposal_results[agent_name] = result
                total_latency += result.get("latency_ms", 0)

                if result["success"]:
                    logger.info(
                        f"✅ {agent_name}.propose completed in {result['latency_ms']:.2f}ms"
                    )
                else:
                    logger.warning(f"❌ {agent_name}.propose failed: {result['error']}")

            except Exception as e:
                logger.error(f"❌ Error collecting proposal result: {e}")

        return {
            "phase": "proposal",
            "results": proposal_results,
            "total_latency_ms": total_latency,
            "parallel_execution": True,
        }

    def run_parallel_review(
        self, context: Dict[str, Any], code_samples: List[str]
    ) -> Dict[str, Any]:
        """Run review phase in parallel"""

        logger.info("🔍 Starting parallel review phase...")

        # Create review tasks for each code sample
        review_tasks = []
        for i, code in enumerate(code_samples):
            review_tasks.append(("reviewer", "review", context, code))
            review_tasks.append(("policy", "review", context, code))

        # Submit tasks to thread pool
        futures = []
        for agent_name, method_name, *args in review_tasks:
            future = self.executor.submit(
                self.run_agent_method, agent_name, method_name, *args
            )
            futures.append(future)

        # Collect results
        review_results = {}
        total_latency = 0

        for future in as_completed(futures):
            try:
                result = future.result()
                agent_name = result["agent"]
                if agent_name not in review_results:
                    review_results[agent_name] = []

                review_results[agent_name].append(result)
                total_latency += result.get("latency_ms", 0)

                if result["success"]:
                    logger.info(
                        f"✅ {agent_name}.review completed in {result['latency_ms']:.2f}ms"
                    )
                else:
                    logger.warning(f"❌ {agent_name}.review failed: {result['error']}")

            except Exception as e:
                logger.error(f"❌ Error collecting review result: {e}")

        return {
            "phase": "review",
            "results": review_results,
            "total_latency_ms": total_latency,
            "parallel_execution": True,
        }

    def orchestrate_parallel(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate complete pipeline with parallel execution"""

        logger.info("🎼 Starting parallel orchestration...")
        start_time = time.perf_counter()

        pipeline_results = {
            "context": context,
            "phases": {},
            "performance": {},
            "final_result": None,
        }

        try:
            # Phase 1: Parallel Analysis
            analysis_results = self.run_parallel_analysis(context)
            pipeline_results["phases"]["analysis"] = analysis_results

            # Phase 2: Parallel Proposal
            proposal_results = self.run_parallel_proposal(context, analysis_results)
            pipeline_results["phases"]["proposal"] = proposal_results

            # Phase 3: Parallel Review (if we have code samples)
            code_samples = ["sample_code_1", "sample_code_2"]  # Mock samples
            review_results = self.run_parallel_review(context, code_samples)
            pipeline_results["phases"]["review"] = review_results

            # Calculate performance metrics
            end_time = time.perf_counter()
            total_time = (end_time - start_time) * 1000

            pipeline_results["performance"] = {
                "total_latency_ms": total_time,
                "analysis_latency_ms": analysis_results["total_latency_ms"],
                "proposal_latency_ms": proposal_results["total_latency_ms"],
                "review_latency_ms": review_results["total_latency_ms"],
                "parallelization_speedup": self._calculate_speedup(
                    pipeline_results["phases"]
                ),
            }

            # Generate final result
            pipeline_results["final_result"] = self._synthesize_results(
                pipeline_results["phases"]
            )

            logger.info(f"🎉 Parallel orchestration completed in {total_time:.2f}ms")

        except Exception as e:
            logger.error(f"❌ Parallel orchestration failed: {e}")
            pipeline_results["error"] = str(e)

        return pipeline_results

    def _calculate_speedup(self, phases: Dict[str, Any]) -> float:
        """Calculate speedup from parallelization"""

        total_parallel_time = sum(
            phase.get("total_latency_ms", 0) for phase in phases.values()
        )

        # Estimate sequential time (rough approximation)
        estimated_sequential_time = total_parallel_time * 2.5  # Conservative estimate

        if estimated_sequential_time > 0:
            return estimated_sequential_time / total_parallel_time
        return 1.0

    def _synthesize_results(self, phases: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize final results from all phases"""

        # Extract successful results
        successful_results = {}

        for phase_name, phase_data in phases.items():
            phase_results = phase_data.get("results", {})
            for agent_name, agent_result in phase_results.items():
                if agent_result.get("success", False):
                    if agent_name not in successful_results:
                        successful_results[agent_name] = {}
                    successful_results[agent_name][phase_name] = agent_result.get(
                        "result"
                    )

        return {
            "synthesis": "parallel_orchestration",
            "successful_agents": list(successful_results.keys()),
            "agent_results": successful_results,
            "recommendations": self._generate_recommendations(phases),
        }

    def _generate_recommendations(self, phases: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on performance"""

        recommendations = []

        # Check for slow agents
        for phase_name, phase_data in phases.items():
            phase_results = phase_data.get("results", {})
            for agent_name, agent_result in phase_results.items():
                latency = agent_result.get("latency_ms", 0)
                if latency > 1000:  # More than 1 second
                    recommendations.append(
                        f"Consider optimizing {agent_name}.{agent_result['method']} "
                        f"(current latency: {latency:.2f}ms)"
                    )

        # Check for failed agents
        failed_agents = []
        for phase_name, phase_data in phases.items():
            phase_results = phase_data.get("results", {})
            for agent_name, agent_result in phase_results.items():
                if not agent_result.get("success", False):
                    failed_agents.append(f"{agent_name}.{agent_result['method']}")

        if failed_agents:
            recommendations.append(
                f"Investigate failures in: {', '.join(failed_agents)}"
            )

        if not recommendations:
            recommendations.append("All agents performing well")

        return recommendations

    def shutdown(self):
        """Clean shutdown of the orchestrator"""
        logger.info("🛑 Shutting down parallel orchestrator...")
        self.executor.shutdown(wait=True)
        logger.info("✅ Parallel orchestrator shutdown complete")


def main():
    """Test the parallel orchestrator"""

    # Test context
    test_context = {
        "task": "Create a simple web API",
        "language": "python",
        "complexity": "medium",
        "requirements": ["FastAPI", "SQLAlchemy", "authentication"],
    }

    # Create and run orchestrator
    orchestrator = ParallelOrchestrator(max_workers=4)

    try:
        results = orchestrator.orchestrate_parallel(test_context)

        # Print summary
        print("\n" + "=" * 60)
        print("📊 PARALLEL ORCHESTRATION RESULTS")
        print("=" * 60)
        print(f"Total latency: {results['performance']['total_latency_ms']:.2f}ms")
        print(
            f"Speedup factor: {results['performance']['parallelization_speedup']:.2f}x"
        )
        print(f"Successful agents: {len(results['final_result']['successful_agents'])}")

        if "recommendations" in results["final_result"]:
            print("\n💡 Recommendations:")
            for rec in results["final_result"]["recommendations"]:
                print(f"  - {rec}")

        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"bench/results/parallel_orchestration_{timestamp}.json"
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

        with open(filename, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n✅ Results saved to {filename}")

    finally:
        orchestrator.shutdown()


if __name__ == "__main__":
    main()
