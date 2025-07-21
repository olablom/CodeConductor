#!/usr/bin/env python3
"""
Performance Profiler for CodeConductor

Measures latency for each agent method and identifies bottlenecks.
"""

import time
import statistics
from pathlib import Path
from typing import Dict, List, Any, Callable
import json
from datetime import datetime
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.codegen_agent import CodeGenAgent
from agents.architect_agent import ArchitectAgent
from agents.review_agent import ReviewAgent
from agents.policy_agent import PolicyAgent
from agents.prompt_optimizer import PromptOptimizerAgent
from agents.orchestrator import AgentOrchestrator


class PerformanceProfiler:
    """Profiles agent performance and identifies bottlenecks"""

    def __init__(self):
        self.results = {}
        self.agent_instances = {}
        self._setup_agents()

    def _setup_agents(self):
        """Initialize agent instances for testing"""
        try:
            # Create mock context for testing
            mock_context = {
                "task": "Create a simple calculator",
                "language": "python",
                "complexity": "medium",
                "requirements": ["addition", "subtraction", "multiplication"],
            }

            # Initialize agents
            self.agent_instances = {
                "codegen": CodeGenAgent(),
                "architect": ArchitectAgent(),
                "reviewer": ReviewAgent(),
                "policy": PolicyAgent(),
                "optimizer": PromptOptimizerAgent(),
                "orchestrator": AgentOrchestrator(),
            }

            print("✅ Agents initialized successfully")

        except Exception as e:
            print(f"❌ Error initializing agents: {e}")
            self.agent_instances = {}

    def profile_method(
        self, agent_name: str, method_name: str, method_func: Callable, *args, **kwargs
    ) -> Dict[str, Any]:
        """Profile a single method call"""

        times = []
        errors = 0

        # Run method multiple times for accurate measurement
        num_runs = 10

        for i in range(num_runs):
            try:
                start_time = time.perf_counter()
                result = method_func(*args, **kwargs)
                end_time = time.perf_counter()

                latency = (end_time - start_time) * 1000  # Convert to milliseconds
                times.append(latency)

            except Exception as e:
                errors += 1
                print(f"Error in {agent_name}.{method_name}: {e}")

        if times:
            return {
                "agent": agent_name,
                "method": method_name,
                "runs": num_runs,
                "errors": errors,
                "min_latency": min(times),
                "max_latency": max(times),
                "avg_latency": statistics.mean(times),
                "median_latency": statistics.median(times),
                "std_dev": statistics.stdev(times) if len(times) > 1 else 0,
                "success_rate": (num_runs - errors) / num_runs * 100,
            }
        else:
            return {
                "agent": agent_name,
                "method": method_name,
                "runs": num_runs,
                "errors": errors,
                "success_rate": 0,
                "error": "All runs failed",
            }

    def profile_agent_methods(self) -> Dict[str, List[Dict[str, Any]]]:
        """Profile all agent methods"""

        results = {}

        # Test context for agent methods
        test_context = {
            "task": "Create a simple API endpoint",
            "language": "python",
            "complexity": "medium",
            "requirements": ["REST API", "JSON response", "error handling"],
        }

        test_prompt = "Create a simple calculator function in Python"

        for agent_name, agent in self.agent_instances.items():
            print(f"\n🔍 Profiling {agent_name}...")
            agent_results = []

            # Profile analyze method
            if hasattr(agent, "analyze"):
                result = self.profile_method(
                    agent_name, "analyze", agent.analyze, test_context
                )
                agent_results.append(result)
                print(f"  analyze: {result['avg_latency']:.2f}ms avg")

            # Profile propose method
            if hasattr(agent, "propose"):
                result = self.profile_method(
                    agent_name, "propose", agent.propose, test_context
                )
                agent_results.append(result)
                print(f"  propose: {result['avg_latency']:.2f}ms avg")

            # Profile review method
            if hasattr(agent, "review"):
                result = self.profile_method(
                    agent_name, "review", agent.review, test_context, "sample code"
                )
                agent_results.append(result)
                print(f"  review: {result['avg_latency']:.2f}ms avg")

            # Profile specific agent methods
            if agent_name == "optimizer" and hasattr(agent, "optimize_prompt"):
                result = self.profile_method(
                    agent_name, "optimize_prompt", agent.optimize_prompt, test_prompt
                )
                agent_results.append(result)
                print(f"  optimize_prompt: {result['avg_latency']:.2f}ms avg")

            if agent_name == "orchestrator" and hasattr(agent, "orchestrate"):
                result = self.profile_method(
                    agent_name, "orchestrate", agent.orchestrate, test_context
                )
                agent_results.append(result)
                print(f"  orchestrate: {result['avg_latency']:.2f}ms avg")

            results[agent_name] = agent_results

        return results

    def profile_pipeline_end_to_end(self) -> Dict[str, Any]:
        """Profile complete pipeline execution"""

        print("\n🚀 Profiling end-to-end pipeline...")

        test_context = {
            "task": "Create a simple web scraper",
            "language": "python",
            "complexity": "medium",
            "requirements": ["requests", "beautifulsoup", "error handling"],
        }

        times = []
        errors = 0
        num_runs = 5  # Fewer runs for end-to-end due to time

        for i in range(num_runs):
            try:
                start_time = time.perf_counter()

                # Simulate pipeline execution
                orchestrator = self.agent_instances.get("orchestrator")
                if orchestrator and hasattr(orchestrator, "orchestrate"):
                    result = orchestrator.orchestrate(test_context)

                end_time = time.perf_counter()
                latency = (end_time - start_time) * 1000
                times.append(latency)

                print(f"  Run {i + 1}: {latency:.2f}ms")

            except Exception as e:
                errors += 1
                print(f"  Run {i + 1} failed: {e}")

        if times:
            return {
                "pipeline": "end_to_end",
                "runs": num_runs,
                "errors": errors,
                "min_latency": min(times),
                "max_latency": max(times),
                "avg_latency": statistics.mean(times),
                "median_latency": statistics.median(times),
                "success_rate": (num_runs - errors) / num_runs * 100,
            }
        else:
            return {
                "pipeline": "end_to_end",
                "runs": num_runs,
                "errors": errors,
                "success_rate": 0,
                "error": "All runs failed",
            }

    def identify_bottlenecks(
        self, results: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks"""

        bottlenecks = []

        for agent_name, agent_results in results.items():
            for result in agent_results:
                # Flag slow methods (>100ms average)
                if result.get("avg_latency", 0) > 100:
                    bottlenecks.append(
                        {
                            "type": "slow_method",
                            "agent": agent_name,
                            "method": result["method"],
                            "avg_latency": result["avg_latency"],
                            "recommendation": "Consider caching or optimization",
                        }
                    )

                # Flag high error rates (>20%)
                if result.get("success_rate", 100) < 80:
                    bottlenecks.append(
                        {
                            "type": "high_error_rate",
                            "agent": agent_name,
                            "method": result["method"],
                            "success_rate": result["success_rate"],
                            "recommendation": "Investigate error handling",
                        }
                    )

        return bottlenecks

    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save profiling results to file"""

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bench/results/performance_profile_{timestamp}.json"

        # Create results directory
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

        with open(filename, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"✅ Results saved to {filename}")

    def run_full_profile(self) -> Dict[str, Any]:
        """Run complete performance profiling"""

        print("🎯 Starting Performance Profiling...")
        print("=" * 50)

        # Profile individual agent methods
        agent_results = self.profile_agent_methods()

        # Profile end-to-end pipeline
        pipeline_result = self.profile_pipeline_end_to_end()

        # Identify bottlenecks
        bottlenecks = self.identify_bottlenecks(agent_results)

        # Compile results
        full_results = {
            "timestamp": datetime.now().isoformat(),
            "agent_profiles": agent_results,
            "pipeline_profile": pipeline_result,
            "bottlenecks": bottlenecks,
            "summary": {
                "total_agents": len(agent_results),
                "total_methods": sum(
                    len(results) for results in agent_results.values()
                ),
                "bottlenecks_found": len(bottlenecks),
                "pipeline_avg_latency": pipeline_result.get("avg_latency", 0),
            },
        }

        # Save results
        self.save_results(full_results)

        # Print summary
        print("\n" + "=" * 50)
        print("📊 PERFORMANCE SUMMARY")
        print("=" * 50)
        print(f"Total agents profiled: {full_results['summary']['total_agents']}")
        print(f"Total methods tested: {full_results['summary']['total_methods']}")
        print(f"Bottlenecks found: {full_results['summary']['bottlenecks_found']}")
        print(
            f"Pipeline avg latency: {full_results['summary']['pipeline_avg_latency']:.2f}ms"
        )

        if bottlenecks:
            print("\n🚨 BOTTLENECKS IDENTIFIED:")
            for bottleneck in bottlenecks:
                print(
                    f"  - {bottleneck['agent']}.{bottleneck['method']}: {bottleneck['recommendation']}"
                )

        return full_results


def main():
    """Main profiling function"""

    profiler = PerformanceProfiler()
    results = profiler.run_full_profile()

    print("\n🎉 Performance profiling completed!")
    print("Check the results file for detailed metrics.")


if __name__ == "__main__":
    main()
