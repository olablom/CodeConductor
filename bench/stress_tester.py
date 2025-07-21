#!/usr/bin/env python3
"""
Stress Tester for CodeConductor

Simulates high load scenarios to test system performance and stability.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional
import json
import statistics
from pathlib import Path
from datetime import datetime
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StressTester:
    """Stress tests the CodeConductor system under various load conditions"""

    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.results = []
        self.failures = []
        self.start_time = None
        self.end_time = None

    def generate_test_context(self, task_id: int) -> Dict[str, Any]:
        """Generate a test context for stress testing"""

        tasks = [
            "Create a simple calculator",
            "Build a REST API endpoint",
            "Implement user authentication",
            "Create a database schema",
            "Write unit tests",
            "Create a web scraper",
            "Build a CLI tool",
            "Implement caching system",
            "Create a logging system",
            "Build a configuration manager",
        ]

        languages = ["python", "javascript", "typescript", "java", "go"]
        complexities = ["simple", "medium", "complex"]

        return {
            "task_id": task_id,
            "task": random.choice(tasks),
            "language": random.choice(languages),
            "complexity": random.choice(complexities),
            "requirements": [
                "error handling",
                "documentation",
                "testing",
                "performance",
            ],
            "timestamp": time.time(),
        }

    def simulate_agent_operation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate an agent operation for stress testing"""

        task_id = context["task_id"]
        start_time = time.perf_counter()

        try:
            # Simulate processing time (0.1 to 2 seconds)
            processing_time = random.uniform(0.1, 2.0)
            time.sleep(processing_time)

            # Simulate occasional failures (5% failure rate)
            if random.random() < 0.05:
                raise Exception(f"Simulated failure for task {task_id}")

            end_time = time.perf_counter()
            latency = (end_time - start_time) * 1000

            return {
                "task_id": task_id,
                "success": True,
                "latency_ms": latency,
                "processing_time_ms": processing_time * 1000,
                "agent": random.choice(["codegen", "architect", "reviewer", "policy"]),
                "timestamp": time.time(),
            }

        except Exception as e:
            end_time = time.perf_counter()
            latency = (end_time - start_time) * 1000

            return {
                "task_id": task_id,
                "success": False,
                "error": str(e),
                "latency_ms": latency,
                "agent": random.choice(["codegen", "architect", "reviewer", "policy"]),
                "timestamp": time.time(),
            }

    def run_concurrent_tasks(self, num_tasks: int) -> Dict[str, Any]:
        """Run multiple tasks concurrently"""

        logger.info(f"🚀 Starting stress test with {num_tasks} concurrent tasks...")

        self.start_time = time.perf_counter()

        # Generate test contexts
        contexts = [self.generate_test_context(i) for i in range(num_tasks)]

        # Run tasks concurrently
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.simulate_agent_operation, context)
                for context in contexts
            ]

            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result()
                    self.results.append(result)

                    if not result["success"]:
                        self.failures.append(result)

                except Exception as e:
                    logger.error(f"❌ Error collecting result: {e}")

        self.end_time = time.perf_counter()

        return self._analyze_results()

    def run_gradual_load_test(
        self, max_tasks: int = 100, step_size: int = 10
    ) -> Dict[str, Any]:
        """Run gradual load test to find breaking point"""

        logger.info(
            f"📈 Starting gradual load test (max: {max_tasks}, step: {step_size})..."
        )

        load_results = []

        for num_tasks in range(step_size, max_tasks + 1, step_size):
            logger.info(f"Testing with {num_tasks} tasks...")

            # Reset for this test
            self.results = []
            self.failures = []

            # Run test
            test_result = self.run_concurrent_tasks(num_tasks)

            # Add load level info
            test_result["load_level"] = num_tasks
            load_results.append(test_result)

            # Check if we've hit breaking point
            failure_rate = test_result["failure_rate"]
            avg_latency = test_result["avg_latency_ms"]

            if failure_rate > 0.2 or avg_latency > 5000:  # 20% failure or 5s latency
                logger.warning(f"⚠️ Breaking point reached at {num_tasks} tasks")
                logger.warning(f"   Failure rate: {failure_rate:.2%}")
                logger.warning(f"   Avg latency: {avg_latency:.2f}ms")
                break

        return {
            "test_type": "gradual_load",
            "load_results": load_results,
            "breaking_point": self._find_breaking_point(load_results),
        }

    def run_burst_test(
        self, burst_size: int = 50, num_bursts: int = 5
    ) -> Dict[str, Any]:
        """Run burst load test to test system recovery"""

        logger.info(
            f"💥 Starting burst test ({num_bursts} bursts of {burst_size} tasks each)..."
        )

        burst_results = []

        for burst_num in range(num_bursts):
            logger.info(f"Burst {burst_num + 1}/{num_bursts}...")

            # Reset for this burst
            self.results = []
            self.failures = []

            # Run burst
            burst_result = self.run_concurrent_tasks(burst_size)
            burst_result["burst_number"] = burst_num + 1
            burst_results.append(burst_result)

            # Wait between bursts
            if burst_num < num_bursts - 1:
                time.sleep(2)

        return {
            "test_type": "burst_load",
            "burst_results": burst_results,
            "recovery_analysis": self._analyze_recovery(burst_results),
        }

    def _analyze_results(self) -> Dict[str, Any]:
        """Analyze test results"""

        if not self.results:
            return {"error": "No results to analyze"}

        total_time = (self.end_time - self.start_time) * 1000
        successful_results = [r for r in self.results if r["success"]]
        failed_results = [r for r in self.results if not r["success"]]

        latencies = [r["latency_ms"] for r in successful_results]

        analysis = {
            "total_tasks": len(self.results),
            "successful_tasks": len(successful_results),
            "failed_tasks": len(failed_results),
            "success_rate": len(successful_results) / len(self.results)
            if self.results
            else 0,
            "failure_rate": len(failed_results) / len(self.results)
            if self.results
            else 0,
            "total_time_ms": total_time,
            "throughput_tasks_per_sec": len(self.results) / (total_time / 1000)
            if total_time > 0
            else 0,
            "avg_latency_ms": statistics.mean(latencies) if latencies else 0,
            "min_latency_ms": min(latencies) if latencies else 0,
            "max_latency_ms": max(latencies) if latencies else 0,
            "median_latency_ms": statistics.median(latencies) if latencies else 0,
            "latency_std_dev": statistics.stdev(latencies) if len(latencies) > 1 else 0,
            "errors": [r["error"] for r in failed_results],
            "agent_distribution": self._analyze_agent_distribution(),
        }

        return analysis

    def _analyze_agent_distribution(self) -> Dict[str, int]:
        """Analyze distribution of agent usage"""

        agent_counts = {}
        for result in self.results:
            agent = result.get("agent", "unknown")
            agent_counts[agent] = agent_counts.get(agent, 0) + 1

        return agent_counts

    def _find_breaking_point(self, load_results: List[Dict[str, Any]]) -> Optional[int]:
        """Find the breaking point in gradual load test"""

        for result in load_results:
            failure_rate = result["failure_rate"]
            avg_latency = result["avg_latency_ms"]

            if failure_rate > 0.2 or avg_latency > 5000:
                return result["load_level"]

        return None

    def _analyze_recovery(self, burst_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze system recovery between bursts"""

        recovery_metrics = []

        for i in range(1, len(burst_results)):
            prev_burst = burst_results[i - 1]
            curr_burst = burst_results[i]

            # Calculate recovery metrics
            latency_recovery = (
                prev_burst["avg_latency_ms"] - curr_burst["avg_latency_ms"]
            )
            failure_recovery = prev_burst["failure_rate"] - curr_burst["failure_rate"]

            recovery_metrics.append(
                {
                    "burst_pair": f"{i}-{i + 1}",
                    "latency_recovery_ms": latency_recovery,
                    "failure_rate_recovery": failure_recovery,
                    "recovered": latency_recovery > 0 and failure_recovery > 0,
                }
            )

        return {
            "recovery_metrics": recovery_metrics,
            "avg_latency_recovery_ms": statistics.mean(
                [m["latency_recovery_ms"] for m in recovery_metrics]
            )
            if recovery_metrics
            else 0,
            "recovery_success_rate": len(
                [m for m in recovery_metrics if m["recovered"]]
            )
            / len(recovery_metrics)
            if recovery_metrics
            else 0,
        }

    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save stress test results"""

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bench/results/stress_test_{timestamp}.json"

        # Create results directory
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

        with open(filename, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"✅ Results saved to {filename}")

    def print_summary(self, results: Dict[str, Any]):
        """Print a summary of stress test results"""

        print("\n" + "=" * 60)
        print("📊 STRESS TEST SUMMARY")
        print("=" * 60)

        if "test_type" in results:
            print(f"Test Type: {results['test_type']}")

            if results["test_type"] == "gradual_load":
                breaking_point = results.get("breaking_point")
                if breaking_point:
                    print(f"Breaking Point: {breaking_point} tasks")
                else:
                    print("Breaking Point: Not reached")

            elif results["test_type"] == "burst_load":
                recovery_rate = results.get("recovery_analysis", {}).get(
                    "recovery_success_rate", 0
                )
                print(f"Recovery Success Rate: {recovery_rate:.2%}")

        if "total_tasks" in results:
            print(f"Total Tasks: {results['total_tasks']}")
            print(f"Success Rate: {results['success_rate']:.2%}")
            print(f"Failure Rate: {results['failure_rate']:.2%}")
            print(f"Throughput: {results['throughput_tasks_per_sec']:.2f} tasks/sec")
            print(f"Avg Latency: {results['avg_latency_ms']:.2f}ms")
            print(
                f"Latency Range: {results['min_latency_ms']:.2f}ms - {results['max_latency_ms']:.2f}ms"
            )

        if "agent_distribution" in results:
            print("\nAgent Distribution:")
            for agent, count in results["agent_distribution"].items():
                print(f"  {agent}: {count} tasks")


def main():
    """Run comprehensive stress tests"""

    print("🎯 Starting CodeConductor Stress Testing Suite...")
    print("=" * 60)

    tester = StressTester(max_workers=10)

    # Test 1: Concurrent load test
    print("\n1️⃣ Running concurrent load test...")
    concurrent_results = tester.run_concurrent_tasks(50)
    tester.print_summary(concurrent_results)

    # Test 2: Gradual load test
    print("\n2️⃣ Running gradual load test...")
    gradual_results = tester.run_gradual_load_test(max_tasks=100, step_size=20)
    tester.print_summary(gradual_results)

    # Test 3: Burst test
    print("\n3️⃣ Running burst test...")
    burst_results = tester.run_burst_test(burst_size=30, num_bursts=3)
    tester.print_summary(burst_results)

    # Compile all results
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "concurrent_test": concurrent_results,
        "gradual_load_test": gradual_results,
        "burst_test": burst_results,
        "summary": {
            "max_throughput": concurrent_results.get("throughput_tasks_per_sec", 0),
            "breaking_point": gradual_results.get("breaking_point"),
            "recovery_rate": burst_results.get("recovery_analysis", {}).get(
                "recovery_success_rate", 0
            ),
        },
    }

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"bench/results/comprehensive_stress_test_{timestamp}.json"
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    with open(filename, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n✅ All results saved to {filename}")
    print("\n🎉 Stress testing completed!")


if __name__ == "__main__":
    main()
