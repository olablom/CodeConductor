#!/usr/bin/env python3
"""
Performance Optimization Test Suite
Tests different optimizations for the 2-agent debate system
"""

import asyncio
import json
import time
from datetime import datetime

# Test cases for performance optimization
PERFORMANCE_TEST_CASES = [
    {
        "name": "fibonacci_performance",
        "prompt": "Create a Python function to calculate the nth Fibonacci number",
        "expected_time": 60,  # Target: under 60 seconds
    },
    {
        "name": "rest_api_performance",
        "prompt": "Create a REST API endpoint for user login using Flask",
        "expected_time": 90,  # Target: under 90 seconds
    },
]


async def test_parallel_execution():
    """Test parallel execution of agents"""

    print("⚡ Testing Parallel Execution")

    start_time = time.time()

    try:
        from src.codeconductor.debate.debate_manager import DebateManager

        # Create debate manager with parallel execution
        debate_manager = DebateManager(
            num_agents=2,
            model_name="mixtral-8x7b-instruct",
            max_tokens=1500,  # Reduced for speed
            temperature=0.7,
            parallel_execution=True,  # Enable parallel
        )

        result = await debate_manager.run_debate(
            prompt="Create a Python function to calculate the nth Fibonacci number",
            max_rounds=2,  # Reduced rounds
            timeout=60,
        )

        execution_time = time.time() - start_time

        return {
            "optimization": "parallel_execution",
            "execution_time": execution_time,
            "success": result is not None and result.get("final_code"),
            "result": result,
        }

    except Exception as e:
        execution_time = time.time() - start_time
        return {
            "optimization": "parallel_execution",
            "execution_time": execution_time,
            "success": False,
            "error": str(e),
        }


async def test_caching_optimization():
    """Test caching of model loads and responses"""

    print("💾 Testing Caching Optimization")

    start_time = time.time()

    try:
        from src.codeconductor.debate.debate_manager import DebateManager

        # Create debate manager with caching
        debate_manager = DebateManager(
            num_agents=2,
            model_name="mixtral-8x7b-instruct",
            max_tokens=1500,
            temperature=0.7,
            enable_caching=True,  # Enable caching
        )

        # Run same prompt twice to test caching
        result1 = await debate_manager.run_debate(
            prompt="Create a Python function to calculate the nth Fibonacci number",
            max_rounds=2,
            timeout=60,
        )

        # Second run should be faster due to caching
        result2 = await debate_manager.run_debate(
            prompt="Create a Python function to calculate the nth Fibonacci number",
            max_rounds=2,
            timeout=60,
        )

        execution_time = time.time() - start_time

        return {
            "optimization": "caching",
            "execution_time": execution_time,
            "success": result1 is not None and result2 is not None,
            "result": {"first_run": result1, "second_run": result2},
        }

    except Exception as e:
        execution_time = time.time() - start_time
        return {
            "optimization": "caching",
            "execution_time": execution_time,
            "success": False,
            "error": str(e),
        }


async def test_smaller_model():
    """Test with smaller model for speed"""

    print("🔧 Testing Smaller Model")

    start_time = time.time()

    try:
        from src.codeconductor.debate.debate_manager import DebateManager

        # Try with smaller model
        debate_manager = DebateManager(
            num_agents=2,
            model_name="deepseek-coder-6.7b-instruct",  # Smaller model
            max_tokens=1000,  # Reduced tokens
            temperature=0.7,
        )

        result = await debate_manager.run_debate(
            prompt="Create a Python function to calculate the nth Fibonacci number",
            max_rounds=2,
            timeout=60,
        )

        execution_time = time.time() - start_time

        return {
            "optimization": "smaller_model",
            "execution_time": execution_time,
            "success": result is not None and result.get("final_code"),
            "result": result,
        }

    except Exception as e:
        execution_time = time.time() - start_time
        return {
            "optimization": "smaller_model",
            "execution_time": execution_time,
            "success": False,
            "error": str(e),
        }


async def test_reduced_rounds():
    """Test with reduced debate rounds"""

    print("🔄 Testing Reduced Rounds")

    start_time = time.time()

    try:
        from src.codeconductor.debate.debate_manager import DebateManager

        # Create debate manager with reduced rounds
        debate_manager = DebateManager(
            num_agents=2,
            model_name="mixtral-8x7b-instruct",
            max_tokens=1500,
            temperature=0.7,
        )

        result = await debate_manager.run_debate(
            prompt="Create a Python function to calculate the nth Fibonacci number",
            max_rounds=1,  # Only 1 round
            timeout=60,
        )

        execution_time = time.time() - start_time

        return {
            "optimization": "reduced_rounds",
            "execution_time": execution_time,
            "success": result is not None and result.get("final_code"),
            "result": result,
        }

    except Exception as e:
        execution_time = time.time() - start_time
        return {
            "optimization": "reduced_rounds",
            "execution_time": execution_time,
            "success": False,
            "error": str(e),
        }


async def run_performance_optimization_tests():
    """Run all performance optimization tests"""

    print("🚀 Performance Optimization Test Suite")
    print("=" * 60)

    optimizations = [
        test_parallel_execution,
        test_caching_optimization,
        test_smaller_model,
        test_reduced_rounds,
    ]

    results = []

    for optimization_test in optimizations:
        result = await optimization_test()
        results.append(result)

        status = "✅ PASS" if result["success"] else "❌ FAIL"
        print(f"{status} {result['optimization']} ({result['execution_time']:.1f}s)")

        if result.get("error"):
            print(f"   Error: {result['error']}")

    # Calculate statistics
    successful_optimizations = sum(1 for r in results if r["success"])
    total_optimizations = len(results)
    success_rate = (
        (successful_optimizations / total_optimizations) * 100 if total_optimizations > 0 else 0
    )

    avg_time = sum(r["execution_time"] for r in results) / len(results) if results else 0

    # Find best optimization
    successful_results = [r for r in results if r["success"]]
    best_optimization = None
    best_time = float("inf")

    for result in successful_results:
        if result["execution_time"] < best_time:
            best_time = result["execution_time"]
            best_optimization = result["optimization"]

    # Create summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "test_type": "performance_optimization",
        "total_optimizations": total_optimizations,
        "successful_optimizations": successful_optimizations,
        "success_rate": success_rate,
        "average_time": avg_time,
        "best_optimization": best_optimization,
        "best_time": best_time,
        "optimizations": {},
    }

    # Group by optimization type
    for result in results:
        opt_name = result["optimization"]
        summary["optimizations"][opt_name] = {
            "success": result["success"],
            "execution_time": result["execution_time"],
            "error": result.get("error", None),
        }

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(f"performance_optimization_results_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(f"performance_optimization_summary_{timestamp}.yaml", "w") as f:
        import yaml

        yaml.dump(summary, f, default_flow_style=False)

    # Print final results
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE OPTIMIZATION RESULTS:")
    print(f"✅ Successful: {successful_optimizations}/{total_optimizations} ({success_rate:.1f}%)")
    print(f"⏱️  Average time: {avg_time:.1f}s")

    if best_optimization:
        print(f"🏆 Best optimization: {best_optimization} ({best_time:.1f}s)")

    print("\n📋 DETAILED RESULTS:")
    for opt_name, stats in summary["optimizations"].items():
        status = "✅" if stats["success"] else "❌"
        print(f"  {opt_name}: {status} ({stats['execution_time']:.1f}s)")
        if stats.get("error"):
            print(f"    Error: {stats['error']}")

    return summary, results


if __name__ == "__main__":
    asyncio.run(run_performance_optimization_tests())
