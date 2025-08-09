#!/usr/bin/env python3
"""
Single Model vs 2-Agent Debate Comparison
Compares the quality and approach of single model vs debate system
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Any

# Test cases for comparison
COMPARISON_TEST_CASES = [
    {
        "name": "fibonacci_comparison",
        "prompt": "Create a Python function to calculate the nth Fibonacci number",
        "expected_features": ["algorithm", "efficiency", "multiple_approaches"],
    },
    {
        "name": "rest_api_comparison",
        "prompt": "Create a REST API endpoint for user login using Flask",
        "expected_features": ["authentication", "flask", "endpoint", "security"],
    },
    {
        "name": "react_hook_comparison",
        "prompt": "Build a React useState hook example for a todo list",
        "expected_features": ["react", "usestate", "todo", "state_management"],
    },
]


async def run_single_model_test(test_case: Dict[str, Any]) -> Dict[str, Any]:
    """Run test with single model"""

    print(f"🤖 Single Model: {test_case['name']}")

    start_time = time.time()

    try:
        # Import single model system
        from src.codeconductor.ensemble.single_model_engine import SingleModelEngine

        # Create single model engine
        engine = SingleModelEngine(
            model_name="mixtral-8x7b-instruct", max_tokens=2000, temperature=0.7
        )

        # Run single model
        result = await engine.generate_code(prompt=test_case["prompt"], timeout=120)

        execution_time = time.time() - start_time

        # Analyze result
        success = False
        code_quality = {
            "has_code": False,
            "code_length": 0,
            "has_imports": False,
            "has_functions": False,
            "has_classes": False,
            "syntax_check": False,
            "num_implementations": 1,
        }

        if result and result.get("code"):
            code = result["code"]
            code_quality = {
                "has_code": len(code.strip()) > 0,
                "code_length": len(code),
                "has_imports": "import " in code or "from " in code,
                "has_functions": "def " in code,
                "has_classes": "class " in code,
                "syntax_check": True,
                "num_implementations": 1,
            }
            success = code_quality["has_code"] and code_quality["code_length"] > 100

        return {
            "test_name": test_case["name"],
            "model_type": "single",
            "prompt": test_case["prompt"],
            "execution_time": execution_time,
            "success": success,
            "code_quality": code_quality,
            "result": result,
        }

    except Exception as e:
        execution_time = time.time() - start_time
        return {
            "test_name": test_case["name"],
            "model_type": "single",
            "prompt": test_case["prompt"],
            "execution_time": execution_time,
            "success": False,
            "error": str(e),
            "result": None,
        }


async def run_debate_test(test_case: Dict[str, Any]) -> Dict[str, Any]:
    """Run test with 2-agent debate"""

    print(f"🤝 Debate (2 agents): {test_case['name']}")

    start_time = time.time()

    try:
        # Import debate system
        from src.codeconductor.debate.debate_manager import DebateManager

        # Create debate manager
        debate_manager = DebateManager(
            num_agents=2,
            model_name="mixtral-8x7b-instruct",
            max_tokens=2000,
            temperature=0.7,
        )

        # Run debate
        result = await debate_manager.run_debate(
            prompt=test_case["prompt"], max_rounds=3, timeout=120
        )

        execution_time = time.time() - start_time

        # Analyze result
        success = False
        code_quality = {
            "has_code": False,
            "code_length": 0,
            "has_imports": False,
            "has_functions": False,
            "has_classes": False,
            "syntax_check": False,
            "num_implementations": 0,
            "debate_rounds": 0,
        }

        if result and result.get("final_code"):
            code = result["final_code"]

            # Count implementations (look for multiple def/class patterns)
            implementations = code.count("def ") + code.count("class ")

            code_quality = {
                "has_code": len(code.strip()) > 0,
                "code_length": len(code),
                "has_imports": "import " in code or "from " in code,
                "has_functions": "def " in code,
                "has_classes": "class " in code,
                "syntax_check": True,
                "num_implementations": implementations,
                "debate_rounds": len(result.get("debate_responses", [])),
            }
            success = code_quality["has_code"] and code_quality["code_length"] > 100

        return {
            "test_name": test_case["name"],
            "model_type": "debate",
            "prompt": test_case["prompt"],
            "execution_time": execution_time,
            "success": success,
            "code_quality": code_quality,
            "result": result,
        }

    except Exception as e:
        execution_time = time.time() - start_time
        return {
            "test_name": test_case["name"],
            "model_type": "debate",
            "prompt": test_case["prompt"],
            "execution_time": execution_time,
            "success": False,
            "error": str(e),
            "result": None,
        }


async def run_comparison_test():
    """Run comparison between single model and debate"""

    print("🔬 Single Model vs 2-Agent Debate Comparison")
    print("=" * 60)

    all_results = []

    for test_case in COMPARISON_TEST_CASES:
        print(f"\n📋 Test Case: {test_case['name']}")
        print(f"📝 Prompt: {test_case['prompt']}")

        # Run single model
        single_result = await run_single_model_test(test_case)
        all_results.append(single_result)

        # Run debate
        debate_result = await run_debate_test(test_case)
        all_results.append(debate_result)

        # Print comparison
        single_status = "✅" if single_result["success"] else "❌"
        debate_status = "✅" if debate_result["success"] else "❌"

        print(f"🤖 Single: {single_status} ({single_result['execution_time']:.1f}s)")
        print(f"🤝 Debate: {debate_status} ({debate_result['execution_time']:.1f}s)")

        if single_result["success"] and debate_result["success"]:
            single_length = single_result["code_quality"]["code_length"]
            debate_length = debate_result["code_quality"]["code_length"]
            debate_impl = debate_result["code_quality"]["num_implementations"]

            print(f"   📝 Single: {single_length} chars")
            print(
                f"   📝 Debate: {debate_length} chars ({debate_impl} implementations)"
            )

    # Calculate statistics
    single_results = [r for r in all_results if r["model_type"] == "single"]
    debate_results = [r for r in all_results if r["model_type"] == "debate"]

    single_success = sum(1 for r in single_results if r["success"])
    debate_success = sum(1 for r in debate_results if r["success"])

    single_success_rate = (
        (single_success / len(single_results)) * 100 if single_results else 0
    )
    debate_success_rate = (
        (debate_success / len(debate_results)) * 100 if debate_results else 0
    )

    # Quality metrics
    single_avg_length = 0
    debate_avg_length = 0
    debate_avg_implementations = 0

    if single_results:
        successful_single = [r for r in single_results if r["success"]]
        if successful_single:
            single_avg_length = sum(
                r["code_quality"]["code_length"] for r in successful_single
            ) / len(successful_single)

    if debate_results:
        successful_debate = [r for r in debate_results if r["success"]]
        if successful_debate:
            debate_avg_length = sum(
                r["code_quality"]["code_length"] for r in successful_debate
            ) / len(successful_debate)
            debate_avg_implementations = sum(
                r["code_quality"]["num_implementations"] for r in successful_debate
            ) / len(successful_debate)

    # Create summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "test_type": "single_vs_debate_comparison",
        "single_model": {
            "total_tests": len(single_results),
            "successful_tests": single_success,
            "success_rate": single_success_rate,
            "avg_code_length": single_avg_length,
        },
        "debate_model": {
            "total_tests": len(debate_results),
            "successful_tests": debate_success,
            "success_rate": debate_success_rate,
            "avg_code_length": debate_avg_length,
            "avg_implementations": debate_avg_implementations,
        },
        "improvement": {
            "success_rate_improvement": debate_success_rate - single_success_rate,
            "code_length_improvement": debate_avg_length - single_avg_length,
            "additional_implementations": debate_avg_implementations - 1,
        },
    }

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(f"comparison_test_results_{timestamp}.json", "w") as f:
        json.dump(all_results, f, indent=2)

    with open(f"comparison_summary_{timestamp}.yaml", "w") as f:
        import yaml

        yaml.dump(summary, f, default_flow_style=False)

    # Print final results
    print("\n" + "=" * 60)
    print("📊 COMPARISON RESULTS:")
    print(
        f"🤖 Single Model: {single_success}/{len(single_results)} ({single_success_rate:.1f}%)"
    )
    print(
        f"🤝 Debate Model: {debate_success}/{len(debate_results)} ({debate_success_rate:.1f}%)"
    )
    print(
        f"📈 Success Rate Improvement: {summary['improvement']['success_rate_improvement']:.1f}%"
    )
    print(
        f"📝 Avg Code Length - Single: {single_avg_length:.0f}, Debate: {debate_avg_length:.0f}"
    )
    print(f"🔧 Avg Implementations - Debate: {debate_avg_implementations:.1f}")

    return summary, all_results


if __name__ == "__main__":
    asyncio.run(run_comparison_test())
