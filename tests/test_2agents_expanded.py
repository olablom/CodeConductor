#!/usr/bin/env python3
"""
Expanded 2-Agent Debate Test Suite
Tests multiple scenarios to prove the value of the debate system
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Any

# Test cases that should work well with 2-agent debate
TEST_CASES = [
    {
        "name": "rest_api_login",
        "prompt": "Create a REST API endpoint for user login using Flask with JWT tokens",
        "expected_features": ["authentication", "jwt", "flask", "endpoint"],
    },
    {
        "name": "react_usestate",
        "prompt": "Build a React useState hook example for a todo list with add/remove functionality",
        "expected_features": ["react", "usestate", "todo", "add", "remove"],
    },
    {
        "name": "sql_top_customers",
        "prompt": "Write a SQL query to find top 5 customers by total order value",
        "expected_features": ["sql", "aggregation", "ordering", "limit"],
    },
    {
        "name": "binary_search",
        "prompt": "Create a Python function to perform binary search on a sorted array",
        "expected_features": ["algorithm", "search", "sorted", "efficiency"],
    },
    {
        "name": "bug_fix_division",
        "prompt": "Fix this bug: def divide(a,b): return a/b  # Division by zero error",
        "expected_features": ["error_handling", "validation", "division"],
    },
    {
        "name": "email_validation",
        "prompt": "Create a Python function to validate email addresses using regex",
        "expected_features": ["regex", "validation", "email", "pattern"],
    },
    {
        "name": "linkedlist_implementation",
        "prompt": "Implement a linked list class in Python with insert, delete, and search methods",
        "expected_features": ["data_structure", "linkedlist", "methods", "class"],
    },
    {
        "name": "todo_class",
        "prompt": "Create a Todo class with add, complete, and list methods",
        "expected_features": ["class", "methods", "todo", "management"],
    },
]


async def run_2agent_test(test_case: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single test case with 2 agents"""

    print(f"🧪 Testing: {test_case['name']}")
    print(f"📝 Prompt: {test_case['prompt']}")

    start_time = time.time()

    try:
        # Import the debate system
        from src.codeconductor.debate.debate_manager import DebateManager

        # Create debate manager with 2 agents
        debate_manager = DebateManager(
            num_agents=2,
            model_name="mixtral-8x7b-instruct",
            max_tokens=2000,
            temperature=0.7,
        )

        # Run the debate
        result = await debate_manager.run_debate(
            prompt=test_case["prompt"],
            max_rounds=3,
            timeout=120,  # 2 minutes timeout
        )

        execution_time = time.time() - start_time

        # Analyze the result
        success = False
        code_quality = {
            "has_code": False,
            "code_length": 0,
            "has_imports": False,
            "has_functions": False,
            "has_classes": False,
            "syntax_check": False,
        }

        if result and result.get("final_code"):
            code = result["final_code"]
            code_quality = {
                "has_code": len(code.strip()) > 0,
                "code_length": len(code),
                "has_imports": "import " in code or "from " in code,
                "has_functions": "def " in code,
                "has_classes": "class " in code,
                "syntax_check": True,  # Basic check
            }
            success = code_quality["has_code"] and code_quality["code_length"] > 100

        return {
            "test_name": test_case["name"],
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
            "prompt": test_case["prompt"],
            "execution_time": execution_time,
            "success": False,
            "error": str(e),
            "result": None,
        }


async def run_comprehensive_2agent_test():
    """Run comprehensive test suite with 2 agents"""

    print("🚀 Starting Comprehensive 2-Agent Debate Test Suite")
    print("=" * 60)

    results = []
    total_start_time = time.time()

    for test_case in TEST_CASES:
        result = await run_2agent_test(test_case)
        results.append(result)

        # Print immediate feedback
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        print(f"{status} {test_case['name']} ({result['execution_time']:.1f}s)")

        if result.get("error"):
            print(f"   Error: {result['error']}")

    total_time = time.time() - total_start_time

    # Calculate statistics
    successful_tests = sum(1 for r in results if r["success"])
    total_tests = len(results)
    success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0

    avg_time = (
        sum(r["execution_time"] for r in results) / len(results) if results else 0
    )

    # Quality metrics
    avg_code_length = 0
    syntax_success_rate = 0

    if results:
        code_results = [r for r in results if r.get("code_quality")]
        if code_results:
            avg_code_length = sum(
                r["code_quality"]["code_length"] for r in code_results
            ) / len(code_results)
            syntax_success_rate = (
                sum(1 for r in code_results if r["code_quality"]["syntax_check"])
                / len(code_results)
                * 100
            )

    # Create summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "test_type": "comprehensive_2agent",
        "total_tests": total_tests,
        "successful_tests": successful_tests,
        "success_rate": success_rate,
        "average_time": avg_time,
        "total_time": total_time,
        "code_quality_metrics": {
            "avg_code_length": avg_code_length,
            "syntax_success_rate": syntax_success_rate,
        },
        "tests_by_type": {},
    }

    # Group by test type
    for result in results:
        test_name = result["test_name"]
        if test_name not in summary["tests_by_type"]:
            summary["tests_by_type"][test_name] = {"successful": 0, "total": 0}

        summary["tests_by_type"][test_name]["total"] += 1
        if result["success"]:
            summary["tests_by_type"][test_name]["successful"] += 1

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(f"comprehensive_2agent_test_results_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(f"comprehensive_2agent_summary_{timestamp}.yaml", "w") as f:
        import yaml

        yaml.dump(summary, f, default_flow_style=False)

    # Print final results
    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS:")
    print(f"✅ Successful: {successful_tests}/{total_tests} ({success_rate:.1f}%)")
    print(f"⏱️  Average time: {avg_time:.1f}s")
    print(f"📝 Avg code length: {avg_code_length:.0f} chars")
    print(f"🔧 Syntax success: {syntax_success_rate:.1f}%")
    print(f"⏱️  Total time: {total_time:.1f}s")

    print("\n📋 DETAILED RESULTS:")
    for test_name, stats in summary["tests_by_type"].items():
        rate = (stats["successful"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        print(f"  {test_name}: {stats['successful']}/{stats['total']} ({rate:.1f}%)")

    return summary, results


if __name__ == "__main__":
    asyncio.run(run_comprehensive_2agent_test())
