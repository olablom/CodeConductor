#!/usr/bin/env python3
"""
Simple Single Agent Test - Prove the model works!

This bypasses the debate system and tests a single agent directly.
"""

import asyncio
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from codeconductor.debate.local_ai_agent import LocalAIAgent
from codeconductor.ensemble.single_model_engine import SingleModelEngine


class SimpleSingleAgentTester:
    """Simple tester that bypasses debate system"""

    def __init__(self):
        self.results = []
        self.shared_engine = None

    async def initialize(self):
        """Initialize the shared model engine"""
        print("🚀 Initializing Simple Single Agent Test")
        self.shared_engine = SingleModelEngine("meta-llama-3.1-8b-instruct")
        await self.shared_engine.initialize()
        print("✅ Test suite initialized successfully")

    def create_single_agent(self) -> LocalAIAgent:
        """Create a single agent for direct testing"""
        agent = LocalAIAgent(
            "Coder",
            "You are Coder – an AI development expert who focuses on practical implementation, working code, and getting things done. You prioritize functionality, performance, and user experience. You think in terms of features, bugs, and deployment.",
        )
        agent.set_shared_engine(self.shared_engine)
        return agent

    def extract_code(self, response: str) -> str:
        """Extract code from response"""
        # Find code blocks (```python ... ```) - more flexible
        matches = re.findall(r"```(?:python)?\s*\n(.*?)\n```", response, re.DOTALL)
        if matches:
            return "\n\n".join(matches)

        # Fallback: look for code without markdown
        lines = response.split("\n")
        code_lines = []
        in_code = False

        for line in lines:
            if "def " in line or "import " in line or "from " in line or "class " in line:
                in_code = True
            if in_code and line.strip():
                code_lines.append(line)
            elif in_code and not line.strip():
                break

        return "\n".join(code_lines) if code_lines else "No code found"

    def validate_code(self, code: str, test_type: str) -> dict[str, Any]:
        """Validate generated code"""
        validation = {
            "has_code": bool(code.strip()),
            "has_imports": "import" in code or "from" in code,
            "has_functions": "def " in code,
            "has_classes": "class " in code,
            "syntax_check": True,
            "test_passed": False,
            "code_length": len(code),
        }

        # Basic syntax check
        try:
            compile(code, "<string>", "exec")
            validation["syntax_check"] = True
        except SyntaxError:
            validation["syntax_check"] = False

        # Test-specific validation
        if test_type == "fibonacci":
            validation["test_passed"] = self._test_fibonacci(code)
        elif test_type == "binary_search":
            validation["test_passed"] = self._test_binary_search(code)
        elif test_type == "rest_api":
            validation["test_passed"] = self._test_rest_api(code)

        return validation

    def _test_fibonacci(self, code: str) -> bool:
        """Test if fibonacci function works"""
        try:
            exec_globals = {}
            exec(code, exec_globals)

            # Find fibonacci function
            fibonacci = None
            for name, obj in exec_globals.items():
                if "fib" in name.lower() and callable(obj):
                    fibonacci = obj
                    break

            if fibonacci:
                # Test basic cases
                return (
                    fibonacci(0) == 0
                    and fibonacci(1) == 1
                    and fibonacci(5) == 5
                    and fibonacci(10) == 55
                )
        except:
            pass
        return False

    def _test_binary_search(self, code: str) -> bool:
        """Test if binary search function works"""
        try:
            exec_globals = {}
            exec(code, exec_globals)

            # Find binary search function
            binary_search = None
            for name, obj in exec_globals.items():
                if "binary" in name.lower() and callable(obj):
                    binary_search = obj
                    break

            if binary_search:
                # Test with sorted array
                arr = [1, 3, 5, 7, 9, 11, 13, 15]
                return (
                    binary_search(arr, 7) == 3
                    and binary_search(arr, 1) == 0
                    and binary_search(arr, 15) == 7
                )
        except:
            pass
        return False

    def _test_rest_api(self, code: str) -> bool:
        """Test if REST API code has basic structure"""
        return (
            "@app.route" in code
            or "def login" in code
            or "def register" in code
            or "flask" in code.lower()
            or "fastapi" in code.lower()
        )

    async def run_single_test(
        self, test_name: str, prompt: str, timeout: float = 60.0
    ) -> dict[str, Any]:
        """Run a single test case with one agent"""
        print(f"\n🧪 Running test: {test_name} (single agent)")
        print(f"📝 Prompt: {prompt}")

        start_time = time.time()

        try:
            # Create single agent
            agent = self.create_single_agent()

            # Generate response directly (no debate)
            response = await asyncio.wait_for(
                agent.generate_response(prompt, timeout=timeout), timeout=timeout
            )

            execution_time = time.time() - start_time

            # Extract and validate code
            code = self.extract_code(response)
            validation = self.validate_code(code, test_name)

            result = {
                "test_name": test_name,
                "num_agents": 1,
                "prompt": prompt,
                "execution_time": execution_time,
                "success": validation["test_passed"],
                "validation": validation,
                "code": code,
                "response": response,
            }

            print(f"✅ Test completed in {execution_time:.1f}s")
            print(f"📊 Success: {validation['test_passed']}")
            print(f"📝 Code extracted: {validation['code_length']} characters")
            print(f"🔍 Syntax check: {validation['syntax_check']}")
            print(f"📄 Response length: {len(response)} characters")

            return result

        except TimeoutError:
            execution_time = time.time() - start_time
            print(f"⏰ Test timed out after {execution_time:.1f}s")
            return {
                "test_name": test_name,
                "num_agents": 1,
                "prompt": prompt,
                "execution_time": execution_time,
                "success": False,
                "error": "timeout",
                "code": "",
                "response": "",
            }
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"❌ Test failed: {e}")
            return {
                "test_name": test_name,
                "num_agents": 1,
                "prompt": prompt,
                "execution_time": execution_time,
                "success": False,
                "error": str(e),
                "code": "",
                "response": "",
            }

    async def run_simple_test_suite(self):
        """Run simple test suite"""
        print("🧪 Simple Single Agent Test Suite")
        print("=" * 50)

        # Simple test cases
        test_cases = [
            (
                "fibonacci",
                "Create a Python function to calculate the nth Fibonacci number",
            ),
            (
                "binary_search",
                "Create a Python function to perform binary search on a sorted array",
            ),
            (
                "rest_api",
                "Create a simple REST API endpoint for user login using Flask",
            ),
        ]

        for test_name, prompt in test_cases:
            result = await self.run_single_test(test_name, prompt)
            self.results.append(result)

            # Save intermediate results
            self.save_results()

        # Print summary
        self.print_summary()

    def save_results(self):
        """Save test results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        with open(f"simple_test_results_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        # Save summary
        summary = {
            "total_tests": len(self.results),
            "successful_tests": sum(1 for r in self.results if r.get("success", False)),
            "average_time": sum(r.get("execution_time", 0) for r in self.results)
            / len(self.results),
            "tests_by_type": {},
            "code_quality_metrics": {
                "avg_code_length": sum(
                    r.get("validation", {}).get("code_length", 0) for r in self.results
                )
                / len(self.results),
                "syntax_success_rate": sum(
                    1 for r in self.results if r.get("validation", {}).get("syntax_check", False)
                )
                / len(self.results)
                * 100,
            },
        }

        # Group by test type
        for result in self.results:
            test_type = result["test_name"]
            if test_type not in summary["tests_by_type"]:
                summary["tests_by_type"][test_type] = {"total": 0, "successful": 0}
            summary["tests_by_type"][test_type]["total"] += 1
            if result.get("success", False):
                summary["tests_by_type"][test_type]["successful"] += 1

        with open(f"simple_summary_{timestamp}.yaml", "w", encoding="utf-8") as f:
            yaml.dump(summary, f, default_flow_style=False, allow_unicode=True)

        print(f"📁 Results saved to simple_test_results_{timestamp}.json")
        print(f"📁 Summary saved to simple_summary_{timestamp}.yaml")

    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 50)
        print("📊 SIMPLE SINGLE AGENT TEST SUMMARY")
        print("=" * 50)

        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.get("success", False))
        avg_time = sum(r.get("execution_time", 0) for r in self.results) / total_tests

        print(f"Total tests: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Success rate: {successful_tests/total_tests*100:.1f}%")
        print(f"Average time: {avg_time:.1f}s")

        # Results by test type
        print("\n📊 Results by test type:")
        type_stats = {}
        for result in self.results:
            test_type = result["test_name"]
            if test_type not in type_stats:
                type_stats[test_type] = {"total": 0, "successful": 0, "times": []}
            type_stats[test_type]["total"] += 1
            type_stats[test_type]["times"].append(result.get("execution_time", 0))
            if result.get("success", False):
                type_stats[test_type]["successful"] += 1

        for test_type, stats in type_stats.items():
            success_rate = stats["successful"] / stats["total"] * 100
            avg_time = sum(stats["times"]) / len(stats["times"])
            print(
                f"  {test_type}: {stats['successful']}/{stats['total']} ({success_rate:.1f}%) - {avg_time:.1f}s avg"
            )

        # Code quality metrics
        print("\n📈 Code Quality Metrics:")
        syntax_success = sum(
            1 for r in self.results if r.get("validation", {}).get("syntax_check", False)
        )
        avg_code_length = (
            sum(r.get("validation", {}).get("code_length", 0) for r in self.results) / total_tests
        )
        print(
            f"  Syntax success rate: {syntax_success}/{total_tests} ({syntax_success/total_tests*100:.1f}%)"
        )
        print(f"  Average code length: {avg_code_length:.0f} characters")

        # Success prediction
        if successful_tests >= 2:
            print(f"\n🎉 EXCELLENT! {successful_tests}/{total_tests} tests passed!")
            print("   Model works perfectly! Ready for debate system optimization.")
        elif successful_tests >= 1:
            print(f"\n✅ GOOD! {successful_tests}/{total_tests} tests passed!")
            print("   Model works, needs minor prompt optimization.")
        else:
            print(f"\n⚠️  NEEDS WORK: {successful_tests}/{total_tests} tests passed.")
            print("   Focus on model configuration and prompts.")

    async def cleanup(self):
        """Cleanup resources"""
        if self.shared_engine:
            await self.shared_engine.cleanup()


async def main():
    """Run simple test suite"""
    tester = SimpleSingleAgentTester()

    try:
        await tester.initialize()
        await tester.run_simple_test_suite()
    finally:
        await tester.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
