#!/usr/bin/env python3
"""
Comprehensive CodeConductor Test Suite

Tests real code generation, different agent configurations, and performance metrics.
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

from codeconductor.debate.local_ai_agent import LocalAIAgent, LocalDebateManager
from codeconductor.ensemble.single_model_engine import SingleModelEngine


class CodeConductorTester:
    """Comprehensive tester for CodeConductor"""

    def __init__(self):
        self.results = []
        self.shared_engine = None

    async def initialize(self):
        """Initialize the shared model engine"""
        print("🚀 Initializing CodeConductor Test Suite...")
        self.shared_engine = SingleModelEngine("meta-llama-3.1-8b-instruct")
        await self.shared_engine.initialize()
        print("✅ Test suite initialized successfully")

    def create_agents(self, num_agents: int) -> list[LocalAIAgent]:
        """Create agents based on configuration"""
        agent_configs = {
            2: [
                (
                    "Architect",
                    "You are Architect – an AI development expert who focuses on system design, architecture patterns, and scalable solutions. You prioritize clean code structure, maintainability, and best practices.",
                ),
                (
                    "Coder",
                    "You are Coder – an AI development expert who focuses on practical implementation, working code, and getting things done. You prioritize functionality, performance, and user experience.",
                ),
            ],
            3: [
                (
                    "Architect",
                    "You are Architect – an AI development expert who focuses on system design, architecture patterns, and scalable solutions. You prioritize clean code structure, maintainability, and best practices.",
                ),
                (
                    "Coder",
                    "You are Coder – an AI development expert who focuses on practical implementation, working code, and getting things done. You prioritize functionality, performance, and user experience.",
                ),
                (
                    "Optimizer",
                    "You are Optimizer – an AI development expert who focuses on performance, efficiency, and resource management. You prioritize speed, memory usage, and scalability.",
                ),
            ],
            4: [
                (
                    "Architect",
                    "You are Architect – an AI development expert who focuses on system design, architecture patterns, and scalable solutions. You prioritize clean code structure, maintainability, and best practices.",
                ),
                (
                    "Coder",
                    "You are Coder – an AI development expert who focuses on practical implementation, working code, and getting things done. You prioritize functionality, performance, and user experience.",
                ),
                (
                    "Optimizer",
                    "You are Optimizer – an AI development expert who focuses on performance, efficiency, and resource management. You prioritize speed, memory usage, and scalability.",
                ),
                (
                    "Reviewer",
                    "You are Reviewer – an AI development expert who focuses on code quality, testing, and best practices. You prioritize readability, maintainability, and edge case handling.",
                ),
            ],
        }

        if num_agents not in agent_configs:
            raise ValueError(f"Unsupported number of agents: {num_agents}")

        agents = []
        for name, persona in agent_configs[num_agents]:
            agent = LocalAIAgent(name, persona)
            agent.set_shared_engine(self.shared_engine)
            agents.append(agent)

        return agents

    def extract_code(self, debate_responses: list[dict]) -> str:
        """Extract code from debate responses"""
        code_blocks = []
        for response in debate_responses:
            content = response.get("content", "")
            # Find code blocks (```python ... ```)
            matches = re.findall(r"```(?:python)?\s*\n(.*?)\n```", content, re.DOTALL)
            code_blocks.extend(matches)

        return "\n\n".join(code_blocks) if code_blocks else "No code found"

    def validate_code(self, code: str, test_type: str) -> dict[str, Any]:
        """Validate generated code"""
        validation = {
            "has_code": bool(code.strip()),
            "has_imports": "import" in code or "from" in code,
            "has_functions": "def " in code,
            "has_classes": "class " in code,
            "syntax_check": True,  # Placeholder for actual syntax check
            "test_passed": False,
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
        elif test_type == "email_validation":
            validation["test_passed"] = self._test_email_validation(code)
        elif test_type == "todo_class":
            validation["test_passed"] = self._test_todo_class(code)

        return validation

    def _test_fibonacci(self, code: str) -> bool:
        """Test if fibonacci function works"""
        try:
            # Create a safe execution environment
            exec_globals = {}
            exec(code, exec_globals)

            # Test the function
            if "fibonacci" in exec_globals:
                fib = exec_globals["fibonacci"]
                return fib(5) == 5 and fib(10) == 55
        except:
            pass
        return False

    def _test_email_validation(self, code: str) -> bool:
        """Test if email validation function works"""
        try:
            exec_globals = {}
            exec(code, exec_globals)

            if "validate_email" in exec_globals:
                validate = exec_globals["validate_email"]
                return validate("test@example.com") and not validate("invalid-email")
        except:
            pass
        return False

    def _test_todo_class(self, code: str) -> bool:
        """Test if TODO class works"""
        try:
            exec_globals = {}
            exec(code, exec_globals)

            # Find the TODO class
            todo_class = None
            for name, obj in exec_globals.items():
                if isinstance(obj, type) and "todo" in name.lower():
                    todo_class = obj
                    break

            if todo_class:
                todo = todo_class()
                # Test basic CRUD operations
                todo.add("Test task")
                return len(todo.tasks) > 0
        except:
            pass
        return False

    async def run_single_test(
        self, test_name: str, prompt: str, num_agents: int, timeout: float = 300.0
    ) -> dict[str, Any]:
        """Run a single test case"""
        print(f"\n🧪 Running test: {test_name} ({num_agents} agents)")
        print(f"📝 Prompt: {prompt}")

        start_time = time.time()

        try:
            # Create agents and debate manager
            agents = self.create_agents(num_agents)
            debate = LocalDebateManager(agents)
            debate.set_shared_engine(self.shared_engine)

            # Run debate
            debate_responses = await asyncio.wait_for(
                debate.conduct_debate(prompt, timeout_per_turn=timeout / num_agents),
                timeout=timeout,
            )

            execution_time = time.time() - start_time

            # Extract and validate code
            code = self.extract_code(debate_responses)
            validation = self.validate_code(code, test_name)

            result = {
                "test_name": test_name,
                "num_agents": num_agents,
                "prompt": prompt,
                "execution_time": execution_time,
                "success": validation["test_passed"],
                "validation": validation,
                "code": code,
                "debate_responses": debate_responses,
            }

            print(f"✅ Test completed in {execution_time:.1f}s")
            print(f"📊 Success: {validation['test_passed']}")
            print(f"📝 Code extracted: {len(code)} characters")

            return result

        except TimeoutError:
            execution_time = time.time() - start_time
            print(f"⏰ Test timed out after {execution_time:.1f}s")
            return {
                "test_name": test_name,
                "num_agents": num_agents,
                "prompt": prompt,
                "execution_time": execution_time,
                "success": False,
                "error": "timeout",
                "code": "",
                "debate_responses": [],
            }
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"❌ Test failed: {e}")
            return {
                "test_name": test_name,
                "num_agents": num_agents,
                "prompt": prompt,
                "execution_time": execution_time,
                "success": False,
                "error": str(e),
                "code": "",
                "debate_responses": [],
            }

    async def run_comprehensive_test(self):
        """Run comprehensive test suite"""
        print("🧪 CodeConductor Comprehensive Test Suite")
        print("=" * 60)

        # Test cases
        test_cases = [
            (
                "fibonacci",
                "Create a Python function to calculate the nth Fibonacci number",
                2,
            ),
            (
                "fibonacci",
                "Create a Python function to calculate the nth Fibonacci number",
                3,
            ),
            (
                "email_validation",
                "Create a Python function to validate email addresses",
                2,
            ),
            (
                "email_validation",
                "Create a Python function to validate email addresses",
                3,
            ),
            (
                "todo_class",
                "Create a Python TODO class with add, remove, and list methods",
                2,
            ),
            (
                "todo_class",
                "Create a Python TODO class with add, remove, and list methods",
                3,
            ),
            (
                "linkedlist",
                "Create a Python LinkedList class with insert and delete methods",
                3,
            ),
            (
                "linkedlist",
                "Create a Python LinkedList class with insert and delete methods",
                4,
            ),
        ]

        for test_name, prompt, num_agents in test_cases:
            result = await self.run_single_test(test_name, prompt, num_agents)
            self.results.append(result)

            # Save intermediate results
            self.save_results()

        # Print summary
        self.print_summary()

    def save_results(self):
        """Save test results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        with open(f"codeconductor_test_results_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        # Save summary
        summary = {
            "total_tests": len(self.results),
            "successful_tests": sum(1 for r in self.results if r.get("success", False)),
            "average_time": sum(r.get("execution_time", 0) for r in self.results)
            / len(self.results),
            "tests_by_agents": {},
            "tests_by_type": {},
        }

        # Group by number of agents
        for result in self.results:
            num_agents = result["num_agents"]
            if num_agents not in summary["tests_by_agents"]:
                summary["tests_by_agents"][num_agents] = {"total": 0, "successful": 0}
            summary["tests_by_agents"][num_agents]["total"] += 1
            if result.get("success", False):
                summary["tests_by_agents"][num_agents]["successful"] += 1

        # Group by test type
        for result in self.results:
            test_type = result["test_name"]
            if test_type not in summary["tests_by_type"]:
                summary["tests_by_type"][test_type] = {"total": 0, "successful": 0}
            summary["tests_by_type"][test_type]["total"] += 1
            if result.get("success", False):
                summary["tests_by_type"][test_type]["successful"] += 1

        with open(f"codeconductor_summary_{timestamp}.yaml", "w", encoding="utf-8") as f:
            yaml.dump(summary, f, default_flow_style=False, allow_unicode=True)

        print(f"📁 Results saved to codeconductor_test_results_{timestamp}.json")
        print(f"📁 Summary saved to codeconductor_summary_{timestamp}.yaml")

    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("📊 CODE CONDUCTOR TEST SUMMARY")
        print("=" * 60)

        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.get("success", False))
        avg_time = sum(r.get("execution_time", 0) for r in self.results) / total_tests

        print(f"Total tests: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Success rate: {successful_tests/total_tests*100:.1f}%")
        print(f"Average time: {avg_time:.1f}s")

        # Results by number of agents
        print("\n📈 Results by number of agents:")
        agent_stats = {}
        for result in self.results:
            num_agents = result["num_agents"]
            if num_agents not in agent_stats:
                agent_stats[num_agents] = {"total": 0, "successful": 0, "times": []}
            agent_stats[num_agents]["total"] += 1
            agent_stats[num_agents]["times"].append(result.get("execution_time", 0))
            if result.get("success", False):
                agent_stats[num_agents]["successful"] += 1

        for num_agents, stats in agent_stats.items():
            success_rate = stats["successful"] / stats["total"] * 100
            avg_time = sum(stats["times"]) / len(stats["times"])
            print(
                f"  {num_agents} agents: {stats['successful']}/{stats['total']} ({success_rate:.1f}%) - {avg_time:.1f}s avg"
            )

        # Results by test type
        print("\n📊 Results by test type:")
        type_stats = {}
        for result in self.results:
            test_type = result["test_name"]
            if test_type not in type_stats:
                type_stats[test_type] = {"total": 0, "successful": 0}
            type_stats[test_type]["total"] += 1
            if result.get("success", False):
                type_stats[test_type]["successful"] += 1

        for test_type, stats in type_stats.items():
            success_rate = stats["successful"] / stats["total"] * 100
            print(f"  {test_type}: {stats['successful']}/{stats['total']} ({success_rate:.1f}%)")

    async def cleanup(self):
        """Cleanup resources"""
        if self.shared_engine:
            await self.shared_engine.cleanup()


async def main():
    """Run comprehensive test suite"""
    tester = CodeConductorTester()

    try:
        await tester.initialize()
        await tester.run_comprehensive_test()
    finally:
        await tester.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
