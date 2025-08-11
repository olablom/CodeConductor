#!/usr/bin/env python3
"""
Test Runner for CodeConductor MVP.

Handles pytest execution, result parsing, and feedback generation for the feedback loop.
"""

import subprocess
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feedback.learning_system import log_test_reward


@dataclass
class TestResult:
    """
    Represents the result of a pytest run.
    """

    success: bool
    stdout: str
    errors: List[str] = field(default_factory=list)
    test_results: List[Dict[str, Any]] = field(
        default_factory=list
    )  # Test-as-Reward data


class PytestRunner:
    """
    Executes pytest with JSON reporting and parses real test results.
    """

    def __init__(self, prompt: str = "", code: str = "", tests_dir: str = "tests"):
        self.prompt = prompt
        self.code = code
        self.tests_dir = tests_dir

    def run(self) -> Dict[str, Any]:
        """
        Run pytest with JSON reporting and return structured results.

        Returns:
            Dictionary with test results, success status, and reward
        """
        try:
            # 1. Save generated code to temporary file if provided
            code_path = None
            if self.code and self.code.strip():
                with tempfile.NamedTemporaryFile(
                    suffix=".py", delete=False, mode="w"
                ) as tmp:
                    tmp.write(self.code)
                    code_path = tmp.name

            # 2. Run pytest with JSON report
            report_file = f"pytest_report_{os.getpid()}.json"
            cmd = [
                "pytest",
                self.tests_dir,
                "--maxfail=1",
                "--disable-warnings",
                "--json-report",
                f"--json-report-file={report_file}",
                "-q",  # Quiet mode for cleaner output
                "--tb=short",  # Shorter tracebacks
            ]

            # Run pytest
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            output = process.stdout + process.stderr

            # 3. Read JSON report
            test_results = []
            success = False
            if os.path.exists(report_file):
                try:
                    with open(report_file, "r") as f:
                        report = json.load(f)

                    # 4. Transform to our structure
                    for test in report.get("tests", []):
                        name = test.get("nodeid", "unknown_test")
                        passed = test.get("outcome") == "passed"
                        error = None if passed else test.get("longrepr", "")
                        duration = test.get("duration", 0.0)  # Test duration in seconds

                        test_results.append(
                            {
                                "name": name,
                                "passed": passed,
                                "error": error,
                                "type": "pytest",
                                "duration_s": duration,
                            }
                        )

                    # Determine overall success
                    passed_tests = sum(1 for t in test_results if t["passed"])
                    total_tests = len(test_results)
                    success = total_tests > 0 and passed_tests == total_tests

                    # Clean up report file
                    os.remove(report_file)

                except (json.JSONDecodeError, KeyError) as e:
                    print(f"⚠️ Error parsing JSON report: {e}")
                    test_results = []
                    success = False
            else:
                print(f"⚠️ No pytest report file found: {report_file}")

            # 4.5. If no pytest tests collected, attempt doctest on generated code
            if (not success) and not test_results and code_path:
                try:
                    dt_proc = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "doctest",
                            "-v",
                            code_path,
                        ],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    dt_out = dt_proc.stdout + dt_proc.stderr
                    # Simple parsing: rc==0 → treat as success (even if 0 tests)
                    success = dt_proc.returncode == 0
                    # Extract a brief summary if present
                    summary = None
                    for line in dt_out.strip().splitlines()[::-1]:
                        if "passed" in line and "failed" in line:
                            summary = line.strip()
                            break
                    test_results.append(
                        {
                            "name": "doctest_summary",
                            "passed": success,
                            "type": "doctest",
                            "summary": summary or "doctest run",
                        }
                    )
                except Exception:
                    pass

            # 5. Log reward if we have prompt and code
            reward = 0.0
            if self.prompt and self.code and test_results:
                reward = self._log_test_reward(test_results)

            return {
                "success": success,
                "test_results": test_results,
                "reward": reward,
                "passed_tests": sum(1 for t in test_results if t.get("passed")),
                "total_tests": len(test_results),
                "stdout": output,
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "test_results": [],
                "reward": 0.0,
                "passed_tests": 0,
                "total_tests": 0,
                "stdout": "Test execution timed out",
                "error": "Timeout",
            }
        except Exception as e:
            return {
                "success": False,
                "test_results": [],
                "reward": 0.0,
                "passed_tests": 0,
                "total_tests": 0,
                "stdout": f"Error running tests: {str(e)}",
                "error": str(e),
            }
        finally:
            # Clean up temporary code file
            if code_path and os.path.exists(code_path):
                os.remove(code_path)

    def _log_test_reward(self, test_results: List[Dict[str, Any]]) -> float:
        """
        Log test results as reward using the learning system.

        Args:
            test_results: List of test result dictionaries

        Returns:
            Calculated reward value
        """
        try:
            reward = log_test_reward(self.prompt, self.code, test_results)
            print(
                f"🎯 Pytest-as-Reward logged: {reward:.2f} "
                f"({sum(1 for t in test_results if t.get('passed'))}/{len(test_results)} tests passed)"
            )
            return reward
        except Exception as e:
            print(f"⚠️ Error logging pytest reward: {e}")
            return 0.0


class TestRunner:
    """
    Executes pytest on a given directory and extracts failure and error details.
    """

    def __init__(self, timeout: int = 60) -> None:
        self.timeout = timeout

    def run_pytest(
        self,
        test_dir: Path,
        prompt: str = "",
        code: str = "",
        metadata: Dict[str, Any] = None,
    ) -> TestResult:
        """
        Run pytest in default mode (full tracebacks) on the given directory.

        Args:
            test_dir: Directory containing tests
            prompt: The prompt that was used to generate the code (for Test-as-Reward)
            code: The generated code (for Test-as-Reward)
            metadata: Additional metadata for logging

        Returns:
            TestResult: success flag, combined stdout/stderr, and parsed errors.
        """
        cmd = ["pytest", str(test_dir)]
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        output = process.stdout + process.stderr

        # Extract errors first
        errors = self._extract_errors(output)

        # Parse test results for Test-as-Reward
        test_results = self._parse_test_results(output)

        # Handle different cases
        if errors:
            # Has errors = failure
            result = TestResult(
                success=False,
                stdout=output,
                errors=errors,
                test_results=test_results,
            )
        elif re.search(r"collected\s+0\s+items", output):
            # No tests collected and no errors = no test files
            result = TestResult(
                success=True,
                stdout=output,
                errors=["No test files found"],
                test_results=test_results,
            )
        else:
            # No errors = success
            result = TestResult(
                success=True,
                stdout=output,
                errors=errors,
                test_results=test_results,
            )

        # Log Test-as-Reward if we have prompt and code
        if prompt and code:
            self._log_test_reward(prompt, code, test_results, metadata)

        return result

    def _parse_test_results(self, output: str) -> List[Dict[str, Any]]:
        """
        Parse pytest output to extract individual test results for Test-as-Reward.

        Args:
            output: Raw pytest output

        Returns:
            List of test result dictionaries with 'passed' key
        """
        test_results = []

        # Extract test collection info
        collected_match = re.search(r"collected\s+(\d+)\s+items?", output)
        if collected_match:
            total_tests = int(collected_match.group(1))
        else:
            total_tests = 0

        # Extract passed/failed counts
        passed_match = re.search(r"(\d+)\s+passed", output)
        failed_match = re.search(r"(\d+)\s+failed", output)
        error_match = re.search(r"(\d+)\s+error", output)

        passed_count = int(passed_match.group(1)) if passed_match else 0
        failed_count = int(failed_match.group(1)) if failed_match else 0
        error_count = int(error_match.group(1)) if error_match else 0

        # Create test result entries
        for i in range(passed_count):
            test_results.append(
                {"name": f"test_{i + 1}_passed", "passed": True, "type": "passed"}
            )

        for i in range(failed_count):
            test_results.append(
                {"name": f"test_{i + 1}_failed", "passed": False, "type": "failed"}
            )

        for i in range(error_count):
            test_results.append(
                {"name": f"test_{i + 1}_error", "passed": False, "type": "error"}
            )

        # If no detailed parsing possible, create a summary result
        if not test_results and total_tests > 0:
            success_rate = passed_count / total_tests if total_tests > 0 else 0.0
            test_results.append(
                {
                    "name": "summary",
                    "passed": success_rate
                    >= 0.5,  # Consider it passed if at least 50% success
                    "type": "summary",
                    "success_rate": success_rate,
                    "total_tests": total_tests,
                    "passed_tests": passed_count,
                }
            )

        return test_results

    def _log_test_reward(
        self,
        prompt: str,
        code: str,
        test_results: List[Dict[str, Any]],
        metadata: Dict[str, Any] = None,
    ) -> float:
        """
        Log test results as reward using the learning system.

        Args:
            prompt: The prompt that was used
            code: The generated code
            test_results: List of test result dictionaries
            metadata: Additional metadata

        Returns:
            Calculated reward value
        """
        try:
            reward = log_test_reward(prompt, code, test_results, metadata)
            print(
                f"🎯 Test-as-Reward logged: {reward:.2f} ({len([t for t in test_results if t.get('passed')])}/{len(test_results)} tests passed)"
            )
            return reward
        except Exception as e:
            print(f"⚠️ Error logging test reward: {e}")
            return 0.0

    def _extract_errors(self, output: str) -> List[str]:
        """
        Extract individual failure and error blocks from pytest output.

        Captures blocks under '=== FAILURES ===' and '=== ERRORS ===',
        then splits each block on test function headers to separate multiple entries.
        """
        combined = []
        # patterns for failures and errors
        for section in ("FAILURES", "ERRORS"):
            pattern = rf"=+ {section} =+\n(.+?)(?:\n=+|\Z)"
            match = re.search(pattern, output, flags=re.S)
            if match:
                block = match.group(1)
                # For now, just return the entire block as one error
                # This is simpler and more reliable
                combined.append(block.strip())
        return combined

    def execute_test(self, test_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single test specification with duration tracking.

        Args:
            test_spec: Dictionary containing test name and function

        Returns:
            Test result dictionary with duration
        """
        import time

        start_time = time.time()

        try:
            # Assume test_spec contains name and a function to call
            test_fn = test_spec["fn"]
            result = test_fn(test_spec.get("code", ""))
            duration = time.time() - start_time
            return {"name": test_spec["name"], "passed": result, "duration_s": duration}
        except Exception as e:
            duration = time.time() - start_time
            return {
                "name": test_spec["name"],
                "passed": False,
                "error": str(e),
                "duration_s": duration,
            }

    def run_custom_tests(
        self,
        tests: List[Dict[str, Any]],
        prompt: str = "",
        code: str = "",
        metadata: Dict[str, Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run custom tests and log rewards.

        Args:
            tests: List of test specifications
            prompt: The prompt that was used
            code: The generated code
            metadata: Additional metadata

        Returns:
            List of test results
        """
        results = []
        for test_spec in tests:
            res = self.execute_test(test_spec)
            results.append(res)

        # Log reward after running tests
        if prompt and code:
            self._log_test_reward(prompt, code, results, metadata)

        return results


# Example usage for testing
if __name__ == "__main__":
    # Test the new PytestRunner
    print("🧪 Testing PytestRunner...")

    # Create a simple test
    test_code = """
def add_numbers(a, b):
    return a + b

def test_add_numbers():
    assert add_numbers(2, 3) == 5
    assert add_numbers(-1, 1) == 0
    print("All tests passed!")
"""

    runner = PytestRunner(
        prompt="Create a function to add numbers", code=test_code, tests_dir="tests"
    )

    results = runner.run()
    print(f"Results: {results}")

    # Also test the old TestRunner for compatibility
    print("\n🧪 Testing legacy TestRunner...")

    def always_pass(code):
        return True

    def always_fail(code):
        return False

    tests = [
        {"name": "test_pass", "fn": always_pass},
        {"name": "test_fail", "fn": always_fail},
    ]
    prompt = "Dummy prompt"
    code = 'print("Hello World")'

    runner = TestRunner()
    results = runner.run_custom_tests(tests, prompt, code)
    print("Test Results:", results)
    print("Patterns logged in feedback/patterns.json")
