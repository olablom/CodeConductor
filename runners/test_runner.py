#!/usr/bin/env python3
"""
Test Runner for CodeConductor MVP.

Handles pytest execution, result parsing, and feedback generation for the feedback loop.
"""

import subprocess
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
        Execute a single test specification.

        Args:
            test_spec: Dictionary containing test name and function

        Returns:
            Test result dictionary
        """
        try:
            # Assume test_spec contains name and a function to call
            test_fn = test_spec["fn"]
            result = test_fn(test_spec.get("code", ""))
            return {"name": test_spec["name"], "passed": result}
        except Exception as e:
            return {"name": test_spec["name"], "passed": False, "error": str(e)}

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
    # Dummy tests for demonstration
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
