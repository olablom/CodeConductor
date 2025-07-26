#!/usr/bin/env python3
"""
Test Runner for CodeConductor MVP.

Handles pytest execution, result parsing, and feedback generation for the feedback loop.
"""

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import re


@dataclass
class TestResult:
    """
    Represents the result of a pytest run.
    """

    success: bool
    stdout: str
    errors: List[str] = field(default_factory=list)


class TestRunner:
    """
    Executes pytest on a given directory and extracts failure and error details.
    """

    def __init__(self, timeout: int = 60) -> None:
        self.timeout = timeout

    def run_pytest(self, test_dir: Path) -> TestResult:
        """
        Run pytest in default mode (full tracebacks) on the given directory.

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

        # Handle different cases
        if errors:
            # Has errors = failure
            return TestResult(
                success=False,
                stdout=output,
                errors=errors,
            )
        elif re.search(r"collected\s+0\s+items", output):
            # No tests collected and no errors = no test files
            return TestResult(
                success=True,
                stdout=output,
                errors=["No test files found"],
            )
        else:
            # No errors = success
            return TestResult(
                success=True,
                stdout=output,
                errors=errors,
            )

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
