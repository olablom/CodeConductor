import os
import tempfile
import subprocess
import shutil
import re
from typing import List, Tuple, Dict, Any


class TestRunner:
    """
    MVP Test Runner & Analyzer
    - Writes code files to temp dir
    - Runs pytest
    - Captures and parses results
    - Extracts errors and stack traces
    - Returns structured feedback
    """

    def __init__(self, timeout: int = 20):
        self.timeout = timeout

    def run_tests(self, code_blocks: List[Tuple[str, str]]) -> Dict[str, Any]:
        # 1. Create temp dir
        temp_dir = tempfile.mkdtemp(prefix="codeconductor_")
        results = {
            "status": "error",
            "passed": 0,
            "failed": 0,
            "errors": [],
            "stdout": "",
            "stderr": "",
            "coverage": None,
            "test_files": [],
        }
        try:
            # 2. Write code files
            for filename, code in code_blocks:
                file_path = os.path.join(temp_dir, filename)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(code)
                if filename.startswith("test_") or filename.endswith("_test.py"):
                    results["test_files"].append(filename)
            # 3. Run pytest
            proc = subprocess.run(
                ["pytest", temp_dir, "--maxfail=5", "--disable-warnings", "-q"],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            results["stdout"] = proc.stdout
            results["stderr"] = proc.stderr
            # 4. Parse results
            self._parse_pytest_output(proc.stdout, results)
            results["status"] = (
                "pass" if results["failed"] == 0 and results["passed"] > 0 else "fail"
            )
            # 5. Extract errors
            results["errors"] = self._extract_errors(proc.stdout + proc.stderr)
        except subprocess.TimeoutExpired:
            results["status"] = "timeout"
            results["errors"].append({"error": "Test execution timed out"})
        except Exception as e:
            results["status"] = "error"
            results["errors"].append({"error": str(e)})
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        return results

    def _parse_pytest_output(self, output: str, results: Dict[str, Any]):
        # Example: "3 passed, 1 failed, 1 error in 0.12s"
        match = re.search(r"(\d+) passed", output)
        if match:
            results["passed"] = int(match.group(1))
        match = re.search(r"(\d+) failed", output)
        if match:
            results["failed"] = int(match.group(1))
        match = re.search(r"(\d+) error", output)
        if match:
            results["errors_count"] = int(match.group(1))

    def _extract_errors(self, output: str) -> List[Dict[str, Any]]:
        # Extract error blocks from pytest output
        errors = []
        error_blocks = re.split(r"=+ FAILURES =+", output)
        if len(error_blocks) > 1:
            failures = error_blocks[1]
            # Split by test case
            for block in re.split(r"_+ test", failures):
                if not block.strip():
                    continue
                # Extract test name
                test_name_match = re.search(r"([\w_]+)\s+\[.*\]|([\w_]+)", block)
                test_name = test_name_match.group(1) if test_name_match else "unknown"
                # Extract error message
                error_msg = "\n".join(block.strip().splitlines()[-5:])
                errors.append({"test": test_name, "error": error_msg})
        # Also look for assertion errors
        for match in re.finditer(r"AssertionError:.*", output):
            errors.append({"error": match.group(0)})
        return errors
