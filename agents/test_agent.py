"""
TestAgent - Automated Testing and Feedback

Provides automated testing capabilities for generated code.
Essential for the RL feedback loop and quality assessment.
"""

import subprocess
import ast
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple
import tempfile
import json


class TestAgent:
    """Agent responsible for automated testing and code quality assessment"""

    def __init__(self):
        self.test_results = []
        self.quality_metrics = {}

    def analyze_code(self, code: str, project_description: str) -> Dict[str, Any]:
        """
        Analyze generated code for quality and potential issues

        Args:
            code: Generated code to analyze
            project_description: Original project description

        Returns:
            Dict with analysis results
        """

        analysis = {
            "syntax_valid": self._check_syntax(code),
            "complexity_score": self._calculate_complexity(code),
            "test_coverage": self._estimate_test_coverage(code),
            "security_issues": self._check_security(code),
            "best_practices": self._check_best_practices(code),
            "performance_indicators": self._check_performance(code),
            "overall_score": 0.0,
            "recommendations": [],
        }

        # Calculate overall score
        analysis["overall_score"] = self._calculate_overall_score(analysis)

        # Generate recommendations
        analysis["recommendations"] = self._generate_recommendations(analysis)

        return analysis

    def run_tests(self, code: str, test_type: str = "basic", project_path: Path = None) -> Dict[str, Any]:
        """
        Run automated tests on generated code

        Args:
            code: Code to test
            test_type: Type of tests to run (basic, comprehensive, security)
            project_path: Path to project directory (for multi-file projects)

        Returns:
            Dict with test results
        """

        test_results = {
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "coverage": 0.0,
            "execution_time": 0.0,
            "errors": [],
            "warnings": [],
        }

        try:
            if project_path and project_path.is_dir():
                # Multi-file project testing
                test_results.update(self._run_project_tests(project_path, test_type))
            else:
                # Single file testing
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                    f.write(code)
                    temp_file = f.name

                # Run different types of tests
                if test_type == "basic":
                    test_results.update(self._run_basic_tests(temp_file))
                elif test_type == "comprehensive":
                    test_results.update(self._run_comprehensive_tests(temp_file))
                elif test_type == "security":
                    test_results.update(self._run_security_tests(temp_file))

                # Cleanup
                Path(temp_file).unlink()

        except Exception as e:
            test_results["errors"].append(f"Test execution failed: {str(e)}")

        return test_results

    def _check_syntax(self, code: str) -> bool:
        """Check if code has valid Python syntax"""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def _calculate_complexity(self, code: str) -> float:
        """Calculate code complexity score (0-10, lower is better)"""
        try:
            tree = ast.parse(code)

            # Count various complexity indicators
            functions = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
            classes = len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
            imports = len([node for node in ast.walk(tree) if isinstance(node, ast.Import)])
            imports_from = len([node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)])

            # Count nested structures
            nested_levels = 0
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                    nested_levels += 1

            # Calculate complexity score
            complexity = (functions * 0.5 + classes * 1.0 + nested_levels * 0.3) / 10
            return min(complexity, 10.0)

        except SyntaxError:
            return 10.0  # Maximum complexity for invalid syntax

    def _estimate_test_coverage(self, code: str) -> float:
        """Estimate test coverage based on code structure"""
        try:
            tree = ast.parse(code)

            # Count testable elements
            functions = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
            classes = len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])

            # Simple heuristic: more functions/classes = lower coverage estimate
            total_elements = functions + classes
            if total_elements == 0:
                return 100.0  # No code to test

            # Estimate coverage (this is a simplified heuristic)
            coverage = max(0, 100 - (total_elements * 5))
            return min(coverage, 100.0)

        except SyntaxError:
            return 0.0

    def _check_security(self, code: str) -> List[str]:
        """Check for common security issues"""
        security_issues = []

        # Check for hardcoded secrets
        if re.search(r'password\s*=\s*["\'][^"\']+["\']', code, re.IGNORECASE):
            security_issues.append("Hardcoded password detected")

        if re.search(r'secret\s*=\s*["\'][^"\']+["\']', code, re.IGNORECASE):
            security_issues.append("Hardcoded secret detected")

        if re.search(r'api_key\s*=\s*["\'][^"\']+["\']', code, re.IGNORECASE):
            security_issues.append("Hardcoded API key detected")

        # Check for SQL injection patterns
        if re.search(r'execute\s*\(\s*["\'][^"\']*\+', code, re.IGNORECASE):
            security_issues.append("Potential SQL injection vulnerability")

        # Check for eval usage
        if "eval(" in code:
            security_issues.append("eval() usage detected - security risk")

        # Check for exec usage
        if "exec(" in code:
            security_issues.append("exec() usage detected - security risk")

        return security_issues

    def _check_best_practices(self, code: str) -> List[str]:
        """Check for Python best practices"""
        issues = []

        # Check for proper imports
        if "import *" in code:
            issues.append("Wildcard import detected - use specific imports")

        # Check for proper exception handling
        if "except:" in code and "except Exception:" not in code:
            issues.append("Bare except clause detected - specify exception type")

        # Check for proper string formatting
        if code.count("%") > code.count(".format(") + code.count('f"'):
            issues.append("Consider using f-strings or .format() instead of % formatting")

        # Check for proper variable naming
        if re.search(r"\b[a-z]+\d*\s*=\s*", code):
            # This is a simplified check - in practice you'd want more sophisticated analysis
            pass

        return issues

    def _check_performance(self, code: str) -> List[str]:
        """Check for performance issues"""
        issues = []

        # Check for inefficient patterns
        if "for i in range(len(" in code:
            issues.append("Consider using enumerate() instead of range(len())")

        if ".keys()" in code and "in" in code:
            issues.append("Unnecessary .keys() call in membership test")

        # Check for potential memory issues
        if code.count("list(") > 3:
            issues.append("Multiple list() calls detected - consider alternatives")

        return issues

    def _calculate_overall_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall quality score (0-10)"""
        score = 10.0

        # Deduct for syntax errors
        if not analysis["syntax_valid"]:
            score -= 5.0

        # Deduct for complexity
        score -= analysis["complexity_score"] * 0.3

        # Deduct for security issues
        score -= len(analysis["security_issues"]) * 1.0

        # Deduct for best practice violations
        score -= len(analysis["best_practices"]) * 0.5

        # Deduct for performance issues
        score -= len(analysis["performance_indicators"]) * 0.3

        return max(0.0, min(10.0, score))

    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []

        if not analysis["syntax_valid"]:
            recommendations.append("Fix syntax errors before proceeding")

        if analysis["complexity_score"] > 5:
            recommendations.append("Consider breaking down complex functions")

        if analysis["security_issues"]:
            recommendations.append("Address security vulnerabilities")

        if analysis["best_practices"]:
            recommendations.append("Follow Python best practices")

        if analysis["performance_indicators"]:
            recommendations.append("Optimize performance-critical sections")

        if analysis["test_coverage"] < 50:
            recommendations.append("Add more comprehensive tests")

        return recommendations

    def _run_basic_tests(self, file_path: str) -> Dict[str, Any]:
        """Run basic syntax and import tests"""
        results = {
            "tests_run": 2,
            "tests_passed": 0,
            "tests_failed": 0,
            "errors": [],
            "warnings": [],
        }

        try:
            # Test 1: Syntax check
            result = subprocess.run(
                ["python", "-m", "py_compile", file_path],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                results["tests_passed"] += 1
            else:
                results["tests_failed"] += 1
                results["errors"].append(f"Syntax error: {result.stderr}")

            # Test 2: Import test
            result = subprocess.run(
                ["python", "-c", f"import ast; exec(open('{file_path}').read())"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                results["tests_passed"] += 1
            else:
                results["tests_failed"] += 1
                results["warnings"].append(f"Import/runtime warning: {result.stderr}")

        except subprocess.TimeoutExpired:
            results["tests_failed"] += 1
            results["errors"].append("Test timeout")
        except Exception as e:
            results["tests_failed"] += 1
            results["errors"].append(f"Test error: {str(e)}")

        return results

    def _run_comprehensive_tests(self, file_path: str) -> Dict[str, Any]:
        """Run comprehensive tests including linting"""
        results = {
            "tests_run": 3,
            "tests_passed": 0,
            "tests_failed": 0,
            "errors": [],
            "warnings": [],
        }

        # Run basic tests first
        basic_results = self._run_basic_tests(file_path)
        results["tests_passed"] += basic_results["tests_passed"]
        results["tests_failed"] += basic_results["tests_failed"]
        results["errors"].extend(basic_results["errors"])
        results["warnings"].extend(basic_results["warnings"])

        # Additional comprehensive test
        try:
            # Check for common issues
            with open(file_path, "r") as f:
                code = f.read()

            # Check for proper docstrings
            if "def " in code and '"""' not in code and "'''" not in code:
                results["warnings"].append("Consider adding docstrings to functions")

            results["tests_passed"] += 1

        except Exception as e:
            results["tests_failed"] += 1
            results["errors"].append(f"Comprehensive test error: {str(e)}")

        return results

    def _run_security_tests(self, file_path: str) -> Dict[str, Any]:
        """Run security-focused tests"""
        results = {
            "tests_run": 2,
            "tests_passed": 0,
            "tests_failed": 0,
            "errors": [],
            "warnings": [],
        }

        try:
            with open(file_path, "r") as f:
                code = f.read()

            # Security check 1: No eval/exec
            if "eval(" in code or "exec(" in code:
                results["tests_failed"] += 1
                results["errors"].append("Security risk: eval() or exec() detected")
            else:
                results["tests_passed"] += 1

            # Security check 2: No hardcoded secrets
            if re.search(r'password\s*=\s*["\'][^"\']+["\']', code, re.IGNORECASE):
                results["tests_failed"] += 1
                results["errors"].append("Security risk: Hardcoded password detected")
            else:
                results["tests_passed"] += 1

        except Exception as e:
            results["tests_failed"] += 1
            results["errors"].append(f"Security test error: {str(e)}")

        return results

    def _run_project_tests(self, project_path: Path, test_type: str) -> Dict[str, Any]:
        """Run tests on multi-file project"""
        results = {
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "errors": [],
            "warnings": [],
        }

        try:
            # Check if tests directory exists
            tests_dir = project_path / "tests"
            if not tests_dir.exists():
                results["warnings"].append("No tests directory found")
                return results

            # Run pytest on the project
            result = subprocess.run(
                ["python", "-m", "pytest", str(tests_dir), "-v"],
                capture_output=True,
                text=True,
                cwd=str(project_path),
                timeout=60,
            )

            # Parse pytest output
            if result.returncode == 0:
                # Count passed tests
                passed_lines = [line for line in result.stdout.split("\n") if "PASSED" in line]
                results["tests_passed"] = len(passed_lines)
                results["tests_run"] = len(passed_lines)
            else:
                # Count failed tests
                failed_lines = [line for line in result.stdout.split("\n") if "FAILED" in line]
                results["tests_failed"] = len(failed_lines)
                results["tests_run"] = len(failed_lines)
                results["errors"].append(f"Tests failed: {result.stdout}")

            # Run additional quality checks
            self._run_project_quality_checks(project_path, results)

        except subprocess.TimeoutExpired:
            results["errors"].append("Project test timeout")
        except Exception as e:
            results["errors"].append(f"Project test error: {str(e)}")

        return results

    def _run_project_quality_checks(self, project_path: Path, results: Dict[str, Any]):
        """Run quality checks on project files"""
        try:
            # Check all Python files for syntax
            python_files = list(project_path.rglob("*.py"))

            for py_file in python_files:
                try:
                    with open(py_file, "r") as f:
                        code = f.read()

                    # Check syntax
                    ast.parse(code)
                    results["tests_passed"] += 1
                    results["tests_run"] += 1

                except SyntaxError as e:
                    results["tests_failed"] += 1
                    results["tests_run"] += 1
                    results["errors"].append(f"Syntax error in {py_file}: {e}")

            # Check for required files
            required_files = ["main.py", "requirements.txt", "README.md"]
            for req_file in required_files:
                if (project_path / req_file).exists():
                    results["tests_passed"] += 1
                    results["tests_run"] += 1
                else:
                    results["tests_failed"] += 1
                    results["tests_run"] += 1
                    results["warnings"].append(f"Missing required file: {req_file}")

        except Exception as e:
            results["errors"].append(f"Quality check error: {str(e)}")

    def get_test_summary(self) -> Dict[str, Any]:
        """Get summary of all test results"""
        if not self.test_results:
            return {"total_tests": 0, "success_rate": 0.0, "average_score": 0.0}

        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results if result.get("tests_passed", 0) > 0)
        average_score = sum(result.get("overall_score", 0) for result in self.test_results) / total_tests

        return {
            "total_tests": total_tests,
            "success_rate": (successful_tests / total_tests) * 100 if total_tests > 0 else 0.0,
            "average_score": average_score,
        }
