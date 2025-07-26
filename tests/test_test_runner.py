#!/usr/bin/env python3
"""
Unit tests for Test Runner component.
"""

import pytest
import tempfile
import os
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from runners.test_runner import TestRunner, TestResult


class TestTestRunner:
    """Test Test Runner functionality."""

    def test_run_pytest_success(self):
        """Test successful pytest execution."""
        runner = TestRunner()

        # Create temporary test files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a simple test file that passes
            test_file = temp_path / "test_simple.py"
            test_file.write_text("""
import pytest

def test_pass():
    assert True

def test_basic_math():
    assert 2 + 2 == 4
""")

            # Run pytest
            result = runner.run_pytest(temp_path)

            assert result.success is True
            assert result.errors == []
            # Check that tests passed (pytest shows ".." for passed tests)
            assert "passed" in result.stdout

    def test_run_pytest_failure(self):
        """Test pytest execution with failures."""
        runner = TestRunner()

        # Create temporary test files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a test file with failures
            test_file = temp_path / "test_failing.py"
            test_file.write_text("""
import pytest

def test_pass():
    assert True

def test_fail():
    assert False, "This test should fail"

def test_error():
    raise ValueError("This test raises an error")
""")

            # Run pytest
            result = runner.run_pytest(temp_path)

            assert result.success is False
            assert len(result.errors) >= 1  # At least one error
            # Check that both error messages are present in the combined error output
            combined_errors = " ".join(result.errors)
            assert "This test should fail" in combined_errors
            assert "This test raises an error" in combined_errors

    def test_run_pytest_no_tests(self):
        """Test pytest execution with no test files."""
        runner = TestRunner()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # No test files
            result = runner.run_pytest(temp_path)

            assert result.success is True  # No tests found is considered success
            assert result.errors == ["No test files found"]

    def test_run_pytest_syntax_error(self):
        """Test pytest execution with syntax errors."""
        runner = TestRunner()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a test file with syntax error
            test_file = temp_path / "test_syntax_error.py"
            test_file.write_text("""
import pytest

def test_syntax_error():
    if True
        assert True  # Missing colon
""")

            # Run pytest
            result = runner.run_pytest(temp_path)

            assert result.success is False
            assert len(result.errors) > 0
            combined_errors = " ".join(result.errors)
            assert "SyntaxError" in combined_errors

    def test_run_pytest_import_error(self):
        """Test pytest execution with import errors."""
        runner = TestRunner()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a test file with import error
            test_file = temp_path / "test_import_error.py"
            test_file.write_text("""
import pytest
from nonexistent_module import nonexistent_function

def test_import_error():
    assert nonexistent_function() == True
""")

            # Run pytest
            result = runner.run_pytest(temp_path)

            assert result.success is False
            assert len(result.errors) > 0
            combined_errors = " ".join(result.errors)
            assert "ModuleNotFoundError" in combined_errors

    def test_extract_errors_failures(self):
        """Test error extraction from FAILURES section."""
        runner = TestRunner()

        output = """
=== FAILURES ===
__________________________________ test_fail __________________________________

    def test_fail():
>       assert False, "This test should fail"
E       AssertionError: This test should fail

test_file.py:5: AssertionError
---
__________________________________ test_error __________________________________

    def test_error():
>       raise ValueError("This test raises an error")
E       ValueError: This test raises an error

test_file.py:8: ValueError
"""

        errors = runner._extract_errors(output)
        assert len(errors) >= 1
        combined_errors = " ".join(errors)
        assert "This test should fail" in combined_errors
        assert "This test raises an error" in combined_errors

    def test_extract_errors_errors(self):
        """Test error extraction from ERRORS section."""
        runner = TestRunner()

        output = """
=== ERRORS ===
_________________________________ ERROR collecting test_file.py _________________________________

ImportError while importing test module 'test_file.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
    from nonexistent_module import nonexistent_function
E   ModuleNotFoundError: No module named 'nonexistent_module'
"""

        errors = runner._extract_errors(output)
        assert len(errors) >= 1
        combined_errors = " ".join(errors)
        assert "ModuleNotFoundError" in combined_errors

    def test_extract_errors_both_sections(self):
        """Test error extraction from both FAILURES and ERRORS sections."""
        runner = TestRunner()

        output = """
=== FAILURES ===
__________________________________ test_fail __________________________________

    def test_fail():
>       assert False, "This test should fail"
E       AssertionError: This test should fail

test_file.py:5: AssertionError
---
__________________________________ test_error __________________________________

    def test_error():
>       raise ValueError("This test raises an error")
E       ValueError: This test raises an error

test_file.py:8: ValueError

=== ERRORS ===
_________________________________ ERROR collecting test_file2.py _________________________________

ImportError while importing test module 'test_file2.py'.
    from nonexistent_module import nonexistent_function
E   ModuleNotFoundError: No module named 'nonexistent_module'
"""

        errors = runner._extract_errors(output)
        assert len(errors) >= 1
        combined_errors = " ".join(errors)
        assert "This test should fail" in combined_errors
        assert "This test raises an error" in combined_errors
        assert "ModuleNotFoundError" in combined_errors


class TestTestResult:
    """Test TestResult dataclass."""

    def test_test_result_creation(self):
        """Test TestResult creation."""
        result = TestResult(
            success=True, stdout="test output", errors=["error1", "error2"]
        )

        assert result.success is True
        assert result.stdout == "test output"
        assert result.errors == ["error1", "error2"]

    def test_test_result_defaults(self):
        """Test TestResult default values."""
        result = TestResult(success=False, stdout="")

        assert result.success is False
        assert result.stdout == ""
        assert result.errors == []
