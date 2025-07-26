#!/usr/bin/env python3
"""
Unit tests for Cursor Integration components.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from integrations.cursor_integration import ClipboardManager, CodeExtractor


class TestClipboardManager:
    """Test Clipboard Manager functionality."""

    def test_copy_to_clipboard_success(self):
        """Test successful clipboard copy."""
        manager = ClipboardManager()
        test_text = "Test prompt for Cursor"

        with patch("pyperclip.copy") as mock_copy:
            result = manager.copy_to_clipboard(test_text)

            mock_copy.assert_called_once_with(test_text)
            assert result is True

    def test_copy_to_clipboard_failure(self):
        """Test clipboard copy failure."""
        manager = ClipboardManager()
        test_text = "Test prompt for Cursor"

        with patch("pyperclip.copy", side_effect=Exception("Clipboard error")):
            result = manager.copy_to_clipboard(test_text)

            assert result is False

    def test_read_from_clipboard_success(self):
        """Test successful clipboard read."""
        manager = ClipboardManager()
        expected_text = "Cursor generated code"

        with patch("pyperclip.paste", return_value=expected_text):
            result = manager.read_from_clipboard()

            assert result == expected_text

    def test_read_from_clipboard_failure(self):
        """Test clipboard read failure."""
        manager = ClipboardManager()

        with patch("pyperclip.paste", side_effect=Exception("Clipboard error")):
            result = manager.read_from_clipboard()

            assert result == ""

    def test_clipboard_roundtrip(self):
        """Test clipboard copy and read roundtrip."""
        manager = ClipboardManager()
        test_text = "Roundtrip test: ```python\nprint('hello')\n```"

        with (
            patch("pyperclip.copy") as mock_copy,
            patch("pyperclip.paste", return_value=test_text),
        ):
            # Copy to clipboard
            copy_result = manager.copy_to_clipboard(test_text)
            mock_copy.assert_called_once_with(test_text)
            assert copy_result is True

            # Read from clipboard
            read_result = manager.read_from_clipboard()
            assert read_result == test_text


class TestCodeExtractor:
    """Test Code Extractor functionality."""

    def test_extract_single_file(self):
        """Test extracting single file from Cursor output."""
        extractor = CodeExtractor()
        cursor_output = """
        Here's the implementation:
        
        ```python
        # calculator.py
        def add(a, b):
            return a + b
        ```
        
        The function adds two numbers.
        """

        result = extractor.extract_cursor_code(cursor_output)

        assert len(result) == 1
        file_path, code = result[0]
        assert file_path.name == "calculator.py"
        assert "def add(a, b):" in code
        assert "return a + b" in code

    def test_extract_multiple_files(self):
        """Test extracting multiple files from Cursor output."""
        extractor = CodeExtractor()
        cursor_output = """
        Here are the files:
        
        ```python
        # main.py
        from calculator import Calculator
        
        calc = Calculator()
        result = calc.add(5, 3)
        print(result)
        ```
        
        ```python
        # calculator.py
        class Calculator:
            def add(self, a, b):
                return a + b
        ```
        
        ```python
        # test_calculator.py
        import pytest
        from calculator import Calculator
        
        def test_add():
            calc = Calculator()
            assert calc.add(2, 3) == 5
        ```
        """

        result = extractor.extract_cursor_code(cursor_output)

        assert len(result) == 3

        # Check main.py
        main_file = next((f for f, _ in result if f.name == "main.py"), None)
        assert main_file is not None

        # Check calculator.py
        calc_file = next((f for f, _ in result if f.name == "calculator.py"), None)
        assert calc_file is not None

        # Check test_calculator.py
        test_file = next((f for f, _ in result if f.name == "test_calculator.py"), None)
        assert test_file is not None

    def test_extract_no_code_blocks(self):
        """Test handling output with no code blocks."""
        extractor = CodeExtractor()
        cursor_output = "This is just text with no code blocks."

        result = extractor.extract_cursor_code(cursor_output)

        assert result == []

    def test_extract_code_without_filename(self):
        """Test extracting code blocks without explicit filenames."""
        extractor = CodeExtractor()
        cursor_output = """
        Here's the code:
        
        ```python
        def factorial(n):
            if n <= 1:
                return 1
            return n * factorial(n - 1)
        ```
        """

        result = extractor.extract_cursor_code(cursor_output)

        assert len(result) == 1
        file_path, code = result[0]
        # Should generate default filename
        assert file_path.name.startswith("generated_")
        assert "def factorial(n):" in code

    def test_extract_mixed_content(self):
        """Test extracting from mixed content with explanations."""
        extractor = CodeExtractor()
        cursor_output = """
        I'll create a simple calculator. Here's the implementation:
        
        First, the main calculator class:
        
        ```python
        # calculator.py
        class Calculator:
            def __init__(self):
                self.history = []
            
            def add(self, a, b):
                result = a + b
                self.history.append(f"{a} + {b} = {result}")
                return result
        ```
        
        And here's a test file:
        
        ```python
        # test_calculator.py
        import pytest
        from calculator import Calculator
        
        def test_calculator_add():
            calc = Calculator()
            assert calc.add(2, 3) == 5
            assert len(calc.history) == 1
        ```
        
        The calculator keeps a history of operations.
        """

        result = extractor.extract_cursor_code(cursor_output)

        assert len(result) == 2

        # Check calculator.py
        calc_file, calc_code = next(
            (f, c) for f, c in result if f.name == "calculator.py"
        )
        assert "class Calculator:" in calc_code
        assert "def add(self, a, b):" in calc_code

        # Check test_calculator.py
        test_file, test_code = next(
            (f, c) for f, c in result if f.name == "test_calculator.py"
        )
        assert "def test_calculator_add():" in test_code
        assert "assert calc.add(2, 3) == 5" in test_code

    def test_extract_empty_input(self):
        """Test handling empty input."""
        extractor = CodeExtractor()

        result = extractor.extract_cursor_code("")
        assert result == []

        result = extractor.extract_cursor_code(None)
        assert result == []

    def test_extract_with_special_characters(self):
        """Test extracting code with special characters in filenames."""
        extractor = CodeExtractor()
        cursor_output = """
        ```python
        # my-special_file.py
        def special_function():
            return "special"
        ```
        """

        result = extractor.extract_cursor_code(cursor_output)

        assert len(result) == 1
        file_path, code = result[0]
        assert file_path.name == "my-special_file.py"
        assert "def special_function():" in code


if __name__ == "__main__":
    pytest.main([__file__])
