#!/usr/bin/env python3
"""
Test file for PytestRunner integration.
"""


def test_simple_function():
    """Test a simple function."""

    def add(a, b):
        return a + b

    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0, 0) == 0


def test_string_operations():
    """Test string operations."""
    text = "Hello, World!"
    assert len(text) == 13
    assert text.upper() == "HELLO, WORLD!"
    assert text.lower() == "hello, world!"


def test_list_operations():
    """Test list operations."""
    numbers = [1, 2, 3, 4, 5]
    assert len(numbers) == 5
    assert sum(numbers) == 15
    assert max(numbers) == 5
    assert min(numbers) == 1


def test_dictionary_operations():
    """Test dictionary operations."""
    data = {"name": "Test", "value": 42}
    assert "name" in data
    assert data["name"] == "Test"
    assert data["value"] == 42
    assert len(data) == 2


if __name__ == "__main__":
    # Run tests manually
    test_simple_function()
    test_string_operations()
    test_list_operations()
    test_dictionary_operations()
    print("All tests passed!")
