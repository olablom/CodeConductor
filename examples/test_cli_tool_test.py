#!/usr/bin/env python3
"""
Test file for CLI Tool Smoke Testing
"""

import pytest
import subprocess
import tempfile
import os
from pathlib import Path

def test_cli_basic_functionality():
    """Test basic CLI functionality."""
    # Create a temporary input file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("hello world")
        input_file = f.name
    
    try:
        # Test basic processing
        result = subprocess.run([
            'python', 'test_cli_tool.py', 
            '--input', input_file,
            '--verbose'
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        assert "HELLO WORLD" in result.stdout
        
    finally:
        # Cleanup
        os.unlink(input_file)

def test_cli_with_output_file():
    """Test CLI with output file."""
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("test data")
        input_file = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        output_file = f.name
    
    try:
        # Test with output file
        result = subprocess.run([
            'python', 'test_cli_tool.py',
            '--input', input_file,
            '--output', output_file
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        
        # Check output file content
        with open(output_file, 'r') as f:
            content = f.read()
            assert content == "TEST DATA"
            
    finally:
        # Cleanup
        os.unlink(input_file)
        os.unlink(output_file)

def test_cli_error_handling():
    """Test CLI error handling."""
    # Test with non-existent file
    result = subprocess.run([
        'python', 'test_cli_tool.py',
        '--input', 'non_existent_file.txt'
    ], capture_output=True, text=True)
    
    assert result.returncode == 1
    assert "not found" in result.stderr

def test_cli_help():
    """Test CLI help functionality."""
    result = subprocess.run([
        'python', 'test_cli_tool.py',
        '--help'
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
    assert "Simple CLI Tool for Testing" in result.stdout
    assert "--input" in result.stdout
    assert "--output" in result.stdout

if __name__ == "__main__":
    pytest.main([__file__]) 