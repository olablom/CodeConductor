#!/usr/bin/env python3
"""
Test script using subprocess to run pylint
"""

import tempfile
import os
import subprocess
import re


def estimate_code_quality_subprocess(code: str) -> float:
    """Estimate code quality using pylint via subprocess."""
    if not code or not code.strip():
        return 0.0

    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name

        print(f"📝 Created temporary file: {temp_file_path}")

        # Run pylint via subprocess
        result = subprocess.run([
            'pylint',
            temp_file_path,
            '--disable=missing-module-docstring,missing-class-docstring,missing-function-docstring',
            '--score=yes',
            '--output-format=text'
        ], capture_output=True, text=True, timeout=30)
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        print(f"📊 Pylint output:\n{result.stdout}")
        if result.stderr:
            print(f"⚠️ Pylint stderr:\n{result.stderr}")
        
        # Extract score from output
        score_match = re.search(r'Your code has been rated at ([0-9.]+)/10', result.stdout)
        if score_match:
            score = float(score_match.group(1)) / 10.0
            print(f"📊 Extracted score: {score:.2f}/1.0")
            return score
        else:
            print("⚠️ Could not extract score from pylint output")
            return 0.5
        
    except subprocess.TimeoutExpired:
        print("⏰ Pylint timed out")
        return 0.5
    except Exception as e:
        print(f"❌ Error with pylint subprocess: {e}")
        return 0.5


def test_pylint_subprocess():
    """Test pylint with different code samples using subprocess."""
    print("🧪 Testing Pylint Code Quality Assessment (Subprocess)")
    print("=" * 60)
    
    test_cases = [
        {
            "name": "High Quality Code",
            "code": """
def fibonacci(n: int) -> int:
    '''Calculate the nth Fibonacci number.'''
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
""",
        },
        {
            "name": "Medium Quality Code",
            "code": """
def process_data(data):
    result = []
    for i in range(len(data)):
        result.append(data[i] * 2)
    return result
""",
        },
        {
            "name": "Low Quality Code",
            "code": """
def bad_function(x,y,z):
    a=x+y
    b=a*z
    print(b)
    return b
""",
        }
    ]
    
    for test_case in test_cases:
        print(f"\n📝 Testing: {test_case['name']}")
        print(f"💻 Code:\n{test_case['code']}")
        
        quality_score = estimate_code_quality_subprocess(test_case['code'])
        print(f"📊 Final Quality Score: {quality_score:.2f}/1.0")
        print("-" * 50)


if __name__ == "__main__":
    test_pylint_subprocess() 