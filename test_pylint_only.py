#!/usr/bin/env python3
"""
Minimal test script for pylint functionality only
"""

import tempfile
import os
import logging

# Try to import pylint
try:
    from pylint.lint import Run
    from pylint.reporters import CollectingReporter
    PYLINT_AVAILABLE = True
    print("✅ pylint is available")
except ImportError:
    PYLINT_AVAILABLE = False
    print("❌ pylint not available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def estimate_code_quality(code: str) -> float:
    """Estimate code quality using pylint."""
    if not code or not code.strip():
        return 0.0

    if not PYLINT_AVAILABLE:
        print("⚠️ pylint not available, using fallback")
        return 0.5

    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name

        print(f"📝 Created temporary file: {temp_file_path}")

        # Run pylint
        reporter = CollectingReporter()
        Run([
            temp_file_path, 
            '--disable=missing-module-docstring,missing-class-docstring,missing-function-docstring',
            '--score=yes'
        ], reporter=reporter)
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        # Pylint score is 0-10, normalize to 0-1
        score = reporter.linter.stats.global_note / 10.0
        print(f"📊 Pylint score: {score:.2f}/1.0")
        return score
        
    except Exception as e:
        print(f"❌ Error with pylint: {e}")
        return 0.5


def test_pylint():
    """Test pylint with different code samples."""
    print("🧪 Testing Pylint Code Quality Assessment")
    print("=" * 50)
    
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
        
        quality_score = estimate_code_quality(test_case['code'])
        print(f"📊 Final Quality Score: {quality_score:.2f}/1.0")
        print("-" * 40)


if __name__ == "__main__":
    test_pylint() 