#!/usr/bin/env python3
"""
Simple test script for pylint integration in code_reviewer.py
"""

import logging
from ensemble.code_reviewer import CodeReviewer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_pylint_integration():
    """Test pylint integration in code reviewer."""
    logger.info("🧪 Testing Pylint Integration")
    logger.info("=" * 50)
    
    # Create a simple code reviewer
    models = ["phi3:mini", "codellama-7b-instruct"]
    reviewer = CodeReviewer(models)
    
    # Test cases
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
        logger.info(f"\n📝 Testing: {test_case['name']}")
        logger.info(f"💻 Code:\n{test_case['code']}")
        
        try:
            quality_score = reviewer.estimate_code_quality(test_case['code'])
            logger.info(f"📊 Quality Score: {quality_score:.2f}/1.0")
        except Exception as e:
            logger.error(f"❌ Error: {e}")
        
        logger.info("-" * 40)


if __name__ == "__main__":
    test_pylint_integration() 