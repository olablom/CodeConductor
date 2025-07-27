#!/usr/bin/env python3
"""
Demo script for testing improved linter functionality and phi3 model integration.
"""

import logging
import time
from typing import Dict, Any
from ensemble.ensemble_engine import EnsembleEngine
from ensemble.model_manager import ModelManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_code_quality_with_pylint():
    """Test the improved code quality assessment with pylint."""
    logger.info("🧪 Testing Code Quality Assessment with Pylint")
    logger.info("=" * 60)
    
    from ensemble.code_reviewer import CodeReviewer
    
    # Create a simple code reviewer for testing
    models = ["phi3:mini", "codellama-7b-instruct", "mistral-7b-instruct-v0.1"]
    reviewer = CodeReviewer(models)
    
    # Test cases with different code quality levels
    test_cases = [
        {
            "name": "High Quality Code",
            "code": """
def fibonacci(n: int) -> int:
    '''Calculate the nth Fibonacci number.'''
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def validate_input(n: int) -> bool:
    if not isinstance(n, int) or n < 0:
        raise ValueError("Input must be a non-negative integer")
    return True
""",
            "expected_quality": "high"
        },
        {
            "name": "Medium Quality Code",
            "code": """
def process_data(data):
    result = []
    for i in range(len(data)):
        result.append(data[i] * 2)
    return result

def main():
    print("Starting...")
    data = [1, 2, 3, 4, 5]
    output = process_data(data)
    print(output)
    # TODO: Add error handling
""",
            "expected_quality": "medium"
        },
        {
            "name": "Low Quality Code",
            "code": """
def bad_function(x,y,z):
    a=x+y
    b=a*z
    print(b)
    return b

# TODO: fix this
# FIXME: needs work
""",
            "expected_quality": "low"
        }
    ]
    
    for test_case in test_cases:
        logger.info(f"\n📝 Testing: {test_case['name']}")
        logger.info(f"💻 Code:\n{test_case['code']}")
        
        quality_score = reviewer.estimate_code_quality(test_case['code'])
        logger.info(f"📊 Quality Score: {quality_score:.2f}/1.0")
        
        # Review the code
        review_result = reviewer.review_code(
            "Test task for code quality assessment",
            test_case['code']
        )
        
        logger.info(f"🤖 Reviewer: {review_result['reviewer']}")
        logger.info(f"🔧 Suggested Fixes: {len(review_result['suggested_fixes'])}")
        logger.info(f"💡 Fixes: {review_result['suggested_fixes']}")
        logger.info("-" * 40)


async def test_phi3_integration():
    """Test phi3 model integration for faster response times."""
    logger.info("\n🚀 Testing Phi3 Integration")
    logger.info("=" * 60)
    
    try:
        # Initialize model manager and ensemble engine
        model_manager = ModelManager()
        models = await model_manager.list_models()
        
        if not models:
            logger.error("❌ No models available")
            return
        
        logger.info(f"📦 Available models: {[m.id for m in models]}")
        
        # Check if phi3 is available
        phi3_models = [m for m in models if "phi3" in m.id.lower()]
        if not phi3_models:
            logger.warning("⚠️ No phi3 models found")
            return
        
        logger.info(f"✅ Found phi3 models: {[m.id for m in phi3_models]}")
        
        # Initialize ensemble engine
        engine = EnsembleEngine(models)
        
        # Test tasks
        test_tasks = [
            "Create a function to validate email addresses",
            "Create a function to sort a list of numbers",
            "Create a simple calculator function"
        ]
        
        for i, task in enumerate(test_tasks, 1):
            logger.info(f"\n📝 Task {i}: {task}")
            
            start_time = time.time()
            
            try:
                result = await engine.process_request(task)
                execution_time = time.time() - start_time
                
                logger.info(f"✅ Success!")
                logger.info(f"📊 Confidence: {result.confidence:.2f}")
                logger.info(f"⏱️  Execution Time: {execution_time:.2f}s")
                logger.info(f"🤖 Selected Model: {result.selected_model}")
                
                if hasattr(result, 'generated_code') and result.generated_code:
                    logger.info(f"💻 Code Length: {len(result.generated_code)} chars")
                else:
                    logger.info("💻 No code generated (timeout or error)")
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"❌ Error: {e}")
                logger.info(f"⏱️  Execution Time: {execution_time:.2f}s")
            
            logger.info("-" * 40)
    
    except Exception as e:
        logger.error(f"❌ Error in phi3 integration test: {e}")


async def compare_model_performance():
    """Compare performance between different models."""
    logger.info("\n⚖️  Model Performance Comparison")
    logger.info("=" * 60)
    
    try:
        model_manager = ModelManager()
        models = await model_manager.list_models()
        
        if not models:
            logger.error("❌ No models available")
            return
        
        # Test with a simple task
        test_task = "Create a function to calculate the factorial of a number"
        
        # Test each model individually
        for model in models[:3]:  # Test first 3 models
            logger.info(f"\n🤖 Testing model: {model.id}")
            
            start_time = time.time()
            
            try:
                # Create a simple ensemble with just this model
                single_model_engine = EnsembleEngine([model])
                result = await single_model_engine.process_request(test_task)
                execution_time = time.time() - start_time
                
                logger.info(f"✅ Success!")
                logger.info(f"📊 Confidence: {result.confidence:.2f}")
                logger.info(f"⏱️  Execution Time: {execution_time:.2f}s")
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"❌ Error: {e}")
                logger.info(f"⏱️  Execution Time: {execution_time:.2f}s")
            
            logger.info("-" * 30)
    
    except Exception as e:
        logger.error(f"❌ Error in performance comparison: {e}")


async def main():
    """Run all tests."""
    logger.info("🎼 CodeConductor Linter and Phi3 Demo")
    logger.info("=" * 60)
    
    # Test 1: Code quality assessment with pylint
    test_code_quality_with_pylint()
    
    # Test 2: Phi3 integration
    await test_phi3_integration()
    
    # Test 3: Model performance comparison
    await compare_model_performance()
    
    logger.info("\n🎉 All tests completed!")


if __name__ == "__main__":
    import argparse
    import asyncio
    
    parser = argparse.ArgumentParser(description="Demo script for linter and phi3 testing")
    parser.add_argument("--model", type=str, default="phi3:mini",
                       help="Specific model to test (default: phi3:mini)")
    parser.add_argument("--timeout", type=float, default=15.0,
                       help="Timeout for model requests (default: 15.0)")
    
    args = parser.parse_args()
    
    logger.info(f"🔧 Configuration: model={args.model}, timeout={args.timeout}s")
    
    asyncio.run(main()) 