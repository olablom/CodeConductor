#!/usr/bin/env python3
"""
Demo script for Multi-Agent Debugging with CodeReviewer
Tests the integration of CodeReviewer with the ensemble pipeline.
"""

import asyncio
import logging
from ensemble.ensemble_engine import EnsembleEngine, EnsembleRequest
from ensemble.code_reviewer import CodeReviewer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_code_reviewer_standalone():
    """Demo the CodeReviewer in standalone mode."""
    print("🔍 CodeReviewer Standalone Demo")
    print("=" * 50)

    # Create CodeReviewer with sample models
    models = ["phi3", "codellama", "mistral"]
    reviewer = CodeReviewer(models)

    # Sample code to review
    sample_code = """
def calculate_sum(a, b):
    return a + b

def process_data(data):
    result = []
    for item in data:
        result.append(item * 2)
    return result
"""

    task_description = "Create functions for basic calculations and data processing"

    print(f"📝 Task: {task_description}")
    print(f"💻 Code to review:\n{sample_code}")

    # Review the code
    review_results = reviewer.review_code(task_description, sample_code)

    print("\n📋 Review Results:")
    print(f"🤖 Reviewer: {review_results['reviewer']}")
    print(f"📊 Code Quality: {review_results['code_quality_score']:.2f}/1.0")
    print(f"🧪 Test Reward: {review_results['test_reward']:.2f}/1.0")

    print(f"\n💬 Comments:\n{review_results['comments']}")

    if review_results["suggested_fixes"]:
        print(f"\n🔧 Suggested Fixes:")
        for i, fix in enumerate(review_results["suggested_fixes"], 1):
            print(f"  {i}. {fix}")
    else:
        print("\n✅ No suggested fixes")

    print(f"\n📋 Summary:\n{reviewer.get_review_summary(review_results)}")


async def demo_ensemble_with_code_review():
    """Demo the ensemble engine with code review integration."""
    print("\n🚀 Ensemble with Code Review Demo")
    print("=" * 50)

    # Create ensemble engine with code review enabled
    engine = EnsembleEngine(use_rlhf=True, use_rag=True, use_code_reviewer=True)

    # Initialize the engine
    print("🔧 Initializing ensemble engine...")
    success = await engine.initialize()

    if not success:
        print("❌ Failed to initialize ensemble engine")
        return

    print("✅ Ensemble engine initialized successfully")

    # Create a test request
    request = EnsembleRequest(
        task_description="Create a function to validate email addresses",
        min_models=2,
        timeout=15.0,  # Balanced timeout for testing
    )

    print(f"📝 Processing request: {request.task_description}")

    # Process the request
    response = await engine.process_request(request)

    print(f"\n📊 Response Summary:")
    print(f"  🎯 Confidence: {response.confidence:.2f}")
    print(f"  ⏱️  Execution Time: {response.execution_time:.2f}s")
    print(f"  🤖 Selected Model: {response.selected_model}")

    if response.rlhf_action is not None:
        print(
            f"  🧠 RLHF Action: {response.rlhf_action} ({response.rlhf_action_description})"
        )

    # Check if code review was performed
    consensus_content = response.consensus.get("content", "")
    if consensus_content:
        print(f"\n💻 Generated Code:\n{consensus_content}")

        # Check for review summary
        review_summary = response.consensus.get("review_summary")
        if review_summary:
            print(f"\n🔍 Code Review Summary:\n{review_summary}")
        else:
            print(
                "\nℹ️  No code review performed (no code generated or review disabled)"
            )

    if response.disagreements:
        print(f"\n⚠️  Disagreements: {response.disagreements}")


async def demo_code_reviewer_vs_no_review():
    """Compare ensemble performance with and without code review."""
    print("\n⚖️  Code Review vs No Review Comparison")
    print("=" * 50)

    # Test with code review enabled
    print("🔍 Testing WITH code review...")
    engine_with_review = EnsembleEngine(use_code_reviewer=True)
    await engine_with_review.initialize()

    request = EnsembleRequest(
        task_description="Create a function to sort a list of numbers",
        min_models=2,
        timeout=15.0,  # Balanced timeout for testing
    )

    response_with_review = await engine_with_review.process_request(request)

    # Test without code review
    print("🔍 Testing WITHOUT code review...")
    engine_no_review = EnsembleEngine(use_code_reviewer=False)
    await engine_no_review.initialize()

    response_no_review = await engine_no_review.process_request(request)

    # Compare results
    print(f"\n📊 Comparison Results:")
    print(f"  🔍 With Review:")
    print(f"    - Confidence: {response_with_review.confidence:.2f}")
    print(f"    - Execution Time: {response_with_review.execution_time:.2f}s")
    print(
        f"    - Code Length: {len(response_with_review.consensus.get('content', ''))} chars"
    )

    print(f"  🔍 Without Review:")
    print(f"    - Confidence: {response_no_review.confidence:.2f}")
    print(f"    - Execution Time: {response_no_review.execution_time:.2f}s")
    print(
        f"    - Code Length: {len(response_no_review.consensus.get('content', ''))} chars"
    )

    # Check for review summary
    review_summary = response_with_review.consensus.get("review_summary")
    if review_summary:
        print(f"\n🔍 Review Summary:\n{review_summary}")


async def demo_code_reviewer_capabilities():
    """Demo the various capabilities of the CodeReviewer."""
    print("\n🎯 CodeReviewer Capabilities Demo")
    print("=" * 50)

    models = ["phi3", "codellama", "mistral"]
    reviewer = CodeReviewer(models)

    # Test different types of code
    test_cases = [
        {
            "name": "Good Code",
            "code": """
def fibonacci(n):
    '''Calculate the nth Fibonacci number.'''
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def validate_input(n):
    if not isinstance(n, int) or n < 0:
        raise ValueError("Input must be a non-negative integer")
    return True
""",
            "task": "Create a recursive Fibonacci function with input validation",
        },
        {
            "name": "Code with Issues",
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
            "task": "Process a list of numbers and print results",
        },
    ]

    for test_case in test_cases:
        print(f"\n📝 Testing: {test_case['name']}")
        print(f"💻 Code:\n{test_case['code']}")

        review_results = reviewer.review_code(test_case["task"], test_case["code"])

        print(f"📊 Quality Score: {review_results['code_quality_score']:.2f}/1.0")
        print(f"🔧 Suggested Fixes: {len(review_results['suggested_fixes'])}")

        if review_results["suggested_fixes"]:
            print("💡 Fixes:")
            for fix in review_results["suggested_fixes"]:
                print(f"  - {fix}")

        print(f"📋 Summary:\n{reviewer.get_review_summary(review_results)}")


async def main():
    """Run all demos."""
    print("🎼 CodeConductor Multi-Agent Debugging Demo")
    print("=" * 60)

    try:
        # Demo 1: Standalone CodeReviewer
        await demo_code_reviewer_standalone()

        # Demo 2: Ensemble with Code Review
        await demo_ensemble_with_code_review()

        # Demo 3: Comparison
        await demo_code_reviewer_vs_no_review()

        # Demo 4: Capabilities
        await demo_code_reviewer_capabilities()

        print("\n🎉 All demos completed successfully!")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
