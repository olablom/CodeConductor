#!/usr/bin/env python3
"""
Test-as-Reward Demonstration Script

This script demonstrates how to use the Test-as-Reward system
to log test results as rewards for the learning system.
"""

import os
import sys
import shutil
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from feedback.learning_system import LearningSystem, log_test_reward
from runners.test_runner import TestRunner


def demo_custom_tests():
    """Demonstrate Test-as-Reward with custom tests"""
    print("🎯 Demo: Custom Tests with Test-as-Reward")
    print("=" * 50)

    # Create test functions
    def test_addition(code):
        """Test if addition function works"""
        try:
            # Simulate testing the code
            if "def add" in code and "return" in code and "a + b" in code:
                return True
            return False
        except:
            return False

    def test_multiplication(code):
        """Test if multiplication function works"""
        try:
            if "def multiply" in code and "return" in code and "a * b" in code:
                return True
            return False
        except:
            return False

    def test_division(code):
        """Test if division function works"""
        try:
            if "def divide" in code and "return" in code and "a / b" in code:
                return True
            return False
        except:
            return False

    # Test scenarios
    test_scenarios = [
        {
            "name": "Good Code - All Tests Pass",
            "prompt": "Create functions for basic math operations",
            "code": """
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

def divide(a, b):
    return a / b if b != 0 else 0
""",
            "tests": [
                {"name": "test_addition", "fn": test_addition},
                {"name": "test_multiplication", "fn": test_multiplication},
                {"name": "test_division", "fn": test_division},
            ],
        },
        {
            "name": "Partial Code - Some Tests Pass",
            "prompt": "Create functions for basic math operations",
            "code": """
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b
""",
            "tests": [
                {"name": "test_addition", "fn": test_addition},
                {"name": "test_multiplication", "fn": test_multiplication},
                {"name": "test_division", "fn": test_division},
            ],
        },
        {
            "name": "Bad Code - No Tests Pass",
            "prompt": "Create functions for basic math operations",
            "code": """
print("Hello World")
""",
            "tests": [
                {"name": "test_addition", "fn": test_addition},
                {"name": "test_multiplication", "fn": test_multiplication},
                {"name": "test_division", "fn": test_division},
            ],
        },
    ]

    # Run tests and log rewards
    runner = TestRunner()
    learning_system = LearningSystem()

    for scenario in test_scenarios:
        print(f"\n📝 Scenario: {scenario['name']}")
        print(f"Prompt: {scenario['prompt']}")
        print(f"Code: {scenario['code'].strip()}")

        # Run tests
        results = runner.run_custom_tests(
            tests=scenario["tests"],
            prompt=scenario["prompt"],
            code=scenario["code"],
            metadata={"scenario": scenario["name"]},
        )

        # Show results
        passed = sum(1 for r in results if r.get("passed"))
        total = len(results)
        print(f"Results: {passed}/{total} tests passed")
        print(f"Test details: {results}")

    # Show learning system statistics
    print(f"\n📊 Learning System Statistics:")
    stats = learning_system.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")


def demo_pytest_integration():
    """Demonstrate Test-as-Reward with pytest"""
    print("\n🧪 Demo: Pytest Integration with Test-as-Reward")
    print("=" * 50)

    # Create a simple test file
    test_file_content = """
import pytest

def test_simple_addition():
    assert 1 + 1 == 2

def test_simple_multiplication():
    assert 2 * 3 == 6

def test_failing_test():
    assert 1 == 2  # This will fail
"""

    test_dir = Path("temp_test_dir")
    test_dir.mkdir(exist_ok=True)

    # Write test file
    test_file = test_dir / "test_demo.py"
    with open(test_file, "w") as f:
        f.write(test_file_content)

    try:
        # Run pytest with Test-as-Reward
        runner = TestRunner()
        prompt = "Create a simple math function"
        code = """
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b
"""

        print(f"Running pytest on {test_dir}")
        result = runner.run_pytest(
            test_dir=test_dir,
            prompt=prompt,
            code=code,
            metadata={"test_type": "pytest_demo"},
        )

        print(f"Pytest Success: {result.success}")
        print(f"Test Results: {len(result.test_results)} tests processed")
        print(f"Errors: {len(result.errors)}")

        if result.test_results:
            passed = sum(1 for t in result.test_results if t.get("passed"))
            total = len(result.test_results)
            print(f"Parsed Results: {passed}/{total} tests passed")

    finally:
        # Cleanup - use shutil.rmtree for recursive removal
        if test_dir.exists():
            shutil.rmtree(test_dir)


def show_patterns():
    """Show saved patterns"""
    print("\n📚 Saved Patterns:")
    print("=" * 50)

    learning_system = LearningSystem()
    patterns = learning_system.get_patterns()

    if not patterns:
        print("No patterns saved yet.")
        return

    for i, pattern in enumerate(patterns[-5:], 1):  # Show last 5 patterns
        print(f"\nPattern {i}:")
        print(f"  Timestamp: {pattern.timestamp}")
        print(f"  Task: {pattern.task_description}")
        print(f"  Reward: {pattern.reward:.2f}" if pattern.reward else "  Reward: None")
        print(
            f"  Code Preview: {pattern.code[:100]}..."
            if len(pattern.code) > 100
            else f"  Code: {pattern.code}"
        )


def main():
    """Main demonstration function"""
    print("🚀 Test-as-Reward System Demonstration")
    print("=" * 60)

    # Demo 1: Custom tests
    demo_custom_tests()

    # Demo 2: Pytest integration
    demo_pytest_integration()

    # Show saved patterns
    show_patterns()

    print(f"\n✅ Demonstration complete!")
    print(f"📁 Check feedback/patterns.json for saved patterns")
    print(f"🎯 Test-as-Reward system is ready for RLHF integration!")


if __name__ == "__main__":
    main()
