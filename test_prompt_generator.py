#!/usr/bin/env python3
"""
Test script for Prompt Generator component.
"""

import asyncio
import json
import logging
from generators import PromptGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def test_prompt_generator():
    """Test the PromptGenerator with different task types."""
    print("🚀 Testing CodeConductor Prompt Generator")
    print("=" * 50)

    generator = PromptGenerator()

    # Test cases
    test_cases = [
        {
            "name": "Function Task",
            "task": "Create a simple Python function that calculates the factorial of a number",
            "consensus": {
                "approach": "Recursive function to calculate factorial",
                "complexity": "low",
                "reasoning": "A recursive function is a good approach for this problem because it allows us to avoid having to keep track of the intermediate results and instead, we can simply call the function again with the next number in the sequence. This makes the code more concise and easier to understand.",
                "files_needed": ["factorial.py"],
                "dependencies": [],
            },
        },
        {
            "name": "Class Task",
            "task": "Create a Calculator class with basic arithmetic operations",
            "consensus": {
                "approach": "Class-based calculator with arithmetic methods",
                "complexity": "medium",
                "reasoning": "A class-based approach provides encapsulation and allows for easy extension of functionality. Each arithmetic operation can be a separate method.",
                "files_needed": ["calculator.py"],
                "dependencies": [],
            },
        },
        {
            "name": "API Task",
            "task": "Create a REST API endpoint for user authentication",
            "consensus": {
                "approach": "Flask API endpoint with JWT authentication",
                "complexity": "high",
                "reasoning": "Flask provides a lightweight framework for creating REST APIs. JWT tokens are secure and stateless for authentication.",
                "files_needed": ["auth_api.py", "config.py"],
                "dependencies": ["flask", "pyjwt"],
            },
        },
        {
            "name": "Test Task",
            "task": "Write comprehensive tests for the factorial function",
            "consensus": {
                "approach": "Pytest test suite with edge cases",
                "complexity": "low",
                "reasoning": "Pytest is the standard testing framework for Python. We need to test normal cases, edge cases, and error conditions.",
                "files_needed": ["test_factorial.py"],
                "dependencies": ["pytest"],
            },
        },
    ]

    # Test context
    context = {
        "project_structure": "Standard Python project with src/ and tests/ directories",
        "coding_standards": "PEP 8, type hints, docstrings",
        "existing_patterns": "Async/await for I/O operations, error handling with try/except",
        "dependencies": ["pytest", "aiohttp", "streamlit"],
    }

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {test_case['name']}")
        print("-" * 30)

        try:
            # Generate prompt
            consensus_data = {
                "task": test_case["task"],
                "approach": test_case["consensus"]["approach"],
                "requirements": [
                    f"Implement {test_case['name'].lower()} functionality",
                    "Include proper error handling",
                    "Write comprehensive tests",
                ],
                "files_needed": test_case["consensus"]["files_needed"],
                "dependencies": test_case["consensus"]["dependencies"],
            }

            prompt = generator.generate(consensus_data)

            print(f"✅ Generated prompt ({len(prompt)} characters)")
            print("\n📄 Generated Prompt:")
            print("=" * 40)
            print(prompt)
            print("=" * 40)

        except Exception as e:
            print(f"❌ Failed to generate prompt: {e}")

    print(f"\n🎯 Prompt Generator Test Complete!")
    print(f"✅ Tested {len(test_cases)} different task types")


if __name__ == "__main__":
    test_prompt_generator()
