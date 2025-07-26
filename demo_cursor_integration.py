#!/usr/bin/env python3
"""
Demo: Cursor Integration for CodeConductor

Shows how the ensemble → prompt → Cursor → code extraction pipeline works.
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ensemble.model_manager import ModelManager
from ensemble.query_dispatcher import QueryDispatcher
from ensemble.consensus_calculator import ConsensusCalculator
from generators.prompt_generator import PromptGenerator, PromptContext
from integrations.cursor_integration import CursorIntegration


class MockQueryResult:
    """Mock result class for testing."""

    def __init__(self, model_id: str, response: str, success: bool = True):
        self.model_id = model_id
        self.response = response
        self.success = success


async def create_mock_ensemble_data():
    """Create mock ensemble data that simulates real LLM responses."""
    print("Creating mock ensemble data...")

    # Simulate responses from different models to the same coding task
    mock_results = [
        MockQueryResult(
            "model-1",
            '{"task": "Create a simple calculator class", "approach": "Class-based implementation with arithmetic methods", "requirements": ["Add, subtract, multiply, divide methods", "Handle division by zero", "Include type hints"], "files_needed": ["calculator.py", "test_calculator.py"], "dependencies": ["pytest"], "complexity": "low"}',
        ),
        MockQueryResult(
            "model-2",
            '{"task": "Create a simple calculator class", "approach": "Object-oriented calculator with error handling", "requirements": ["Implement basic arithmetic operations", "Add input validation", "Write unit tests"], "files_needed": ["calculator.py", "test_calculator.py"], "dependencies": ["pytest"], "complexity": "low"}',
        ),
        MockQueryResult(
            "model-3",
            '{"task": "Create a simple calculator class", "approach": "Simple calculator class with methods", "requirements": ["Create Calculator class", "Add arithmetic methods", "Include tests"], "files_needed": ["calculator.py", "test_calculator.py"], "dependencies": ["pytest"], "complexity": "low"}',
        ),
    ]

    print(f"Created {len(mock_results)} mock model responses")
    return mock_results


async def demo_cursor_integration():
    """Demo the complete ensemble → Cursor → code extraction pipeline."""
    print("🚀 CodeConductor: Ensemble → Cursor Integration Demo")
    print("=" * 70)

    # Step 1: Create mock ensemble data
    print("\n📋 Step 1: Creating mock ensemble data...")
    mock_results = await create_mock_ensemble_data()

    # Step 2: Calculate consensus
    print("\n🧮 Step 2: Calculating consensus from ensemble...")
    calculator = ConsensusCalculator(min_confidence=0.6)
    consensus_result = calculator.calculate_consensus(mock_results)

    print(f"Consensus calculated:")
    print(f"   - Confidence: {consensus_result.confidence:.2f}")
    print(f"   - Consensus: {json.dumps(consensus_result.consensus, indent=2)}")
    print(f"   - Disagreements: {len(consensus_result.disagreements)}")

    # Step 3: Generate prompt from consensus
    print("\n📝 Step 3: Generating prompt from consensus...")
    generator = PromptGenerator()

    # Create context for the prompt
    context = PromptContext(
        project_structure="Standard Python project with src/ and tests/ directories",
        coding_standards=[
            "Use type hints",
            "Include docstrings",
            "Follow PEP 8",
            "Handle errors gracefully",
            "Write comprehensive tests",
        ],
        existing_patterns=["Use dataclasses", "Implement repository pattern"],
        dependencies=["pytest", "aiohttp"],
    )

    prompt = generator.generate(consensus_result.consensus, context)

    print(f"Generated prompt ({len(prompt)} characters)")
    print("\n📄 Generated Prompt:")
    print("=" * 60)
    print(prompt)
    print("=" * 60)

    # Step 4: Cursor Integration
    print("\n🎯 Step 4: Cursor Integration...")
    integration = CursorIntegration()

    # Create output directory for generated files
    output_dir = Path("generated_code")
    output_dir.mkdir(exist_ok=True)

    print(f"Output directory: {output_dir.absolute()}")

    # Run Cursor workflow
    generated_files = integration.run_cursor_workflow(prompt, output_dir)

    # Step 5: Summary
    print("\n" + "=" * 70)
    print("📊 Pipeline Summary:")
    print(f"   - Ensemble Models: {len(mock_results)}")
    print(f"   - Consensus Confidence: {consensus_result.confidence:.2f}")
    print(f"   - Generated Prompt Length: {len(prompt)} characters")
    print(f"   - Generated Files: {len(generated_files)}")

    if generated_files:
        print(f"\n✅ Successfully generated {len(generated_files)} files:")
        for file_path in generated_files:
            print(f"   - {file_path}")

        print(f"\n🎉 Pipeline complete! Ready for test execution!")
        print(f"   - Files saved to: {output_dir.absolute()}")
        print(f"   - Next step: Run tests on generated code")
    else:
        print(f"\n❌ No files were generated")
        print(f"   - Check Cursor output and try again")

    print("\n" + "=" * 70)


def demo_code_extraction():
    """Demo code extraction with sample Cursor output."""
    print("🔍 Code Extraction Demo")
    print("=" * 40)

    # Sample Cursor output
    sample_output = """
    Here's the implementation:
    
    ```python
    # calculator.py
    class Calculator:
        def __init__(self):
            self.history = []
        
        def add(self, a: float, b: float) -> float:
            result = a + b
            self.history.append(f"{a} + {b} = {result}")
            return result
        
        def subtract(self, a: float, b: float) -> float:
            result = a - b
            self.history.append(f"{a} - {b} = {result}")
            return result
    ```
    
    And here's the test file:
    
    ```python
    # test_calculator.py
    import pytest
    from calculator import Calculator
    
    def test_calculator_add():
        calc = Calculator()
        assert calc.add(2, 3) == 5
        assert len(calc.history) == 1
    
    def test_calculator_subtract():
        calc = Calculator()
        assert calc.subtract(5, 3) == 2
        assert len(calc.history) == 1
    ```
    """

    from integrations.cursor_integration import CodeExtractor

    extractor = CodeExtractor()
    extracted_files = extractor.extract_cursor_code(sample_output)

    print(f"Extracted {len(extracted_files)} files:")
    for file_path, code in extracted_files:
        print(f"\n📁 {file_path}:")
        print("-" * 30)
        print(code[:200] + "..." if len(code) > 200 else code)


async def main():
    """Main demo function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="CodeConductor Cursor Integration Demo"
    )
    parser.add_argument(
        "--extraction-only", action="store_true", help="Only run code extraction demo"
    )

    args = parser.parse_args()

    if args.extraction_only:
        demo_code_extraction()
    else:
        await demo_cursor_integration()


if __name__ == "__main__":
    asyncio.run(main())
