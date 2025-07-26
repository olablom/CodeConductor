#!/usr/bin/env python3
"""
CodeConductor MVP - Simple Working Pipeline Demo

Demonstrates the complete working pipeline without complex consensus calculation.
"""

import asyncio
from pathlib import Path
from typing import List, Dict, Any

# Import our components
from generators.prompt_generator import PromptGenerator
from integrations.cursor_integration import CursorIntegration
from runners.test_runner import TestRunner


class SimplePipelineDemo:
    """
    Simple working pipeline demo with mock data.
    """

    def __init__(self):
        self.prompt_generator = PromptGenerator()
        self.cursor_integration = CursorIntegration()
        self.test_runner = TestRunner()

    def create_mock_cursor_output(self, task: str) -> str:
        """Create realistic mock Cursor output for demonstration."""
        if "calculator" in task.lower():
            return """
Here's a simple calculator class implementation:

```python
# calculator.py
from typing import Union, List

class Calculator:
    \"\"\"A simple calculator class with basic arithmetic operations.\"\"\"
    
    def __init__(self):
        self.history: List[str] = []
    
    def add(self, a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
        \"\"\"Add two numbers.\"\"\"
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def subtract(self, a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
        \"\"\"Subtract b from a.\"\"\"
        result = a - b
        self.history.append(f"{a} - {b} = {result}")
        return result
    
    def multiply(self, a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
        \"\"\"Multiply two numbers.\"\"\"
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result
    
    def divide(self, a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
        \"\"\"Divide a by b.\"\"\"
        if b == 0:
            raise ValueError("Cannot divide by zero")
        result = a / b
        self.history.append(f"{a} / {b} = {result}")
        return result
    
    def get_history(self) -> List[str]:
        \"\"\"Get calculation history.\"\"\"
        return self.history.copy()
    
    def clear_history(self) -> None:
        \"\"\"Clear calculation history.\"\"\"
        self.history.clear()
```

```python
# test_calculator.py
import pytest
from calculator import Calculator

def test_calculator_add():
    \"\"\"Test addition functionality.\"\"\"
    calc = Calculator()
    assert calc.add(2, 3) == 5
    assert calc.add(-1, 1) == 0
    assert calc.add(0.5, 0.5) == 1.0

def test_calculator_subtract():
    \"\"\"Test subtraction functionality.\"\"\"
    calc = Calculator()
    assert calc.subtract(5, 3) == 2
    assert calc.subtract(1, 1) == 0
    assert calc.subtract(0.5, 0.3) == 0.2

def test_calculator_multiply():
    \"\"\"Test multiplication functionality.\"\"\"
    calc = Calculator()
    assert calc.multiply(2, 3) == 6
    assert calc.multiply(-2, 3) == -6
    assert calc.multiply(0, 5) == 0

def test_calculator_divide():
    \"\"\"Test division functionality.\"\"\"
    calc = Calculator()
    assert calc.divide(6, 2) == 3
    assert calc.divide(5, 2) == 2.5
    assert calc.divide(0, 5) == 0

def test_calculator_divide_by_zero():
    \"\"\"Test division by zero error.\"\"\"
    calc = Calculator()
    with pytest.raises(ValueError, match="Cannot divide by zero"):
        calc.divide(5, 0)

def test_calculator_history():
    \"\"\"Test history functionality.\"\"\"
    calc = Calculator()
    calc.add(2, 3)
    calc.multiply(4, 5)
    
    history = calc.get_history()
    assert len(history) == 2
    assert "2 + 3 = 5" in history
    assert "4 * 5 = 20" in history

def test_calculator_clear_history():
    \"\"\"Test clear history functionality.\"\"\"
    calc = Calculator()
    calc.add(2, 3)
    calc.clear_history()
    assert len(calc.get_history()) == 0
"""
        else:
            return """
Here's a simple implementation:

```python
# main.py
def hello_world():
    return "Hello, World!"

if __name__ == "__main__":
    print(hello_world())
```

```python
# test_main.py
import pytest
from main import hello_world

def test_hello_world():
    assert hello_world() == "Hello, World!"
"""
        return mock_output

    async def run_simple_pipeline(self, task: str, output_dir: Path = None) -> bool:
        """
        Run the simple working pipeline with mock data.

        Args:
            task: The programming task to solve
            output_dir: Directory to save generated files

        Returns:
            bool: True if successful
        """
        if output_dir is None:
            output_dir = Path("generated")
        output_dir.mkdir(exist_ok=True)

        print("🚀 CodeConductor MVP - Simple Working Pipeline Demo")
        print("=" * 60)
        print(f"Task: {task}")
        print(f"Output Directory: {output_dir}")
        print()

        # Step 1: Create mock consensus data (simulating ensemble output)
        print("1️⃣ Creating mock consensus data...")
        mock_consensus = {
            "task": task,
            "files_needed": ["calculator.py", "test_calculator.py"],
            "dependencies": ["pytest"],
            "requirements": "Add, subtract, multiply, divide methods, Handle division by zero, Include type hints",
            "complexity": "low",
            "approach": "Class-based implementation with arithmetic methods",
        }
        print(f"   Mock consensus created with {len(mock_consensus)} fields")

        # Step 2: Generate prompt
        print("2️⃣ Generating prompt from consensus...")
        prompt = self.prompt_generator.generate(mock_consensus)
        if not prompt:
            print("❌ Failed to generate prompt")
            return False
        print(f"   Generated prompt ({len(prompt)} characters)")

        # Step 3: Simulate Cursor integration
        print("3️⃣ Simulating Cursor integration...")
        print("   📋 Prompt would be copied to clipboard")
        print("   🤖 Cursor would generate code")
        print("   📋 Code would be copied back")

        # Create mock Cursor output
        mock_cursor_output = self.create_mock_cursor_output(task)
        print(f"   📄 Mock Cursor output ({len(mock_cursor_output)} characters)")

        # Step 4: Extract and save files
        print("4️⃣ Extracting and saving files...")
        saved_files = self.cursor_integration.extract_and_save_files(
            mock_cursor_output, output_dir
        )
        print(f"   ✅ Saved {len(saved_files)} files")

        if saved_files:
            print("   📁 Generated files:")
            for file_path in saved_files:
                print(f"      - {file_path}")

        # Step 5: Run tests
        print("5️⃣ Running tests...")
        if saved_files:
            test_result = self.test_runner.run_pytest(output_dir)
            print(f"   Tests passed: {test_result.success}")
            if not test_result.success:
                print(f"   Errors: {test_result.errors}")
        else:
            print("   ⚠️  No files to test")

        # Summary
        print("\n" + "=" * 60)
        print("📊 PIPELINE SUMMARY:")
        print(f"   - Mock Consensus Fields: {len(mock_consensus)}")
        print(f"   - Generated Prompt: {len(prompt)} chars")
        print(f"   - Mock Cursor Output: {len(mock_cursor_output)} chars")
        print(f"   - Generated Files: {len(saved_files)}")
        if saved_files:
            print(
                f"   - Test Status: {'✅ PASS' if test_result.success else '❌ FAIL'}"
            )
        print("=" * 60)

        return len(saved_files) > 0


async def main():
    """Main demo function."""
    demo = SimplePipelineDemo()

    # Test with calculator task
    success = await demo.run_simple_pipeline("Create a simple calculator class")

    if success:
        print("\n🎉 Pipeline completed successfully!")
        print("📁 Check the 'generated' directory for output files")
    else:
        print("\n❌ Pipeline failed")


if __name__ == "__main__":
    asyncio.run(main())
