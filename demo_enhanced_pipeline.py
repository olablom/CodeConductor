#!/usr/bin/env python3
"""
CodeConductor MVP - Enhanced Pipeline Demo

Demonstrates the enhanced clipboard workflow with auto-detection, notifications, and hotkeys.
"""

import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any

# Import our components
from generators.prompt_generator import PromptGenerator
from integrations.cursor_integration import CursorIntegration
from runners.test_runner import TestRunner
from integrations.notifications import (
    notify_tests_running,
    notify_tests_passed,
    notify_tests_failed,
    notify_pipeline_complete,
    notify_error,
)


class EnhancedPipelineDemo:
    """
    Enhanced pipeline demo with clipboard improvements.
    """

    def __init__(self, enable_enhancements: bool = True):
        self.prompt_generator = PromptGenerator()
        self.cursor_integration = CursorIntegration(
            enable_enhancements=enable_enhancements
        )
        self.test_runner = TestRunner()
        self.enhancements_enabled = enable_enhancements

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

    async def run_enhanced_pipeline(
        self, task: str, output_dir: Path = None, use_auto_detect: bool = True
    ) -> bool:
        """
        Run the enhanced pipeline with clipboard improvements.

        Args:
            task: The programming task to solve
            output_dir: Directory to save generated files
            use_auto_detect: Whether to use auto-detection mode

        Returns:
            bool: True if successful
        """
        if output_dir is None:
            output_dir = Path("generated")
        output_dir.mkdir(exist_ok=True)

        print("🚀 CodeConductor MVP - Enhanced Pipeline Demo")
        print("=" * 60)
        print(f"Task: {task}")
        print(f"Output Directory: {output_dir}")
        print(
            f"Enhancements: {'✅ Enabled' if self.enhancements_enabled else '❌ Disabled'}"
        )
        print(f"Auto-detect: {'✅ Enabled' if use_auto_detect else '❌ Manual'}")
        print()

        try:
            # Step 1: Create mock consensus data
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

            # Step 3: Enhanced Cursor integration
            print("3️⃣ Enhanced Cursor integration...")
            print("   📋 Prompt copied to clipboard with notification")
            print("   🤖 Auto-detection mode active")
            print("   🎹 Global hotkeys available")

            # Copy prompt (will show notification if enhancements enabled)
            self.cursor_integration.copy_prompt_to_clipboard(prompt)

            # Create mock Cursor output (simulating real workflow)
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

            # Step 5: Enhanced test running
            print("5️⃣ Running tests with notifications...")
            if saved_files:
                # Show notification that tests are running
                if self.enhancements_enabled:
                    notify_tests_running()

                test_result = self.test_runner.run_pytest(output_dir)
                print(f"   Tests passed: {test_result.success}")

                # Show appropriate notification
                if self.enhancements_enabled:
                    if test_result.success:
                        notify_tests_passed()
                    else:
                        notify_tests_failed(
                            len(test_result.errors) if test_result.errors else 0
                        )

                if not test_result.success:
                    print(f"   Errors: {test_result.errors}")
            else:
                print("   ⚠️  No files to test")

            # Summary
            print("\n" + "=" * 60)
            print("📊 ENHANCED PIPELINE SUMMARY:")
            print(f"   - Mock Consensus Fields: {len(mock_consensus)}")
            print(f"   - Generated Prompt: {len(prompt)} chars")
            print(f"   - Mock Cursor Output: {len(mock_cursor_output)} chars")
            print(f"   - Generated Files: {len(saved_files)}")
            if saved_files:
                print(
                    f"   - Test Status: {'✅ PASS' if test_result.success else '❌ FAIL'}"
                )
            print("=" * 60)

            # Show completion notification
            if self.enhancements_enabled:
                notify_pipeline_complete(len(saved_files) > 0)

            return len(saved_files) > 0

        except Exception as e:
            error_msg = f"Pipeline error: {str(e)}"
            print(f"❌ {error_msg}")

            if self.enhancements_enabled:
                notify_error(error_msg)

            return False

    def start_enhanced_workflow(self):
        """Start enhanced workflow with hotkeys."""
        if not self.enhancements_enabled:
            print("⚠️  Enhancements not available")
            return

        # Define hotkey callbacks
        callbacks = {
            "copy_prompt": lambda: print("🎯 Copy prompt hotkey pressed"),
            "paste_from_cursor": lambda: print("🤖 Paste from Cursor hotkey pressed"),
            "rerun_last_task": lambda: print("🔄 Re-run last task hotkey pressed"),
            "run_tests": lambda: print("🧪 Run tests hotkey pressed"),
            "stop_pipeline": lambda: print("🛑 Stop pipeline hotkey pressed"),
        }

        self.cursor_integration.start_enhanced_workflow(callbacks)
        print("🎹 Enhanced workflow started with hotkeys!")

    def stop_enhanced_workflow(self):
        """Stop enhanced workflow."""
        if self.enhancements_enabled:
            self.cursor_integration.stop_enhanced_workflow()
            print("🛑 Enhanced workflow stopped")


async def main():
    """Main demo function."""
    print("🎯 CodeConductor Enhanced Pipeline Demo")
    print("=" * 50)
    print("This demo showcases:")
    print("  ✅ Auto-detection of Cursor output")
    print("  ✅ Windows notifications")
    print("  ✅ Global hotkeys")
    print("  ✅ Enhanced user experience")
    print("=" * 50)

    # Create demo instance
    demo = EnhancedPipelineDemo(enable_enhancements=True)

    # Start enhanced workflow
    demo.start_enhanced_workflow()

    try:
        # Test with calculator task
        success = await demo.run_enhanced_pipeline(
            "Create a simple calculator class", use_auto_detect=True
        )

        if success:
            print("\n🎉 Enhanced pipeline completed successfully!")
            print("📁 Check the 'generated' directory for output files")
        else:
            print("\n❌ Enhanced pipeline failed")

    finally:
        # Stop enhanced workflow
        demo.stop_enhanced_workflow()


if __name__ == "__main__":
    asyncio.run(main())
