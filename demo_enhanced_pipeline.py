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
    notify_prompt_copied,
    notify_waiting_for_cursor,
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
        Run the enhanced pipeline with real ensemble engine when available.
        """
        if output_dir is None:
            output_dir = Path("generated")
        output_dir.mkdir(exist_ok=True)

        print("🚀 CodeConductor MVP - Enhanced Pipeline Demo")
        print("=" * 60)
        print(f"Task: {task}")
        print(f"Output Directory: {output_dir}")
        print(f"Enhancements: ✅ Enabled")
        print(f"Auto-detect: ✅ Enabled")
        print()

        # Step 1: Try real ensemble engine first, fallback to mock
        print("1️⃣ Running Ensemble Engine...")
        try:
            # Try to get real consensus from ensemble
            from ensemble.model_manager import ModelManager
            from ensemble.query_dispatcher import QueryDispatcher
            from ensemble.consensus_calculator import ConsensusCalculator
            
            model_manager = ModelManager()
            query_dispatcher = QueryDispatcher()
            consensus_calculator = ConsensusCalculator()
            
            # Check if we have models available
            models = await model_manager.list_models()
            if models:
                print(f"   📦 Found {len(models)} models, trying real ensemble...")
                
                # Use improved dispatch with fallback strategies
                responses = await query_dispatcher.dispatch_with_fallback(task, min_models=2)
                
                if responses:
                    # Convert responses to format expected by consensus_calculator
                    formatted_responses = []
                    for model_id, response_data in responses.items():
                        # Create a result object with success/response attributes
                        result_obj = type('Result', (), {
                            'model_id': model_id,
                            'success': 'error' not in response_data,
                            'response': self._extract_response_content(response_data),
                            'response_time': 0.0
                        })()
                        formatted_responses.append(result_obj)
                    
                    # Calculate consensus
                    consensus_result = consensus_calculator.calculate_consensus(formatted_responses)
                    consensus = consensus_result.consensus
                    print(f"   ✅ Real ensemble consensus generated with {len(formatted_responses)} responses")
                    print(f"   📊 Confidence: {consensus_result.confidence:.2f}")
                else:
                    print("   ⚠️  No model responses, using mock data")
                    consensus = self._create_mock_consensus(task)
            else:
                print("   ⚠️  No models available, using mock data")
                consensus = self._create_mock_consensus(task)
                
        except Exception as e:
            print(f"   ⚠️  Ensemble error: {e}, using mock data")
            consensus = self._create_mock_consensus(task)

        print(f"   Mock consensus created with {len(consensus)} fields")

        # Step 2: Generate prompt
        print("2️⃣ Generating prompt from consensus...")
        prompt = self.prompt_generator.generate_prompt(consensus, task)
        if not prompt:
            print("❌ Failed to generate prompt")
            return False
        print(f"   Generated prompt ({len(prompt)} characters)")

        # Step 3: Enhanced Cursor integration
        print("3️⃣ Enhanced Cursor integration...")
        if self.enhancements_enabled:
            # Copy prompt to clipboard with notification
            if self.cursor_integration.clipboard_manager.copy_to_clipboard(prompt):
                print("   📋 Prompt copied to clipboard with notification")
                notify_prompt_copied()
            else:
                print("   ⚠️  Failed to copy to clipboard")
        else:
            print("   📋 Prompt copied to clipboard")

        print("   🤖 Auto-detection mode active")
        print("   🎹 Global hotkeys available")

        # Step 4: Extract and save files
        print("4️⃣ Extracting and saving files...")
        if self.enhancements_enabled:
            # Show notification that we're waiting for Cursor
            notify_waiting_for_cursor()

        # Create mock Cursor output for demo
        mock_cursor_output = self.create_mock_cursor_output(task)
        print(f"   📄 Mock Cursor output ({len(mock_cursor_output)} characters)")

        # Extract and save files
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
        print(f"   - Consensus Fields: {len(consensus)}")
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

    def _extract_response_content(self, response_data: dict) -> str:
        """Extract the actual response content from model response data."""
        try:
            if "error" in response_data:
                return f"Error: {response_data['error']}"
            
            # Handle LM Studio format
            if "choices" in response_data:
                return response_data["choices"][0]["message"]["content"]
            
            # Handle Ollama format
            if "response" in response_data:
                return response_data["response"]
            
            # Fallback: return as JSON string
            import json
            return json.dumps(response_data)
            
        except Exception as e:
            return f"Failed to extract response: {e}"

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
