#!/usr/bin/env python3
"""
CodeConductor MVP - Full Auto Demo

End-to-end demonstration of the complete pipeline:
1. Ensemble Engine → 2. Prompt Generator → 3. Cursor Integration → 4. Test Runner → 5. Feedback Loop
"""

import asyncio
import argparse
import sys
from pathlib import Path
from typing import Optional, List

# Import our components
from ensemble.model_manager import ModelManager
from ensemble.query_dispatcher import QueryDispatcher
from ensemble.consensus_calculator import ConsensusCalculator
from generators.prompt_generator import PromptGenerator
from integrations.cursor_integration import CursorIntegration
from runners.test_runner import TestRunner


class FullAutoDemo:
    """
    Complete end-to-end demo of CodeConductor MVP pipeline.
    """
    
    def __init__(self, max_iterations: int = 3):
        self.max_iterations = max_iterations
        self.model_manager = ModelManager()
        self.query_dispatcher = QueryDispatcher()
        self.consensus_calculator = ConsensusCalculator()
        self.prompt_generator = PromptGenerator()
        self.cursor_integration = CursorIntegration()
        self.test_runner = TestRunner()
        
    async def run_full_pipeline(self, task: str, output_dir: Path = None) -> bool:
        """
        Run the complete pipeline from task to working code.
        
        Args:
            task: The programming task to solve
            output_dir: Directory to save generated files (default: generated/)
            
        Returns:
            bool: True if successful, False if max iterations reached
        """
        if output_dir is None:
            output_dir = Path("generated")
        output_dir.mkdir(exist_ok=True)
        
        print("🚀 CodeConductor MVP - Full Auto Pipeline")
        print("=" * 50)
        print(f"Task: {task}")
        print(f"Output Directory: {output_dir}")
        print(f"Max Iterations: {self.max_iterations}")
        print()
        
        current_task = task
        iteration = 1
        
        while iteration <= self.max_iterations:
            print(f"🔄 ITERATION {iteration}/{self.max_iterations}")
            print("-" * 30)
            
            # Step 1: Ensemble Engine
            print("1️⃣ Running Ensemble Engine...")
            consensus = await self._run_ensemble(current_task)
            if not consensus:
                print("❌ Ensemble failed to generate consensus")
                return False
            
            # Step 2: Prompt Generator
            print("2️⃣ Generating Prompt...")
            prompt = self._generate_prompt(consensus, current_task)
            if not prompt:
                print("❌ Failed to generate prompt")
                return False
            
            # Step 3: Cursor Integration (Manual MVP)
            print("3️⃣ Cursor Integration (Manual)...")
            print("📋 Prompt copied to clipboard!")
            print("💡 Please:")
            print("   1. Paste the prompt into Cursor")
            print("   2. Generate the code")
            print("   3. Copy the output back")
            print("   4. Press Enter here when ready...")
            
            input("⏳ Press Enter after Cursor generation...")
            
            # Step 4: Extract and Save Code
            print("4️⃣ Extracting Code...")
            generated_files = self._extract_and_save_code(output_dir)
            if not generated_files:
                print("❌ No code files extracted")
                return False
            
            # Step 5: Test Runner
            print("5️⃣ Running Tests...")
            test_result = self._run_tests(output_dir)
            
            if test_result.success:
                print("✅ SUCCESS! All tests passed!")
                print(f"📁 Generated files: {[f.name for f in generated_files]}")
                return True
            else:
                print("❌ Tests failed. Generating feedback...")
                feedback = self._generate_feedback(test_result, current_task)
                current_task = feedback
                print(f"🔄 Updated task with feedback: {feedback[:100]}...")
            
            iteration += 1
            print()
        
        print(f"❌ Max iterations ({self.max_iterations}) reached. Pipeline incomplete.")
        return False
    
    async def _run_ensemble(self, task: str) -> Optional[dict]:
        """Run ensemble engine to get consensus."""
        try:
            # Get healthy models
            models = await self.model_manager.list_healthy_models()
            if not models:
                print("⚠️  No healthy models found, using mock data")
                return self._create_mock_consensus(task)
            
            # Dispatch query
            responses = await self.query_dispatcher.dispatch_to_models(
                task, models[:2]  # Use first 2 models
            )
            
            if not responses:
                print("⚠️  No model responses, using mock data")
                return self._create_mock_consensus(task)
            
            # Calculate consensus
            consensus = self.consensus_calculator.calculate_consensus(responses)
            print(f"✅ Ensemble consensus generated with {len(responses)} responses")
            return consensus
            
        except Exception as e:
            print(f"⚠️  Ensemble error: {e}, using mock data")
            return self._create_mock_consensus(task)
    
    def _create_mock_consensus(self, task: str) -> dict:
        """Create mock consensus data for demo purposes."""
        return {
            "task": task,
            "approach": "Create a simple, well-tested implementation",
            "requirements": [
                "Follow Python best practices",
                "Include comprehensive tests",
                "Handle edge cases gracefully",
                "Use clear variable names"
            ],
            "confidence": 0.85,
            "suggestions": [
                "Start with a simple implementation",
                "Add tests for edge cases",
                "Consider error handling"
            ]
        }
    
    def _generate_prompt(self, consensus: dict, original_task: str) -> Optional[str]:
        """Generate prompt from consensus."""
        try:
            context = {
                "project_structure": "Simple Python project",
                "coding_standards": "PEP 8, type hints, docstrings",
                "testing_approach": "pytest with comprehensive coverage"
            }
            
            prompt = self.prompt_generator.generate(consensus, context)
            print(f"✅ Prompt generated ({len(prompt)} characters)")
            return prompt
            
        except Exception as e:
            print(f"❌ Prompt generation failed: {e}")
            return None
    
    def _extract_and_save_code(self, output_dir: Path) -> List[Path]:
        """Extract code from clipboard and save files."""
        try:
            # Read from clipboard
            cursor_output = self.cursor_integration.read_from_clipboard()
            if not cursor_output:
                print("❌ No content found in clipboard")
                return []
            
            # Extract and save files
            generated_files = self.cursor_integration.extract_and_save_files(
                cursor_output, output_dir
            )
            
            print(f"✅ Extracted {len(generated_files)} files")
            for file_path in generated_files:
                print(f"   📄 {file_path.name}")
            
            return generated_files
            
        except Exception as e:
            print(f"❌ Code extraction failed: {e}")
            return []
    
    def _run_tests(self, test_dir: Path) -> 'TestResult':
        """Run tests on generated code."""
        try:
            result = self.test_runner.run_pytest(test_dir)
            
            if result.success:
                print("✅ All tests passed!")
            else:
                print(f"❌ Tests failed: {len(result.errors)} errors")
                for i, error in enumerate(result.errors[:3], 1):
                    print(f"   {i}. {error[:100]}...")
            
            return result
            
        except Exception as e:
            print(f"❌ Test execution failed: {e}")
            # Return a mock failed result
            from runners.test_runner import TestResult
            return TestResult(success=False, stdout="", errors=[str(e)])
    
    def _generate_feedback(self, test_result: 'TestResult', original_task: str) -> str:
        """Generate feedback for next iteration."""
        feedback_parts = [original_task]
        feedback_parts.append("\n\nTest Results:")
        feedback_parts.append(f"- Success: {test_result.success}")
        feedback_parts.append(f"- Errors: {len(test_result.errors)}")
        
        if test_result.errors:
            feedback_parts.append("\nError Details:")
            for i, error in enumerate(test_result.errors[:2], 1):
                # Extract key error info
                error_lines = error.split('\n')
                for line in error_lines:
                    if any(keyword in line for keyword in ['Error:', 'AssertionError:', 'SyntaxError:', 'ImportError:']):
                        feedback_parts.append(f"- {line.strip()}")
                        break
        
        feedback_parts.append("\nPlease fix these issues and regenerate the code.")
        
        return "\n".join(feedback_parts)


def main():
    """Main entry point for the demo."""
    parser = argparse.ArgumentParser(description="CodeConductor MVP - Full Auto Demo")
    parser.add_argument("task", nargs="?", help="Programming task to solve")
    parser.add_argument("--prompt-file", "-f", help="Read task from file")
    parser.add_argument("--output-dir", "-o", default="generated", help="Output directory")
    parser.add_argument("--max-iterations", "-i", type=int, default=3, help="Max iterations")
    
    args = parser.parse_args()
    
    # Get task
    task = None
    if args.task:
        task = args.task
    elif args.prompt_file:
        try:
            with open(args.prompt_file, 'r') as f:
                task = f.read().strip()
        except FileNotFoundError:
            print(f"❌ Prompt file not found: {args.prompt_file}")
            sys.exit(1)
    else:
        # Default demo task
        task = """Create a simple calculator class with methods for add, subtract, multiply, and divide. 
        Include comprehensive tests and handle division by zero."""
    
    # Run demo
    demo = FullAutoDemo(max_iterations=args.max_iterations)
    output_dir = Path(args.output_dir)
    
    try:
        success = asyncio.run(demo.run_full_pipeline(task, output_dir))
        if success:
            print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"📁 Check {output_dir} for generated files")
        else:
            print("\n❌ Pipeline did not complete successfully")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 