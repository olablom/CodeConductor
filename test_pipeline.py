#!/usr/bin/env python3
"""
Test CodeConductor MVP Pipeline
Simulates complete workflow without manual input
"""

from generators import PromptGenerator
from runners.test_runner import TestRunner
from feedback.feedback_controller import FeedbackController, Action

# Mock Cursor responses for testing
MOCK_CURSOR_RESPONSES = [
    # First attempt - has errors (wrong calculation, missing type hints)
    """```factorial.py
def factorial(n):
    if n < 0:
        return None
    if n == 0:
        return 1
    return n + factorial(n - 1)  # BUG: should be multiplication, not addition
```

```test_factorial.py
import pytest
from factorial import factorial

def test_factorial_basic():
    assert factorial(5) == 120

def test_factorial_zero():
    assert factorial(0) == 1

def test_factorial_negative():
    assert factorial(-1) == None
```""",
    
    # Second attempt - fixed errors (correct calculation, added type hints)
    """```factorial.py
def factorial(n: int) -> int:
    \"\"\"Calculate factorial of a number.\"\"\"
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n == 0:
        return 1
    return n * factorial(n - 1)  # FIXED: correct multiplication
```

```test_factorial.py
import pytest
from factorial import factorial

def test_factorial_basic():
    assert factorial(5) == 120

def test_factorial_zero():
    assert factorial(0) == 1

def test_factorial_negative():
    with pytest.raises(ValueError):
        factorial(-1)
```"""
]

def extract_code_blocks(response: str):
    """Extract code blocks from mock response."""
    import re
    pattern = r"```([\w\-/\\.]+)\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    code_blocks = []
    for filename, code in matches:
        code_blocks.append((filename.strip(), code.strip()))
    return code_blocks

def main():
    print("\n=== CodeConductor MVP: Pipeline Test ===\n")
    
    # Initialize components
    prompt_gen = PromptGenerator()
    test_runner = TestRunner()
    feedback_controller = FeedbackController(max_iterations=3)
    
    # Demo data
    demo_consensus = {
        "approach": "Recursive function to calculate factorial",
        "complexity": "low",
        "reasoning": "A recursive function is a good approach for this problem.",
        "files_needed": ["factorial.py"],
        "dependencies": []
    }
    demo_task = "Create a simple Python function that calculates the factorial of a number"
    demo_context = {
        "project_structure": "Standard Python project with src/ and tests/ directories",
        "coding_standards": "PEP 8, type hints, docstrings",
        "existing_patterns": "Async/await for I/O operations, error handling with try/except",
        "dependencies": ["pytest", "aiohttp", "streamlit"]
    }
    
    # Generate initial prompt
    prompt = prompt_gen.generate_prompt(demo_consensus, demo_task, demo_context)
    print("[PROMPT] Initial prompt generated")
    print("-" * 40)
    
    # Main iteration loop
    for iteration in range(len(MOCK_CURSOR_RESPONSES)):
        print(f"\n[ITERATION {iteration + 1}]")
        print("=" * 50)
        
        # Simulate Cursor response
        cursor_response = MOCK_CURSOR_RESPONSES[iteration]
        code_blocks = extract_code_blocks(cursor_response)
        
        print(f"[CURSOR] Generated {len(code_blocks)} files:")
        for filename, code in code_blocks:
            print(f"  - {filename}")
        
        # Run tests
        print("\n[TEST] Running pytest...")
        test_results = test_runner.run_tests(code_blocks)
        
        print(f"[TEST] Status: {test_results['status']}")
        print(f"[TEST] Passed: {test_results['passed']}, Failed: {test_results['failed']}")
        
        if test_results['errors']:
            print("[TEST] Errors found:")
            for error in test_results['errors']:
                print(f"  - {error.get('test', 'unknown')}: {error.get('error', 'Unknown error')[:100]}...")
        
        # Process feedback
        feedback = feedback_controller.process_feedback(test_results, prompt, demo_task)
        
        print(f"\n[FEEDBACK] Decision: {feedback['action'].value}")
        print(f"Reason: {feedback['reason']}")
        print(f"Iteration: {feedback['iteration_count']}")
        
        # Handle action
        if feedback['action'] == Action.COMPLETE:
            print("\n🎉 SUCCESS! All tests passed!")
            break
        elif feedback['action'] == Action.ITERATE:
            print("\n🔄 ITERATING with enhanced prompt...")
            prompt = feedback['enhanced_prompt']
            print("Enhanced prompt includes error details from previous iteration.")
        elif feedback['action'] == Action.ESCALATE:
            print("\n⚠️ ESCALATING to human intervention.")
            print("Maximum iterations reached or unfixable errors detected.")
            break
    
    # Show final summary
    summary = feedback_controller.get_iteration_summary()
    print(f"\n[SUMMARY] Total iterations: {summary['total_iterations']}")
    print(f"[SUMMARY] Final status: {feedback['action'].value}")

if __name__ == "__main__":
    main() 