from generators import PromptGenerator
from integrations.cursor_integration import CursorIntegration
from runners.test_runner import TestRunner
from feedback.feedback_controller import FeedbackController, Action

# Example ensemble consensus and task
demo_consensus = {
    "approach": "Recursive function to calculate factorial",
    "complexity": "low",
    "reasoning": "A recursive function is a good approach for this problem because it allows us to avoid having to keep track of the intermediate results and instead, we can simply call the function again with the next number in the sequence. This makes the code more concise and easier to understand.",
    "files_needed": ["factorial.py"],
    "dependencies": [],
}
demo_task = "Create a simple Python function that calculates the factorial of a number"
demo_context = {
    "project_structure": "Standard Python project with src/ and tests/ directories",
    "coding_standards": "PEP 8, type hints, docstrings",
    "existing_patterns": "Async/await for I/O operations, error handling with try/except",
    "dependencies": ["pytest", "aiohttp", "streamlit"],
}


def main():
    print("\n=== CodeConductor MVP: Complete Pipeline Demo ===\n")

    # Initialize components
    prompt_gen = PromptGenerator()
    ci = CursorIntegration()
    test_runner = TestRunner()
    feedback_controller = FeedbackController(max_iterations=3)

    # Generate initial prompt
    prompt = prompt_gen.generate_prompt(demo_consensus, demo_task, demo_context)
    print("[PROMPT] Initial prompt generated")
    print("-" * 40)

    # Main iteration loop
    while True:
        print(f"\n[ITERATION {feedback_controller.iteration_count + 1}]")
        print("=" * 50)

        # Copy prompt to clipboard
        ci.copy_prompt_to_clipboard(prompt)

        # Get Cursor response
        cursor_response = ci.get_cursor_response()
        code_blocks = ci.extract_code_blocks(cursor_response)

        if not code_blocks:
            print("[WARN] No code blocks found. Escalating to human.")
            break

        # Run tests
        print("\n[TEST] Running pytest...")
        test_results = test_runner.run_tests(code_blocks)

        # Process feedback
        feedback = feedback_controller.process_feedback(test_results, prompt, demo_task)

        print(f"\n[FEEDBACK] Decision: {feedback['action'].value}")
        print(f"Reason: {feedback['reason']}")
        print(f"Iteration: {feedback['iteration_count']}")

        # Handle action
        if feedback["action"] == Action.COMPLETE:
            print("\n🎉 SUCCESS! All tests passed!")
            break
        elif feedback["action"] == Action.ITERATE:
            print("\n🔄 ITERATING with enhanced prompt...")
            prompt = feedback["enhanced_prompt"]
            print("Enhanced prompt includes error details from previous iteration.")
        elif feedback["action"] == Action.ESCALATE:
            print("\n⚠️ ESCALATING to human intervention.")
            print("Maximum iterations reached or unfixable errors detected.")
            break

    # Show final summary
    summary = feedback_controller.get_iteration_summary()
    print(f"\n[SUMMARY] Total iterations: {summary['total_iterations']}")


if __name__ == "__main__":
    main()
