from typing import Dict, Any, List, Optional
from enum import Enum


class Action(Enum):
    COMPLETE = "complete"
    ITERATE = "iterate"
    ESCALATE = "escalate"


class FeedbackController:
    """
    MVP Feedback Loop Controller
    - Analyzes test results
    - Decides next action
    - Enhances prompts with error info
    - Tracks iterations
    """

    def __init__(self, max_iterations: int = 3):
        self.max_iterations = max_iterations
        self.iteration_count = 0
        self.history = []

    def process_feedback(
        self, test_results: Dict[str, Any], original_prompt: str, task_description: str
    ) -> Dict[str, Any]:
        """
        Process test results and decide next action.
        Returns: {action, enhanced_prompt, reason, iteration_count}
        """
        self.iteration_count += 1
        self.history.append(
            {
                "iteration": self.iteration_count,
                "test_results": test_results,
                "timestamp": "now",
            }
        )

        # Analyze test results
        status = test_results.get("status", "error")
        passed = test_results.get("passed", 0)
        failed = test_results.get("failed", 0)
        errors = test_results.get("errors", [])

        # Decision logic
        if status == "pass" and passed > 0:
            return self._complete_action("All tests passed successfully")
        elif self.iteration_count >= self.max_iterations:
            return self._escalate_action("Maximum iterations reached")
        elif self._has_fixable_errors(errors):
            enhanced_prompt = self._enhance_prompt_with_errors(
                original_prompt, errors, task_description
            )
            return self._iterate_action(enhanced_prompt, "Fixable errors detected")
        else:
            return self._escalate_action("Unfixable errors or no progress")

    def _has_fixable_errors(self, errors: List[Dict[str, Any]]) -> bool:
        """Determine if errors are fixable."""
        if not errors:
            return False

        fixable_patterns = [
            "AssertionError",
            "NameError",
            "TypeError",
            "AttributeError",
            "ImportError",
            "SyntaxError",
        ]

        for error in errors:
            error_msg = error.get("error", "").lower()
            if any(pattern.lower() in error_msg for pattern in fixable_patterns):
                return True
        return False

    def _enhance_prompt_with_errors(
        self, original_prompt: str, errors: List[Dict[str, Any]], task_description: str
    ) -> str:
        """Enhance original prompt with error information."""
        error_section = "\n### Previous Errors (Fix These)\n"

        for i, error in enumerate(errors, 1):
            test_name = error.get("test", "unknown")
            error_msg = error.get("error", "Unknown error")
            error_section += f"\n**Error {i}**: {test_name}\n"
            error_section += f"```\n{error_msg}\n```\n"

        error_section += "\n### Instructions\n"
        error_section += "- Fix the errors above\n"
        error_section += "- Ensure all tests pass\n"
        error_section += "- Maintain the same functionality\n"

        # Insert error section before Implementation Notes
        if "### Implementation Notes" in original_prompt:
            parts = original_prompt.split("### Implementation Notes")
            enhanced_prompt = (
                parts[0] + error_section + "\n### Implementation Notes" + parts[1]
            )
        else:
            enhanced_prompt = original_prompt + error_section

        return enhanced_prompt

    def _complete_action(self, reason: str) -> Dict[str, Any]:
        return {
            "action": Action.COMPLETE,
            "enhanced_prompt": None,
            "reason": reason,
            "iteration_count": self.iteration_count,
        }

    def _iterate_action(self, enhanced_prompt: str, reason: str) -> Dict[str, Any]:
        return {
            "action": Action.ITERATE,
            "enhanced_prompt": enhanced_prompt,
            "reason": reason,
            "iteration_count": self.iteration_count,
        }

    def _escalate_action(self, reason: str) -> Dict[str, Any]:
        return {
            "action": Action.ESCALATE,
            "enhanced_prompt": None,
            "reason": reason,
            "iteration_count": self.iteration_count,
        }

    def get_iteration_summary(self) -> Dict[str, Any]:
        """Get summary of all iterations."""
        return {
            "total_iterations": self.iteration_count,
            "max_iterations": self.max_iterations,
            "history": self.history,
        }
