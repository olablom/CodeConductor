#!/usr/bin/env python3
"""
Code Reviewer for Multi-Agent Debugging
Uses RLHF to select the best model for code review and provides actionable feedback.
"""

import logging
import os
import re
import subprocess
import tempfile
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from codeconductor.context.rag_system import RAGSystem

logger = logging.getLogger(__name__)

# Try to import pylint for code quality assessment
try:
    # Test if pylint is available
    result = subprocess.run(["pylint", "--version"], capture_output=True, text=True)
    PYLINT_AVAILABLE = result.returncode == 0
    if PYLINT_AVAILABLE:
        logger.info("✅ pylint is available")
    else:
        logger.warning("⚠️ pylint not available. Install with: pip install pylint")
except Exception:
    PYLINT_AVAILABLE = False
    logger.warning("⚠️ pylint not available. Install with: pip install pylint")


class CodeReviewer:
    """
    Multi-agent code reviewer that uses RLHF to select the best model for review.
    """

    def __init__(self, models: list[str], ppo_model_path: str = "ppo_codeconductor.zip"):
        """
        Initialize the code reviewer.

        Args:
            models: List of available model IDs for review
            ppo_model_path: Path to the trained PPO model
        """
        self.models = models
        self.rag_system = RAGSystem()

        # Load RLHF agent for model selection
        try:
            self.rlhf_agent = PPO.load(ppo_model_path)
            logger.info(f"✅ Loaded RLHF agent from {ppo_model_path}")
        except Exception as e:
            logger.warning(f"⚠️ Could not load RLHF agent: {e}")
            self.rlhf_agent = None

    def select_reviewer(
        self,
        task_description: str,
        code: str,
        test_results: list[dict] | None = None,
    ) -> str:
        """
        Select the best model for code review using RLHF.

        Args:
            task_description: Description of the task
            code: Code to be reviewed
            test_results: Optional test results for context

        Returns:
            Selected model ID for review
        """
        if not self.rlhf_agent:
            # Fallback to first available model
            logger.warning("⚠️ No RLHF agent available, using first model")
            return self.models[0] if self.models else "default"

        # Create observation for RLHF agent
        test_results = test_results or []
        test_reward = self.calculate_test_reward(test_results)
        code_quality = self.estimate_code_quality(code)
        task_complexity = self.estimate_task_complexity(task_description)

        # Create observation vector
        observation = np.array([test_reward, code_quality, 0.0, task_complexity], dtype=np.float32)

        # Get action from RLHF agent to select reviewer
        action, _ = self.rlhf_agent.predict(observation)
        selected_model = self.models[action % len(self.models)]

        logger.info(f"🎯 RLHF selected {selected_model} for review (action: {action})")
        return selected_model

    def review_code(
        self,
        task_description: str,
        code: str,
        test_results: list[dict] | None = None,
    ) -> dict[str, Any]:
        """
        Review code using the selected model and RAG context.

        Args:
            task_description: Description of the task
            code: Code to be reviewed
            test_results: Optional test results for context

        Returns:
            Review results with comments and suggested fixes
        """
        # Select reviewer model using RLHF
        reviewer_model = self.select_reviewer(task_description, code, test_results)

        # Augment prompt with RAG for better context
        review_prompt = self.rag_system.augment_prompt(
            f"Review this code for bugs, readability, and best practices:\n\n{code}\n\nTask: {task_description}"
        )

        # Generate review using selected model
        review = self.run_model(reviewer_model, review_prompt)

        # Extract suggested fixes
        suggested_fixes = self.extract_suggested_fixes(review)

        return {
            "reviewer": reviewer_model,
            "comments": review,
            "suggested_fixes": suggested_fixes,
            "code_quality_score": self.estimate_code_quality(code),
            "test_reward": self.calculate_test_reward(test_results or []),
        }

    def calculate_test_reward(self, test_results: list[dict]) -> float:
        """
        Calculate reward based on test results.

        Args:
            test_results: List of test result dictionaries

        Returns:
            Reward value between 0.0 and 1.0
        """
        if not test_results:
            return 0.0

        total_tests = len(test_results)
        passed_tests = sum(1 for test in test_results if test.get("passed", False))

        return passed_tests / total_tests if total_tests > 0 else 0.0

    def estimate_code_quality(self, code: str) -> float:
        """
        Estimate code quality using pylint for accurate assessment.

        Args:
            code: Code to analyze

        Returns:
            Quality score between 0.0 and 1.0
        """
        if not code or not code.strip():
            return 0.0

        # Use pylint for accurate code quality assessment
        if PYLINT_AVAILABLE:
            try:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as temp_file:
                    temp_file.write(code)
                    temp_file_path = temp_file.name

                # Run pylint via subprocess
                result = subprocess.run(
                    [
                        "pylint",
                        temp_file_path,
                        "--disable=missing-module-docstring,missing-class-docstring,missing-function-docstring",
                        "--score=yes",
                        "--output-format=text",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                # Clean up temporary file
                os.unlink(temp_file_path)

                # Extract score from output
                score_match = re.search(r"Your code has been rated at ([0-9.]+)/10", result.stdout)
                if score_match:
                    score = float(score_match.group(1)) / 10.0
                    logger.info(f"📊 Pylint score: {score:.2f}/1.0")
                    return score
                else:
                    logger.warning("⚠️ Could not extract score from pylint output")
                    return self._fallback_code_quality(code)

            except subprocess.TimeoutExpired:
                logger.warning("⏰ Pylint timed out, using fallback")
                return self._fallback_code_quality(code)
            except Exception as e:
                logger.error(f"Failed to estimate code quality with pylint: {str(e)}")
                # Fall back to simple heuristics
                return self._fallback_code_quality(code)
        else:
            logger.warning("⚠️ pylint not available, using fallback quality assessment")
            return self._fallback_code_quality(code)

    def _fallback_code_quality(self, code: str) -> float:
        """
        Fallback code quality assessment using simple heuristics.

        Args:
            code: Code to analyze

        Returns:
            Quality score between 0.0 and 1.0
        """
        # Simple heuristics for code quality
        quality_score = 0.5  # Base score

        # Check for good practices
        if "def " in code and "return" in code:
            quality_score += 0.1  # Functions with returns

        if "try:" in code and "except" in code:
            quality_score += 0.1  # Error handling

        if "import " in code:
            quality_score += 0.05  # Imports

        if "class " in code:
            quality_score += 0.05  # Classes

        # Check for potential issues
        if "print(" in code and "logging" not in code:
            quality_score -= 0.05  # Print statements without logging

        if "TODO" in code or "FIXME" in code:
            quality_score -= 0.1  # TODO/FIXME comments

        # Normalize to 0.0-1.0 range
        return max(0.0, min(1.0, quality_score))

    def estimate_task_complexity(self, task_description: str) -> float:
        """
        Estimate task complexity based on keywords.

        Args:
            task_description: Description of the task

        Returns:
            Complexity score between 0.0 and 1.0
        """
        complex_keywords = [
            "api",
            "database",
            "authentication",
            "complex",
            "algorithm",
            "optimization",
            "performance",
            "security",
            "encryption",
            "distributed",
            "concurrent",
            "async",
            "threading",
        ]

        task_lower = task_description.lower()
        complexity_score = sum(1 for keyword in complex_keywords if keyword in task_lower)

        # Normalize to 0.0-1.0 range
        return min(1.0, complexity_score / len(complex_keywords))

    def run_model(self, model: str, prompt: str) -> str:
        """
        Run the selected model to generate review (placeholder).

        Args:
            model: Model ID to use
            prompt: Review prompt

        Returns:
            Generated review text
        """
        # Placeholder for actual model inference
        # In a real implementation, this would call the actual model
        logger.info(f"🤖 Running {model} for code review")

        # Simulate model response based on model type
        if "phi" in model.lower():
            return f"Review by {model}:\n- Code structure is good but could use better variable names.\n- Consider adding type hints for better maintainability.\n- Potential optimization in the loop logic."
        elif "codellama" in model.lower():
            return f"Review by {model}:\n- Good separation of concerns.\n- Missing error handling in critical sections.\n- Consider using list comprehensions for better performance."
        else:
            return f"Review by {model}:\n- Code is readable and follows basic conventions.\n- Add more comprehensive documentation.\n- Consider edge cases in input validation."

    def extract_suggested_fixes(self, review: str) -> list[str]:
        """
        Extract actionable fixes from review text.

        Args:
            review: Review text from model

        Returns:
            List of suggested fixes
        """
        fixes = []

        # Simple extraction based on common patterns
        lines = review.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("-") or line.startswith("•"):
                # Remove bullet points and common prefixes
                fix = line.lstrip("- ").lstrip("• ").strip()
                if fix and len(fix) > 10:  # Filter out very short suggestions
                    fixes.append(fix)

        # If no structured fixes found, try to extract from text
        if not fixes:
            # Look for common action words
            action_words = ["add", "use", "consider", "implement", "fix", "improve"]
            for word in action_words:
                if word in review.lower():
                    # Extract sentence containing action word
                    sentences = review.split(".")
                    for sentence in sentences:
                        if word in sentence.lower():
                            fixes.append(sentence.strip())

        return fixes[:5]  # Limit to 5 fixes

    def get_review_summary(self, review_results: dict[str, Any]) -> str:
        """
        Generate a summary of the review results.

        Args:
            review_results: Results from review_code method

        Returns:
            Summary string
        """
        reviewer = review_results.get("reviewer", "unknown")
        quality_score = review_results.get("code_quality_score", 0.0)
        test_reward = review_results.get("test_reward", 0.0)
        fixes_count = len(review_results.get("suggested_fixes", []))

        summary = "📋 Review Summary:\n"
        summary += f"  🤖 Reviewer: {reviewer}\n"
        summary += f"  📊 Code Quality: {quality_score:.2f}/1.0\n"
        summary += f"  🧪 Test Reward: {test_reward:.2f}/1.0\n"
        summary += f"  🔧 Suggested Fixes: {fixes_count}\n"

        return summary


# Convenience function
def create_code_reviewer(
    models: list[str], ppo_model_path: str = "ppo_codeconductor.zip"
) -> CodeReviewer:
    """
    Create a CodeReviewer instance.

    Args:
        models: List of available model IDs
        ppo_model_path: Path to PPO model

    Returns:
        CodeReviewer instance
    """
    return CodeReviewer(models, ppo_model_path)
