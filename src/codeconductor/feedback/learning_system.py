"""
Learning System for CodeConductor
Saves successful prompt-code patterns for analysis and model optimization
"""

import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PATTERNS_FILE = os.path.join(os.path.dirname(__file__), "patterns.json")


def calculate_complexity(code: str) -> float:
    """
    Calculate cyclomatic complexity of the provided code.
    Requires 'radon' library.
    """
    try:
        from radon.complexity import cc_visit

        blocks = cc_visit(code)
        total = sum(block.complexity for block in blocks)
        return total / len(blocks) if blocks else 0.0
    except ImportError:
        return 0.0


@dataclass
class Pattern:
    """Represents a successful prompt-code pattern"""

    prompt: str
    code: str
    validation: dict[str, Any]
    task_description: str
    timestamp: str
    model_used: str | None = None
    execution_time: float | None = None
    user_rating: int | None = None  # 1-5 scale
    notes: str | None = None
    reward: float | None = None  # Test-as-Reward value
    # New performance and quality metrics
    exec_time_s: float | None = None  # Execution time in seconds
    cyclomatic_complexity: float | None = None  # Code complexity
    tests_total: int | None = None  # Total number of tests
    tests_passed: int | None = None  # Number of passed tests


class LearningSystem:
    """
    Manages saving and loading of successful code generation patterns
    """

    def __init__(self, patterns_file: str = "patterns.json"):
        self.patterns_file = Path(patterns_file)
        self.patterns: list[Pattern] = []
        self._load_patterns()

    def _load_patterns(self):
        """Load existing patterns from JSON file"""
        if self.patterns_file.exists():
            try:
                with open(self.patterns_file, encoding="utf-8") as f:
                    data = json.load(f)
                    self.patterns = [Pattern(**pattern_data) for pattern_data in data]
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load patterns from {self.patterns_file}: {e}")
                self.patterns = []
        else:
            self.patterns = []

    def _save_patterns(self):
        """Save patterns to JSON file"""
        try:
            # Convert patterns to dictionaries
            pattern_data = [asdict(pattern) for pattern in self.patterns]

            # Create directory if it doesn't exist
            self.patterns_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.patterns_file, "w", encoding="utf-8") as f:
                json.dump(pattern_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"Error saving patterns: {e}")
            raise

    def save_successful_pattern(
        self,
        prompt: str,
        code: str,
        validation: dict[str, Any],
        task_description: str,
        model_used: str | None = None,
        execution_time: float | None = None,
        user_rating: int | None = None,
        notes: str | None = None,
        reward: float | None = None,
        exec_time_s: float | None = None,
        cyclomatic_complexity: float | None = None,
        tests_total: int | None = None,
        tests_passed: int | None = None,
    ) -> bool:
        """
        Save a successful prompt-code pattern

        Args:
            prompt: The prompt that was used
            code: The generated code
            validation: Validation results from validation_system
            task_description: Description of the task
            model_used: Which model was used (optional)
            execution_time: How long it took to generate (optional)
            user_rating: User rating 1-5 (optional)
            notes: Additional notes (optional)
            reward: Test-as-Reward value (optional)

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            pattern = Pattern(
                prompt=prompt,
                code=code,
                validation=validation,
                task_description=task_description,
                timestamp=datetime.now().isoformat(),
                model_used=model_used,
                execution_time=execution_time,
                user_rating=user_rating,
                notes=notes,
                reward=reward,
                exec_time_s=exec_time_s,
                cyclomatic_complexity=cyclomatic_complexity,
                tests_total=tests_total,
                tests_passed=tests_passed,
            )

            self.patterns.append(pattern)
            self._save_patterns()
            return True

        except Exception as e:
            print(f"Error saving pattern: {e}")
            return False

    def log_test_reward(
        self, prompt: str, code: str, test_results: list, metadata: dict = None
    ) -> float:
        """
        Calculate and log reward based on test results with enhanced metrics

        Args:
            prompt: The prompt that was used
            code: The generated code
            test_results: List of test result dictionaries with 'passed' key
            metadata: Additional metadata to save

        Returns:
            Calculated reward value (0.0 to 1.0)
        """
        # Compute test-based reward
        total = len(test_results)
        passed = sum(1 for t in test_results if t.get("passed", False))
        reward = passed / total if total > 0 else 0.0

        # Compute performance (execution) metrics
        start = time.time()
        # Optionally re-run code for performance, or record previous
        exec_time = time.time() - start

        # Compute code complexity
        complexity = calculate_complexity(code)

        # Calculate test durations if available
        test_durations = []
        for test in test_results:
            if "duration_s" in test:
                test_durations.append(test["duration_s"])

        # Create validation dict from test results
        validation = {
            "test_results": test_results,
            "total_tests": total,
            "passed_tests": passed,
            "reward": reward,
            "metadata": metadata or {},
            "test_durations": test_durations,
        }

        # Merge additional metadata
        meta = metadata.copy() if metadata else {}
        meta.update(
            {
                "tests_total": total,
                "tests_passed": passed,
                "exec_time_s": exec_time,
                "cyclomatic_complexity": complexity,
            }
        )

        # Save with enhanced metrics
        self.save_successful_pattern(
            prompt=prompt,
            code=code,
            validation=validation,
            task_description=f"Test execution with {passed}/{total} tests passed",
            reward=reward,
            notes=f"Test-as-Reward: {reward:.2f} ({passed}/{total} tests passed)",
            exec_time_s=exec_time,
            cyclomatic_complexity=complexity,
            tests_total=total,
            tests_passed=passed,
        )

        return reward

    def get_patterns(self, limit: int | None = None) -> list[Pattern]:
        """Get all patterns, optionally limited"""
        if limit:
            return self.patterns[-limit:]
        return self.patterns

    def get_patterns_by_score(self, min_score: float = 0.0) -> list[Pattern]:
        """Get patterns with reward above minimum score"""
        return [p for p in self.patterns if p.reward and p.reward >= min_score]

    def get_patterns_by_task(self, task_keyword: str) -> list[Pattern]:
        """Get patterns containing task keyword"""
        keyword_lower = task_keyword.lower()
        return [
            p
            for p in self.patterns
            if keyword_lower in p.task_description.lower() or keyword_lower in p.prompt.lower()
        ]

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about saved patterns"""
        if not self.patterns:
            return {
                "total_patterns": 0,
                "average_reward": 0.0,
                "recent_patterns": 0,
                "high_reward_patterns": 0,
            }

        # Calculate average reward
        rewards = [p.reward for p in self.patterns if p.reward is not None]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0

        # Count recent patterns (last 7 days)
        recent_count = sum(1 for p in self.patterns if self._is_recent(p.timestamp))

        # Count high reward patterns (reward >= 0.8)
        high_reward_count = sum(1 for p in self.patterns if p.reward and p.reward >= 0.8)

        return {
            "total_patterns": len(self.patterns),
            "average_reward": round(avg_reward, 3),
            "recent_patterns": recent_count,
            "high_reward_patterns": high_reward_count,
            "reward_distribution": {
                "0.0-0.2": sum(1 for p in self.patterns if p.reward and 0.0 <= p.reward < 0.2),
                "0.2-0.4": sum(1 for p in self.patterns if p.reward and 0.2 <= p.reward < 0.4),
                "0.4-0.6": sum(1 for p in self.patterns if p.reward and 0.4 <= p.reward < 0.6),
                "0.6-0.8": sum(1 for p in self.patterns if p.reward and 0.6 <= p.reward < 0.8),
                "0.8-1.0": sum(1 for p in self.patterns if p.reward and 0.8 <= p.reward <= 1.0),
            },
        }

    def _is_recent(self, timestamp: str, days: int = 7) -> bool:
        """Check if timestamp is within specified days"""
        try:
            pattern_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            now = datetime.now(pattern_time.tzinfo)
            return (now - pattern_time).days <= days
        except:
            return False

    def delete_pattern(self, index: int) -> bool:
        """Delete pattern at specified index"""
        try:
            if 0 <= index < len(self.patterns):
                del self.patterns[index]
                self._save_patterns()
                return True
            return False
        except Exception as e:
            print(f"Error deleting pattern: {e}")
            return False

    def export_patterns(self, export_file: str) -> bool:
        """Export patterns to specified file"""
        try:
            pattern_data = [asdict(pattern) for pattern in self.patterns]
            with open(export_file, "w", encoding="utf-8") as f:
                json.dump(pattern_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error exporting patterns: {e}")
            return False


# Legacy functions for backward compatibility
def _load_patterns():
    if not os.path.exists(PATTERNS_FILE):
        return []
    with open(PATTERNS_FILE, encoding="utf-8") as f:
        return json.load(f)


def _save_patterns(patterns):
    with open(PATTERNS_FILE, "w", encoding="utf-8") as f:
        json.dump(patterns, f, ensure_ascii=False, indent=2)


def save_successful_pattern(
    prompt: str, code: str, validation: dict[str, Any], task_description: str, **kwargs
) -> bool:
    """Legacy function for backward compatibility"""
    learning_system = LearningSystem()
    return learning_system.save_successful_pattern(
        prompt=prompt,
        code=code,
        validation=validation,
        task_description=task_description,
        **kwargs,
    )


def log_test_reward(prompt: str, code: str, test_results: list, metadata: dict = None) -> float:
    """Legacy function for backward compatibility"""
    learning_system = LearningSystem()
    return learning_system.log_test_reward(prompt, code, test_results, metadata)


# Test the learning system
if __name__ == "__main__":
    learning_system = LearningSystem()

    # Test saving a pattern
    test_validation = {"tests_passed": 3, "total_tests": 5}
    success = learning_system.save_successful_pattern(
        prompt="Create a function that adds two numbers",
        code="def add(a, b): return a + b",
        validation=test_validation,
        task_description="Simple addition function",
        reward=0.6,
    )

    print(f"Pattern saved: {success}")
    print(f"Statistics: {learning_system.get_statistics()}")
