"""
Learning System for CodeConductor
Saves successful prompt-code patterns for analysis and model optimization
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class Pattern:
    """Represents a successful prompt-code pattern"""

    prompt: str
    code: str
    validation: Dict[str, Any]
    task_description: str
    timestamp: str
    model_used: Optional[str] = None
    execution_time: Optional[float] = None
    user_rating: Optional[int] = None  # 1-5 scale
    notes: Optional[str] = None


class LearningSystem:
    """
    Manages saving and loading of successful code generation patterns
    """

    def __init__(self, patterns_file: str = "patterns.json"):
        self.patterns_file = Path(patterns_file)
        self.patterns: List[Pattern] = []
        self._load_patterns()

    def _load_patterns(self):
        """Load existing patterns from JSON file"""
        if self.patterns_file.exists():
            try:
                with open(self.patterns_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.patterns = [Pattern(**pattern_data) for pattern_data in data]
            except (json.JSONDecodeError, KeyError) as e:
                print(
                    f"Warning: Could not load patterns from {self.patterns_file}: {e}"
                )
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
        validation: Dict[str, Any],
        task_description: str,
        model_used: Optional[str] = None,
        execution_time: Optional[float] = None,
        user_rating: Optional[int] = None,
        notes: Optional[str] = None,
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
            )

            self.patterns.append(pattern)
            self._save_patterns()

            print(
                f"✅ Pattern saved successfully! Total patterns: {len(self.patterns)}"
            )
            return True

        except Exception as e:
            print(f"❌ Error saving pattern: {e}")
            return False

    def get_patterns(self, limit: Optional[int] = None) -> List[Pattern]:
        """Get all patterns, optionally limited"""
        if limit:
            return self.patterns[-limit:]  # Return most recent patterns
        return self.patterns.copy()

    def get_patterns_by_score(self, min_score: float = 0.0) -> List[Pattern]:
        """Get patterns with validation score above threshold"""
        return [
            pattern
            for pattern in self.patterns
            if pattern.validation.get("score", 0.0) >= min_score
        ]

    def get_patterns_by_task(self, task_keyword: str) -> List[Pattern]:
        """Get patterns related to a specific task type"""
        keyword_lower = task_keyword.lower()
        return [
            pattern
            for pattern in self.patterns
            if keyword_lower in pattern.task_description.lower()
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about saved patterns"""
        if not self.patterns:
            return {
                "total_patterns": 0,
                "average_score": 0.0,
                "best_score": 0.0,
                "task_types": [],
                "models_used": [],
            }

        scores = [p.validation.get("score", 0.0) for p in self.patterns]
        task_types = list(set(p.task_description.split()[0] for p in self.patterns))
        models_used = list(set(p.model_used for p in self.patterns if p.model_used))

        return {
            "total_patterns": len(self.patterns),
            "average_score": sum(scores) / len(scores),
            "best_score": max(scores),
            "task_types": task_types,
            "models_used": models_used,
            "recent_patterns": len(
                [p for p in self.patterns if self._is_recent(p.timestamp)]
            ),
        }

    def _is_recent(self, timestamp: str, days: int = 7) -> bool:
        """Check if pattern is from recent days"""
        try:
            pattern_time = datetime.fromisoformat(timestamp)
            days_ago = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            days_ago = days_ago.replace(day=days_ago.day - days)
            return pattern_time >= days_ago
        except:
            return False

    def delete_pattern(self, index: int) -> bool:
        """Delete a pattern by index"""
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
        """Export patterns to a different file"""
        try:
            pattern_data = [asdict(pattern) for pattern in self.patterns]
            with open(export_file, "w", encoding="utf-8") as f:
                json.dump(pattern_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error exporting patterns: {e}")
            return False


# Global instance for easy access
learning_system = LearningSystem()


def save_successful_pattern(
    prompt: str, code: str, validation: Dict[str, Any], task_description: str, **kwargs
) -> bool:
    """
    Convenience function to save a successful pattern
    """
    return learning_system.save_successful_pattern(
        prompt=prompt,
        code=code,
        validation=validation,
        task_description=task_description,
        **kwargs,
    )


# Example usage
if __name__ == "__main__":
    # Test the learning system
    test_validation = {
        "score": 0.85,
        "is_valid": True,
        "issues": [],
        "suggestions": ["Add more error handling"],
        "compliance": {"syntax_valid": True, "has_type_hints": True},
        "metrics": {"total_lines": 50, "total_functions": 3},
    }

    success = save_successful_pattern(
        prompt="Create a FastAPI endpoint for user authentication",
        code="from fastapi import FastAPI\n\n@app.post('/login')\ndef login():\n    return {'message': 'success'}",
        validation=test_validation,
        task_description="Create authentication endpoint",
        model_used="phi3",
        user_rating=4,
    )

    print(f"Pattern saved: {success}")
    print(f"Statistics: {learning_system.get_statistics()}")
