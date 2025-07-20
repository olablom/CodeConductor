"""
RewardAgent - Calculates rewards for reinforcement learning

This module implements the reward calculation logic for the RL system.
Rewards are based on test results, code complexity, policy violations,
and human feedback to guide the learning process.
"""

import math
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


@dataclass
class TestResults:
    """Container for test execution results."""

    passed: int
    failed: int
    total: int
    execution_time: float
    coverage: Optional[float] = None
    lint_score: Optional[float] = None

    @property
    def pass_rate(self) -> float:
        """Calculate test pass rate."""
        return self.passed / self.total if self.total > 0 else 0.0


@dataclass
class CodeMetrics:
    """Container for code quality metrics."""

    complexity: float  # Cyclomatic complexity
    lines_of_code: int
    function_count: int
    class_count: int
    comment_ratio: float = 0.0

    @property
    def complexity_score(self) -> float:
        """Calculate normalized complexity score (0-1, lower is better)."""
        # Normalize complexity: 1-10 = good, 10+ = penalized
        if self.complexity <= 10:
            return 1.0
        else:
            return max(0.0, 1.0 - (self.complexity - 10) / 20)


class RewardAgent(BaseAgent):
    """
    Agent responsible for calculating rewards for the RL system.

    This agent analyzes various metrics and calculates a reward that
    guides the learning process toward better code generation.
    """

    def __init__(
        self, name: str = "reward_agent", config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the reward agent."""
        default_config = {
            "test_weight": 0.4,  # Weight for test results
            "complexity_weight": 0.2,  # Weight for code complexity
            "policy_weight": 0.2,  # Weight for policy compliance
            "feedback_weight": 0.2,  # Weight for human feedback
            "complexity_threshold": 10.0,  # Threshold for complexity penalty
            "min_reward": -1.0,  # Minimum possible reward
            "max_reward": 1.0,  # Maximum possible reward
        }

        if config:
            default_config.update(config)

        super().__init__(name, default_config)

        logger.info(f"Initialized RewardAgent with config: {self.config}")

    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the context to understand what metrics are available.

        Args:
            context: Dictionary containing test results, code metrics, etc.

        Returns:
            Analysis of available metrics and their quality
        """
        analysis = {
            "metrics_available": [],
            "data_quality": "unknown",
            "recommendations": [],
        }

        # Check what metrics are available
        if "test_results" in context:
            analysis["metrics_available"].append("test_results")
        if "code_metrics" in context:
            analysis["metrics_available"].append("code_metrics")
        if "policy_violations" in context:
            analysis["metrics_available"].append("policy_violations")
        if "human_feedback" in context:
            analysis["metrics_available"].append("human_feedback")

        # Assess data quality
        if len(analysis["metrics_available"]) >= 3:
            analysis["data_quality"] = "good"
        elif len(analysis["metrics_available"]) >= 2:
            analysis["data_quality"] = "fair"
        else:
            analysis["data_quality"] = "poor"
            analysis["recommendations"].append(
                "More metrics needed for accurate reward calculation"
            )

        return analysis

    def propose(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Propose reward calculation strategy based on available metrics.

        Args:
            analysis: Analysis results from analyze()

        Returns:
            Proposed reward calculation approach
        """
        available_metrics = analysis.get("metrics_available", [])

        strategy = {
            "approach": "weighted_combination",
            "weights": {},
            "confidence": 0.8,
        }

        # Adjust weights based on available metrics
        if "test_results" in available_metrics:
            strategy["weights"]["test"] = self.config["test_weight"]
        if "code_metrics" in available_metrics:
            strategy["weights"]["complexity"] = self.config["complexity_weight"]
        if "policy_violations" in available_metrics:
            strategy["weights"]["policy"] = self.config["policy_weight"]
        if "human_feedback" in available_metrics:
            strategy["weights"]["feedback"] = self.config["feedback_weight"]

        return strategy

    def review(self, code: str) -> Dict[str, Any]:
        """
        Review code and provide reward-related insights.

        Args:
            code: Code string to review

        Returns:
            Review results with reward-related recommendations
        """
        # This is a placeholder - in practice, this would analyze the code
        # and provide specific recommendations for improving rewards

        return {
            "quality_score": 0.8,
            "issues": [],
            "recommendations": [
                "Add more comprehensive tests to improve reward calculation",
                "Consider code complexity optimization",
                "Ensure policy compliance for better rewards",
            ],
            "reward_potential": "high",
        }

    def calculate_reward(
        self,
        test_results: Optional[TestResults] = None,
        code_metrics: Optional[CodeMetrics] = None,
        policy_violations: Optional[List[str]] = None,
        human_feedback: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate reward based on various metrics.

        Args:
            test_results: Test execution results
            code_metrics: Code quality metrics
            policy_violations: List of policy violations
            human_feedback: Human feedback scores and comments

        Returns:
            Dictionary containing reward value and breakdown
        """
        reward_components = {}
        total_reward = 0.0

        # 1. Test Results Reward
        if test_results:
            test_reward = self._calculate_test_reward(test_results)
            reward_components["test"] = test_reward
            total_reward += test_reward * self.config["test_weight"]

        # 2. Complexity Reward
        if code_metrics:
            complexity_reward = self._calculate_complexity_reward(code_metrics)
            reward_components["complexity"] = complexity_reward
            total_reward += complexity_reward * self.config["complexity_weight"]

        # 3. Policy Compliance Reward
        if policy_violations is not None:
            policy_reward = self._calculate_policy_reward(policy_violations)
            reward_components["policy"] = policy_reward
            total_reward += policy_reward * self.config["policy_weight"]

        # 4. Human Feedback Reward
        if human_feedback:
            feedback_reward = self._calculate_feedback_reward(human_feedback)
            reward_components["feedback"] = feedback_reward
            total_reward += feedback_reward * self.config["feedback_weight"]

        # Normalize reward to configured range
        total_reward = max(
            self.config["min_reward"], min(self.config["max_reward"], total_reward)
        )

        result = {
            "total_reward": total_reward,
            "components": reward_components,
            "weights": {
                "test": self.config["test_weight"],
                "complexity": self.config["complexity_weight"],
                "policy": self.config["policy_weight"],
                "feedback": self.config["feedback_weight"],
            },
            "metadata": {"agent": self.name, "timestamp": self._get_timestamp()},
        }

        logger.info(
            f"Calculated reward: {total_reward:.3f} with components: {reward_components}"
        )
        return result

    def _calculate_test_reward(self, test_results: TestResults) -> float:
        """Calculate reward based on test results."""
        # Base reward on pass rate
        pass_rate = test_results.pass_rate

        # Bonus for high coverage
        coverage_bonus = 0.0
        if test_results.coverage:
            if test_results.coverage >= 0.9:
                coverage_bonus = 0.2
            elif test_results.coverage >= 0.8:
                coverage_bonus = 0.1
            elif test_results.coverage >= 0.7:
                coverage_bonus = 0.05

        # Bonus for good lint score
        lint_bonus = 0.0
        if test_results.lint_score:
            if test_results.lint_score >= 9.0:
                lint_bonus = 0.1
            elif test_results.lint_score >= 8.0:
                lint_bonus = 0.05

        # Penalty for slow execution
        time_penalty = 0.0
        if test_results.execution_time > 10.0:  # More than 10 seconds
            time_penalty = -0.1

        reward = pass_rate + coverage_bonus + lint_bonus + time_penalty
        return max(0.0, min(1.0, reward))  # Clamp to [0, 1]

    def _calculate_complexity_reward(self, code_metrics: CodeMetrics) -> float:
        """Calculate reward based on code complexity."""
        # Base reward on complexity score
        complexity_score = code_metrics.complexity_score

        # Bonus for good comment ratio
        comment_bonus = 0.0
        if code_metrics.comment_ratio >= 0.2:
            comment_bonus = 0.1
        elif code_metrics.comment_ratio >= 0.1:
            comment_bonus = 0.05

        # Bonus for reasonable function/class count
        structure_bonus = 0.0
        if 1 <= code_metrics.function_count <= 20:
            structure_bonus = 0.05
        if 0 <= code_metrics.class_count <= 5:
            structure_bonus += 0.05

        reward = complexity_score + comment_bonus + structure_bonus
        return max(0.0, min(1.0, reward))  # Clamp to [0, 1]

    def _calculate_policy_reward(self, policy_violations: List[str]) -> float:
        """Calculate reward based on policy compliance."""
        if not policy_violations:
            return 1.0  # Perfect compliance

        # Penalty based on number and severity of violations
        base_penalty = len(policy_violations) * 0.15  # Reduced from 0.2

        # Additional penalty for severe violations
        severe_penalties = 0.0
        for violation in policy_violations:
            if any(
                severe in violation.lower()
                for severe in ["security", "dangerous", "unsafe"]
            ):
                severe_penalties += 0.3
            elif any(
                moderate in violation.lower() for moderate in ["deprecated", "warning"]
            ):
                severe_penalties += 0.05  # Reduced from 0.1

        total_penalty = base_penalty + severe_penalties
        reward = max(0.0, 1.0 - total_penalty)  # Clamp to [0, 1]

        return reward

    def _calculate_feedback_reward(self, human_feedback: Dict[str, Any]) -> float:
        """Calculate reward based on human feedback."""
        # Extract feedback scores
        thumbs_up = human_feedback.get("thumbs_up", 0)
        thumbs_down = human_feedback.get("thumbs_down", 0)
        rating = human_feedback.get("rating", 0.5)  # 0-1 scale
        comments = human_feedback.get("comments", [])

        # Calculate base reward from rating
        base_reward = rating

        # Adjust based on thumbs up/down ratio
        if thumbs_up + thumbs_down > 0:
            ratio = thumbs_up / (thumbs_up + thumbs_down)
            base_reward = (base_reward + ratio) / 2

        # Bonus for positive comments
        comment_bonus = 0.0
        positive_keywords = ["good", "great", "excellent", "perfect", "nice", "clean"]
        negative_keywords = ["bad", "poor", "terrible", "ugly", "messy", "broken"]

        for comment in comments:
            comment_lower = comment.lower()
            positive_count = sum(
                1 for keyword in positive_keywords if keyword in comment_lower
            )
            negative_count = sum(
                1 for keyword in negative_keywords if keyword in comment_lower
            )

            if positive_count > negative_count:
                comment_bonus += 0.1
            elif negative_count > positive_count:
                comment_bonus -= 0.1

        reward = base_reward + comment_bonus
        return max(0.0, min(1.0, reward))  # Clamp to [0, 1]

    def _get_timestamp(self) -> str:
        """Get current timestamp for logging."""
        from datetime import datetime

        return datetime.now().isoformat()

    def get_reward_statistics(self) -> Dict[str, Any]:
        """Get statistics about reward calculations."""
        return {
            "agent_name": self.name,
            "config": self.config,
            "total_calculations": getattr(self, "_total_calculations", 0),
            "average_reward": getattr(self, "_average_reward", 0.0),
            "min_reward": getattr(self, "_min_reward", 0.0),
            "max_reward": getattr(self, "_max_reward", 0.0),
        }
