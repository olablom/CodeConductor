"""
Reward Agent for CodeConductor

This agent calculates rewards for the reinforcement learning system based on:
- Test results (pass/fail, coverage)
- Code quality metrics
- Human feedback scores
- Policy compliance results
- Performance metrics
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from agents.base_agent import BaseAgent


@dataclass
class TestResult:
    """Represents test execution results."""

    passed: bool
    total_tests: int
    passed_tests: int
    failed_tests: int
    coverage_percentage: float
    execution_time: float
    error_messages: List[str]


@dataclass
class CodeQualityMetrics:
    """Represents code quality assessment."""

    quality_score: float
    complexity_score: float
    maintainability_score: float
    documentation_score: float
    style_score: float
    security_score: float


@dataclass
class HumanFeedback:
    """Represents human feedback on generated code."""

    overall_rating: float  # 0.0 to 1.0
    usefulness_rating: float
    correctness_rating: float
    completeness_rating: float
    comments: str
    approved: bool


@dataclass
class PolicyResult:
    """Represents policy compliance results."""

    safe: bool
    decision: str  # "pass", "warn", "block"
    risk_level: str  # "low", "medium", "high", "critical"
    violations_count: int
    critical_violations: int
    high_violations: int


class RewardAgent(BaseAgent):
    """
    Reward calculation agent for reinforcement learning.

    This agent focuses on:
    - Calculating comprehensive rewards
    - Balancing multiple reward factors
    - Providing reward breakdowns
    - Generating improvement recommendations
    """

    def __init__(
        self, name: str = "reward_agent", config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the reward agent."""
        super().__init__(name, config)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Reward configuration
        self.reward_config = config.get("reward", {}) if config else {}

        # Default reward weights
        self.weights = {
            "test_results": 0.3,
            "code_quality": 0.25,
            "human_feedback": 0.25,
            "policy_compliance": 0.15,
            "performance": 0.05,
        }

        # Override with config if provided
        if "weights" in self.reward_config:
            self.weights.update(self.reward_config["weights"])

        # Reward thresholds
        self.thresholds = {
            "excellent": 0.9,
            "good": 0.7,
            "acceptable": 0.5,
            "needs_improvement": 0.3,
        }

        # Penalty factors
        self.penalties = {
            "test_failure": 0.5,
            "policy_violation": 0.3,
            "low_quality": 0.2,
            "human_rejection": 0.4,
        }

        self.logger.info(
            f"RewardAgent '{name}' initialized with weights: {self.weights}"
        )

    def calculate_reward(
        self,
        test_results: TestResult,
        code_quality: CodeQualityMetrics,
        human_feedback: HumanFeedback,
        policy_result: PolicyResult,
        performance_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive reward for the reinforcement learning system.

        Args:
            test_results: Test execution results
            code_quality: Code quality metrics
            human_feedback: Human feedback scores
            policy_result: Policy compliance results
            performance_metrics: Optional performance metrics

        Returns:
            Comprehensive reward calculation with breakdown
        """
        self.logger.info(f"Calculating reward for {self.name}")

        try:
            # Calculate individual reward components
            test_reward = self._calculate_test_reward(test_results)
            quality_reward = self._calculate_quality_reward(code_quality)
            feedback_reward = self._calculate_feedback_reward(human_feedback)
            policy_reward = self._calculate_policy_reward(policy_result)
            performance_reward = self._calculate_performance_reward(performance_metrics)

            # Calculate weighted total reward
            total_reward = (
                test_reward * self.weights["test_results"]
                + quality_reward * self.weights["code_quality"]
                + feedback_reward * self.weights["human_feedback"]
                + policy_reward * self.weights["policy_compliance"]
                + performance_reward * self.weights["performance"]
            )

            # Apply penalties
            total_reward = self._apply_penalties(
                total_reward, test_results, policy_result, human_feedback
            )

            # Ensure reward is in valid range
            total_reward = max(0.0, min(1.0, total_reward))

            # Generate recommendations
            recommendations = self._generate_recommendations(
                test_reward,
                quality_reward,
                feedback_reward,
                policy_reward,
                performance_reward,
            )

            # Determine reward level
            reward_level = self._determine_reward_level(total_reward)

            result = {
                "total_reward": total_reward,
                "reward_level": reward_level,
                "breakdown": {
                    "test_reward": test_reward,
                    "quality_reward": quality_reward,
                    "feedback_reward": feedback_reward,
                    "policy_reward": policy_reward,
                    "performance_reward": performance_reward,
                },
                "weights": self.weights,
                "recommendations": recommendations,
                "thresholds": self.thresholds,
                "metadata": {
                    "test_passed": test_results.passed,
                    "human_approved": human_feedback.approved,
                    "policy_safe": policy_result.safe,
                    "quality_score": code_quality.quality_score,
                },
            }

            self.logger.info(f"Reward calculated: {total_reward:.3f} ({reward_level})")
            return result

        except Exception as e:
            self.logger.error(f"Error calculating reward: {e}")
            return {
                "total_reward": 0.0,
                "reward_level": "error",
                "error": str(e),
                "breakdown": {},
                "recommendations": ["Reward calculation failed"],
            }

    def _calculate_test_reward(self, test_results: TestResult) -> float:
        """Calculate reward based on test results."""
        if test_results.total_tests == 0:
            return 0.0

        # Base reward from test pass rate
        pass_rate = test_results.passed_tests / test_results.total_tests

        # Coverage bonus (up to 20% bonus for high coverage)
        coverage_bonus = min(0.2, test_results.coverage_percentage / 100.0 * 0.2)

        # Execution time penalty (if too slow)
        time_penalty = 0.0
        if test_results.execution_time > 10.0:  # More than 10 seconds
            time_penalty = min(0.1, (test_results.execution_time - 10.0) / 100.0)

        # Error penalty
        error_penalty = min(0.1, len(test_results.error_messages) * 0.02)

        reward = pass_rate + coverage_bonus - time_penalty - error_penalty
        return max(0.0, min(1.0, reward))

    def _calculate_quality_reward(self, code_quality: CodeQualityMetrics) -> float:
        """Calculate reward based on code quality metrics."""
        # Weighted average of quality scores
        quality_components = [
            code_quality.quality_score * 0.3,
            code_quality.complexity_score * 0.2,
            code_quality.maintainability_score * 0.2,
            code_quality.documentation_score * 0.15,
            code_quality.style_score * 0.1,
            code_quality.security_score * 0.05,
        ]

        return sum(quality_components)

    def _calculate_feedback_reward(self, human_feedback: HumanFeedback) -> float:
        """Calculate reward based on human feedback."""
        # Base reward from overall rating
        base_reward = human_feedback.overall_rating

        # Bonus for high ratings in specific areas
        bonus = 0.0
        if human_feedback.usefulness_rating > 0.8:
            bonus += 0.05
        if human_feedback.correctness_rating > 0.8:
            bonus += 0.05
        if human_feedback.completeness_rating > 0.8:
            bonus += 0.05

        # Approval bonus
        if human_feedback.approved:
            bonus += 0.1

        reward = base_reward + bonus
        return max(0.0, min(1.0, reward))

    def _calculate_policy_reward(self, policy_result: PolicyResult) -> float:
        """Calculate reward based on policy compliance."""
        if not policy_result.safe:
            return 0.0

        # Base reward based on decision
        if policy_result.decision == "pass":
            base_reward = 1.0
        elif policy_result.decision == "warn":
            base_reward = 0.7
        else:
            base_reward = 0.0

        # Penalty for violations
        violation_penalty = (
            policy_result.critical_violations * 0.3
            + policy_result.high_violations * 0.15
            + policy_result.violations_count * 0.05
        )

        reward = base_reward - violation_penalty
        return max(0.0, min(1.0, reward))

    def _calculate_performance_reward(
        self, performance_metrics: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate reward based on performance metrics."""
        if not performance_metrics:
            return 0.5  # Neutral reward if no metrics

        # Example performance metrics
        response_time = performance_metrics.get("response_time", 1.0)
        memory_usage = performance_metrics.get("memory_usage", 100.0)
        cpu_usage = performance_metrics.get("cpu_usage", 50.0)

        # Calculate performance score
        time_score = max(
            0.0, 1.0 - (response_time - 1.0) / 10.0
        )  # Penalty for slow response
        memory_score = max(
            0.0, 1.0 - (memory_usage - 100.0) / 1000.0
        )  # Penalty for high memory
        cpu_score = max(0.0, 1.0 - (cpu_usage - 50.0) / 100.0)  # Penalty for high CPU

        performance_score = (time_score + memory_score + cpu_score) / 3.0
        return max(0.0, min(1.0, performance_score))

    def _apply_penalties(
        self,
        total_reward: float,
        test_results: TestResult,
        policy_result: PolicyResult,
        human_feedback: HumanFeedback,
    ) -> float:
        """Apply penalties based on failures and violations."""
        penalty = 0.0

        # Test failure penalty
        if not test_results.passed:
            penalty += self.penalties["test_failure"]

        # Policy violation penalty
        if not policy_result.safe or policy_result.decision == "block":
            penalty += self.penalties["policy_violation"]

        # Human rejection penalty
        if not human_feedback.approved:
            penalty += self.penalties["human_rejection"]

        # Low quality penalty
        if (
            test_results.passed
            and test_results.passed_tests / max(test_results.total_tests, 1) < 0.8
        ):
            penalty += self.penalties["low_quality"]

        return max(0.0, total_reward - penalty)

    def _generate_recommendations(
        self,
        test_reward: float,
        quality_reward: float,
        feedback_reward: float,
        policy_reward: float,
        performance_reward: float,
    ) -> List[str]:
        """Generate improvement recommendations based on reward components."""
        recommendations = []

        if test_reward < 0.7:
            recommendations.append("Improve test coverage and fix failing tests")

        if quality_reward < 0.7:
            recommendations.append("Enhance code quality and maintainability")

        if feedback_reward < 0.7:
            recommendations.append(
                "Address human feedback and improve user satisfaction"
            )

        if policy_reward < 0.7:
            recommendations.append(
                "Fix policy violations and improve security compliance"
            )

        if performance_reward < 0.7:
            recommendations.append("Optimize performance and resource usage")

        if not recommendations:
            recommendations.append("Excellent performance across all metrics")

        return recommendations

    def _determine_reward_level(self, total_reward: float) -> str:
        """Determine reward level based on total reward."""
        if total_reward >= self.thresholds["excellent"]:
            return "excellent"
        elif total_reward >= self.thresholds["good"]:
            return "good"
        elif total_reward >= self.thresholds["acceptable"]:
            return "acceptable"
        elif total_reward >= self.thresholds["needs_improvement"]:
            return "needs_improvement"
        else:
            return "poor"

    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze context for reward calculation preparation.

        Args:
            context: Context information

        Returns:
            Analysis results for reward calculation
        """
        self.logger.info(f"Analyzing context for reward calculation")

        return {
            "reward_ready": True,
            "weights": self.weights,
            "thresholds": self.thresholds,
            "recommendations": ["Context ready for reward calculation"],
        }

    def propose(
        self, analysis: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Propose reward calculation strategy.

        Args:
            analysis: Analysis results
            context: Context information

        Returns:
            Reward calculation strategy
        """
        self.logger.info(f"Proposing reward calculation strategy")

        return {
            "strategy": "comprehensive_reward",
            "weights": self.weights,
            "thresholds": self.thresholds,
            "penalties": self.penalties,
        }

    def review(
        self, proposal: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Review reward calculation proposal.

        Args:
            proposal: Proposal to review
            context: Context information

        Returns:
            Review results
        """
        self.logger.info(f"Reviewing reward calculation proposal")

        return {
            "approved": True,
            "confidence": 0.9,
            "recommendations": ["Reward calculation strategy looks good"],
        }
