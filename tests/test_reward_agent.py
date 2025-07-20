"""
Tests for RewardAgent class.

This module tests the reward calculation logic and ensures proper
handling of different metrics and scenarios.
"""

import pytest
from agents.reward_agent import RewardAgent, TestResults, CodeMetrics


class TestTestResults:
    """Test cases for TestResults dataclass."""

    def test_test_results_creation(self):
        """Test creating TestResults instance."""
        results = TestResults(
            passed=8,
            failed=2,
            total=10,
            execution_time=5.5,
            coverage=0.85,
            lint_score=9.2,
        )

        assert results.passed == 8
        assert results.failed == 2
        assert results.total == 10
        assert results.execution_time == 5.5
        assert results.coverage == 0.85
        assert results.lint_score == 9.2

    def test_pass_rate_calculation(self):
        """Test pass rate calculation."""
        results = TestResults(passed=8, failed=2, total=10, execution_time=5.0)
        assert results.pass_rate == 0.8

    def test_pass_rate_zero_total(self):
        """Test pass rate with zero total tests."""
        results = TestResults(passed=0, failed=0, total=0, execution_time=0.0)
        assert results.pass_rate == 0.0


class TestCodeMetrics:
    """Test cases for CodeMetrics dataclass."""

    def test_code_metrics_creation(self):
        """Test creating CodeMetrics instance."""
        metrics = CodeMetrics(
            complexity=8.5,
            lines_of_code=150,
            function_count=12,
            class_count=3,
            comment_ratio=0.15,
        )

        assert metrics.complexity == 8.5
        assert metrics.lines_of_code == 150
        assert metrics.function_count == 12
        assert metrics.class_count == 3
        assert metrics.comment_ratio == 0.15

    def test_complexity_score_low(self):
        """Test complexity score for low complexity."""
        metrics = CodeMetrics(
            complexity=5.0, lines_of_code=100, function_count=5, class_count=1
        )
        assert metrics.complexity_score == 1.0

    def test_complexity_score_high(self):
        """Test complexity score for high complexity."""
        metrics = CodeMetrics(
            complexity=25.0, lines_of_code=200, function_count=20, class_count=5
        )
        # Should be penalized: 1.0 - (25-10)/20 = 1.0 - 0.75 = 0.25
        assert metrics.complexity_score == 0.25

    def test_complexity_score_boundary(self):
        """Test complexity score at boundary."""
        metrics = CodeMetrics(
            complexity=10.0, lines_of_code=100, function_count=5, class_count=1
        )
        assert metrics.complexity_score == 1.0


class TestRewardAgent:
    """Test cases for RewardAgent class."""

    def test_reward_agent_initialization(self):
        """Test RewardAgent initialization."""
        agent = RewardAgent("test_reward_agent")

        assert agent.name == "test_reward_agent"
        assert agent.config["test_weight"] == 0.4
        assert agent.config["complexity_weight"] == 0.2
        assert agent.config["policy_weight"] == 0.2
        assert agent.config["feedback_weight"] == 0.2

    def test_reward_agent_with_custom_config(self):
        """Test RewardAgent with custom configuration."""
        config = {
            "test_weight": 0.5,
            "complexity_weight": 0.3,
            "policy_weight": 0.1,
            "feedback_weight": 0.1,
        }
        agent = RewardAgent("custom_agent", config)

        assert agent.config["test_weight"] == 0.5
        assert agent.config["complexity_weight"] == 0.3
        assert agent.config["policy_weight"] == 0.1
        assert agent.config["feedback_weight"] == 0.1

    def test_analyze_method(self):
        """Test analyze method."""
        agent = RewardAgent("test_agent")

        context = {
            "test_results": TestResults(
                passed=8, failed=2, total=10, execution_time=5.0
            ),
            "code_metrics": CodeMetrics(
                complexity=8.0, lines_of_code=100, function_count=5, class_count=1
            ),
            "policy_violations": [],
            "human_feedback": {"rating": 0.8},
        }

        analysis = agent.analyze(context)

        assert "metrics_available" in analysis
        assert "data_quality" in analysis
        assert "recommendations" in analysis
        assert len(analysis["metrics_available"]) == 4
        assert analysis["data_quality"] == "good"

    def test_analyze_method_poor_data(self):
        """Test analyze method with poor data quality."""
        agent = RewardAgent("test_agent")

        context = {
            "test_results": TestResults(
                passed=5, failed=5, total=10, execution_time=5.0
            )
        }

        analysis = agent.analyze(context)

        assert len(analysis["metrics_available"]) == 1
        assert analysis["data_quality"] == "poor"
        assert len(analysis["recommendations"]) > 0

    def test_propose_method(self):
        """Test propose method."""
        agent = RewardAgent("test_agent")

        analysis = {
            "metrics_available": ["test_results", "code_metrics", "policy_violations"]
        }

        proposal = agent.propose(analysis)

        assert "approach" in proposal
        assert "weights" in proposal
        assert "confidence" in proposal
        assert proposal["approach"] == "weighted_combination"
        assert proposal["confidence"] == 0.8

    def test_review_method(self):
        """Test review method."""
        agent = RewardAgent("test_agent")

        code = "def hello_world():\n    print('Hello, World!')"
        review = agent.review(code)

        assert "quality_score" in review
        assert "issues" in review
        assert "recommendations" in review
        assert "reward_potential" in review
        assert review["quality_score"] == 0.8


class TestRewardCalculation:
    """Test cases for reward calculation methods."""

    def test_calculate_reward_all_metrics(self):
        """Test reward calculation with all metrics."""
        agent = RewardAgent("test_agent")

        test_results = TestResults(
            passed=9,
            failed=1,
            total=10,
            execution_time=3.0,
            coverage=0.9,
            lint_score=9.5,
        )
        code_metrics = CodeMetrics(
            complexity=8.0,
            lines_of_code=100,
            function_count=5,
            class_count=1,
            comment_ratio=0.2,
        )
        policy_violations = []
        human_feedback = {
            "thumbs_up": 3,
            "thumbs_down": 0,
            "rating": 0.9,
            "comments": ["Great code!"],
        }

        reward = agent.calculate_reward(
            test_results=test_results,
            code_metrics=code_metrics,
            policy_violations=policy_violations,
            human_feedback=human_feedback,
        )

        assert "total_reward" in reward
        assert "components" in reward
        assert "weights" in reward
        assert "metadata" in reward

        # Check components
        components = reward["components"]
        assert "test" in components
        assert "complexity" in components
        assert "policy" in components
        assert "feedback" in components

        # Total reward should be positive
        assert reward["total_reward"] > 0.0

    def test_calculate_reward_partial_metrics(self):
        """Test reward calculation with partial metrics."""
        agent = RewardAgent("test_agent")

        test_results = TestResults(passed=5, failed=5, total=10, execution_time=5.0)
        policy_violations = ["deprecated_function_used"]

        reward = agent.calculate_reward(
            test_results=test_results, policy_violations=policy_violations
        )

        assert "total_reward" in reward
        assert "test" in reward["components"]
        assert "policy" in reward["components"]
        assert "complexity" not in reward["components"]
        assert "feedback" not in reward["components"]

    def test_calculate_reward_no_metrics(self):
        """Test reward calculation with no metrics."""
        agent = RewardAgent("test_agent")

        reward = agent.calculate_reward()

        assert reward["total_reward"] == 0.0
        assert len(reward["components"]) == 0

    def test_test_reward_calculation(self):
        """Test test reward calculation."""
        agent = RewardAgent("test_agent")

        # Perfect test results
        perfect_results = TestResults(
            passed=10,
            failed=0,
            total=10,
            execution_time=2.0,
            coverage=0.95,
            lint_score=10.0,
        )
        perfect_reward = agent._calculate_test_reward(perfect_results)
        assert perfect_reward > 0.9

        # Poor test results
        poor_results = TestResults(
            passed=2,
            failed=8,
            total=10,
            execution_time=15.0,
            coverage=0.3,
            lint_score=5.0,
        )
        poor_reward = agent._calculate_test_reward(poor_results)
        assert poor_reward < 0.5

    def test_complexity_reward_calculation(self):
        """Test complexity reward calculation."""
        agent = RewardAgent("test_agent")

        # Good complexity
        good_metrics = CodeMetrics(
            complexity=5.0,
            lines_of_code=50,
            function_count=3,
            class_count=1,
            comment_ratio=0.25,
        )
        good_reward = agent._calculate_complexity_reward(good_metrics)
        assert good_reward > 0.8

        # Poor complexity
        poor_metrics = CodeMetrics(
            complexity=25.0,
            lines_of_code=500,
            function_count=50,
            class_count=10,
            comment_ratio=0.05,
        )
        poor_reward = agent._calculate_complexity_reward(poor_metrics)
        assert poor_reward < 0.5

    def test_policy_reward_calculation(self):
        """Test policy reward calculation."""
        agent = RewardAgent("test_agent")

        # No violations
        no_violations = []
        perfect_reward = agent._calculate_policy_reward(no_violations)
        assert perfect_reward == 1.0

        # Minor violations
        minor_violations = ["deprecated_function_used", "missing_docstring"]
        minor_reward = agent._calculate_policy_reward(minor_violations)
        assert 0.5 < minor_reward < 1.0

        # Severe violations
        severe_violations = ["security_vulnerability", "dangerous_code_pattern"]
        severe_reward = agent._calculate_policy_reward(severe_violations)
        assert severe_reward < 0.5

    def test_feedback_reward_calculation(self):
        """Test feedback reward calculation."""
        agent = RewardAgent("test_agent")

        # Positive feedback
        positive_feedback = {
            "thumbs_up": 5,
            "thumbs_down": 0,
            "rating": 0.9,
            "comments": ["Excellent work!", "Great code quality!"],
        }
        positive_reward = agent._calculate_feedback_reward(positive_feedback)
        assert positive_reward > 0.8

        # Negative feedback
        negative_feedback = {
            "thumbs_up": 0,
            "thumbs_down": 3,
            "rating": 0.2,
            "comments": ["Poor code", "Bad implementation"],
        }
        negative_reward = agent._calculate_feedback_reward(negative_feedback)
        assert negative_reward < 0.5

    def test_reward_normalization(self):
        """Test that rewards are properly normalized."""
        agent = RewardAgent("test_agent", {"min_reward": -1.0, "max_reward": 1.0})

        # Test with very high rewards
        test_results = TestResults(
            passed=10,
            failed=0,
            total=10,
            execution_time=1.0,
            coverage=1.0,
            lint_score=10.0,
        )
        code_metrics = CodeMetrics(
            complexity=1.0,
            lines_of_code=10,
            function_count=1,
            class_count=0,
            comment_ratio=0.5,
        )
        policy_violations = []
        human_feedback = {
            "thumbs_up": 10,
            "thumbs_down": 0,
            "rating": 1.0,
            "comments": ["Perfect!"],
        }

        reward = agent.calculate_reward(
            test_results=test_results,
            code_metrics=code_metrics,
            policy_violations=policy_violations,
            human_feedback=human_feedback,
        )

        # Should be clamped to max_reward
        assert reward["total_reward"] <= 1.0


class TestRewardAgentIntegration:
    """Integration tests for RewardAgent."""

    def test_full_workflow(self):
        """Test complete reward calculation workflow."""
        agent = RewardAgent("workflow_agent")

        # Step 1: Analyze context
        context = {
            "test_results": TestResults(
                passed=8, failed=2, total=10, execution_time=4.0
            ),
            "code_metrics": CodeMetrics(
                complexity=7.0, lines_of_code=80, function_count=4, class_count=1
            ),
            "policy_violations": ["missing_docstring"],
            "human_feedback": {
                "thumbs_up": 2,
                "thumbs_down": 0,
                "rating": 0.8,
                "comments": ["Good work"],
            },
        }

        analysis = agent.analyze(context)
        assert analysis["data_quality"] == "good"

        # Step 2: Propose strategy
        proposal = agent.propose(analysis)
        assert proposal["approach"] == "weighted_combination"

        # Step 3: Calculate reward
        reward = agent.calculate_reward(
            test_results=context["test_results"],
            code_metrics=context["code_metrics"],
            policy_violations=context["policy_violations"],
            human_feedback=context["human_feedback"],
        )

        assert reward["total_reward"] > 0.0
        assert len(reward["components"]) == 4

    def test_reward_statistics(self):
        """Test reward statistics functionality."""
        agent = RewardAgent("stats_agent")

        stats = agent.get_reward_statistics()

        assert "agent_name" in stats
        assert "config" in stats
        assert "total_calculations" in stats
        assert "average_reward" in stats
        assert stats["agent_name"] == "stats_agent"
