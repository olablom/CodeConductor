"""
Unit tests for RewardAgent reward calculation functionality
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to the path to import the agents module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.reward_agent import (
    RewardAgent,
    TestResult,
    CodeQualityMetrics,
    HumanFeedback,
    PolicyResult,
)


class TestRewardAgent(unittest.TestCase):
    """Test cases for RewardAgent reward calculation"""

    def setUp(self):
        """Set up test fixtures"""
        self.agent = RewardAgent("TestRewardAgent")

        # Sample test data
        self.sample_test_result = TestResult(
            passed=True,
            total_tests=10,
            passed_tests=9,
            failed_tests=1,
            coverage_percentage=85.0,
            execution_time=2.5,
            error_messages=[],
        )

        self.sample_code_quality = CodeQualityMetrics(
            quality_score=0.8,
            complexity_score=0.7,
            maintainability_score=0.8,
            documentation_score=0.9,
            style_score=0.8,
            security_score=0.9,
        )

        self.sample_human_feedback = HumanFeedback(
            overall_rating=0.8,
            usefulness_rating=0.9,
            correctness_rating=0.8,
            completeness_rating=0.7,
            comments="Good code, well structured",
            approved=True,
        )

        self.sample_policy_result = PolicyResult(
            safe=True,
            decision="pass",
            risk_level="low",
            violations_count=0,
            critical_violations=0,
            high_violations=0,
        )

    def test_init(self):
        """Test RewardAgent initialization"""
        agent = RewardAgent("TestAgent")
        self.assertEqual(agent.name, "TestAgent")
        self.assertIn("test_results", agent.weights)
        self.assertIn("code_quality", agent.weights)
        self.assertIn("human_feedback", agent.weights)
        self.assertIn("policy_compliance", agent.weights)
        self.assertIn("performance", agent.weights)

    def test_calculate_reward_basic(self):
        """Test basic reward calculation"""
        result = self.agent.calculate_reward(
            self.sample_test_result,
            self.sample_code_quality,
            self.sample_human_feedback,
            self.sample_policy_result,
        )

        self.assertIn("total_reward", result)
        self.assertIn("reward_level", result)
        self.assertIn("breakdown", result)
        self.assertIn("recommendations", result)
        self.assertIn("weights", result)
        self.assertIn("thresholds", result)
        self.assertIn("metadata", result)

        self.assertIsInstance(result["total_reward"], float)
        self.assertGreaterEqual(result["total_reward"], 0.0)
        self.assertLessEqual(result["total_reward"], 1.0)

    def test_calculate_reward_perfect_score(self):
        """Test reward calculation with perfect scores"""
        perfect_test = TestResult(
            passed=True,
            total_tests=10,
            passed_tests=10,
            failed_tests=0,
            coverage_percentage=100.0,
            execution_time=1.0,
            error_messages=[],
        )

        perfect_quality = CodeQualityMetrics(
            quality_score=1.0,
            complexity_score=1.0,
            maintainability_score=1.0,
            documentation_score=1.0,
            style_score=1.0,
            security_score=1.0,
        )

        perfect_feedback = HumanFeedback(
            overall_rating=1.0,
            usefulness_rating=1.0,
            correctness_rating=1.0,
            completeness_rating=1.0,
            comments="Perfect!",
            approved=True,
        )

        perfect_policy = PolicyResult(
            safe=True,
            decision="pass",
            risk_level="low",
            violations_count=0,
            critical_violations=0,
            high_violations=0,
        )

        result = self.agent.calculate_reward(
            perfect_test, perfect_quality, perfect_feedback, perfect_policy
        )

        self.assertGreater(result["total_reward"], 0.9)
        self.assertEqual(result["reward_level"], "excellent")

    def test_calculate_reward_poor_score(self):
        """Test reward calculation with poor scores"""
        poor_test = TestResult(
            passed=False,
            total_tests=10,
            passed_tests=2,
            failed_tests=8,
            coverage_percentage=20.0,
            execution_time=15.0,
            error_messages=["Test failed", "Syntax error"],
        )

        poor_quality = CodeQualityMetrics(
            quality_score=0.2,
            complexity_score=0.1,
            maintainability_score=0.2,
            documentation_score=0.1,
            style_score=0.3,
            security_score=0.2,
        )

        poor_feedback = HumanFeedback(
            overall_rating=0.2,
            usefulness_rating=0.1,
            correctness_rating=0.2,
            completeness_rating=0.1,
            comments="This is terrible",
            approved=False,
        )

        poor_policy = PolicyResult(
            safe=False,
            decision="block",
            risk_level="critical",
            violations_count=5,
            critical_violations=2,
            high_violations=3,
        )

        result = self.agent.calculate_reward(
            poor_test, poor_quality, poor_feedback, poor_policy
        )

        self.assertLess(result["total_reward"], 0.3)
        self.assertIn(result["reward_level"], ["poor", "needs_improvement"])

    def test_calculate_test_reward(self):
        """Test test reward calculation"""
        # Good test results
        good_test = TestResult(
            passed=True,
            total_tests=10,
            passed_tests=9,
            failed_tests=1,
            coverage_percentage=90.0,
            execution_time=2.0,
            error_messages=[],
        )

        good_reward = self.agent._calculate_test_reward(good_test)
        self.assertGreater(good_reward, 0.8)

        # Poor test results
        poor_test = TestResult(
            passed=False,
            total_tests=10,
            passed_tests=3,
            failed_tests=7,
            coverage_percentage=30.0,
            execution_time=20.0,
            error_messages=["Error 1", "Error 2", "Error 3"],
        )

        poor_reward = self.agent._calculate_test_reward(poor_test)
        self.assertLess(poor_reward, 0.5)

    def test_calculate_quality_reward(self):
        """Test quality reward calculation"""
        # Good quality
        good_quality = CodeQualityMetrics(
            quality_score=0.9,
            complexity_score=0.8,
            maintainability_score=0.9,
            documentation_score=0.9,
            style_score=0.8,
            security_score=0.9,
        )

        good_reward = self.agent._calculate_quality_reward(good_quality)
        self.assertGreater(good_reward, 0.8)

        # Poor quality
        poor_quality = CodeQualityMetrics(
            quality_score=0.3,
            complexity_score=0.2,
            maintainability_score=0.3,
            documentation_score=0.2,
            style_score=0.4,
            security_score=0.3,
        )

        poor_reward = self.agent._calculate_quality_reward(poor_quality)
        self.assertLess(poor_reward, 0.4)

    def test_calculate_feedback_reward(self):
        """Test feedback reward calculation"""
        # Good feedback
        good_feedback = HumanFeedback(
            overall_rating=0.9,
            usefulness_rating=0.9,
            correctness_rating=0.9,
            completeness_rating=0.8,
            comments="Excellent work!",
            approved=True,
        )

        good_reward = self.agent._calculate_feedback_reward(good_feedback)
        self.assertGreater(good_reward, 0.9)

        # Poor feedback
        poor_feedback = HumanFeedback(
            overall_rating=0.2,
            usefulness_rating=0.1,
            correctness_rating=0.2,
            completeness_rating=0.1,
            comments="This is bad",
            approved=False,
        )

        poor_reward = self.agent._calculate_feedback_reward(poor_feedback)
        self.assertLess(poor_reward, 0.4)

    def test_calculate_policy_reward(self):
        """Test policy reward calculation"""
        # Good policy compliance
        good_policy = PolicyResult(
            safe=True,
            decision="pass",
            risk_level="low",
            violations_count=0,
            critical_violations=0,
            high_violations=0,
        )

        good_reward = self.agent._calculate_policy_reward(good_policy)
        self.assertEqual(good_reward, 1.0)

        # Poor policy compliance
        poor_policy = PolicyResult(
            safe=False,
            decision="block",
            risk_level="critical",
            violations_count=5,
            critical_violations=2,
            high_violations=3,
        )

        poor_reward = self.agent._calculate_policy_reward(poor_policy)
        self.assertEqual(poor_reward, 0.0)

    def test_calculate_performance_reward(self):
        """Test performance reward calculation"""
        # Good performance
        good_performance = {
            "response_time": 0.5,
            "memory_usage": 50.0,
            "cpu_usage": 30.0,
        }

        good_reward = self.agent._calculate_performance_reward(good_performance)
        self.assertGreater(good_reward, 0.8)

        # Poor performance
        poor_performance = {
            "response_time": 15.0,
            "memory_usage": 1000.0,
            "cpu_usage": 90.0,
        }

        poor_reward = self.agent._calculate_performance_reward(poor_performance)
        self.assertLess(poor_reward, 0.3)

        # No performance metrics
        no_performance = None
        neutral_reward = self.agent._calculate_performance_reward(no_performance)
        self.assertEqual(neutral_reward, 0.5)

    def test_apply_penalties(self):
        """Test penalty application"""
        base_reward = 0.8

        # Test failure penalty
        test_failure = TestResult(
            passed=False,
            total_tests=10,
            passed_tests=5,
            failed_tests=5,
            coverage_percentage=50.0,
            execution_time=5.0,
            error_messages=[],
        )

        policy_pass = PolicyResult(
            safe=True,
            decision="pass",
            risk_level="low",
            violations_count=0,
            critical_violations=0,
            high_violations=0,
        )

        feedback_approve = HumanFeedback(
            overall_rating=0.8,
            usefulness_rating=0.8,
            correctness_rating=0.8,
            completeness_rating=0.8,
            comments="Good",
            approved=True,
        )

        penalized_reward = self.agent._apply_penalties(
            base_reward, test_failure, policy_pass, feedback_approve
        )

        self.assertLess(penalized_reward, base_reward)

    def test_generate_recommendations(self):
        """Test recommendation generation"""
        # All good scores
        good_scores = (0.9, 0.9, 0.9, 0.9, 0.9)
        good_recs = self.agent._generate_recommendations(*good_scores)
        self.assertIn("Excellent performance", good_recs[0])

        # Poor scores
        poor_scores = (0.3, 0.4, 0.5, 0.6, 0.7)
        poor_recs = self.agent._generate_recommendations(*poor_scores)
        self.assertGreater(len(poor_recs), 1)
        self.assertTrue(any("test coverage" in rec.lower() for rec in poor_recs))

    def test_determine_reward_level(self):
        """Test reward level determination"""
        self.assertEqual(self.agent._determine_reward_level(0.95), "excellent")
        self.assertEqual(self.agent._determine_reward_level(0.85), "good")
        self.assertEqual(self.agent._determine_reward_level(0.65), "acceptable")
        self.assertEqual(self.agent._determine_reward_level(0.45), "needs_improvement")
        self.assertEqual(self.agent._determine_reward_level(0.25), "poor")

    def test_calculate_reward_with_error(self):
        """Test reward calculation when an error occurs"""
        with patch.object(
            self.agent, "_calculate_test_reward", side_effect=Exception("Test error")
        ):
            result = self.agent.calculate_reward(
                self.sample_test_result,
                self.sample_code_quality,
                self.sample_human_feedback,
                self.sample_policy_result,
            )

            self.assertEqual(result["total_reward"], 0.0)
            self.assertEqual(result["reward_level"], "error")
            self.assertIn("error", result)

    def test_dataclass_creation(self):
        """Test dataclass creation"""
        # Test TestResult
        test_result = TestResult(
            passed=True,
            total_tests=5,
            passed_tests=4,
            failed_tests=1,
            coverage_percentage=80.0,
            execution_time=1.5,
            error_messages=["Minor issue"],
        )

        self.assertTrue(test_result.passed)
        self.assertEqual(test_result.total_tests, 5)
        self.assertEqual(test_result.passed_tests, 4)
        self.assertEqual(test_result.failed_tests, 1)
        self.assertEqual(test_result.coverage_percentage, 80.0)
        self.assertEqual(test_result.execution_time, 1.5)
        self.assertEqual(test_result.error_messages, ["Minor issue"])

        # Test CodeQualityMetrics
        quality = CodeQualityMetrics(
            quality_score=0.8,
            complexity_score=0.7,
            maintainability_score=0.8,
            documentation_score=0.9,
            style_score=0.8,
            security_score=0.9,
        )

        self.assertEqual(quality.quality_score, 0.8)
        self.assertEqual(quality.complexity_score, 0.7)
        self.assertEqual(quality.maintainability_score, 0.8)
        self.assertEqual(quality.documentation_score, 0.9)
        self.assertEqual(quality.style_score, 0.8)
        self.assertEqual(quality.security_score, 0.9)

        # Test HumanFeedback
        feedback = HumanFeedback(
            overall_rating=0.8,
            usefulness_rating=0.9,
            correctness_rating=0.8,
            completeness_rating=0.7,
            comments="Good work",
            approved=True,
        )

        self.assertEqual(feedback.overall_rating, 0.8)
        self.assertEqual(feedback.usefulness_rating, 0.9)
        self.assertEqual(feedback.correctness_rating, 0.8)
        self.assertEqual(feedback.completeness_rating, 0.7)
        self.assertEqual(feedback.comments, "Good work")
        self.assertTrue(feedback.approved)

        # Test PolicyResult
        policy = PolicyResult(
            safe=True,
            decision="pass",
            risk_level="low",
            violations_count=0,
            critical_violations=0,
            high_violations=0,
        )

        self.assertTrue(policy.safe)
        self.assertEqual(policy.decision, "pass")
        self.assertEqual(policy.risk_level, "low")
        self.assertEqual(policy.violations_count, 0)
        self.assertEqual(policy.critical_violations, 0)
        self.assertEqual(policy.high_violations, 0)


if __name__ == "__main__":
    unittest.main()
