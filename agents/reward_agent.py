"""
RewardAgent - Reinforcement Learning Reward Calculation

Calculates rewards for the RL system based on:
- Test results from TestAgent
- Human feedback from HumanGate
- Code quality metrics
- Iteration efficiency
"""

import time
from typing import Dict, Any, List, Optional
from datetime import datetime


class RewardAgent:
    """Agent responsible for calculating RL rewards"""

    def __init__(self):
        self.reward_history = []
        self.episode_rewards = {}
        self.learning_curves = []

    def calculate_reward(
        self,
        test_results: Dict[str, Any],
        human_feedback: Optional[Dict[str, Any]] = None,
        code_quality: Optional[Dict[str, Any]] = None,
        iteration_count: int = 1,
        execution_time: float = 0.0,
        prompt_optimization: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive reward for RL system

        Args:
            test_results: Results from TestAgent
            human_feedback: Feedback from HumanGate
            code_quality: Quality metrics from TestAgent
            iteration_count: Number of iterations needed
            execution_time: Time taken for generation
            prompt_optimization: RL optimization details

        Returns:
            Dict with reward breakdown and total
        """

        reward_components = {
            "test_reward": self._calculate_test_reward(test_results),
            "quality_reward": self._calculate_quality_reward(code_quality),
            "human_reward": self._calculate_human_reward(human_feedback),
            "efficiency_reward": self._calculate_efficiency_reward(
                iteration_count, execution_time
            ),
            "optimization_reward": self._calculate_optimization_reward(
                prompt_optimization
            ),
        }

        # Calculate total reward
        total_reward = sum(reward_components.values())

        # Create reward record
        reward_record = {
            "timestamp": datetime.now().isoformat(),
            "total_reward": total_reward,
            "components": reward_components,
            "metadata": {
                "iteration_count": iteration_count,
                "execution_time": execution_time,
                "test_results": test_results,
                "human_feedback": human_feedback,
                "code_quality": code_quality,
            },
        }

        # Store reward history
        self.reward_history.append(reward_record)

        return reward_record

    def _calculate_test_reward(self, test_results: Dict[str, Any]) -> float:
        """Calculate reward based on test results"""
        if not test_results:
            return 0.0

        reward = 0.0

        # Base reward for tests passing
        tests_run = test_results.get("tests_run", 0)
        tests_passed = test_results.get("tests_passed", 0)

        if tests_run > 0:
            pass_rate = tests_passed / tests_run
            reward += pass_rate * 20.0  # Up to 20 points for test success

        # Penalty for errors
        errors = test_results.get("errors", [])
        reward -= len(errors) * 5.0  # -5 points per error

        # Penalty for warnings
        warnings = test_results.get("warnings", [])
        reward -= len(warnings) * 1.0  # -1 point per warning

        # Bonus for high coverage
        coverage = test_results.get("coverage", 0.0)
        if coverage > 80:
            reward += 10.0  # Bonus for high coverage
        elif coverage > 60:
            reward += 5.0  # Bonus for medium coverage

        return max(-50.0, min(50.0, reward))  # Clamp between -50 and 50

    def _calculate_quality_reward(
        self, code_quality: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate reward based on code quality metrics"""
        if not code_quality:
            return 0.0

        reward = 0.0

        # Overall quality score (0-10)
        overall_score = code_quality.get("overall_score", 0.0)
        reward += overall_score * 3.0  # Up to 30 points for quality

        # Syntax validity
        if code_quality.get("syntax_valid", False):
            reward += 10.0  # Bonus for valid syntax
        else:
            reward -= 20.0  # Heavy penalty for syntax errors

        # Complexity penalty
        complexity = code_quality.get("complexity_score", 0.0)
        if complexity > 7:
            reward -= 10.0  # Penalty for high complexity
        elif complexity < 3:
            reward += 5.0  # Bonus for low complexity

        # Security issues penalty
        security_issues = code_quality.get("security_issues", [])
        reward -= len(security_issues) * 8.0  # Heavy penalty for security issues

        # Best practices
        best_practices = code_quality.get("best_practices", [])
        reward -= (
            len(best_practices) * 2.0
        )  # Small penalty for best practice violations

        return max(-50.0, min(50.0, reward))

    def _calculate_human_reward(
        self, human_feedback: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate reward based on human feedback"""
        if not human_feedback:
            return 0.0

        reward = 0.0

        # Approval status
        approved = human_feedback.get("approved", False)
        if approved:
            reward += 30.0  # High reward for approval
        else:
            reward -= 20.0  # Penalty for rejection

        # Feedback reason
        reason = human_feedback.get("reason", "")
        if "human_approved" in reason:
            reward += 10.0  # Extra bonus for direct approval
        elif "human_edited" in reason:
            reward += 5.0  # Small bonus for edits (shows engagement)
        elif "human_rejected" in reason:
            reward -= 10.0  # Extra penalty for rejection

        # NEW: Thumbs up/down feedback bonus
        feedback_score = human_feedback.get("feedback_score", 0)
        if feedback_score > 0:
            reward += 15.0  # +15 bonus for positive feedback
        elif feedback_score < 0:
            reward -= 10.0  # -10 penalty for negative feedback

        # Feedback quality
        feedback_text = human_feedback.get("feedback", "")
        if feedback_text and len(feedback_text) > 10:
            reward += 2.0  # Small bonus for detailed feedback

        # Comment quality bonus
        comment = human_feedback.get("comment", "")
        if comment and len(comment) > 20:
            reward += 3.0  # Bonus for detailed comments

        return max(-50.0, min(50.0, reward))

    def _calculate_efficiency_reward(
        self, iteration_count: int, execution_time: float
    ) -> float:
        """Calculate reward based on efficiency metrics"""
        reward = 0.0

        # Iteration efficiency
        if iteration_count == 1:
            reward += 15.0  # High reward for first-try success
        elif iteration_count <= 3:
            reward += 10.0  # Good reward for few iterations
        elif iteration_count <= 5:
            reward += 5.0  # Moderate reward
        else:
            reward -= (iteration_count - 5) * 2.0  # Penalty for many iterations

        # Execution time efficiency
        if execution_time < 30:  # Less than 30 seconds
            reward += 5.0
        elif execution_time < 60:  # Less than 1 minute
            reward += 2.0
        elif execution_time > 300:  # More than 5 minutes
            reward -= 5.0

        return max(-30.0, min(30.0, reward))

    def _calculate_optimization_reward(
        self, prompt_optimization: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate reward based on RL optimization effectiveness"""
        if not prompt_optimization:
            return 0.0

        reward = 0.0

        # Optimization type effectiveness
        optimization_type = prompt_optimization.get("type", "")
        if optimization_type == "add_examples":
            reward += 3.0
        elif optimization_type == "clarify_requirements":
            reward += 2.0
        elif optimization_type == "add_context":
            reward += 1.0

        # Optimization confidence
        confidence = prompt_optimization.get("confidence", 0.0)
        reward += confidence * 5.0  # Up to 5 points for high confidence

        # Previous performance improvement
        improvement = prompt_optimization.get("improvement", 0.0)
        reward += improvement * 10.0  # Up to 10 points for improvement

        return max(-20.0, min(20.0, reward))

    def get_episode_reward(self, episode_id: str) -> Optional[Dict[str, Any]]:
        """Get reward for a specific episode"""
        return self.episode_rewards.get(episode_id)

    def get_learning_curve(self, window_size: int = 10) -> List[Dict[str, Any]]:
        """Get learning curve data for visualization"""
        if len(self.reward_history) < window_size:
            return []

        learning_curve = []
        for i in range(window_size, len(self.reward_history)):
            window_rewards = [
                r["total_reward"] for r in self.reward_history[i - window_size : i]
            ]
            avg_reward = sum(window_rewards) / len(window_rewards)

            learning_curve.append(
                {
                    "episode": i,
                    "average_reward": avg_reward,
                    "window_size": window_size,
                    "timestamp": self.reward_history[i]["timestamp"],
                }
            )

        return learning_curve

    def get_reward_statistics(self) -> Dict[str, Any]:
        """Get comprehensive reward statistics"""
        if not self.reward_history:
            return {
                "total_episodes": 0,
                "average_reward": 0.0,
                "best_reward": 0.0,
                "worst_reward": 0.0,
                "reward_trend": "stable",
            }

        rewards = [r["total_reward"] for r in self.reward_history]

        # Calculate trend
        if len(rewards) >= 10:
            recent_avg = sum(rewards[-10:]) / 10
            earlier_avg = (
                sum(rewards[-20:-10]) / 10 if len(rewards) >= 20 else rewards[0]
            )

            if recent_avg > earlier_avg + 5:
                trend = "improving"
            elif recent_avg < earlier_avg - 5:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        return {
            "total_episodes": len(self.reward_history),
            "average_reward": sum(rewards) / len(rewards),
            "best_reward": max(rewards),
            "worst_reward": min(rewards),
            "reward_trend": trend,
            "recent_performance": rewards[-5:] if len(rewards) >= 5 else rewards,
        }

    def get_component_analysis(self) -> Dict[str, Any]:
        """Analyze which reward components are most important"""
        if not self.reward_history:
            return {}

        component_totals = {
            "test_reward": 0.0,
            "quality_reward": 0.0,
            "human_reward": 0.0,
            "efficiency_reward": 0.0,
            "optimization_reward": 0.0,
        }

        for record in self.reward_history:
            for component, value in record["components"].items():
                component_totals[component] += value

        # Calculate averages
        episode_count = len(self.reward_history)
        component_averages = {
            component: total / episode_count
            for component, total in component_totals.items()
        }

        # Find most important component
        most_important = max(component_averages.items(), key=lambda x: abs(x[1]))

        return {
            "component_averages": component_averages,
            "most_important_component": most_important[0],
            "most_important_value": most_important[1],
            "total_episodes": episode_count,
        }

    def reset_episode(self, episode_id: str):
        """Reset episode-specific data"""
        if episode_id in self.episode_rewards:
            del self.episode_rewards[episode_id]

    def export_reward_data(self, file_path: str):
        """Export reward history to JSON file"""
        import json

        export_data = {
            "reward_history": self.reward_history,
            "statistics": self.get_reward_statistics(),
            "component_analysis": self.get_component_analysis(),
            "export_timestamp": datetime.now().isoformat(),
        }

        with open(file_path, "w") as f:
            json.dump(export_data, f, indent=2)

    def import_reward_data(self, file_path: str):
        """Import reward history from JSON file"""
        import json

        with open(file_path, "r") as f:
            import_data = json.load(f)

        self.reward_history = import_data.get("reward_history", [])
        # Note: episode_rewards and learning_curves would need to be reconstructed
        # from the reward_history if needed
