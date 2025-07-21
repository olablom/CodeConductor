"""
Unit tests for QLearningAgent Q-learning functionality
"""

import unittest
from unittest.mock import patch
import sys
import os
import tempfile
import sqlite3

# Add the parent directory to the path to import the agents module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.qlearning_agent import QLearningAgent, QState, QAction


class TestQLearningAgent(unittest.TestCase):
    """Test cases for QLearningAgent Q-learning functionality"""

    def setUp(self):
        """Set up test fixtures"""
        # Create temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db.close()

        config = {
            "qlearning": {
                "db_path": self.temp_db.name,
                "learning_rate": 0.1,
                "discount_factor": 0.9,
                "epsilon": 0.1,
                "epsilon_decay": 0.995,
                "epsilon_min": 0.01,
            }
        }

        self.agent = QLearningAgent("TestQLearningAgent", config)

        # Sample context
        self.sample_context = {
            "task_type": "api",
            "complexity": "medium",
            "language": "python",
            "agent_count": 3,
            "requirements": "Create a REST API",
        }

    def tearDown(self):
        """Clean up test fixtures"""
        # Remove temporary database
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)

    def test_init(self):
        """Test QLearningAgent initialization"""
        agent = QLearningAgent("TestAgent")
        self.assertEqual(agent.name, "TestAgent")
        self.assertEqual(agent.learning_rate, 0.1)
        self.assertEqual(agent.discount_factor, 0.9)
        self.assertEqual(agent.epsilon, 0.1)
        self.assertEqual(agent.epsilon_decay, 0.995)
        self.assertEqual(agent.epsilon_min, 0.01)

    def test_init_database(self):
        """Test database initialization"""
        # Check that tables were created
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()

        # Check q_table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='q_table'")
        self.assertIsNotNone(cursor.fetchone())

        # Check learning_stats exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='learning_stats'")
        self.assertIsNotNone(cursor.fetchone())

        conn.close()

    def test_get_state(self):
        """Test state creation from context"""
        state = self.agent.get_state(self.sample_context)

        self.assertIsInstance(state, QState)
        self.assertEqual(state.task_type, "api")
        self.assertEqual(state.complexity, "medium")
        self.assertEqual(state.language, "python")
        self.assertEqual(state.agent_count, 3)
        self.assertIsInstance(state.context_hash, str)
        self.assertEqual(len(state.context_hash), 16)

    def test_get_actions(self):
        """Test action generation"""
        state = self.agent.get_state(self.sample_context)
        actions = self.agent.get_actions(state)

        self.assertIsInstance(actions, list)
        self.assertGreater(len(actions), 0)

        # Check that all actions are QAction instances
        for action in actions:
            self.assertIsInstance(action, QAction)
            self.assertIn(
                action.agent_combination,
                [
                    "architect_review_codegen",
                    "architect_codegen",
                    "review_codegen",
                    "all_agents",
                ],
            )
            self.assertIn(
                action.prompt_strategy,
                ["detailed", "concise", "step_by_step", "example_based"],
            )
            self.assertIn(action.iteration_count, [1, 2, 3])
            self.assertIn(action.confidence_threshold, [0.5, 0.7, 0.8, 0.9])

    def test_state_action_to_key(self):
        """Test state-action key generation"""
        state = self.agent.get_state(self.sample_context)
        action = self.agent.get_actions(state)[0]

        key = self.agent.state_action_to_key(state, action)

        self.assertIsInstance(key, str)
        self.assertIn(state.task_type, key)
        self.assertIn(action.agent_combination, key)

    def test_get_q_value(self):
        """Test Q-value retrieval"""
        state = self.agent.get_state(self.sample_context)
        action = self.agent.get_actions(state)[0]

        # Initially should be 0.0
        q_value = self.agent.get_q_value(state, action)
        self.assertEqual(q_value, 0.0)

    def test_set_q_value(self):
        """Test Q-value setting"""
        state = self.agent.get_state(self.sample_context)
        action = self.agent.get_actions(state)[0]

        # Set Q-value
        self.agent.set_q_value(state, action, 0.5)

        # Retrieve Q-value
        q_value = self.agent.get_q_value(state, action)
        self.assertEqual(q_value, 0.5)

    def test_select_action_exploration(self):
        """Test action selection with exploration"""
        state = self.agent.get_state(self.sample_context)

        # Force exploration by setting high epsilon
        self.agent.epsilon = 1.0

        actions = self.agent.get_actions(state)
        selected_action = self.agent.select_action(state, actions)

        self.assertIn(selected_action, actions)

    def test_select_action_exploitation(self):
        """Test action selection with exploitation"""
        state = self.agent.get_state(self.sample_context)
        actions = self.agent.get_actions(state)

        # Set Q-values for actions
        self.agent.set_q_value(state, actions[0], 0.1)
        self.agent.set_q_value(state, actions[1], 0.9)  # Best action
        self.agent.set_q_value(state, actions[2], 0.3)

        # Force exploitation by setting low epsilon
        self.agent.epsilon = 0.0

        selected_action = self.agent.select_action(state, actions)

        # Should select action with highest Q-value
        self.assertEqual(selected_action, actions[1])

    def test_update_q_value(self):
        """Test Q-value update"""
        state = self.agent.get_state(self.sample_context)
        next_state = self.agent.get_state({**self.sample_context, "iteration": 2})
        action = self.agent.get_actions(state)[0]

        # Set initial Q-value
        self.agent.set_q_value(state, action, 0.1)

        # Update Q-value
        reward = 0.8
        self.agent.update_q_value(state, action, reward, next_state)

        # Check that Q-value was updated
        new_q_value = self.agent.get_q_value(state, action)
        self.assertGreater(new_q_value, 0.1)

        # Check that epsilon was decayed
        self.assertLess(self.agent.epsilon, 0.1)

    def test_get_best_action(self):
        """Test best action retrieval"""
        state = self.agent.get_state(self.sample_context)
        actions = self.agent.get_actions(state)

        # Set Q-values
        self.agent.set_q_value(state, actions[0], 0.1)
        self.agent.set_q_value(state, actions[1], 0.9)  # Best
        self.agent.set_q_value(state, actions[2], 0.3)

        best_action, best_q_value = self.agent.get_best_action(state)

        self.assertEqual(best_action, actions[1])
        self.assertEqual(best_q_value, 0.9)

    def test_get_learning_statistics(self):
        """Test learning statistics retrieval"""
        stats = self.agent.get_learning_statistics()

        self.assertIn("total_entries", stats)
        self.assertIn("average_q_value", stats)
        self.assertIn("max_q_value", stats)
        self.assertIn("min_q_value", stats)
        self.assertIn("current_epsilon", stats)
        self.assertIn("learning_rate", stats)
        self.assertIn("discount_factor", stats)

        # Initially should be empty
        self.assertEqual(stats["total_entries"], 0)

    def test_save_learning_statistics(self):
        """Test learning statistics saving"""
        episode_count = 10
        average_reward = 0.75

        self.agent.save_learning_statistics(episode_count, average_reward)

        # Check that statistics were saved
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()

        cursor.execute("SELECT episode_count, average_reward FROM learning_stats ORDER BY id DESC LIMIT 1")
        result = cursor.fetchone()

        self.assertIsNotNone(result)
        self.assertEqual(result[0], episode_count)
        self.assertEqual(result[1], average_reward)

        conn.close()

    def test_reset_q_table(self):
        """Test Q-table reset"""
        state = self.agent.get_state(self.sample_context)
        action = self.agent.get_actions(state)[0]

        # Add some data
        self.agent.set_q_value(state, action, 0.5)
        self.agent.save_learning_statistics(5, 0.6)

        # Reset
        self.agent.reset_q_table()

        # Check that data was cleared
        q_value = self.agent.get_q_value(state, action)
        self.assertEqual(q_value, 0.0)

        stats = self.agent.get_learning_statistics()
        self.assertEqual(stats["total_entries"], 0)

    def test_epsilon_decay(self):
        """Test epsilon decay over multiple updates"""
        initial_epsilon = self.agent.epsilon

        state = self.agent.get_state(self.sample_context)
        next_state = self.agent.get_state({**self.sample_context, "iteration": 2})
        action = self.agent.get_actions(state)[0]

        # Perform multiple updates
        for _ in range(10):
            self.agent.update_q_value(state, action, 0.5, next_state)

        # Epsilon should have decayed
        self.assertLess(self.agent.epsilon, initial_epsilon)

        # But not below minimum
        self.assertGreaterEqual(self.agent.epsilon, self.agent.epsilon_min)

    def test_q_learning_formula(self):
        """Test Q-learning update formula"""
        state = self.agent.get_state(self.sample_context)
        next_state = self.agent.get_state({**self.sample_context, "iteration": 2})
        action = self.agent.get_actions(state)[0]

        # Set initial Q-value
        initial_q = 0.1
        self.agent.set_q_value(state, action, initial_q)

        # Set Q-values for next state
        next_actions = self.agent.get_actions(next_state)
        self.agent.set_q_value(next_state, next_actions[0], 0.3)
        self.agent.set_q_value(next_state, next_actions[1], 0.7)  # Max

        # Update with reward
        reward = 0.8
        self.agent.update_q_value(state, action, reward, next_state)

        # Check Q-learning formula: Q(s,a) = Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        expected_q = initial_q + self.agent.learning_rate * (reward + self.agent.discount_factor * 0.7 - initial_q)

        actual_q = self.agent.get_q_value(state, action)
        self.assertAlmostEqual(actual_q, expected_q, places=4)

    def test_dataclass_creation(self):
        """Test dataclass creation"""
        # Test QState
        state = QState(
            task_type="api",
            complexity="medium",
            language="python",
            agent_count=3,
            context_hash="abc123def456",
        )

        self.assertEqual(state.task_type, "api")
        self.assertEqual(state.complexity, "medium")
        self.assertEqual(state.language, "python")
        self.assertEqual(state.agent_count, 3)
        self.assertEqual(state.context_hash, "abc123def456")

        # Test QAction
        action = QAction(
            agent_combination="all_agents",
            prompt_strategy="detailed",
            iteration_count=2,
            confidence_threshold=0.8,
        )

        self.assertEqual(action.agent_combination, "all_agents")
        self.assertEqual(action.prompt_strategy, "detailed")
        self.assertEqual(action.iteration_count, 2)
        self.assertEqual(action.confidence_threshold, 0.8)

    def test_analyze_method(self):
        """Test analyze method"""
        result = self.agent.analyze(self.sample_context)

        self.assertIn("state", result)
        self.assertIn("available_actions", result)
        self.assertIn("current_epsilon", result)
        self.assertIn("learning_ready", result)
        self.assertTrue(result["learning_ready"])

    def test_propose_method(self):
        """Test propose method"""
        analysis = {"learning_ready": True}
        result = self.agent.propose(analysis, self.sample_context)

        self.assertIn("action", result)
        self.assertIn("exploration", result)
        self.assertIn("epsilon", result)
        self.assertIn("agent_combination", result["action"])
        self.assertIn("prompt_strategy", result["action"])

    def test_review_method(self):
        """Test review method"""
        proposal = {"action": {"agent_combination": "all_agents"}}
        result = self.agent.review(proposal, self.sample_context)

        self.assertIn("approved", result)
        self.assertIn("confidence", result)
        self.assertIn("recommendations", result)
        self.assertTrue(result["approved"])

    def test_error_handling(self):
        """Test error handling in database operations"""
        # Test with invalid database path
        with patch.object(self.agent, "db_path", "/invalid/path/db.sqlite"):
            q_value = self.agent.get_q_value(
                self.agent.get_state(self.sample_context),
                self.agent.get_actions(self.agent.get_state(self.sample_context))[0],
            )
            self.assertEqual(q_value, 0.0)


if __name__ == "__main__":
    unittest.main()
