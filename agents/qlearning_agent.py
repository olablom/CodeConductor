"""
Q-Learning Agent for CodeConductor

This agent implements Q-learning for optimizing prompt generation and agent behavior.
It maintains a Q-table in SQLite and learns from rewards to improve future decisions.
"""

import logging
import sqlite3
import json
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from agents.base_agent import BaseAgent


@dataclass
class QState:
    """Represents a state in the Q-learning system."""

    task_type: str
    complexity: str
    language: str
    agent_count: int
    context_hash: str


@dataclass
class QAction:
    """Represents an action in the Q-learning system."""

    agent_combination: str
    prompt_strategy: str
    iteration_count: int
    confidence_threshold: float


class QLearningAgent(BaseAgent):
    """
    Q-Learning agent for optimizing multi-agent behavior.

    This agent focuses on:
    - Maintaining Q-table in SQLite
    - Learning from rewards
    - Selecting optimal actions
    - Epsilon-greedy exploration
    """

    def __init__(self, name: str = "qlearning_agent", config: Optional[Dict[str, Any]] = None):
        """Initialize the Q-learning agent."""
        super().__init__(name, config)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Q-learning configuration
        self.q_config = config.get("qlearning", {}) if config else {}

        # Learning parameters
        self.learning_rate = self.q_config.get("learning_rate", 0.1)
        self.discount_factor = self.q_config.get("discount_factor", 0.9)
        self.epsilon = self.q_config.get("epsilon", 0.1)
        self.epsilon_decay = self.q_config.get("epsilon_decay", 0.995)
        self.epsilon_min = self.q_config.get("epsilon_min", 0.01)

        # Database configuration
        self.db_path = self.q_config.get("db_path", "data/qtable.db")

        # Initialize database
        self._init_database()

        # Statistics
        self.total_episodes = 0
        self.successful_episodes = 0

        self.logger.info(
            f"QLearningAgent '{name}' initialized with learning_rate={self.learning_rate}, epsilon={self.epsilon}"
        )

    def _init_database(self):
        """Initialize SQLite database for Q-table."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create Q-table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS q_table (
                    state_hash TEXT PRIMARY KEY,
                    state_data TEXT NOT NULL,
                    action_data TEXT NOT NULL,
                    q_value REAL DEFAULT 0.0,
                    visit_count INTEGER DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Create statistics table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS learning_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    episode_count INTEGER,
                    average_reward REAL,
                    epsilon REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            conn.commit()
            conn.close()

            self.logger.info(f"Q-table database initialized at {self.db_path}")

        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise

    def get_state(self, context: Dict[str, Any]) -> QState:
        """
        Convert context to Q-learning state.

        Args:
            context: Context information

        Returns:
            QState representation
        """
        # Extract state components
        task_type = context.get("task_type", "unknown")
        complexity = context.get("complexity", "medium")
        language = context.get("language", "python")
        agent_count = context.get("agent_count", 3)

        # Create context hash for uniqueness
        context_str = json.dumps(context, sort_keys=True)
        context_hash = str(hash(context_str))[:16]  # Truncate for readability

        return QState(
            task_type=task_type,
            complexity=complexity,
            language=language,
            agent_count=agent_count,
            context_hash=context_hash,
        )

    def get_actions(self, state: QState) -> List[QAction]:
        """
        Get available actions for a given state.

        Args:
            state: Current state

        Returns:
            List of available actions
        """
        actions = []

        # Agent combinations
        agent_combinations = [
            "architect_review_codegen",
            "architect_codegen",
            "review_codegen",
            "all_agents",
        ]

        # Prompt strategies
        prompt_strategies = ["detailed", "concise", "step_by_step", "example_based"]

        # Iteration counts
        iteration_counts = [1, 2, 3]

        # Confidence thresholds
        confidence_thresholds = [0.5, 0.7, 0.8, 0.9]

        # Generate action combinations
        for agent_combo in agent_combinations:
            for strategy in prompt_strategies:
                for iterations in iteration_counts:
                    for threshold in confidence_thresholds:
                        actions.append(
                            QAction(
                                agent_combination=agent_combo,
                                prompt_strategy=strategy,
                                iteration_count=iterations,
                                confidence_threshold=threshold,
                            )
                        )

        return actions

    def state_action_to_key(self, state: QState, action: QAction) -> str:
        """Convert state-action pair to database key."""
        state_data = {
            "task_type": state.task_type,
            "complexity": state.complexity,
            "language": state.language,
            "agent_count": state.agent_count,
            "context_hash": state.context_hash,
        }

        action_data = {
            "agent_combination": action.agent_combination,
            "prompt_strategy": action.prompt_strategy,
            "iteration_count": action.iteration_count,
            "confidence_threshold": action.confidence_threshold,
        }

        return f"{json.dumps(state_data, sort_keys=True)}:{json.dumps(action_data, sort_keys=True)}"

    def get_q_value(self, state: QState, action: QAction) -> float:
        """
        Get Q-value for state-action pair.

        Args:
            state: Current state
            action: Action to evaluate

        Returns:
            Q-value for the state-action pair
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            key = self.state_action_to_key(state, action)

            cursor.execute("SELECT q_value FROM q_table WHERE state_hash = ?", (key,))

            result = cursor.fetchone()
            conn.close()

            return result[0] if result else 0.0

        except Exception as e:
            self.logger.error(f"Error getting Q-value: {e}")
            return 0.0

    def set_q_value(self, state: QState, action: QAction, q_value: float):
        """
        Set Q-value for state-action pair.

        Args:
            state: Current state
            action: Action
            q_value: Q-value to set
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            key = self.state_action_to_key(state, action)
            state_data = json.dumps(
                {
                    "task_type": state.task_type,
                    "complexity": state.complexity,
                    "language": state.language,
                    "agent_count": state.agent_count,
                    "context_hash": state.context_hash,
                }
            )
            action_data = json.dumps(
                {
                    "agent_combination": action.agent_combination,
                    "prompt_strategy": action.prompt_strategy,
                    "iteration_count": action.iteration_count,
                    "confidence_threshold": action.confidence_threshold,
                }
            )

            cursor.execute(
                """
                INSERT OR REPLACE INTO q_table 
                (state_hash, state_data, action_data, q_value, visit_count, last_updated)
                VALUES (?, ?, ?, ?, 
                    COALESCE((SELECT visit_count FROM q_table WHERE state_hash = ?), 0) + 1,
                    CURRENT_TIMESTAMP)
            """,
                (key, state_data, action_data, q_value, key),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            self.logger.error(f"Error setting Q-value: {e}")

    def select_action(self, state: QState, available_actions: Optional[List[QAction]] = None) -> QAction:
        """
        Select action using epsilon-greedy strategy.

        Args:
            state: Current state
            available_actions: Optional list of available actions

        Returns:
            Selected action
        """
        if available_actions is None:
            available_actions = self.get_actions(state)

        # Epsilon-greedy: explore with probability epsilon
        if random.random() < self.epsilon:
            # Exploration: choose random action
            selected_action = random.choice(available_actions)
            self.logger.debug(f"Exploration: selected random action {selected_action}")
        else:
            # Exploitation: choose action with highest Q-value
            best_action = None
            best_q_value = float("-inf")

            for action in available_actions:
                q_value = self.get_q_value(state, action)
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_action = action

            selected_action = best_action or random.choice(available_actions)
            self.logger.debug(f"Exploitation: selected best action {selected_action} with Q-value {best_q_value}")

        return selected_action

    def update_q_value(self, state: QState, action: QAction, reward: float, next_state: QState):
        """
        Update Q-value using Q-learning formula.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        try:
            # Get current Q-value
            current_q = self.get_q_value(state, action)

            # Get maximum Q-value for next state
            next_actions = self.get_actions(next_state)
            max_next_q = max([self.get_q_value(next_state, a) for a in next_actions], default=0.0)

            # Q-learning update formula
            new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)

            # Update Q-table
            self.set_q_value(state, action, new_q)

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            self.logger.debug(f"Updated Q-value: {current_q:.4f} -> {new_q:.4f} (reward: {reward:.4f})")

        except Exception as e:
            self.logger.error(f"Error updating Q-value: {e}")

    def get_best_action(self, state: QState) -> Tuple[QAction, float]:
        """
        Get the best action for a given state.

        Args:
            state: Current state

        Returns:
            Tuple of (best_action, best_q_value)
        """
        actions = self.get_actions(state)
        best_action = None
        best_q_value = float("-inf")

        for action in actions:
            q_value = self.get_q_value(state, action)
            if q_value > best_q_value:
                best_q_value = q_value
                best_action = action

        return best_action, best_q_value

    def get_learning_statistics(self) -> Dict[str, Any]:
        """
        Get learning statistics.

        Returns:
            Dictionary with learning statistics
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get Q-table statistics
            cursor.execute("SELECT COUNT(*) FROM q_table")
            total_entries = cursor.fetchone()[0]

            cursor.execute("SELECT AVG(q_value) FROM q_table")
            avg_q_value = cursor.fetchone()[0] or 0.0

            cursor.execute("SELECT MAX(q_value) FROM q_table")
            max_q_value = cursor.fetchone()[0] or 0.0

            cursor.execute("SELECT MIN(q_value) FROM q_table")
            min_q_value = cursor.fetchone()[0] or 0.0

            # Get recent learning stats
            cursor.execute(
                """
                SELECT episode_count, average_reward, epsilon 
                FROM learning_stats 
                ORDER BY timestamp DESC 
                LIMIT 1
            """
            )
            recent_stats = cursor.fetchone()

            conn.close()

            return {
                "total_entries": total_entries,
                "average_q_value": avg_q_value,
                "max_q_value": max_q_value,
                "min_q_value": min_q_value,
                "current_epsilon": self.epsilon,
                "learning_rate": self.learning_rate,
                "discount_factor": self.discount_factor,
                "recent_episode_count": recent_stats[0] if recent_stats else 0,
                "recent_average_reward": recent_stats[1] if recent_stats else 0.0,
                "recent_epsilon": recent_stats[2] if recent_stats else self.epsilon,
            }

        except Exception as e:
            self.logger.error(f"Error getting learning statistics: {e}")
            return {}

    def save_learning_statistics(self, episode_count: int, average_reward: float):
        """
        Save learning statistics to database.

        Args:
            episode_count: Number of episodes completed
            average_reward: Average reward for recent episodes
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO learning_stats (episode_count, average_reward, epsilon)
                VALUES (?, ?, ?)
            """,
                (episode_count, average_reward, self.epsilon),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            self.logger.error(f"Error saving learning statistics: {e}")

    def reset_q_table(self):
        """Reset the Q-table (clear all entries)."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("DELETE FROM q_table")
            cursor.execute("DELETE FROM learning_stats")

            conn.commit()
            conn.close()

            self.logger.info("Q-table reset successfully")

        except Exception as e:
            self.logger.error(f"Error resetting Q-table: {e}")

    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze context for Q-learning preparation.

        Args:
            context: Context information

        Returns:
            Analysis results for Q-learning
        """
        self.logger.info("Analyzing context for Q-learning")

        state = self.get_state(context)
        actions = self.get_actions(state)

        return {
            "state": {
                "task_type": state.task_type,
                "complexity": state.complexity,
                "language": state.language,
                "agent_count": state.agent_count,
            },
            "available_actions": len(actions),
            "current_epsilon": self.epsilon,
            "learning_ready": True,
        }

    def propose(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Propose Q-learning action.

        Args:
            analysis: Analysis results
            context: Context information

        Returns:
            Proposed action
        """
        self.logger.info("Proposing Q-learning action")

        state = self.get_state(context)
        action = self.select_action(state)

        return {
            "action": {
                "agent_combination": action.agent_combination,
                "prompt_strategy": action.prompt_strategy,
                "iteration_count": action.iteration_count,
                "confidence_threshold": action.confidence_threshold,
            },
            "exploration": random.random() < self.epsilon,
            "epsilon": self.epsilon,
        }

    def review(self, proposal: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Review Q-learning proposal.

        Args:
            proposal: Proposal to review
            context: Context information

        Returns:
            Review results
        """
        self.logger.info("Reviewing Q-learning proposal")

        return {
            "approved": True,
            "confidence": 0.8,
            "recommendations": ["Q-learning action looks reasonable"],
        }
