"""
QLearningAgent - Tabular Q-learning implementation

This module implements a tabular Q-learning agent for prompt optimization.
The agent maintains a Q-table in SQLite and uses epsilon-greedy exploration
to learn optimal actions for different states.
"""

import json
import sqlite3
import logging
import random
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


@dataclass
class State:
    """Represents a state in the Q-learning environment."""

    prompt_type: str  # e.g., "code_generation", "analysis", "review"
    complexity_level: str  # "low", "medium", "high"
    previous_action: Optional[str] = None
    iteration_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "prompt_type": self.prompt_type,
            "complexity_level": self.complexity_level,
            "previous_action": self.previous_action,
            "iteration_count": self.iteration_count,
        }

    def to_string(self) -> str:
        """Convert state to string for hashing."""
        return json.dumps(self.to_dict(), sort_keys=True)


@dataclass
class Action:
    """Represents an action in the Q-learning environment."""

    action_type: str  # e.g., "add_examples", "clarify_requirements", "add_context"
    parameters: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert action to dictionary for serialization."""
        return {"action_type": self.action_type, "parameters": self.parameters}

    def to_string(self) -> str:
        """Convert action to string for hashing."""
        return json.dumps(self.to_dict(), sort_keys=True)


class QLearningAgent(BaseAgent):
    """
    Q-learning agent for prompt optimization.

    This agent learns optimal actions for different states by maintaining
    a Q-table and using reinforcement learning principles.
    """

    def __init__(
        self, name: str = "q_learning_agent", config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the Q-learning agent."""
        default_config = {
            "learning_rate": 0.1,  # Alpha - how much to update Q-values
            "discount_factor": 0.9,  # Gamma - future reward importance
            "epsilon": 0.1,  # Exploration rate
            "epsilon_decay": 0.995,  # How fast to reduce exploration
            "epsilon_min": 0.01,  # Minimum exploration rate
            "db_path": "data/q_learning.db",  # SQLite database path
            "actions": [  # Available actions
                "add_examples",
                "clarify_requirements",
                "add_context",
                "simplify_prompt",
                "add_constraints",
                "optimize_format",
            ],
        }

        if config:
            default_config.update(config)

        super().__init__(name, default_config)

        # Initialize database
        self._init_database()

        # Learning statistics
        self.episode_count = 0
        self.total_rewards = 0.0

        logger.info(f"Initialized QLearningAgent with config: {self.config}")

    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the context to understand the current state.

        Args:
            context: Dictionary containing current situation information

        Returns:
            Analysis of the current state and available actions
        """
        # Extract state information from context
        prompt_type = context.get("prompt_type", "unknown")
        complexity = context.get("complexity", "medium")
        previous_action = context.get("previous_action")
        iteration = context.get("iteration_count", 0)

        # Create state object
        state = State(
            prompt_type=prompt_type,
            complexity_level=complexity,
            previous_action=previous_action,
            iteration_count=iteration,
        )

        analysis = {
            "current_state": state.to_dict(),
            "available_actions": self.config["actions"],
            "q_values": self._get_q_values(state),
            "recommended_action": self._get_best_action(state),
            "exploration_rate": self.config["epsilon"],
        }

        return analysis

    def propose(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Propose an action based on current state and Q-learning policy.

        Args:
            analysis: Analysis results from analyze()

        Returns:
            Proposed action with confidence and reasoning
        """
        current_state = analysis.get("current_state", {})
        state = State(**current_state)

        # Select action using epsilon-greedy policy
        action_type = self._select_action(state)

        # Generate action parameters based on action type
        parameters = self._generate_action_parameters(action_type, state)

        action = Action(action_type=action_type, parameters=parameters)

        # Get Q-value for this state-action pair
        q_value = self._get_q_value(state, action)

        proposal = {
            "action": action.to_dict(),
            "confidence": self._calculate_confidence(q_value),
            "reasoning": self._generate_reasoning(state, action),
            "exploration": self._is_exploration(state, action),
            "state": state.to_dict(),
        }

        return proposal

    def review(self, code: str) -> Dict[str, Any]:
        """
        Review code and provide Q-learning insights.

        Args:
            code: Code string to review

        Returns:
            Review results with learning insights
        """
        # This is a placeholder - in practice, this would analyze the code
        # and provide insights about how it relates to the learning process

        return {
            "quality_score": 0.8,
            "issues": [],
            "recommendations": [
                "Consider using Q-learning insights to optimize prompt generation",
                "Monitor exploration vs exploitation balance",
                "Track reward progression for learning validation",
            ],
            "learning_insights": "Code quality affects reward calculation and learning",
        }

    def select_action(self, state: State, epsilon: Optional[float] = None) -> Action:
        """
        Select an action for the given state using epsilon-greedy policy.

        Args:
            state: Current state
            epsilon: Exploration rate (uses config if None)

        Returns:
            Selected action
        """
        if epsilon is None:
            epsilon = self.config["epsilon"]

        # Epsilon-greedy policy
        if random.random() < epsilon:
            # Exploration: random action
            action_type = random.choice(self.config["actions"])
        else:
            # Exploitation: best action
            action_type = self._get_best_action(state)

        parameters = self._generate_action_parameters(action_type, state)
        return Action(action_type=action_type, parameters=parameters)

    def update_q(self, state: State, action: Action, reward: float, next_state: State):
        """
        Update Q-value for state-action pair using Q-learning update rule.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state reached
        """
        # Get current Q-value
        current_q = self._get_q_value(state, action)

        # Get maximum Q-value for next state
        max_next_q = self._get_max_q_value(next_state)

        # Q-learning update rule: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        alpha = self.config["learning_rate"]
        gamma = self.config["discount_factor"]

        new_q = current_q + alpha * (reward + gamma * max_next_q - current_q)

        # Store updated Q-value
        self._set_q_value(state, action, new_q)

        # Update statistics
        self.total_rewards += reward
        self.episode_count += 1

        # Decay epsilon
        self._decay_epsilon()

        logger.debug(
            f"Updated Q({state.to_string()}, {action.to_string()}) = {new_q:.4f}"
        )

    def _init_database(self):
        """Initialize SQLite database for Q-table storage."""
        import os

        # Create data directory if it doesn't exist (skip for in-memory databases)
        if self.config["db_path"] != ":memory:":
            os.makedirs(os.path.dirname(self.config["db_path"]), exist_ok=True)

        # Use check_same_thread=False to avoid threading issues
        self.db_connection = sqlite3.connect(
            self.config["db_path"], check_same_thread=False
        )
        cursor = self.db_connection.cursor()

        # Create Q-table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS q_table (
                state_hash TEXT,
                action_hash TEXT,
                q_value REAL DEFAULT 0.0,
                visit_count INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (state_hash, action_hash)
            )
        """)

        # Create metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learning_metrics (
                episode_id INTEGER PRIMARY KEY AUTOINCREMENT,
                state_hash TEXT,
                action_hash TEXT,
                reward REAL,
                next_state_hash TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.db_connection.commit()

    def __del__(self):
        """Cleanup database connection on deletion."""
        if hasattr(self, "db_connection"):
            self.db_connection.close()

    def _get_q_value(self, state: State, action: Action) -> float:
        """Get Q-value for state-action pair."""
        cursor = self.db_connection.cursor()
        cursor.execute(
            "SELECT q_value FROM q_table WHERE state_hash = ? AND action_hash = ?",
            (state.to_string(), action.to_string()),
        )
        result = cursor.fetchone()
        return result[0] if result else 0.0

    def _set_q_value(self, state: State, action: Action, q_value: float):
        """Set Q-value for state-action pair."""
        cursor = self.db_connection.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO q_table (state_hash, action_hash, q_value, visit_count, last_updated)
            VALUES (?, ?, ?, 
                COALESCE((SELECT visit_count FROM q_table WHERE state_hash = ? AND action_hash = ?), 0) + 1,
                CURRENT_TIMESTAMP)
        """,
            (
                state.to_string(),
                action.to_string(),
                q_value,
                state.to_string(),
                action.to_string(),
            ),
        )
        self.db_connection.commit()

    def _get_q_values(self, state: State) -> Dict[str, float]:
        """Get all Q-values for a given state."""
        q_values = {}
        cursor = self.db_connection.cursor()
        cursor.execute(
            "SELECT action_hash, q_value FROM q_table WHERE state_hash = ?",
            (state.to_string(),),
        )
        for action_hash, q_value in cursor.fetchall():
            # Parse action hash back to action type
            try:
                action_dict = json.loads(action_hash)
                action_type = action_dict.get("action_type", "unknown")
                q_values[action_type] = q_value
            except json.JSONDecodeError:
                continue
        return q_values

    def _get_max_q_value(self, state: State) -> float:
        """Get maximum Q-value for a given state."""
        q_values = self._get_q_values(state)
        return max(q_values.values()) if q_values else 0.0

    def _get_best_action(self, state: State) -> str:
        """Get the best action for a given state."""
        q_values = self._get_q_values(state)
        if not q_values:
            return random.choice(self.config["actions"])

        return max(q_values.items(), key=lambda x: x[1])[0]

    def _select_action(self, state: State) -> str:
        """Select action using epsilon-greedy policy."""
        return self.select_action(state).action_type

    def _generate_action_parameters(
        self, action_type: str, state: State
    ) -> Dict[str, Any]:
        """Generate parameters for the given action type."""
        parameters = {}

        if action_type == "add_examples":
            parameters = {
                "example_count": min(3, state.iteration_count + 1),
                "example_type": "positive",
            }
        elif action_type == "clarify_requirements":
            parameters = {
                "detail_level": "high" if state.complexity_level == "high" else "medium"
            }
        elif action_type == "add_context":
            parameters = {
                "context_type": "technical"
                if state.complexity_level == "high"
                else "general"
            }
        elif action_type == "simplify_prompt":
            parameters = {"simplification_level": "moderate"}
        elif action_type == "add_constraints":
            parameters = {
                "constraint_type": "safety"
                if state.complexity_level == "high"
                else "basic"
            }
        elif action_type == "optimize_format":
            parameters = {"format_type": "structured"}

        return parameters

    def _calculate_confidence(self, q_value: float) -> float:
        """Calculate confidence based on Q-value."""
        # Normalize Q-value to confidence (0-1)
        # Assuming Q-values are typically in range [-1, 1]
        return max(0.0, min(1.0, (q_value + 1.0) / 2.0))

    def _generate_reasoning(self, state: State, action: Action) -> str:
        """Generate reasoning for the selected action."""
        q_values = self._get_q_values(state)
        best_action = self._get_best_action(state)

        if action.action_type == best_action:
            return f"Selected best action '{action.action_type}' based on Q-learning policy"
        else:
            return f"Explored action '{action.action_type}' (best was '{best_action}')"

    def _is_exploration(self, state: State, action: Action) -> bool:
        """Check if the action was selected through exploration."""
        best_action = self._get_best_action(state)
        return action.action_type != best_action

    def _decay_epsilon(self):
        """Decay the exploration rate."""
        self.config["epsilon"] = max(
            self.config["epsilon_min"],
            self.config["epsilon"] * self.config["epsilon_decay"],
        )

    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get learning statistics."""
        cursor = self.db_connection.cursor()

        # Get total Q-table entries
        cursor.execute("SELECT COUNT(*) FROM q_table")
        total_entries = cursor.fetchone()[0]

        # Get average Q-value
        cursor.execute("SELECT AVG(q_value) FROM q_table")
        avg_q_value = cursor.fetchone()[0] or 0.0

        # Get most visited state-action pairs
        cursor.execute("""
            SELECT state_hash, action_hash, visit_count 
            FROM q_table 
            ORDER BY visit_count DESC 
            LIMIT 5
        """)
        top_visited = cursor.fetchall()

        return {
            "episode_count": self.episode_count,
            "total_rewards": self.total_rewards,
            "average_reward": self.total_rewards / self.episode_count
            if self.episode_count > 0
            else 0.0,
            "epsilon": self.config["epsilon"],
            "total_q_entries": total_entries,
            "average_q_value": avg_q_value,
            "top_visited_pairs": top_visited,
            "learning_rate": self.config["learning_rate"],
            "discount_factor": self.config["discount_factor"],
        }

    def reset_learning(self):
        """Reset learning progress (clear Q-table)."""
        cursor = self.db_connection.cursor()
        cursor.execute("DELETE FROM q_table")
        cursor.execute("DELETE FROM learning_metrics")
        self.db_connection.commit()

        self.episode_count = 0
        self.total_rewards = 0.0
        self.config["epsilon"] = 0.1  # Reset exploration rate

        logger.info("Learning progress reset")

    def export_q_table(self, file_path: str):
        """Export Q-table to JSON file."""
        q_table = {}
        cursor = self.db_connection.cursor()
        cursor.execute(
            "SELECT state_hash, action_hash, q_value, visit_count FROM q_table"
        )

        for state_hash, action_hash, q_value, visit_count in cursor.fetchall():
            if state_hash not in q_table:
                q_table[state_hash] = {}
            q_table[state_hash][action_hash] = {
                "q_value": q_value,
                "visit_count": visit_count,
            }

        import json

        with open(file_path, "w") as f:
            json.dump(q_table, f, indent=2)

        logger.info(f"Q-table exported to {file_path}")

    def import_q_table(self, file_path: str):
        """Import Q-table from JSON file."""
        import json

        with open(file_path, "r") as f:
            q_table = json.load(f)

        cursor = self.db_connection.cursor()
        cursor.execute("DELETE FROM q_table")  # Clear existing table

        for state_hash, actions in q_table.items():
            for action_hash, data in actions.items():
                cursor.execute(
                    """
                    INSERT INTO q_table (state_hash, action_hash, q_value, visit_count)
                    VALUES (?, ?, ?, ?)
                """,
                    (state_hash, action_hash, data["q_value"], data["visit_count"]),
                )

        self.db_connection.commit()

        logger.info(f"Q-table imported from {file_path}")
