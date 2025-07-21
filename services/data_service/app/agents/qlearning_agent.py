"""
Data Service - Q-Learning Agent

This module contains the Q-learning agent migrated from the main CodeConductor
codebase to the microservices architecture.
"""

import logging
import json
import random
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class QState:
    """Represents a state in the Q-learning system."""

    task_type: str
    complexity: str
    language: str
    agent_count: int
    context_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "task_type": self.task_type,
            "complexity": self.complexity,
            "language": self.language,
            "agent_count": self.agent_count,
            "context_hash": self.context_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QState":
        """Create state from dictionary."""
        return cls(**data)


@dataclass
class QAction:
    """Represents an action in the Q-learning system."""

    agent_combination: str
    prompt_strategy: str
    iteration_count: int
    confidence_threshold: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert action to dictionary."""
        return {
            "agent_combination": self.agent_combination,
            "prompt_strategy": self.prompt_strategy,
            "iteration_count": self.iteration_count,
            "confidence_threshold": self.confidence_threshold,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QAction":
        """Create action from dictionary."""
        return cls(**data)


class QLearningAgent:
    """
    Q-Learning agent for optimizing multi-agent behavior.

    This agent focuses on:
    - Maintaining Q-table in memory (with persistence capability)
    - Learning from rewards
    - Selecting optimal actions
    - Epsilon-greedy exploration
    """

    def __init__(
        self, name: str = "qlearning_agent", config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the Q-learning agent."""
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Q-learning configuration
        self.q_config = config.get("qlearning", {}) if config else {}

        # Learning parameters
        self.learning_rate = self.q_config.get("learning_rate", 0.1)
        self.discount_factor = self.q_config.get("discount_factor", 0.9)
        self.epsilon = self.q_config.get("epsilon", 0.1)
        self.epsilon_decay = self.q_config.get("epsilon_decay", 0.995)
        self.epsilon_min = self.q_config.get("epsilon_min", 0.01)

        # Q-table storage (in-memory for microservices)
        self.q_table: Dict[str, float] = {}
        self.visit_counts: Dict[str, int] = {}

        # Statistics
        self.total_episodes = 0
        self.successful_episodes = 0
        self.learning_history: List[Dict[str, Any]] = []

        self.logger.info(
            f"QLearningAgent '{name}' initialized with learning_rate={self.learning_rate}, epsilon={self.epsilon}"
        )

    def get_state(self, context: Dict[str, Any]) -> QState:
        """
        Extract state from context.

        Args:
            context: Context dictionary

        Returns:
            QState object
        """
        # Extract task information
        task_type = context.get("task_type", "unknown")
        complexity = context.get("complexity", "medium")
        language = context.get("language", "python")
        agent_count = context.get("agent_count", 1)

        # Create context hash
        context_str = json.dumps(context, sort_keys=True)
        context_hash = hashlib.md5(context_str.encode()).hexdigest()[:8]

        return QState(
            task_type=task_type,
            complexity=complexity,
            language=language,
            agent_count=agent_count,
            context_hash=context_hash,
        )

    def get_actions(self, state: QState) -> List[QAction]:
        """
        Generate available actions for a state.

        Args:
            state: Current state

        Returns:
            List of available actions
        """
        actions = []

        # Agent combinations
        agent_combinations = [
            "codegen_only",
            "codegen_review",
            "codegen_architect",
            "all_agents",
        ]

        # Prompt strategies
        prompt_strategies = ["standard", "detailed", "simple", "optimized"]

        # Iteration counts
        iteration_counts = [1, 2, 3]

        # Confidence thresholds
        confidence_thresholds = [0.5, 0.7, 0.8, 0.9]

        # Generate action combinations
        for agent_combo in agent_combinations:
            for prompt_strat in prompt_strategies:
                for iterations in iteration_counts:
                    for confidence in confidence_thresholds:
                        actions.append(
                            QAction(
                                agent_combination=agent_combo,
                                prompt_strategy=prompt_strat,
                                iteration_count=iterations,
                                confidence_threshold=confidence,
                            )
                        )

        return actions

    def state_action_to_key(self, state: QState, action: QAction) -> str:
        """
        Convert state-action pair to Q-table key.

        Args:
            state: Current state
            action: Selected action

        Returns:
            Q-table key string
        """
        state_dict = state.to_dict()
        action_dict = action.to_dict()

        combined = {**state_dict, **action_dict}
        return hashlib.md5(json.dumps(combined, sort_keys=True).encode()).hexdigest()

    def get_q_value(self, state: QState, action: QAction) -> float:
        """
        Get Q-value for state-action pair.

        Args:
            state: Current state
            action: Selected action

        Returns:
            Q-value
        """
        key = self.state_action_to_key(state, action)
        return self.q_table.get(key, 0.0)

    def set_q_value(self, state: QState, action: QAction, q_value: float):
        """
        Set Q-value for state-action pair.

        Args:
            state: Current state
            action: Selected action
            q_value: Q-value to set
        """
        key = self.state_action_to_key(state, action)
        self.q_table[key] = q_value
        self.visit_counts[key] = self.visit_counts.get(key, 0) + 1

    def select_action(
        self, state: QState, available_actions: Optional[List[QAction]] = None
    ) -> QAction:
        """
        Select action using epsilon-greedy strategy.

        Args:
            state: Current state
            available_actions: List of available actions (if None, generates all)

        Returns:
            Selected action
        """
        if available_actions is None:
            available_actions = self.get_actions(state)

        if not available_actions:
            raise ValueError("No actions available")

        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            # Exploration: random action
            selected_action = random.choice(available_actions)
            self.logger.info(
                f"Exploration: selected random action {selected_action.agent_combination}"
            )
        else:
            # Exploitation: best action
            best_action = None
            best_q_value = float("-inf")

            for action in available_actions:
                q_value = self.get_q_value(state, action)
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_action = action

            selected_action = best_action or random.choice(available_actions)
            self.logger.info(
                f"Exploitation: selected best action {selected_action.agent_combination} with Q-value {best_q_value:.4f}"
            )

        return selected_action

    def update_q_value(
        self, state: QState, action: QAction, reward: float, next_state: QState
    ):
        """
        Update Q-value using Q-learning update rule.

        Args:
            state: Current state
            action: Selected action
            reward: Observed reward
            next_state: Next state
        """
        current_q = self.get_q_value(state, action)

        # Get max Q-value for next state
        next_actions = self.get_actions(next_state)
        max_next_q = (
            max([self.get_q_value(next_state, a) for a in next_actions])
            if next_actions
            else 0.0
        )

        # Q-learning update rule
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )

        self.set_q_value(state, action, new_q)

        self.logger.info(
            f"Updated Q-value: {current_q:.4f} -> {new_q:.4f} (reward: {reward:.4f})"
        )

    def get_best_action(self, state: QState) -> Tuple[QAction, float]:
        """
        Get the best action for a state.

        Args:
            state: Current state

        Returns:
            Tuple of (best_action, q_value)
        """
        available_actions = self.get_actions(state)

        if not available_actions:
            raise ValueError("No actions available")

        best_action = None
        best_q_value = float("-inf")

        for action in available_actions:
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
        # Calculate average Q-value
        q_values = list(self.q_table.values())
        avg_q_value = sum(q_values) / len(q_values) if q_values else 0.0

        # Calculate visit distribution
        total_visits = sum(self.visit_counts.values())
        visit_distribution = {}
        for key, visits in self.visit_counts.items():
            visit_distribution[key] = visits / total_visits if total_visits > 0 else 0.0

        return {
            "total_episodes": self.total_episodes,
            "successful_episodes": self.successful_episodes,
            "success_rate": self.successful_episodes / self.total_episodes
            if self.total_episodes > 0
            else 0.0,
            "epsilon": self.epsilon,
            "q_table_size": len(self.q_table),
            "average_q_value": avg_q_value,
            "min_q_value": min(q_values) if q_values else 0.0,
            "max_q_value": max(q_values) if q_values else 0.0,
            "total_visits": total_visits,
            "visit_distribution": visit_distribution,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
        }

    def save_learning_statistics(self, episode_count: int, average_reward: float):
        """
        Save learning statistics for tracking.

        Args:
            episode_count: Number of episodes
            average_reward: Average reward for the episode
        """
        stats = {
            "episode": episode_count,
            "average_reward": average_reward,
            "epsilon": self.epsilon,
            "q_table_size": len(self.q_table),
            "timestamp": datetime.now().isoformat(),
        }

        self.learning_history.append(stats)

        # Keep only last 1000 entries
        if len(self.learning_history) > 1000:
            self.learning_history = self.learning_history[-1000:]

    def reset_q_table(self):
        """Reset Q-table to initial state."""
        self.q_table.clear()
        self.visit_counts.clear()
        self.total_episodes = 0
        self.successful_episodes = 0
        self.learning_history.clear()
        self.epsilon = self.q_config.get("epsilon", 0.1)  # Reset epsilon
        self.logger.info("Q-table reset to initial state")

    def decay_epsilon(self):
        """Decay epsilon for exploration."""
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        self.logger.info(f"Epsilon decayed to {self.epsilon:.4f}")

    def get_model_state(self) -> Dict[str, Any]:
        """
        Get current model state for persistence.

        Returns:
            Dictionary with model state
        """
        return {
            "name": self.name,
            "q_table": self.q_table,
            "visit_counts": self.visit_counts,
            "total_episodes": self.total_episodes,
            "successful_episodes": self.successful_episodes,
            "epsilon": self.epsilon,
            "learning_history": self.learning_history,
            "config": self.q_config,
            "timestamp": datetime.now().isoformat(),
        }

    def load_model_state(self, state: Dict[str, Any]):
        """
        Load model state from dictionary.

        Args:
            state: Model state dictionary
        """
        try:
            self.name = state.get("name", self.name)
            self.q_table = state.get("q_table", {})
            self.visit_counts = state.get("visit_counts", {})
            self.total_episodes = state.get("total_episodes", 0)
            self.successful_episodes = state.get("successful_episodes", 0)
            self.epsilon = state.get("epsilon", self.epsilon)
            self.learning_history = state.get("learning_history", [])
            self.q_config.update(state.get("config", {}))

            self.logger.info(f"Loaded model state with {len(self.q_table)} Q-values")

        except Exception as e:
            self.logger.error(f"Error loading model state: {e}")

    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze context for Q-learning.

        Args:
            context: Context to analyze

        Returns:
            Analysis results
        """
        state = self.get_state(context)
        available_actions = self.get_actions(state)

        return {
            "agent_name": self.name,
            "state": state.to_dict(),
            "available_actions": len(available_actions),
            "q_table_size": len(self.q_table),
            "epsilon": self.epsilon,
            "statistics": self.get_learning_statistics(),
            "timestamp": datetime.now().isoformat(),
        }

    def propose(
        self, analysis: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Propose action based on Q-learning.

        Args:
            analysis: Previous analysis results
            context: Original context

        Returns:
            Action proposal
        """
        state = self.get_state(context)
        selected_action = self.select_action(state)

        return {
            "agent_name": self.name,
            "selected_action": selected_action.to_dict(),
            "q_value": self.get_q_value(state, selected_action),
            "epsilon": self.epsilon,
            "exploration": self.epsilon > random.random(),
            "confidence": 1.0 - self.epsilon,
            "reasoning": f"Q-learning selection with epsilon={self.epsilon:.4f}",
        }

    def review(
        self, proposal: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Review proposal and update Q-values.

        Args:
            proposal: Action proposal
            context: Original context

        Returns:
            Review results
        """
        # This would typically be called after observing the reward
        # For now, return the proposal with additional metadata

        return {
            "agent_name": self.name,
            "final_action": proposal.get("selected_action"),
            "q_value": proposal.get("q_value", 0.0),
            "epsilon": self.epsilon,
            "learning_active": True,
            "timestamp": datetime.now().isoformat(),
        }
