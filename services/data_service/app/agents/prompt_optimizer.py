"""
Data Service - Prompt Optimizer Agent

This module contains the PromptOptimizerAgent migrated from the main CodeConductor
codebase to the microservices architecture.
"""

try:
    import numpy as np
except ImportError:
    # Fallback for environments without numpy
    import random

    class NumpyFallback:
        class RandomFallback:
            @staticmethod
            def random():
                return random.random()

            @staticmethod
            def choice(choices, size=None, p=None):
                if size is None:
                    return random.choice(list(choices))
                else:
                    return [random.choice(list(choices)) for _ in range(size)]

        random = RandomFallback()

    np = NumpyFallback()
import hashlib
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime


class PromptAction(Enum):
    """Available prompt mutation actions."""

    ADD_TYPE_HINTS = "add_type_hints"
    ASK_FOR_OOP = "ask_for_oop"
    ADD_DOCSTRINGS = "add_docstrings"
    SIMPLIFY = "simplify"
    ADD_EXAMPLES = "add_examples"
    NO_CHANGE = "no_change"


@dataclass
class OptimizerState:
    """State representation for prompt optimization."""

    task_id: str
    arm_prev: str
    fail_bucket: int  # 0=success, 1=policy_block, 2=test_fail, 3=complexity_high
    complexity_bin: int  # 0=low, 1=medium, 2=high
    model_source: str

    def to_vector(self) -> Tuple[str, str, int, int, str]:
        """Convert to state vector for hashing."""
        return (
            self.task_id,
            self.arm_prev,
            self.fail_bucket,
            self.complexity_bin,
            self.model_source,
        )

    def to_hash(self) -> int:
        """Convert state to hash for Q-table lookup."""
        state_str = json.dumps(self.to_vector(), sort_keys=True)
        return int(hashlib.md5(state_str.encode()).hexdigest()[:8], 16)

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "task_id": self.task_id,
            "arm_prev": self.arm_prev,
            "fail_bucket": self.fail_bucket,
            "complexity_bin": self.complexity_bin,
            "model_source": self.model_source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizerState":
        """Create state from dictionary."""
        return cls(**data)


class PromptOptimizerAgent:
    """
    Q-learning agent for prompt optimization.

    Learns which prompt mutations work best for different scenarios:
    - Failed tests → Try different mutations
    - Policy blocks → Simplify or add safety
    - High complexity → Ask for simpler code
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 0.1,
        name: str = "prompt_optimizer",
        config: Optional[Dict] = None,
    ):
        """
        Initialize PromptOptimizerAgent.

        Args:
            learning_rate: Q-learning alpha parameter
            discount_factor: Q-learning gamma parameter
            epsilon: Exploration rate for ε-greedy
            name: Agent name
            config: Configuration dictionary
        """
        self.name = name
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Q-table: state_hash -> action -> Q-value
        self.q_table: Dict[int, Dict[str, float]] = {}

        # Available actions
        self.actions = [action.value for action in PromptAction]

        # Prompt mutation templates
        self.prompt_mutations = {
            PromptAction.ADD_TYPE_HINTS.value: "\n\n# Please add Python type hints throughout the code.",
            PromptAction.ASK_FOR_OOP.value: "\n\n# Please refactor the solution into an object-oriented programming style.",
            PromptAction.ADD_DOCSTRINGS.value: "\n\n# Please include detailed docstrings in Swedish for all functions.",
            PromptAction.SIMPLIFY.value: "\n\n# Please keep the solution simple and straightforward.",
            PromptAction.ADD_EXAMPLES.value: "\n\n# Please include usage examples in the code.",
            PromptAction.NO_CHANGE.value: "",
        }

        # State tracking
        self.total_episodes = 0
        self.successful_episodes = 0
        self.action_counts = {action: 0 for action in self.actions}
        self.learning_history: List[Dict[str, Any]] = []

        self.logger.info(
            f"PromptOptimizerAgent '{name}' initialized with epsilon={epsilon}"
        )

    def _get_q_value(self, state_hash: int, action: str) -> float:
        """Get Q-value for state-action pair."""
        return self.q_table.get(state_hash, {}).get(action, 0.0)

    def _set_q_value(self, state_hash: int, action: str, value: float):
        """Set Q-value for state-action pair."""
        if state_hash not in self.q_table:
            self.q_table[state_hash] = {}
        self.q_table[state_hash][action] = value

    def _get_best_action(self, state_hash: int) -> str:
        """Get best action for state."""
        if state_hash not in self.q_table:
            return np.random.choice(self.actions)

        q_values = self.q_table[state_hash]
        if not q_values:
            return np.random.choice(self.actions)

        # Fallback max function for environments without numpy
        if hasattr(max, "__call__") and "key" in max.__code__.co_varnames:
            return max(q_values, key=q_values.get)
        else:
            # Simple max implementation
            max_key = None
            max_value = float("-inf")
            for key, value in q_values.items():
                if value > max_value:
                    max_value = value
                    max_key = key
            return max_key if max_key is not None else list(q_values.keys())[0]

    def create_state(
        self,
        task_id: str,
        arm_prev: str,
        passed: bool,
        blocked: bool,
        complexity: float,
        model_source: str,
    ) -> OptimizerState:
        """
        Create optimizer state from context.

        Args:
            task_id: Task identifier
            arm_prev: Previous arm used
            passed: Whether tests passed
            blocked: Whether policy blocked
            complexity: Task complexity (0-1)
            model_source: Source model used

        Returns:
            OptimizerState object
        """
        # Determine fail bucket
        if passed and not blocked:
            fail_bucket = 0  # Success
        elif blocked:
            fail_bucket = 1  # Policy block
        elif not passed:
            fail_bucket = 2  # Test fail
        else:
            fail_bucket = 3  # Other failure

        # Determine complexity bin
        if complexity < 0.33:
            complexity_bin = 0  # Low
        elif complexity < 0.67:
            complexity_bin = 1  # Medium
        else:
            complexity_bin = 2  # High

        return OptimizerState(
            task_id=task_id,
            arm_prev=arm_prev,
            fail_bucket=fail_bucket,
            complexity_bin=complexity_bin,
            model_source=model_source,
        )

    def select_action(self, state: OptimizerState) -> str:
        """
        Select action using epsilon-greedy strategy.

        Args:
            state: Current state

        Returns:
            Selected action
        """
        state_hash = state.to_hash()

        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            # Exploration: random action
            action = np.random.choice(self.actions)
            self.logger.info(f"Exploration: selected random action {action}")
        else:
            # Exploitation: best action
            action = self._get_best_action(state_hash)
            q_value = self._get_q_value(state_hash, action)
            self.logger.info(
                f"Exploitation: selected best action {action} with Q-value {q_value:.4f}"
            )

        # Update action counts
        self.action_counts[action] += 1

        return action

    def mutate_prompt(self, original_prompt: str, action: str) -> str:
        """
        Apply mutation to prompt based on action.

        Args:
            original_prompt: Original prompt
            action: Action to apply

        Returns:
            Mutated prompt
        """
        mutation = self.prompt_mutations.get(action, "")
        return original_prompt + mutation

    def calculate_reward(
        self,
        passed: bool,
        blocked: bool,
        iterations: int,
        complexity: float,
        base_reward: float = 1.0,
    ) -> float:
        """
        Calculate reward for the action.

        Args:
            passed: Whether tests passed
            blocked: Whether policy blocked
            iterations: Number of iterations needed
            complexity: Task complexity
            base_reward: Base reward value

        Returns:
            Calculated reward
        """
        reward = base_reward

        # Success bonus
        if passed and not blocked:
            reward += 2.0
        elif blocked:
            reward -= 1.0  # Policy block penalty
        elif not passed:
            reward -= 0.5  # Test fail penalty

        # Iteration penalty (fewer iterations = better)
        reward -= iterations * 0.1

        # Complexity penalty (lower complexity = better)
        reward -= complexity * 0.5

        return max(reward, -2.0)  # Cap minimum reward

    def update(
        self,
        state: OptimizerState,
        action: str,
        reward: float,
        next_state: Optional[OptimizerState] = None,
    ):
        """
        Update Q-values using Q-learning.

        Args:
            state: Current state
            action: Selected action
            reward: Observed reward
            next_state: Next state (optional)
        """
        state_hash = state.to_hash()
        current_q = self._get_q_value(state_hash, action)

        if next_state:
            next_state_hash = next_state.to_hash()
            next_q_values = self.q_table.get(next_state_hash, {})
            max_next_q = max(next_q_values.values()) if next_q_values else 0.0
        else:
            max_next_q = 0.0

        # Q-learning update rule
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        self._set_q_value(state_hash, action, new_q)

        self.logger.info(
            f"Updated Q-value for state {state_hash}, action {action}: {current_q:.4f} -> {new_q:.4f}"
        )

    def get_q_table_summary(self) -> Dict[str, Any]:
        """
        Get summary of Q-table.

        Returns:
            Q-table summary
        """
        total_entries = sum(len(actions) for actions in self.q_table.values())
        q_values = []
        for actions in self.q_table.values():
            q_values.extend(actions.values())

        return {
            "total_states": len(self.q_table),
            "total_entries": total_entries,
            "average_q_value": np.mean(q_values) if q_values else 0.0,
            "min_q_value": min(q_values) if q_values else 0.0,
            "max_q_value": max(q_values) if q_values else 0.0,
            "epsilon": self.epsilon,
            "total_episodes": self.total_episodes,
            "successful_episodes": self.successful_episodes,
            "success_rate": self.successful_episodes / self.total_episodes
            if self.total_episodes > 0
            else 0.0,
        }

    def get_action_stats(self) -> Dict[str, int]:
        """
        Get action usage statistics.

        Returns:
            Dictionary with action counts
        """
        return self.action_counts.copy()

    def get_model_state(self) -> Dict[str, Any]:
        """
        Get current model state for persistence.

        Returns:
            Dictionary with model state
        """
        return {
            "name": self.name,
            "q_table": self.q_table,
            "action_counts": self.action_counts,
            "total_episodes": self.total_episodes,
            "successful_episodes": self.successful_episodes,
            "epsilon": self.epsilon,
            "learning_history": self.learning_history,
            "config": self.config,
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
            self.action_counts = state.get(
                "action_counts", {action: 0 for action in self.actions}
            )
            self.total_episodes = state.get("total_episodes", 0)
            self.successful_episodes = state.get("successful_episodes", 0)
            self.epsilon = state.get("epsilon", self.epsilon)
            self.learning_history = state.get("learning_history", [])
            self.config.update(state.get("config", {}))

            self.logger.info(f"Loaded model state with {len(self.q_table)} states")

        except Exception as e:
            self.logger.error(f"Error loading model state: {e}")

    def reset(self):
        """Reset the optimizer to initial state."""
        self.q_table.clear()
        self.action_counts = {action: 0 for action in self.actions}
        self.total_episodes = 0
        self.successful_episodes = 0
        self.learning_history.clear()
        self.epsilon = self.config.get("epsilon", 0.1)
        self.logger.info("Prompt optimizer reset to initial state")

    def decay_epsilon(self):
        """Decay epsilon for exploration."""
        epsilon_decay = self.config.get("epsilon_decay", 0.995)
        epsilon_min = self.config.get("epsilon_min", 0.01)
        self.epsilon = max(self.epsilon * epsilon_decay, epsilon_min)
        self.logger.info(f"Epsilon decayed to {self.epsilon:.4f}")

    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze context for prompt optimization.

        Args:
            context: Context to analyze

        Returns:
            Analysis results
        """
        return {
            "agent_name": self.name,
            "available_actions": len(self.actions),
            "q_table_size": len(self.q_table),
            "epsilon": self.epsilon,
            "action_stats": self.get_action_stats(),
            "summary": self.get_q_table_summary(),
            "timestamp": datetime.now().isoformat(),
        }

    def propose(
        self, analysis: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Propose prompt mutation.

        Args:
            analysis: Previous analysis results
            context: Original context

        Returns:
            Mutation proposal
        """
        # Extract context information
        task_id = context.get("task_id", "unknown")
        arm_prev = context.get("arm_prev", "unknown")
        passed = context.get("passed", True)
        blocked = context.get("blocked", False)
        complexity = context.get("complexity", 0.5)
        model_source = context.get("model_source", "unknown")
        original_prompt = context.get("prompt", "")

        # Create state and select action
        state = self.create_state(
            task_id, arm_prev, passed, blocked, complexity, model_source
        )
        action = self.select_action(state)

        # Mutate prompt
        mutated_prompt = self.mutate_prompt(original_prompt, action)

        return {
            "agent_name": self.name,
            "selected_action": action,
            "original_prompt": original_prompt,
            "mutated_prompt": mutated_prompt,
            "mutation": self.prompt_mutations.get(action, ""),
            "q_value": self._get_q_value(state.to_hash(), action),
            "epsilon": self.epsilon,
            "exploration": self.epsilon > np.random.random(),
            "confidence": 1.0 - self.epsilon,
            "reasoning": f"Prompt optimization with action {action}",
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
            "mutation_applied": proposal.get("mutation", ""),
            "q_value": proposal.get("q_value", 0.0),
            "epsilon": self.epsilon,
            "learning_active": True,
            "timestamp": datetime.now().isoformat(),
        }


def optimize_prompt(
    original_prompt: str,
    task_id: str,
    arm_prev: str,
    passed: bool,
    blocked: bool,
    complexity: float,
    model_source: str,
) -> Tuple[str, str]:
    """
    Optimize prompt using the prompt optimizer.

    Args:
        original_prompt: Original prompt
        task_id: Task identifier
        arm_prev: Previous arm used
        passed: Whether tests passed
        blocked: Whether policy blocked
        complexity: Task complexity
        model_source: Source model used

    Returns:
        Tuple of (mutated_prompt, action_used)
    """
    optimizer = PromptOptimizerAgent()
    context = {
        "task_id": task_id,
        "arm_prev": arm_prev,
        "passed": passed,
        "blocked": blocked,
        "complexity": complexity,
        "model_source": model_source,
        "prompt": original_prompt,
    }

    proposal = optimizer.propose({}, context)
    return proposal["mutated_prompt"], proposal["selected_action"]
