"""
PromptOptimizerAgent - RL-based prompt optimization using Q-learning.

This agent learns to optimize prompts by mutating them based on previous outcomes.
Uses Q-learning with ε-greedy exploration to find optimal prompt modifications.
"""

import numpy as np
import hashlib
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json


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
        config: Optional[Dict] = None,
    ):
        """
        Initialize PromptOptimizerAgent.

        Args:
            learning_rate: Q-learning alpha parameter
            discount_factor: Q-learning gamma parameter
            epsilon: Exploration rate for ε-greedy
            config: Configuration dictionary
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.config = config or {}

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
        self.current_state: Optional[OptimizerState] = None
        self.current_action: Optional[str] = None
        self.state_history: List[OptimizerState] = []
        self.action_history: List[str] = []

    def _get_q_value(self, state_hash: int, action: str) -> float:
        """Get Q-value for state-action pair."""
        if state_hash not in self.q_table:
            self.q_table[state_hash] = {action: 0.0 for action in self.actions}
        return self.q_table[state_hash].get(action, 0.0)

    def _set_q_value(self, state_hash: int, action: str, value: float):
        """Set Q-value for state-action pair."""
        if state_hash not in self.q_table:
            self.q_table[state_hash] = {action: 0.0 for action in self.actions}
        self.q_table[state_hash][action] = value

    def _get_best_action(self, state_hash: int) -> str:
        """Get action with highest Q-value for given state."""
        if state_hash not in self.q_table:
            return np.random.choice(self.actions)

        q_values = self.q_table[state_hash]
        max_q = max(q_values.values())
        best_actions = [action for action, q in q_values.items() if q == max_q]
        return np.random.choice(best_actions)

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
        Create optimizer state from pipeline results.

        Args:
            task_id: Identifier for the task/prompt
            arm_prev: Previous bandit arm used
            passed: Whether tests passed
            blocked: Whether code was blocked by PolicyAgent
            complexity: Code complexity score
            model_source: Source of code generation

        Returns:
            OptimizerState object
        """
        # Determine fail bucket
        if blocked:
            fail_bucket = 1  # Policy block
        elif not passed:
            fail_bucket = 2  # Test fail
        elif complexity < 0.3:
            fail_bucket = 3  # High complexity (low score)
        else:
            fail_bucket = 0  # Success

        # Determine complexity bin
        if complexity < 0.3:
            complexity_bin = 2  # High complexity
        elif complexity < 0.7:
            complexity_bin = 1  # Medium complexity
        else:
            complexity_bin = 0  # Low complexity

        return OptimizerState(
            task_id=task_id,
            arm_prev=arm_prev,
            fail_bucket=fail_bucket,
            complexity_bin=complexity_bin,
            model_source=model_source,
        )

    def select_action(self, state: OptimizerState) -> str:
        """
        Select action using ε-greedy policy.

        Args:
            state: Current optimizer state

        Returns:
            Selected action string
        """
        state_hash = state.to_hash()

        # ε-greedy: explore with probability epsilon
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            action = self._get_best_action(state_hash)

        # Store current state and action
        self.current_state = state
        self.current_action = action

        return action

    def mutate_prompt(self, original_prompt: str, action: str) -> str:
        """
        Mutate prompt based on selected action.

        Args:
            original_prompt: Original prompt content
            action: Selected mutation action

        Returns:
            Mutated prompt content
        """
        if action not in self.prompt_mutations:
            return original_prompt

        mutation = self.prompt_mutations[action]
        return original_prompt + mutation

    def calculate_reward(
        self,
        passed: bool,
        blocked: bool,
        iterations: int,
        complexity: float,
        base_reward: float,
    ) -> float:
        """
        Calculate reward for prompt optimization.

        Args:
            passed: Whether tests passed
            blocked: Whether code was blocked
            iterations: Number of iterations to success
            complexity: Code complexity score
            base_reward: Base reward from pipeline

        Returns:
            Optimized reward value
        """
        reward = base_reward

        # Bonus for green on first retry
        if passed and iterations <= 2:
            reward += 10.0

        # Penalty for additional iterations
        if iterations > 1:
            reward -= iterations - 1

        # Penalty for PolicyAgent blocks
        if blocked:
            reward -= 5.0

        # Bonus for good complexity
        if complexity > 0.7:
            reward += 2.0

        return reward

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
            action: Action taken
            reward: Reward received
            next_state: Next state (optional)
        """
        state_hash = state.to_hash()
        current_q = self._get_q_value(state_hash, action)

        # Q-learning update
        if next_state:
            next_state_hash = next_state.to_hash()
            next_max_q = max(
                self._get_q_value(next_state_hash, a) for a in self.actions
            )
            target_q = reward + self.discount_factor * next_max_q
        else:
            target_q = reward

        # Update Q-value
        new_q = current_q + self.learning_rate * (target_q - current_q)
        self._set_q_value(state_hash, action, new_q)

        # Store in history
        self.state_history.append(state)
        self.action_history.append(action)

    def get_q_table_summary(self) -> Dict[str, Any]:
        """Get summary of Q-table for analysis."""
        if not self.q_table:
            return {"total_states": 0, "total_actions": 0, "avg_q_value": 0.0}

        total_states = len(self.q_table)
        total_actions = sum(len(actions) for actions in self.q_table.values())
        all_q_values = [
            q for actions in self.q_table.values() for q in actions.values()
        ]
        avg_q_value = np.mean(all_q_values) if all_q_values else 0.0

        return {
            "total_states": total_states,
            "total_actions": total_actions,
            "avg_q_value": avg_q_value,
            "min_q_value": min(all_q_values) if all_q_values else 0.0,
            "max_q_value": max(all_q_values) if all_q_values else 0.0,
        }

    def get_action_stats(self) -> Dict[str, int]:
        """Get statistics on action usage."""
        if not self.action_history:
            return {action: 0 for action in self.actions}

        action_counts = {}
        for action in self.actions:
            action_counts[action] = self.action_history.count(action)

        return action_counts

    def save_q_table(self, filepath: str):
        """Save Q-table to file."""
        with open(filepath, "w") as f:
            json.dump(self.q_table, f, indent=2)

    def load_q_table(self, filepath: str):
        """Load Q-table from file."""
        with open(filepath, "r") as f:
            self.q_table = json.load(f)


# Global instance for easy access
prompt_optimizer = PromptOptimizerAgent()


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
    Convenience function to optimize a prompt.

    Args:
        original_prompt: Original prompt content
        task_id: Task identifier
        arm_prev: Previous bandit arm
        passed: Whether tests passed
        blocked: Whether code was blocked
        complexity: Code complexity score
        model_source: Model source used

    Returns:
        Tuple of (optimized_prompt, action_taken)
    """
    # Create state
    state = prompt_optimizer.create_state(
        task_id=task_id,
        arm_prev=arm_prev,
        passed=passed,
        blocked=blocked,
        complexity=complexity,
        model_source=model_source,
    )

    # Select action
    action = prompt_optimizer.select_action(state)

    # Mutate prompt
    optimized_prompt = prompt_optimizer.mutate_prompt(original_prompt, action)

    return optimized_prompt, action


if __name__ == "__main__":
    # Test the PromptOptimizerAgent
    print("🧪 Testing PromptOptimizerAgent...")

    # Create test states
    test_states = [
        OptimizerState("hello_world", "conservative", 0, 0, "mock"),  # Success
        OptimizerState("hello_world", "balanced", 1, 1, "mock"),  # Policy block
        OptimizerState("calculator", "exploratory", 2, 2, "lm_studio"),  # Test fail
    ]

    for i, state in enumerate(test_states, 1):
        print(f"\n=== Test {i}: {state} ===")

        # Test action selection
        action = prompt_optimizer.select_action(state)
        print(f"Selected action: {action}")

        # Test prompt mutation
        original = "Write a Python function."
        mutated = prompt_optimizer.mutate_prompt(original, action)
        print(f"Original: {original}")
        print(f"Mutated: {mutated}")

        # Test reward calculation
        reward = prompt_optimizer.calculate_reward(
            passed=True, blocked=False, iterations=1, complexity=0.8, base_reward=30.0
        )
        print(f"Reward: {reward}")

        # Update Q-table
        prompt_optimizer.update(state, action, reward)

    # Show summary
    summary = prompt_optimizer.get_q_table_summary()
    print(f"\n📊 Q-table summary: {summary}")

    stats = prompt_optimizer.get_action_stats()
    print(f"📈 Action stats: {stats}")
