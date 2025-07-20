"""
Data Service - LinUCB Bandit

This module contains the LinUCB bandit algorithm migrated from the main CodeConductor
codebase to the microservices architecture.
"""

try:
    import numpy as np
except ImportError:
    # Fallback for environments without numpy
    import math
    import random

    class SimpleArray:
        def __init__(self, data):
            self.data = data if isinstance(data, list) else list(data)

        def reshape(self, shape):
            return self

        def __array__(self):
            return self.data

        def __getitem__(self, key):
            return self.data[key]

        def __setitem__(self, key, value):
            self.data[key] = value

        def __len__(self):
            return len(self.data)

        def __add__(self, other):
            if isinstance(other, SimpleArray):
                return SimpleArray([a + b for a, b in zip(self.data, other.data)])
            return SimpleArray([a + other for a in self.data])

        def __mul__(self, other):
            if isinstance(other, SimpleArray):
                return SimpleArray([a * b for a, b in zip(self.data, other.data)])
            return SimpleArray([a * other for a in self.data])

        def __matmul__(self, other):
            # Simple matrix multiplication
            if isinstance(other, SimpleArray):
                result = 0
                for a, b in zip(self.data, other.data):
                    result += a * b
                return result
            return self

        def T(self):
            return self

        def tolist(self):
            return self.data

    # Create numpy-like functions
    def array(data):
        return SimpleArray(data)

    def eye(n):
        data = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        return SimpleArray(data)

    def zeros(shape):
        if isinstance(shape, int):
            return SimpleArray([0.0] * shape)
        elif len(shape) == 2:
            return SimpleArray(
                [[0.0 for _ in range(shape[1])] for _ in range(shape[0])]
            )
        return SimpleArray([0.0])

    def random_choice(choices, size=None, p=None):
        if size is None:
            return random.choice(choices)
        else:
            if p is None:
                return [random.choice(choices) for _ in range(size)]
            else:
                # Simple weighted choice
                return [random.choices(choices, weights=p)[0] for _ in range(size)]

    def mean(data):
        return sum(data) / len(data) if data else 0.0

    def std(data):
        if len(data) <= 1:
            return 0.0
        mean_val = mean(data)
        variance = sum((x - mean_val) ** 2 for x in data) / (len(data) - 1)
        return math.sqrt(variance)

    def min(data):
        return min(data) if data else 0.0

    def max(data, key=None):
        if key is None:
            return max(data) if data else 0.0
        else:
            # Custom max with key function
            if not data:
                return 0.0
            max_val = data[0]
            max_key_val = key(max_val)
            for item in data[1:]:
                key_val = key(item)
                if key_val > max_key_val:
                    max_val = item
                    max_key_val = key_val
            return max_val

    # Create numpy namespace
    class NumpyNamespace:
        def __init__(self):
            self.array = array
            self.eye = eye
            self.zeros = zeros
            self.random = type("Random", (), {"choice": random_choice})()
            self.mean = mean
            self.std = std
            self.min = min
            self.max = max
            self.linalg = type(
                "Linalg",
                (),
                {
                    "inv": lambda x: x,  # Simple identity for now
                    "LinAlgError": Exception,
                },
            )()

    np = NumpyNamespace()
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
from datetime import datetime


class LinUCBBandit:
    """
    Linear Upper Confidence Bound (LinUCB) bandit algorithm.

    This bandit learns to select the best action (arm) based on contextual features
    and observed rewards using linear regression with confidence bounds.
    """

    def __init__(self, d: int, alpha: float = 1.0, name: str = "linucb_bandit"):
        """
        Initialize LinUCB bandit.

        Args:
            d: Dimensions of the feature vector
            alpha: Exploration parameter (higher = more exploration)
            name: Name of the bandit instance
        """
        self.d = d
        self.alpha = alpha
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize matrices for each arm
        self.A = defaultdict(lambda: np.eye(d))  # A matrix for each arm
        self.b = defaultdict(lambda: np.zeros((d, 1)))  # b vector for each arm

        # Statistics
        self.total_pulls = 0
        self.arm_pulls = defaultdict(int)
        self.arm_rewards = defaultdict(list)

        self.logger.info(f"LinUCBBandit '{name}' initialized with d={d}, alpha={alpha}")

    def get_ucb(self, arm: str, x) -> float:
        """
        Calculate Upper Confidence Bound for an arm.

        Args:
            arm: Arm identifier
            x: Feature vector

        Returns:
            UCB value for the arm
        """
        try:
            x = x.reshape(-1, 1)
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv @ self.b[arm]
            p = float(theta.T @ x) + self.alpha * float(np.sqrt(x.T @ A_inv @ x))
            return p
        except np.linalg.LinAlgError:
            # Handle singular matrix
            self.logger.warning(f"Singular matrix for arm {arm}, using fallback")
            return 0.0
        except Exception as e:
            self.logger.error(f"Error calculating UCB for arm {arm}: {e}")
            return 0.0

    def select_arm(self, arms: List[str], x) -> str:
        """
        Select the best arm based on UCB values.

        Args:
            arms: List of available arms
            x: Feature vector

        Returns:
            Selected arm identifier
        """
        if not arms:
            raise ValueError("No arms available for selection")

        try:
            ucbs = {arm: self.get_ucb(arm, x) for arm in arms}
            selected_arm = max(ucbs, key=ucbs.get)

            # Update statistics
            self.total_pulls += 1
            self.arm_pulls[selected_arm] += 1

            self.logger.info(
                f"Selected arm {selected_arm} with UCB {ucbs[selected_arm]:.4f}"
            )
            return selected_arm

        except Exception as e:
            self.logger.error(f"Error selecting arm: {e}")
            # Fallback to random selection
            import random

            return random.choice(arms)

    def update(self, arm: str, x, reward: float) -> None:
        """
        Update the model with observed reward.

        Args:
            arm: Arm that was pulled
            x: Feature vector used for selection
            reward: Observed reward
        """
        try:
            x = x.reshape(-1, 1)
            self.A[arm] += x @ x.T
            self.b[arm] += reward * x

            # Update statistics
            self.arm_rewards[arm].append(reward)

            self.logger.info(f"Updated arm {arm} with reward {reward:.4f}")

        except Exception as e:
            self.logger.error(f"Error updating arm {arm}: {e}")

    def get_arm_statistics(self) -> Dict[str, Any]:
        """
        Get statistics for all arms.

        Returns:
            Dictionary with arm statistics
        """
        stats = {}
        for arm in self.A.keys():
            rewards = self.arm_rewards[arm]
            stats[arm] = {
                "pulls": self.arm_pulls[arm],
                "total_reward": sum(rewards),
                "average_reward": np.mean(rewards) if rewards else 0.0,
                "reward_std": np.std(rewards) if len(rewards) > 1 else 0.0,
                "min_reward": min(rewards) if rewards else 0.0,
                "max_reward": max(rewards) if rewards else 0.0,
            }

        return {
            "total_pulls": self.total_pulls,
            "arm_count": len(self.A),
            "arms": stats,
        }

    def get_model_state(self) -> Dict[str, Any]:
        """
        Get current model state for persistence.

        Returns:
            Dictionary with model state
        """
        state = {
            "name": self.name,
            "d": self.d,
            "alpha": self.alpha,
            "total_pulls": self.total_pulls,
            "arm_pulls": dict(self.arm_pulls),
            "arm_rewards": {arm: rewards for arm, rewards in self.arm_rewards.items()},
            "A_matrices": {arm: A.tolist() for arm, A in self.A.items()},
            "b_vectors": {arm: b.tolist() for arm, b in self.b.items()},
            "timestamp": datetime.now().isoformat(),
        }
        return state

    def load_model_state(self, state: Dict[str, Any]) -> None:
        """
        Load model state from dictionary.

        Args:
            state: Model state dictionary
        """
        try:
            self.name = state.get("name", self.name)
            self.d = state.get("d", self.d)
            self.alpha = state.get("alpha", self.alpha)
            self.total_pulls = state.get("total_pulls", 0)
            self.arm_pulls = defaultdict(int, state.get("arm_pulls", {}))
            self.arm_rewards = defaultdict(list, state.get("arm_rewards", {}))

            # Load matrices
            A_matrices = state.get("A_matrices", {})
            b_vectors = state.get("b_vectors", {})

            self.A.clear()
            self.b.clear()

            for arm in A_matrices:
                self.A[arm] = np.array(A_matrices[arm])
                self.b[arm] = np.array(b_vectors[arm])

            self.logger.info(f"Loaded model state for {len(self.A)} arms")

        except Exception as e:
            self.logger.error(f"Error loading model state: {e}")

    def reset(self) -> None:
        """Reset the bandit to initial state."""
        self.A.clear()
        self.b.clear()
        self.total_pulls = 0
        self.arm_pulls.clear()
        self.arm_rewards.clear()
        self.logger.info("Bandit reset to initial state")

    def get_confidence_intervals(self, arms: List[str], x) -> Dict[str, float]:
        """
        Get confidence intervals for all arms.

        Args:
            arms: List of arms
            x: Feature vector

        Returns:
            Dictionary mapping arms to confidence values
        """
        intervals = {}
        for arm in arms:
            try:
                x_reshaped = x.reshape(-1, 1)
                A_inv = np.linalg.inv(self.A[arm])
                confidence = self.alpha * float(
                    np.sqrt(x_reshaped.T @ A_inv @ x_reshaped)
                )
                intervals[arm] = confidence
            except:
                intervals[arm] = 0.0

        return intervals
