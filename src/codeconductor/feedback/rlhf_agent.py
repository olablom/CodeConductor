#!/usr/bin/env python3
"""
RLHF Agent for CodeConductor using PPO

This module implements a PPO-based reinforcement learning agent that learns
optimal model selection and code generation strategies based on Test-as-Reward data.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

# Try to import stable-baselines3, but provide fallback if not available
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.env_util import make_vec_env

    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    STABLE_BASELINES_AVAILABLE = False
    print("⚠️ stable-baselines3 not available. Install with: pip install stable-baselines3")

try:
    import gymnasium as gym
    from gymnasium import Env, spaces

    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    print("⚠️ gymnasium not available. Install with: pip install gymnasium")

logger = logging.getLogger(__name__)


class TrainingCallback(BaseCallback):
    """Custom callback for tracking training progress"""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Check if episode is done
        if self.locals.get("dones"):
            reward = self.locals.get("rewards", 0)
            self.episode_rewards.append(reward)
            self.episode_count += 1

            if self.episode_count % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                logger.info(f"Episode {self.episode_count}, Avg Reward: {avg_reward:.3f}")

        return True


class CodeConductorEnv(Env):
    """
    Gymnasium environment for CodeConductor RLHF training.

    Uses saved patterns from Test-as-Reward system to learn optimal
    model selection and code generation strategies.
    """

    def __init__(self, patterns_file: str = "patterns.json", max_patterns: int = 1000):
        super().__init__()

        if not GYM_AVAILABLE:
            raise ImportError("gymnasium is required for RLHF training")

        self.patterns_file = Path(patterns_file)
        self.max_patterns = max_patterns
        self.patterns = self._load_patterns()

        # Action space: 0=use_model_A, 1=use_model_B, 2=retry_with_fix, 3=escalate_to_gpt4
        self.action_space = spaces.Discrete(4)

        # Observation space: [test_reward, code_quality, user_feedback, task_complexity]
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

        self.current_pattern_idx = 0
        self.current_step = 0
        self.max_steps = 10  # Maximum steps per episode

    def _load_patterns(self) -> list[dict[str, Any]]:
        """Load patterns from JSON file"""
        if not self.patterns_file.exists():
            logger.warning(
                f"Patterns file {self.patterns_file} not found. Creating empty environment."
            )
            return []

        try:
            with open(self.patterns_file, encoding="utf-8") as f:
                patterns = json.load(f)

            # Filter patterns with valid rewards
            valid_patterns = [
                p
                for p in patterns
                if p.get("reward") is not None and isinstance(p.get("reward"), (int, float))
            ]

            if len(valid_patterns) > self.max_patterns:
                valid_patterns = valid_patterns[-self.max_patterns :]

            logger.info(f"Loaded {len(valid_patterns)} valid patterns for RLHF training")
            return valid_patterns

        except Exception as e:
            logger.error(f"Error loading patterns: {e}")
            return []

    def _extract_features(self, pattern: dict[str, Any]) -> np.ndarray:
        """Extract features from pattern for observation"""
        # Test reward (0.0 to 1.0)
        reward_value = pattern.get("reward")
        test_reward = float(reward_value) if reward_value is not None else 0.0

        # Code quality estimate (based on code length, complexity, etc.)
        code = pattern.get("code", "")
        code_quality = self._estimate_code_quality(code)

        # User feedback (if available)
        user_rating = pattern.get("user_rating")
        user_feedback = (
            float(user_rating) / 5.0 if user_rating is not None else 0.0
        )  # Normalize to 0-1

        # Task complexity (based on prompt length and keywords)
        task_complexity = self._estimate_task_complexity(pattern.get("prompt", ""))

        return np.array(
            [test_reward, code_quality, user_feedback, task_complexity],
            dtype=np.float32,
        )

    def _estimate_code_quality(self, code: str) -> float:
        """Estimate code quality based on various metrics"""
        if not code:
            return 0.0

        # Simple heuristics for code quality
        lines = code.split("\n")
        non_empty_lines = [line.strip() for line in lines if line.strip()]

        if not non_empty_lines:
            return 0.0

        # Factors that indicate good code
        has_functions = any("def " in line for line in non_empty_lines)
        has_imports = any("import " in line or "from " in line for line in non_empty_lines)
        has_docstrings = any('"""' in line or "'''" in line for line in non_empty_lines)
        has_type_hints = any(":" in line and "->" in line for line in non_empty_lines)

        # Calculate quality score
        quality_factors = [has_functions, has_imports, has_docstrings, has_type_hints]
        quality_score = sum(quality_factors) / len(quality_factors)

        # Penalize very short or very long code
        length_factor = min(len(non_empty_lines) / 20.0, 1.0)  # Normalize to 0-1

        return (quality_score + length_factor) / 2.0

    def _estimate_task_complexity(self, prompt: str) -> float:
        """Estimate task complexity based on prompt"""
        if not prompt:
            return 0.5  # Default medium complexity

        # Keywords that indicate complexity
        complex_keywords = [
            "api",
            "database",
            "authentication",
            "security",
            "async",
            "threading",
            "machine learning",
            "algorithm",
            "optimization",
            "performance",
            "testing",
            "deployment",
            "microservice",
            "distributed",
        ]

        simple_keywords = [
            "print",
            "hello world",
            "simple",
            "basic",
            "function",
            "variable",
        ]

        prompt_lower = prompt.lower()

        # Count complexity indicators
        complex_count = sum(1 for keyword in complex_keywords if keyword in prompt_lower)
        simple_count = sum(1 for keyword in simple_keywords if keyword in prompt_lower)

        # Calculate complexity score
        if complex_count > 0:
            complexity = min(complex_count / 3.0, 1.0)
        elif simple_count > 0:
            complexity = max(0.1, 1.0 - simple_count / 2.0)
        else:
            complexity = 0.5  # Default medium complexity

        return complexity

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment for new episode"""
        super().reset(seed=seed)

        if not self.patterns:
            # Return zero observation if no patterns available
            return np.zeros(4, dtype=np.float32), {}

        # Select random pattern
        self.current_pattern_idx = np.random.randint(0, len(self.patterns))
        self.current_step = 0

        pattern = self.patterns[self.current_pattern_idx]
        observation = self._extract_features(pattern)

        return observation, {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one step in the environment"""
        if not self.patterns:
            return np.zeros(4, dtype=np.float32), 0.0, True, False, {}

        pattern = self.patterns[self.current_pattern_idx]
        base_reward = float(pattern.get("reward", 0.0))

        # Adjust reward based on action
        action_multiplier = self._get_action_multiplier(action, pattern)
        adjusted_reward = base_reward * action_multiplier

        # Add exploration bonus for trying different actions
        exploration_bonus = 0.01 if action != 0 else 0.0  # Small bonus for non-default action

        # Exec-feedback: favor higher test coverage, penalize runtime if present in pattern
        runtime_ms = float(pattern.get("runtime_ms", 0.0))
        tests_passed = float(pattern.get("tests_passed", pattern.get("passed", 0.0)))
        tests_total = float(pattern.get("tests_total", pattern.get("total", 1.0)))
        exec_coverage = (tests_passed / tests_total) if tests_total > 0 else 0.0
        exec_feedback = exec_coverage - 0.001 * runtime_ms

        final_reward = adjusted_reward + exploration_bonus + 0.2 * float(exec_feedback)

        # Update step counter
        self.current_step += 1
        done = self.current_step >= self.max_steps
        truncated = False  # Gymnasium requires this

        # Create info dict
        info = {
            "prompt": pattern.get("prompt", ""),
            "code": pattern.get("code", ""),
            "base_reward": base_reward,
            "action": action,
            "action_multiplier": action_multiplier,
            "final_reward": final_reward,
        }

        # Return next observation (same pattern for now)
        next_observation = self._extract_features(pattern)

        return next_observation, final_reward, done, truncated, info

    def _get_action_multiplier(self, action: int, pattern: dict[str, Any]) -> float:
        """Get reward multiplier based on action and pattern context"""
        base_reward = pattern.get("reward", 0.0)
        task_complexity = self._estimate_task_complexity(pattern.get("prompt", ""))

        # Action-specific multipliers
        if action == 0:  # use_model_A (default)
            return 1.0
        elif action == 1:  # use_model_B
            # Model B might be better for complex tasks
            return 1.1 if task_complexity > 0.7 else 0.95
        elif action == 2:  # retry_with_fix
            # Retry is good for low-reward patterns
            return 1.2 if base_reward < 0.5 else 0.9
        elif action == 3:  # escalate_to_gpt4
            # Escalation is good for complex tasks or low rewards
            if task_complexity > 0.8 or base_reward < 0.3:
                return 1.3
            else:
                return 0.8  # Penalty for unnecessary escalation
        else:
            return 1.0

    def render(self):
        """Render environment (not implemented for this use case)"""
        pass


class RLHFAgent:
    """
    High-level interface for RLHF training and inference.
    """

    def __init__(
        self,
        model_path: str = "ppo_codeconductor",
        patterns_file: str = "patterns.json",
    ):
        self.model_path = Path(model_path)
        self.patterns_file = Path(patterns_file)
        self.model = None
        self.env = None

    def train(self, total_timesteps: int = 10000, save_model: bool = True) -> bool:
        """Train the RLHF agent"""
        if not STABLE_BASELINES_AVAILABLE:
            logger.error("stable-baselines3 not available. Cannot train agent.")
            return False

        try:
            # Create environment
            self.env = make_vec_env(lambda: CodeConductorEnv(str(self.patterns_file)), n_envs=1)

            # Create model
            self.model = PPO("MlpPolicy", self.env, verbose=1, learning_rate=0.0003)

            # Create callback for tracking
            callback = TrainingCallback()

            # Train model
            logger.info(f"Starting RLHF training for {total_timesteps} timesteps...")
            self.model.learn(total_timesteps=total_timesteps, callback=callback)

            # Save model
            if save_model:
                self.model.save(str(self.model_path))
                logger.info(f"Model saved to {self.model_path}")

            return True

        except Exception as e:
            logger.error(f"Error during training: {e}")
            return False

    def load_model(self) -> bool:
        """Load trained model"""
        if not STABLE_BASELINES_AVAILABLE:
            logger.error("stable-baselines3 not available. Cannot load model.")
            return False

        try:
            # Check for both .zip and no extension
            model_path_zip = self.model_path.with_suffix(".zip")
            if model_path_zip.exists():
                self.model = PPO.load(str(model_path_zip))
                logger.info(f"Model loaded from {model_path_zip}")
                return True
            elif self.model_path.exists():
                self.model = PPO.load(str(self.model_path))
                logger.info(f"Model loaded from {self.model_path}")
                return True
            else:
                logger.warning(f"Model file {self.model_path} or {model_path_zip} not found.")
                return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def predict_action(self, observation: np.ndarray) -> tuple[int, np.ndarray]:
        """Predict optimal action for given observation"""
        if self.model is None:
            logger.error("No model loaded. Call load_model() first.")
            return 0, np.array([1.0, 0.0, 0.0, 0.0])  # Default action

        try:
            action, _states = self.model.predict(observation, deterministic=True)
            return int(action), _states
        except Exception as e:
            logger.error(f"Error predicting action: {e}")
            return 0, np.array([1.0, 0.0, 0.0, 0.0])  # Default action

    def get_action_description(self, action: int) -> str:
        """Get human-readable description of action"""
        action_descriptions = {
            0: "use_model_A (default)",
            1: "use_model_B (alternative)",
            2: "retry_with_fix (improve)",
            3: "escalate_to_gpt4 (complex task)",
        }
        return action_descriptions.get(action, f"unknown_action_{action}")


def demo_rlhf_training():
    """Demonstrate RLHF training"""
    print("🚀 RLHF Training Demonstration")
    print("=" * 50)

    # Check dependencies
    if not STABLE_BASELINES_AVAILABLE:
        print("❌ stable-baselines3 not available")
        print("Install with: pip install stable-baselines3")
        return

    if not GYM_AVAILABLE:
        print("❌ gymnasium not available")
        print("Install with: pip install gymnasium")
        return

    # Create agent
    agent = RLHFAgent()

    # Check if we have patterns
    if not agent.patterns_file.exists():
        print("❌ No patterns.json found. Run Test-as-Reward demo first.")
        return

    # Train agent
    print("🎯 Training RLHF agent...")
    success = agent.train(total_timesteps=5000)  # Reduced for demo

    if success:
        print("✅ Training completed!")

        # Test prediction
        print("\n🧪 Testing predictions...")
        test_observation = np.array([0.5, 0.7, 0.8, 0.6], dtype=np.float32)  # Example observation
        action, states = agent.predict_action(test_observation)
        action_desc = agent.get_action_description(action)

        print(f"Observation: {test_observation}")
        print(f"Predicted Action: {action} ({action_desc})")

    else:
        print("❌ Training failed!")


def demo_rlhf_inference():
    """Demonstrate RLHF inference with trained model"""
    print("🧠 RLHF Inference Demonstration")
    print("=" * 50)

    agent = RLHFAgent()

    if not agent.load_model():
        print("❌ No trained model found. Run training first.")
        return

    # Test with different scenarios
    test_scenarios = [
        ("Low reward, simple task", [0.2, 0.3, 0.0, 0.2]),
        ("High reward, complex task", [0.8, 0.9, 0.8, 0.9]),
        ("Medium reward, medium complexity", [0.5, 0.6, 0.5, 0.5]),
    ]

    for scenario_name, observation in test_scenarios:
        observation = np.array(observation, dtype=np.float32)
        action, states = agent.predict_action(observation)
        action_desc = agent.get_action_description(action)

        print(f"\n📝 {scenario_name}")
        print(f"  Observation: {observation}")
        print(f"  Action: {action} ({action_desc})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RLHF Agent for CodeConductor")
    parser.add_argument(
        "--mode",
        choices=["train", "inference", "demo"],
        default="demo",
        help="Mode to run: train, inference, or demo",
    )
    parser.add_argument(
        "--timesteps", type=int, default=10000, help="Number of timesteps for training"
    )

    args = parser.parse_args()

    if args.mode == "train":
        agent = RLHFAgent()
        success = agent.train(total_timesteps=args.timesteps)
        print("✅ Training completed!" if success else "❌ Training failed!")
    elif args.mode == "inference":
        demo_rlhf_inference()
    else:  # demo
        demo_rlhf_training()
        print("\n" + "=" * 50)
        demo_rlhf_inference()
