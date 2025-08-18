#!/usr/bin/env python3
"""
Simple RLHF System for CodeConductor
Updates model weights based on success/failure feedback
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SimpleRLHFAgent:
    """Simple RLHF agent that updates model weights"""

    def __init__(self, weights_file: str = "model_weights.json"):
        self.weights_file = Path(weights_file)
        self.weights = self.load_weights()

    def load_weights(self) -> dict[str, float]:
        """Load model weights from file"""
        if self.weights_file.exists():
            try:
                with open(self.weights_file) as f:
                    weights = json.load(f)
                logger.info(f"Loaded weights: {weights}")
                return weights
            except Exception as e:
                logger.error(f"Error loading weights: {e}")

        # Default weights (all equal)
        default_weights = {
            "meta-llama-3.1-8b-instruct": 1.0,
            "mistral-7b-instruct-v0.1": 1.0,
            "mistralai/codestral-22b-v0.1": 1.0,
            "google/gemma-3-12b": 1.0,
            "phi3:mini": 1.0,
        }
        logger.info(f"Using default weights: {default_weights}")
        return default_weights

    def save_weights(self):
        """Save weights to file"""
        try:
            with open(self.weights_file, "w") as f:
                json.dump(self.weights, f, indent=2)
            logger.info(f"Saved weights to {self.weights_file}")
        except Exception as e:
            logger.error(f"Error saving weights: {e}")

    def update_weights(self, model: str, success: bool, quality: float = 0.5):
        """Update model weights based on success/failure"""
        if model not in self.weights:
            logger.warning(f"Model {model} not found in weights")
            return

        # Calculate multiplier based on success and quality
        if success:
            # Success: multiply by 1.1 to 1.2 (10-20% increase)
            multiplier = 1.1 + (quality * 0.1)  # 1.1 to 1.2
        else:
            # Failure: multiply by 0.8 to 0.9 (10-20% decrease)
            multiplier = 0.9 - ((1 - quality) * 0.1)  # 0.8 to 0.9

        # Update weight with percentage-based adjustment
        old_weight = self.weights[model]

        if success:
            # Success: multiply by 1.1 to 1.2 (10-20% increase)
            multiplier = 1.1 + (quality * 0.1)  # 1.1 to 1.2
        else:
            # Failure: multiply by 0.8 to 0.9 (10-20% decrease)
            multiplier = 0.9 - ((1 - quality) * 0.1)  # 0.8 to 0.9

        self.weights[model] = max(
            0.1, min(20.0, old_weight * multiplier)
        )  # Increased max from 10.0 to 20.0

        # NO NORMALIZATION - let weights be what they are!

        logger.info(
            f"Updated {model}: {old_weight:.3f} -> {self.weights[model]:.3f} (multiplier: {multiplier:.3f})"
        )

        # Save updated weights
        self.save_weights()

        return self.weights[model]

    def get_best_models(self, n: int = 2) -> list[str]:
        """Get top N models based on weights"""
        sorted_models = sorted(self.weights.items(), key=lambda x: x[1], reverse=True)
        return [model for model, _ in sorted_models[:n]]

    def get_model_weight(self, model: str) -> float:
        """Get weight for specific model"""
        return self.weights.get(model, 1.0)

    def get_learning_improvement(self) -> float:
        """Calculate overall learning improvement"""
        # Simple metric: average weight of top 2 models
        top_models = self.get_best_models(2)
        avg_weight = sum(self.weights[model] for model in top_models) / len(top_models)
        return avg_weight - 1.0  # Improvement from baseline 1.0

    def reset_weights(self):
        """Reset all weights to equal values"""
        len(self.weights)
        for model in self.weights:
            self.weights[model] = 1.0
        self.save_weights()
        logger.info("Reset all weights to 1.0")


# Global instance
simple_rlhf = SimpleRLHFAgent()
