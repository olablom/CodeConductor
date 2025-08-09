#!/usr/bin/env python3
"""
Reset RLHF weights to baseline values
"""

import json
from pathlib import Path


def reset_rlhf_weights():
    """Reset RLHF weights to baseline values"""

    weights_file = Path("model_weights.json")

    # Baseline weights (all equal)
    baseline_weights = {
        "meta-llama-3.1-8b-instruct": 1.0,
        "mistral-7b-instruct-v0.1": 1.0,
        "mistralai/codestral-22b-v0.1": 1.0,
        "google/gemma-3-12b": 1.0,
        "phi3:mini": 1.0,
    }

    # Save baseline weights
    with open(weights_file, "w") as f:
        json.dump(baseline_weights, f, indent=2)

    print("✅ Reset RLHF weights to baseline:")
    for model, weight in baseline_weights.items():
        print(f"  {model}: {weight}")

    print(f"\n📁 Saved to: {weights_file}")


if __name__ == "__main__":
    reset_rlhf_weights()
