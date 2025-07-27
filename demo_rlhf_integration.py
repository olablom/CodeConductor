#!/usr/bin/env python3
"""
Demo script for RLHF integration with Ensemble Engine

This script demonstrates how the RLHF agent is integrated into the ensemble pipeline
to dynamically select optimal models based on task complexity and historical performance.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import ensemble components
from ensemble.ensemble_engine import EnsembleEngine, EnsembleRequest
from runners.test_runner import TestRunner
from feedback.learning_system import log_test_reward


async def demo_rlhf_integration():
    """Demonstrate RLHF integration with ensemble engine."""
    print("🚀 RLHF Integration Demo")
    print("=" * 60)

    # Initialize ensemble engine with RLHF
    print("🔧 Initializing Ensemble Engine with RLHF...")
    ensemble = EnsembleEngine(use_rlhf=True)

    # Initialize the engine
    success = await ensemble.initialize()
    if not success:
        print("❌ Failed to initialize ensemble engine")
        return

    print("✅ Ensemble Engine initialized successfully")

    # Test scenarios with different complexity levels
    test_scenarios = [
        {
            "name": "Simple Task",
            "task": "Create a function to add two numbers",
            "expected_complexity": "low",
            "test_results": [{"passed": True}, {"passed": True}],  # Good test results
            "code_quality": 0.8,
            "user_feedback": 0.9,
        },
        {
            "name": "Complex API Task",
            "task": "Create a REST API endpoint with authentication and database integration",
            "expected_complexity": "high",
            "test_results": [{"passed": False}, {"passed": True}],  # Mixed test results
            "code_quality": 0.6,
            "user_feedback": 0.7,
        },
        {
            "name": "Medium Complexity Task",
            "task": "Implement a sorting algorithm with error handling",
            "expected_complexity": "medium",
            "test_results": [
                {"passed": True},
                {"passed": True},
                {"passed": False},
            ],  # Some failures
            "code_quality": 0.7,
            "user_feedback": 0.8,
        },
    ]

    print("\n🧪 Testing RLHF Integration with Different Scenarios")
    print("-" * 60)

    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n📝 Scenario {i}: {scenario['name']}")
        print(f"   Task: {scenario['task']}")
        print(f"   Expected Complexity: {scenario['expected_complexity']}")
        print(
            f"   Test Results: {len([t for t in scenario['test_results'] if t['passed']])}/{len(scenario['test_results'])} passed"
        )

        # Create ensemble request with RLHF context
        request = EnsembleRequest(
            task_description=scenario["task"],
            test_results=scenario["test_results"],
            code_quality=scenario["code_quality"],
            user_feedback=scenario["user_feedback"],
            min_models=2,
            timeout=30.0,
        )

        # Process request through ensemble with RLHF
        print("   🔄 Processing with RLHF agent...")
        response = await ensemble.process_request(request)

        # Display results
        print(
            f"   ✅ RLHF Action: {response.rlhf_action} ({response.rlhf_action_description})"
        )
        print(f"   🎯 Selected Model: {response.selected_model}")
        print(f"   🧠 Confidence: {response.confidence:.2f}")
        print(f"   ⏱️  Execution Time: {response.execution_time:.2f}s")

        # Log the pattern for learning
        if response.consensus:
            consensus_text = (
                str(response.consensus)[:100] + "..."
                if len(str(response.consensus)) > 100
                else str(response.consensus)
            )
            reward = log_test_reward(
                prompt=scenario["task"],
                code=consensus_text,
                test_results=scenario["test_results"],
                metadata={
                    "scenario": scenario["name"],
                    "rlhf_action": response.rlhf_action,
                    "rlhf_action_description": response.rlhf_action_description,
                    "selected_model": response.selected_model,
                    "confidence": response.confidence,
                },
            )
            print(f"   🎯 Calculated Reward: {reward:.2f}")

        print("   " + "-" * 40)


async def demo_rlhf_vs_no_rlhf():
    """Compare ensemble performance with and without RLHF."""
    print("\n🔄 RLHF vs No-RLHF Comparison")
    print("=" * 60)

    # Test task
    task = "Create a function to validate email addresses with regex"
    test_results = [{"passed": True}, {"passed": False}, {"passed": True}]

    print(f"📝 Task: {task}")
    print(f"🧪 Test Results: 2/3 passed")

    # Test with RLHF enabled
    print("\n🧠 Testing WITH RLHF...")
    ensemble_rlhf = EnsembleEngine(use_rlhf=True)
    await ensemble_rlhf.initialize()

    request = EnsembleRequest(
        task_description=task,
        test_results=test_results,
        code_quality=0.7,
        user_feedback=0.8,
    )

    response_rlhf = await ensemble_rlhf.process_request(request)

    print(
        f"   RLHF Action: {response_rlhf.rlhf_action} ({response_rlhf.rlhf_action_description})"
    )
    print(f"   Selected Model: {response_rlhf.selected_model}")
    print(f"   Confidence: {response_rlhf.confidence:.2f}")
    print(f"   Execution Time: {response_rlhf.execution_time:.2f}s")

    # Test without RLHF
    print("\n⚡ Testing WITHOUT RLHF...")
    ensemble_no_rlhf = EnsembleEngine(use_rlhf=False)
    await ensemble_no_rlhf.initialize()

    response_no_rlhf = await ensemble_no_rlhf.process_request(request)

    print(f"   Default Selection: {response_no_rlhf.selected_model}")
    print(f"   Confidence: {response_no_rlhf.confidence:.2f}")
    print(f"   Execution Time: {response_no_rlhf.execution_time:.2f}s")

    # Compare results
    print("\n📊 Comparison Results:")
    print(f"   RLHF Confidence: {response_rlhf.confidence:.2f}")
    print(f"   No-RLHF Confidence: {response_no_rlhf.confidence:.2f}")
    print(
        f"   Confidence Difference: {response_rlhf.confidence - response_no_rlhf.confidence:+.2f}"
    )

    if response_rlhf.confidence > response_no_rlhf.confidence:
        print("   🎉 RLHF improved confidence!")
    elif response_rlhf.confidence < response_no_rlhf.confidence:
        print("   📉 RLHF reduced confidence (may need more training)")
    else:
        print("   ➖ No difference in confidence")


def demo_rlhf_training_status():
    """Show RLHF training status and model information."""
    print("\n📊 RLHF Training Status")
    print("=" * 60)

    try:
        from feedback.rlhf_agent import RLHFAgent

        agent = RLHFAgent()

        # Check if model exists
        model_path = Path("ppo_codeconductor.zip")
        if model_path.exists():
            print(f"✅ Trained model found: {model_path}")
            print(f"   Size: {model_path.stat().st_size / 1024:.1f} KB")

            # Try to load model
            if agent.load_model():
                print("✅ Model loads successfully")
            else:
                print("❌ Model failed to load")
        else:
            print("❌ No trained model found")
            print("   Run: python feedback/rlhf_agent.py --mode train")

        # Check patterns
        patterns_path = Path("patterns.json")
        if patterns_path.exists():
            import json

            with open(patterns_path, "r") as f:
                patterns = json.load(f)
            print(f"📚 Training data: {len(patterns)} patterns")

            # Show some statistics
            rewards = [
                p.get("reward", 0) for p in patterns if p.get("reward") is not None
            ]
            if rewards:
                print(f"   Average reward: {sum(rewards) / len(rewards):.2f}")
                print(f"   Max reward: {max(rewards):.2f}")
                print(f"   Min reward: {min(rewards):.2f}")
        else:
            print("❌ No patterns.json found")
            print("   Run: python apply_test_as_reward.py")

    except ImportError as e:
        print(f"❌ RLHF not available: {e}")
        print("   Install with: pip install stable-baselines3 gymnasium")


async def main():
    """Main demo function."""
    print("🎼 CodeConductor RLHF Integration Demo")
    print("=" * 60)

    # Show RLHF status
    demo_rlhf_training_status()

    # Run integration demo
    await demo_rlhf_integration()

    # Run comparison demo
    await demo_rlhf_vs_no_rlhf()

    print("\n🎉 Demo completed!")
    print("\n💡 Next steps:")
    print("   1. Run more tasks to collect training data")
    print("   2. Retrain RLHF agent: python feedback/rlhf_agent.py --mode train")
    print("   3. Test in production: streamlit run codeconductor_app.py")


if __name__ == "__main__":
    asyncio.run(main())
