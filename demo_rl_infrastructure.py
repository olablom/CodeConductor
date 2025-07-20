#!/usr/bin/env python3
"""
Demo script for CodeConductor v2.0 RL Infrastructure

This script demonstrates the RewardAgent and QLearningAgent working together
to show the reinforcement learning foundation we've built.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.reward_agent import RewardAgent, TestResults, CodeMetrics
from agents.q_learning_agent import QLearningAgent, State, Action


def demo_reward_calculation():
    """Demonstrate reward calculation with different scenarios."""
    print("🎯 DEMO: Reward Calculation")
    print("=" * 50)

    # Create reward agent
    reward_agent = RewardAgent("demo_reward_agent")

    # Scenario 1: Perfect code
    print("\n📊 Scenario 1: Perfect Code")
    test_results = TestResults(
        passed=10,
        failed=0,
        total=10,
        execution_time=2.0,
        coverage=0.95,
        lint_score=10.0,
    )
    code_metrics = CodeMetrics(
        complexity=5.0,
        lines_of_code=50,
        function_count=3,
        class_count=1,
        comment_ratio=0.25,
    )
    policy_violations = []
    human_feedback = {
        "thumbs_up": 5,
        "thumbs_down": 0,
        "rating": 0.9,
        "comments": ["Excellent work!", "Perfect implementation!"],
    }

    reward = reward_agent.calculate_reward(
        test_results=test_results,
        code_metrics=code_metrics,
        policy_violations=policy_violations,
        human_feedback=human_feedback,
    )

    print(f"Total Reward: {reward['total_reward']:.3f}")
    print(f"Components: {reward['components']}")

    # Scenario 2: Poor code
    print("\n📊 Scenario 2: Poor Code")
    test_results = TestResults(
        passed=2, failed=8, total=10, execution_time=15.0, coverage=0.3, lint_score=5.0
    )
    code_metrics = CodeMetrics(
        complexity=25.0,
        lines_of_code=500,
        function_count=50,
        class_count=10,
        comment_ratio=0.05,
    )
    policy_violations = ["security_vulnerability", "deprecated_function_used"]
    human_feedback = {
        "thumbs_up": 0,
        "thumbs_down": 3,
        "rating": 0.2,
        "comments": ["Poor code", "Bad implementation"],
    }

    reward = reward_agent.calculate_reward(
        test_results=test_results,
        code_metrics=code_metrics,
        policy_violations=policy_violations,
        human_feedback=human_feedback,
    )

    print(f"Total Reward: {reward['total_reward']:.3f}")
    print(f"Components: {reward['components']}")


def demo_q_learning():
    """Demonstrate Q-learning functionality."""
    print("\n🧠 DEMO: Q-Learning Agent")
    print("=" * 50)

    # Create Q-learning agent with in-memory database
    q_agent = QLearningAgent("demo_q_agent", {"db_path": ":memory:"})

    # Create initial state
    state = State(
        prompt_type="code_generation",
        complexity_level="medium",
        previous_action=None,
        iteration_count=0,
    )

    print(f"\n📊 Initial State: {state.to_dict()}")
    print(f"Available Actions: {q_agent.config['actions']}")
    print(f"Exploration Rate (epsilon): {q_agent.config['epsilon']:.3f}")

    # Step 1: Select action
    action = q_agent.select_action(state)
    print(f"\n🎯 Selected Action: {action.action_type}")
    print(f"Action Parameters: {action.parameters}")

    # Step 2: Simulate next state
    next_state = State(
        prompt_type="code_generation",
        complexity_level="medium",
        previous_action=action.action_type,
        iteration_count=1,
    )

    # Step 3: Calculate reward (simulate)
    reward = 0.7  # Simulate good reward

    # Step 4: Update Q-value
    q_agent.update_q(state, action, reward, next_state)

    print("📈 Updated Q-value for state-action pair")
    print(f"Reward received: {reward}")
    print(f"New epsilon: {q_agent.config['epsilon']:.3f}")

    # Step 5: Show learning statistics
    stats = q_agent.get_learning_statistics()
    print("\n📊 Learning Statistics:")
    print(f"Episode count: {stats['episode_count']}")
    print(f"Total rewards: {stats['total_rewards']:.3f}")
    print(f"Average reward: {stats['average_reward']:.3f}")
    print(f"Q-table entries: {stats['total_q_entries']}")


def demo_agent_integration():
    """Demonstrate how agents work together."""
    print("\n🤝 DEMO: Agent Integration")
    print("=" * 50)

    # Create agents
    reward_agent = RewardAgent("integration_reward_agent")
    q_agent = QLearningAgent("integration_q_agent", {"db_path": ":memory:"})

    # Simulate a complete workflow
    print("\n🔄 Complete Workflow:")

    # Step 1: Analyze context
    context = {
        "prompt_type": "code_generation",
        "complexity": "medium",
        "previous_action": None,
        "iteration_count": 0,
    }

    analysis = q_agent.analyze(context)
    print(f"1. Context Analysis: {analysis['current_state']}")

    # Step 2: Propose action
    proposal = q_agent.propose(analysis)
    print(f"2. Action Proposal: {proposal['action']['action_type']}")
    print(f"   Confidence: {proposal['confidence']:.3f}")
    print(f"   Exploration: {proposal['exploration']}")

    # Step 3: Execute action and get results
    state = State(**analysis["current_state"])
    action = Action(**proposal["action"])

    # Simulate results
    test_results = TestResults(
        passed=8, failed=2, total=10, execution_time=4.0, coverage=0.8, lint_score=8.5
    )
    code_metrics = CodeMetrics(
        complexity=7.0,
        lines_of_code=80,
        function_count=4,
        class_count=1,
        comment_ratio=0.15,
    )
    policy_violations = ["missing_docstring"]
    human_feedback = {
        "thumbs_up": 2,
        "thumbs_down": 0,
        "rating": 0.8,
        "comments": ["Good work"],
    }

    # Step 4: Calculate reward
    reward_result = reward_agent.calculate_reward(
        test_results=test_results,
        code_metrics=code_metrics,
        policy_violations=policy_violations,
        human_feedback=human_feedback,
    )

    print("3. Results:")
    print(f"   Test pass rate: {test_results.pass_rate:.1%}")
    print(f"   Code complexity: {code_metrics.complexity}")
    print(f"   Policy violations: {len(policy_violations)}")
    print(f"   Human rating: {human_feedback['rating']:.1f}")

    # Step 5: Update Q-learning
    next_state = State(
        prompt_type="code_generation",
        complexity_level="medium",
        previous_action=action.action_type,
        iteration_count=1,
    )

    q_agent.update_q(state, action, reward_result["total_reward"], next_state)

    print("4. Learning Update:")
    print(f"   Reward: {reward_result['total_reward']:.3f}")
    print("   Q-value updated for state-action pair")
    print(f"   Episode count: {q_agent.episode_count}")


def main():
    """Run all demos."""
    print("🚀 CodeConductor v2.0 - RL Infrastructure Demo")
    print("=" * 60)

    try:
        # Demo 1: Reward calculation
        demo_reward_calculation()

        # Demo 2: Q-learning
        demo_q_learning()

        # Demo 3: Agent integration
        demo_agent_integration()

        print("\n✅ All demos completed successfully!")
        print("\n🎯 Key Features Demonstrated:")
        print("   • Reward calculation based on multiple metrics")
        print("   • Q-learning with epsilon-greedy exploration")
        print("   • State-action management and persistence")
        print("   • Agent integration and workflow")
        print("   • Learning statistics and monitoring")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
