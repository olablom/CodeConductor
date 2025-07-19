#!/usr/bin/env python3
"""
Test script for multi-agent system.
"""

from agents.orchestrator import AgentOrchestrator


def test_multi_agent():
    """Test the multi-agent discussion system."""

    print("🚀 Testing Multi-Agent Discussion System")
    print("=" * 50)

    # Skapa orchestrator
    orchestrator = AgentOrchestrator()

    # Test prompt
    prompt = "Write a Python function that calculates the factorial of a number"

    # Facilitera diskussion
    result = orchestrator.facilitate_discussion(prompt)

    # Visa resultat
    print("\n🎯 Final Proposal:")
    print(f"Proposal ID: {result['proposal_id']}")
    print(f"Approach: {result['approach']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Patterns: {result['patterns']}")
    print(f"Risks: {result['risks']}")
    print(f"RL Score: {result['rl_score']:.2f}")
    print(f"Optimization: {result['optimization']}")

    # Visa agent summary
    summary = orchestrator.get_agent_summary()
    print(f"\n🤖 Agent Summary:")
    print(f"Total Agents: {summary['total_agents']}")
    for name, agent_info in summary["agents"].items():
        print(f"  - {agent_info['name']}: {agent_info['role']}")


if __name__ == "__main__":
    test_multi_agent()
