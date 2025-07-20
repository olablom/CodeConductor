#!/usr/bin/env python3
"""
Test script for multi-agent system.
"""

from agents.orchestrator import AgentOrchestrator
from agents.codegen_agent import CodeGenAgent
from agents.architect_agent import ArchitectAgent
from agents.review_agent import ReviewAgent


def test_multi_agent():
    """Test the multi-agent discussion system."""

    print("🚀 Testing Multi-Agent Discussion System")
    print("=" * 50)

    # Skapa agents och orchestrator
    codegen_agent = CodeGenAgent()
    architect_agent = ArchitectAgent()
    review_agent = ReviewAgent()

    orchestrator = AgentOrchestrator([codegen_agent, architect_agent, review_agent])

    # Test prompt
    prompt = "Write a Python function that calculates the factorial of a number"

    # Facilitera diskussion
    task_context = {"task_type": "code_generation", "prompt": prompt}
    result = orchestrator.run_discussion(task_context)
    proposal = result.get("consensus", {})

    # Visa resultat
    print("\n🎯 Final Proposal:")
    if proposal:
        print(f"Approach: {proposal.get('approach', 'Unknown')}")
        print(f"Confidence: {proposal.get('confidence', 0):.2f}")
        print(f"Status: {proposal.get('status', 'Unknown')}")
    else:
        print("No consensus reached")

    # Visa agent statistics
    stats = orchestrator.get_agent_statistics()
    print(f"\n🤖 Agent Statistics:")
    print(f"Total Agents: {stats['total_agents']}")
    print(f"Agent Names: {stats['agent_names']}")


if __name__ == "__main__":
    test_multi_agent()
