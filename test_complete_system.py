#!/usr/bin/env python3
"""
Test script for complete CodeConductor system with multi-agent discussion and human approval.
"""

from agents.orchestrator import AgentOrchestrator
from integrations.human_gate import HumanGate


def test_complete_system(monkeypatch):
    """Test the complete system with multi-agent discussion and human approval."""

    # Mock user input to avoid stdin capture issues
    inputs = iter(["y", "n", "y"])  # Simulate user responses
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    print("🚀 Testing Complete CodeConductor System")
    print("=" * 60)

    # Skapa orchestrator och human gate
    orchestrator = AgentOrchestrator()
    human_gate = HumanGate()

    # Test prompt
    prompt = "Write a Python function that validates email addresses"

    print(f"📝 Input Prompt: {prompt}")
    print("=" * 60)

    # Steg 1: Multi-agent diskussion
    print("\n🤖 Step 1: Multi-Agent Discussion")
    proposal = orchestrator.facilitate_discussion(prompt)

    # Steg 2: Human approval
    print("\n👤 Step 2: Human-in-the-Loop Approval")
    approved, final_proposal = human_gate.request_approval(proposal)

    # Steg 3: Resultat
    print("\n📊 Step 3: Results")
    if approved:
        print("✅ System: Proposal approved and ready for implementation!")
        print(f"🎯 Final Approach: {final_proposal.get('approach', 'Unknown')}")
        print(f"📊 Confidence: {final_proposal.get('confidence', 0):.1%}")
    else:
        print("❌ System: Proposal rejected - will need revision")

    # Visa approval-statistik
    stats = human_gate.get_approval_stats()
    print(f"\n📈 Approval Statistics:")
    print(f"  - Total Decisions: {stats['total_decisions']}")
    print(f"  - Approved: {stats['approved']}")
    print(f"  - Rejected: {stats['rejected']}")
    print(f"  - Edited: {stats['edited']}")
    print(f"  - Approval Rate: {stats['approval_rate']:.1%}")


if __name__ == "__main__":
    test_complete_system()
