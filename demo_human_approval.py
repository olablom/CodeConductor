#!/usr/bin/env python3
"""
Demo script for Human Approval CLI

This script demonstrates the human approval workflow with a sample
consensus proposal from the multi-agent discussion.
"""

import sys
import os
import json
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cli.human_approval import HumanApprovalCLI, ApprovalResult


def create_demo_proposal():
    """Create a realistic demo proposal for testing."""
    return {
        "title": "Build Microservice Architecture for E-commerce Platform",
        "summary": "Create a scalable microservice architecture with user management, product catalog, order processing, and payment services using FastAPI, Docker, and RabbitMQ.",
        "consensus_reached": True,
        "discussion_rounds": 3,
        "consensus": {
            "decision": "approve",
            "confidence": 0.92,
            "reasoning": "All agents agree this is a well-architected solution that follows microservice best practices. The technology stack is appropriate for the requirements, and the proposed structure allows for scalability and maintainability.",
        },
        "discussion_summary": {
            "agent_agreements": 3,
            "key_points": [
                "Use FastAPI for high-performance API development",
                "Implement Docker containers for easy deployment",
                "Use RabbitMQ for reliable message queuing",
                "Separate concerns into distinct microservices",
                "Include comprehensive error handling and logging",
                "Implement proper authentication and authorization",
                "Use PostgreSQL for data persistence",
                "Add health checks and monitoring",
            ],
            "concerns": [
                "Need to define specific API contracts between services",
                "Consider implementing circuit breakers for resilience",
                "Plan for database migration strategy",
                "Define monitoring and alerting requirements",
            ],
        },
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "agent_names": ["CodeGenAgent", "ArchitectAgent", "ReviewerAgent"],
            "consensus_strategy": "weighted_majority",
            "discussion_duration": "45 seconds",
            "total_agent_responses": 9,
        },
    }


def demo_cli_workflow():
    """Demonstrate the complete CLI workflow."""
    print("🎯 CodeConductor v2.0 - Human Approval Demo")
    print("=" * 60)

    # Create CLI instance
    cli = HumanApprovalCLI()

    # Create demo proposal
    proposal = create_demo_proposal()

    print("📋 Sample Consensus Proposal Created")
    print(f"   Title: {proposal['title']}")
    print(f"   Consensus: {proposal['consensus']['decision']}")
    print(f"   Confidence: {proposal['consensus']['confidence']:.2f}")
    print(f"   Agents: {', '.join(proposal['metadata']['agent_names'])}")

    print("\n🚀 Starting Human Approval Workflow...")
    print("   (This will show the full CLI interface)")

    # Process the approval (this will be interactive)
    try:
        result = cli.process_approval(proposal)

        # Display results
        print("\n" + "=" * 60)
        print("📊 APPROVAL RESULTS")
        print("=" * 60)
        print(f"Decision: {result.user_decision}")
        print(f"Approved: {result.approved}")
        print(f"Timestamp: {result.timestamp}")
        if result.comments:
            print(f"Comments: {result.comments}")

        # Show history
        history = cli.get_approval_history()
        print(f"\n📚 Approval History: {len(history)} entries")

        return result.approved

    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user.")
        return False
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        return False


def demo_non_interactive():
    """Demonstrate CLI features without user interaction."""
    print("🎯 CodeConductor v2.0 - Non-Interactive Demo")
    print("=" * 60)

    cli = HumanApprovalCLI()
    proposal = create_demo_proposal()

    # Display the proposal
    print("📋 Displaying proposal structure:")
    cli.display_proposal(proposal)

    # Show help
    print("\n📖 Showing help information:")
    cli.show_help()

    # Test history functionality
    print("\n📚 Testing history functionality:")
    result = ApprovalResult(
        approved=True,
        proposal=proposal,
        user_decision="approve",
        timestamp=datetime.now(),
        comments="Demo approval",
    )

    cli.save_approval_history(result)
    history = cli.get_approval_history()
    print(f"   History entries: {len(history)}")
    print(f"   Latest decision: {history[0]['decision']}")

    print("\n✅ Non-interactive demo completed!")


def main():
    """Main demo function."""
    import argparse

    parser = argparse.ArgumentParser(description="Human Approval CLI Demo")
    parser.add_argument(
        "--non-interactive", action="store_true", help="Run non-interactive demo"
    )
    parser.add_argument(
        "--save-proposal", type=str, help="Save demo proposal to JSON file"
    )

    args = parser.parse_args()

    if args.save_proposal:
        # Save demo proposal to file
        proposal = create_demo_proposal()
        with open(args.save_proposal, "w") as f:
            json.dump(proposal, f, indent=2, default=str)
        print(f"✅ Demo proposal saved to: {args.save_proposal}")
        return

    if args.non_interactive:
        demo_non_interactive()
    else:
        success = demo_cli_workflow()
        if success:
            print("\n🎉 Demo completed successfully!")
            print("   The proposal was approved and ready for code generation.")
        else:
            print("\n🛑 Demo completed - proposal was not approved.")

        print("\n💡 Try running with --non-interactive for a quick overview")
        print("   or --save-proposal filename.json to save a sample proposal")


if __name__ == "__main__":
    main()
