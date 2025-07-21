#!/usr/bin/env python3
"""
Pipeline Integration: Multi-Agent Discussion + Human Approval (Mock Version)

This script demonstrates the complete workflow using mock agents:
1. Mock multi-agent discussion and consensus
2. Human approval of the consensus proposal
3. Proceed to code generation (Fas 5)
"""

import sys
import os
import json
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cli.human_approval import HumanApprovalCLI


def create_mock_discussion_result(prompt: str, context: dict = None):
    """
    Create a realistic mock discussion result.

    Args:
        prompt: The initial prompt
        context: Additional context

    Returns:
        Mock discussion result
    """
    # Extract key info from prompt
    prompt_lower = prompt.lower()

    if "rest api" in prompt_lower or "api" in prompt_lower:
        title = "Build REST API for User Management"
        summary = "Create a comprehensive REST API with CRUD operations, authentication, and validation using FastAPI and Pydantic."
        key_points = [
            "Use FastAPI for high-performance API development",
            "Implement Pydantic models for data validation",
            "Add JWT authentication and authorization",
            "Include comprehensive error handling",
            "Add request/response logging",
            "Implement rate limiting",
            "Add API documentation with Swagger/OpenAPI",
        ]
        concerns = [
            "Define specific user fields and validation rules",
            "Plan for database schema evolution",
            "Consider security best practices",
        ]
    elif "microservice" in prompt_lower or "e-commerce" in prompt_lower:
        title = "Build Microservice Architecture for E-commerce Platform"
        summary = "Create a scalable microservice architecture with user management, product catalog, order processing, and payment services."
        key_points = [
            "Use FastAPI for high-performance API development",
            "Implement Docker containers for easy deployment",
            "Use RabbitMQ for reliable message queuing",
            "Separate concerns into distinct microservices",
            "Include comprehensive error handling and logging",
            "Implement proper authentication and authorization",
            "Use PostgreSQL for data persistence",
            "Add health checks and monitoring",
        ]
        concerns = [
            "Need to define specific API contracts between services",
            "Consider implementing circuit breakers for resilience",
            "Plan for database migration strategy",
            "Define monitoring and alerting requirements",
        ]
    else:
        title = f"Implement {prompt}"
        summary = f"Create a solution for: {prompt}"
        key_points = [
            "Use appropriate technology stack",
            "Follow best practices",
            "Include proper error handling",
            "Add comprehensive testing",
            "Consider scalability and maintainability",
        ]
        concerns = ["Define specific requirements", "Plan for future enhancements"]

    return {
        "title": title,
        "summary": summary,
        "consensus_reached": True,
        "discussion_rounds": 3,
        "consensus": {
            "decision": "approve",
            "confidence": 0.88,
            "reasoning": "All agents agree this is a well-defined, achievable task. The proposed approach addresses the requirements effectively and follows industry best practices.",
        },
        "discussion_summary": {
            "agent_agreements": 3,
            "key_points": key_points,
            "concerns": concerns,
        },
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "agent_names": ["CodeGenAgent", "ArchitectAgent", "ReviewerAgent"],
            "consensus_strategy": "weighted_majority",
            "discussion_duration": "45 seconds",
            "total_agent_responses": 9,
            "prompt": prompt,
            "context": context or {},
        },
    }


def run_complete_pipeline(prompt: str, context: dict = None):
    """
    Run the complete pipeline: Mock multi-agent discussion + Human approval.

    Args:
        prompt: The initial prompt for the agents
        context: Additional context for the agents

    Returns:
        True if approved and ready for code generation, False otherwise
    """
    print("🎯 CodeConductor v2.0 - Complete Pipeline Demo (Mock)")
    print("=" * 60)

    # Step 1: Mock multi-agent discussion
    print("\n🤖 STEP 1: Multi-Agent Discussion (Mock)")
    print("-" * 40)

    print(f"📝 Prompt: {prompt}")
    if context:
        print(f"📋 Context: {context}")

    print("\n🔄 Simulating agent discussion...")

    # Simulate processing time
    import time

    for i in range(3):
        print(f"   Round {i + 1}/3: Agents discussing...")
        time.sleep(0.5)

    discussion_result = create_mock_discussion_result(prompt, context)

    print("✅ Discussion completed!")
    print(f"   Consensus reached: {discussion_result.get('consensus_reached', False)}")
    print(f"   Discussion rounds: {discussion_result.get('discussion_rounds', 0)}")

    if discussion_result.get("consensus"):
        consensus = discussion_result["consensus"]
        print(f"   Decision: {consensus.get('decision', 'Unknown')}")
        print(f"   Confidence: {consensus.get('confidence', 0.0):.2f}")

    # Step 2: Human approval
    print("\n👤 STEP 2: Human Approval")
    print("-" * 40)

    cli = HumanApprovalCLI()

    try:
        approval_result = cli.process_approval(discussion_result)

        # Save to history
        cli.save_approval_history(approval_result)

        # Step 3: Decision and next steps
        print("\n🎯 STEP 3: Pipeline Decision")
        print("-" * 40)

        if approval_result.approved:
            print("✅ HUMAN APPROVAL GRANTED!")
            print("🚀 Ready to proceed to code generation (Fas 5)")
            print("\n📊 Pipeline Summary:")
            print(
                f"   • Agents reached consensus: {discussion_result.get('consensus_reached', False)}"
            )
            print(f"   • Human decision: {approval_result.user_decision}")
            print(
                f"   • Final proposal: {approval_result.proposal.get('title', 'Unknown')}"
            )
            if approval_result.comments:
                print(f"   • Human comments: {approval_result.comments}")

            return True
        else:
            print("❌ HUMAN APPROVAL DENIED!")
            print("🛑 Pipeline stopped - proposal rejected")
            if approval_result.comments:
                print(f"   • Rejection reason: {approval_result.comments}")

            return False

    except KeyboardInterrupt:
        print("\n\n👋 Pipeline interrupted by user.")
        return False
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        return False


def demo_simple_pipeline():
    """Demo with a simple prompt."""
    prompt = "Create a simple REST API for user management"
    context = {
        "language": "Python",
        "framework": "FastAPI",
        "requirements": ["CRUD operations", "Basic validation"],
    }

    return run_complete_pipeline(prompt, context)


def demo_complex_pipeline():
    """Demo with a complex prompt."""
    prompt = "Build a microservice architecture for an e-commerce platform"
    context = {
        "architecture": "microservices",
        "technologies": ["FastAPI", "Docker", "RabbitMQ", "PostgreSQL"],
        "requirements": [
            "User management",
            "Product catalog",
            "Order processing",
            "Payment integration",
            "Scalability",
            "High availability",
        ],
    }

    return run_complete_pipeline(prompt, context)


def demo_custom_pipeline():
    """Demo with custom prompt from user."""
    print("🎯 Custom Pipeline Demo")
    print("=" * 40)

    print("Enter your prompt for the agents:")
    prompt = input("> ").strip()

    if not prompt:
        print("❌ No prompt provided. Using default.")
        prompt = "Create a simple web application"

    print("\nEnter additional context (optional, press Enter to skip):")
    context_input = input("> ").strip()

    context = None
    if context_input:
        try:
            context = json.loads(context_input)
        except json.JSONDecodeError:
            print("⚠️  Invalid JSON context. Using as plain text.")
            context = {"description": context_input}

    return run_complete_pipeline(prompt, context)


def main():
    """Main pipeline demo function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="CodeConductor v2.0 Pipeline Demo (Mock)"
    )
    parser.add_argument(
        "--demo",
        choices=["simple", "complex", "custom"],
        default="simple",
        help="Demo type to run",
    )
    parser.add_argument("--prompt", type=str, help="Custom prompt for agents")
    parser.add_argument("--context", type=str, help="JSON context for agents")

    args = parser.parse_args()

    if args.prompt:
        # Custom prompt from command line
        context = None
        if args.context:
            try:
                context = json.loads(args.context)
            except json.JSONDecodeError:
                print(f"❌ Invalid JSON context: {args.context}")
                sys.exit(1)

        success = run_complete_pipeline(args.prompt, context)
    else:
        # Run demo based on choice
        if args.demo == "simple":
            success = demo_simple_pipeline()
        elif args.demo == "complex":
            success = demo_complex_pipeline()
        elif args.demo == "custom":
            success = demo_custom_pipeline()

    # Final result
    print("\n" + "=" * 60)
    if success:
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("   Ready for code generation phase.")
        print("\n💡 Next steps:")
        print("   • Implement code generation agents")
        print("   • Add RL feedback loop")
        print("   • Create production deployment")
        sys.exit(0)
    else:
        print("🛑 PIPELINE STOPPED")
        print("   Human approval was denied or pipeline failed.")
        print("\n💡 Suggestions:")
        print("   • Review the agent discussion")
        print("   • Modify the prompt or context")
        print("   • Try a different approach")
        sys.exit(1)


if __name__ == "__main__":
    main()
