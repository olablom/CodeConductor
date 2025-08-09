#!/usr/bin/env python3
"""
Test script for CodeConductor Shared-Model Multi-Agent Debate

Tests the multi-agent debate system using a shared model instance, preventing VRAM explosion.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from codeconductor.debate.shared_model_agent import SharedModelAIAgent, SharedModelDebateManager
from codeconductor.ensemble.single_model_engine import SingleModelEngine


async def test_shared_model_debate():
    """Test the debate system with multiple agents sharing one model"""

    print("🧪 Testing CodeConductor Shared-Model Multi-Agent Debate")
    print("=" * 60)

    # Create ONE shared engine for all agents
    shared_engine = SingleModelEngine("meta-llama-3.1-8b-instruct")
    
    # Initialize the shared engine once
    print("🚀 Initializing shared model engine...")
    success = await shared_engine.initialize()
    if not success:
        print("❌ Failed to initialize shared model engine")
        return False
    
    print("✅ Shared model engine initialized successfully")

    # Define code-focused agent personas (all sharing the same engine)
    agents = [
        SharedModelAIAgent(
            "Architect",
            "You are Architect – an AI assistant that focuses on system design and architecture. "
            "You prioritize clean code structure, scalability, and maintainability. "
            "You think about the big picture and how different components fit together. "
            "Always provide practical, implementable solutions.",
            shared_engine
        ),
        SharedModelAIAgent(
            "Coder",
            "You are Coder – an AI assistant that focuses on implementation and code quality. "
            "You prioritize efficient algorithms, best practices, and readable code. "
            "You think about the details and how to make code work well. "
            "Always provide working, tested code examples.",
            shared_engine
        ),
        SharedModelAIAgent(
            "Tester",
            "You are Tester – an AI assistant that focuses on edge cases and potential bugs. "
            "You prioritize robust error handling, edge case coverage, and code reliability. "
            "You think about what could go wrong and how to prevent it. "
            "Always consider error handling and edge cases.",
            shared_engine
        ),
        SharedModelAIAgent(
            "Reviewer",
            "You are Reviewer – an AI assistant that focuses on code review and best practices. "
            "You prioritize code standards, documentation, and maintainability. "
            "You think about code quality and how others will understand the code. "
            "Always suggest improvements and best practices.",
            shared_engine
        ),
    ]

    # Create debate manager with shared engine
    debate_manager = SharedModelDebateManager(agents, shared_engine)

    # Test prompt
    test_prompt = "Create a simple REST API endpoint that accepts a JSON payload and returns a response. Use Python with Flask."

    print(f"📝 Test Prompt: {test_prompt}")
    print("=" * 60)

    try:
        # Conduct debate with timeout
        result = await asyncio.wait_for(
            debate_manager.conduct_debate(test_prompt),
            timeout=120.0  # 2 minute timeout for full debate
        )

        print("\n" + "=" * 60)
        print("✅ Debate completed successfully!")
        print(f"📊 Total turns: {result['total_turns']}")
        print(f"🤖 Agents: {', '.join(result['agents'])}")

        # Extract consensus
        consensus = debate_manager.extract_consensus()
        print("\n📋 Final Consensus:")
        print("-" * 40)
        print(consensus)

        return True

    except asyncio.TimeoutError:
        print("⏰ Debate timed out after 2 minutes")
        return False
    except Exception as e:
        print(f"❌ Error during debate: {e}")
        return False


if __name__ == "__main__":
    # Run test
    success = asyncio.run(test_shared_model_debate())
    
    if success:
        print("\n🎉 Shared-model debate test PASSED!")
    else:
        print("\n💥 Shared-model debate test FAILED!")
        sys.exit(1) 