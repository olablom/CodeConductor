#!/usr/bin/env python3
"""
Test script for CodeConductor Debate System

Tests the multi-agent debate system with local models.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import pytest

from codeconductor.debate import CodeConductorDebateManager, LocalAIAgent


@pytest.mark.asyncio
async def test_debate_system():
    """Test the debate system with a simple coding task"""

    print("🧪 Testing CodeConductor Debate System")
    print("=" * 50)

    # Define code-focused agent personas
    agents = [
        LocalAIAgent(
            "Architect",
            "You are Architect – an AI assistant that focuses on system design and architecture. "
            "You prioritize clean code structure, scalability, and maintainability. "
            "You think about the big picture and how different components fit together.",
        ),
        LocalAIAgent(
            "Coder",
            "You are Coder – an AI assistant that focuses on implementation and code quality. "
            "You prioritize efficient algorithms, best practices, and readable code. "
            "You think about the details and how to make code work well.",
        ),
        # Reduced to 2 agents to prevent overload
        # LocalAIAgent(
        #     "Tester",
        #     "You are Tester – an AI assistant that focuses on edge cases and potential bugs. "
        #     "You prioritize robust error handling, edge case coverage, and code reliability. "
        #     "You think about what could go wrong and how to prevent it.",
        # ),
        # LocalAIAgent(
        #     "Reviewer",
        #     "You are Reviewer – an AI assistant that focuses on code review and best practices. "
        #     "You prioritize code standards, documentation, and maintainability. "
        #     "You think about code quality and how others will understand the code.",
        # ),
    ]

    # Create debate manager
    debate_manager = CodeConductorDebateManager(agents)

    # Test prompt
    test_prompt = "Create a simple REST API endpoint that accepts a JSON payload and returns a response. Use Python with Flask."

    print(f"📝 Test Prompt: {test_prompt}")
    print("=" * 50)

    try:
        # Conduct debate with timeout
        result = await asyncio.wait_for(
            debate_manager.conduct_debate(test_prompt),
            timeout=60.0,  # 60 second timeout
        )

        print("\n" + "=" * 50)
        print("✅ Debate completed successfully!")
        print(f"📊 Total turns: {result['total_turns']}")
        print(f"🤖 Agents: {', '.join(result['agents'])}")

        # Save transcript
        debate_manager.save_transcript("test_debate_transcript.yaml")

        # Extract consensus
        consensus = debate_manager.extract_consensus()
        print("\n📋 Final Consensus:")
        print("-" * 30)
        print(consensus)

        return True

    except TimeoutError:
        print("⏰ Debate timed out after 60 seconds")
        return False
    except Exception as e:
        print(f"❌ Error during debate: {e}")
        return False
    finally:
        # Cleanup: Unload all models and free memory
        try:
            print("\n🧹 Cleaning up resources...")

            # Get the ensemble engine from the first agent
            if hasattr(debate_manager, "agents") and debate_manager.agents:
                first_agent = debate_manager.agents[0]
                if (
                    hasattr(first_agent, "ensemble_engine")
                    and first_agent.ensemble_engine
                ):
                    await first_agent.ensemble_engine.cleanup()
                    print("✅ Ensemble engine cleanup completed")

            print("✅ Cleanup completed")

        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")


if __name__ == "__main__":
    # Run test
    success = asyncio.run(test_debate_system())

    if success:
        print("\n🎉 Debate system test PASSED!")
    else:
        print("\n💥 Debate system test FAILED!")
        sys.exit(1)
