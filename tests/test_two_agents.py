#!/usr/bin/env python3
"""
Test with 2 agents

Tests the debate system with just 2 agents to verify it works.
"""

import asyncio
import json
import sys
from pathlib import Path

import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from codeconductor.debate.local_ai_agent import LocalAIAgent, LocalDebateManager
from codeconductor.ensemble.single_model_engine import SingleModelEngine


async def main():
    """Test with 2 agents"""

    print("🧪 Testing 2 Agents Debate")
    print("=" * 40)

    # Create shared engine
    print("🚀 Initializing shared model engine...")
    shared_engine = SingleModelEngine("meta-llama-3.1-8b-instruct")
    await shared_engine.initialize()
    print("✅ Shared model engine initialized successfully")

    # Create 2 agents
    agents = [
        LocalAIAgent(
            "PortfolioBuilder",
            "You are PortfolioBuilder – an AI assistant that helps students create project ideas that will look impressive in their GitHub portfolios. You prioritize visual impact, modern frontend/backend tools, and high 'wow'-factor. You don't worry about deadlines or simplicity.",
        ),
        LocalAIAgent(
            "RealismExpert",
            "You are RealismExpert – an AI assistant that helps students create project ideas that are realistic, feasible, and deliverable within a limited timeframe. You focus on MVP (Minimum Viable Product), simplicity, and ensuring the project can be completed on time. You don't worry about trendiness or maximum technical depth.",
        ),
    ]

    # Set shared engine for all agents
    for agent in agents:
        agent.set_shared_engine(shared_engine)

    # Create debate manager
    debate = LocalDebateManager(agents)
    debate.set_shared_engine(shared_engine)

    # Test prompt
    user_prompt = "What should my next AI project be for my AI1 course?"
    print(f"📝 Using prompt: {user_prompt}")

    try:
        # Conduct the debate with shorter timeouts
        print("🎭 Starting 2-agent debate...")
        debate_responses = await asyncio.wait_for(
            debate.conduct_debate(
                user_prompt, timeout_per_turn=120.0
            ),  # 2 minutes per turn
            timeout=600.0,  # 10 minutes total timeout
        )

        # Save transcript
        with open("two_agents_debate.yaml", "w", encoding="utf-8") as f:
            yaml.dump(debate_responses, f, default_flow_style=False, allow_unicode=True)

        with open("two_agents_debate.json", "w", encoding="utf-8") as f:
            json.dump(debate_responses, f, indent=2, ensure_ascii=False)

        print(
            "Debate transcript saved to two_agents_debate.yaml and two_agents_debate.json"
        )

        # Print summary
        print("\n📊 Debate Summary:")
        for response in debate_responses:
            print(
                f"  {response['agent']} ({response['turn']}): {response['content'][:100]}..."
            )

        return True

    except TimeoutError:
        print("⏰ Debate timed out - this can happen with local models")
        return False
    except Exception as e:
        print(f"❌ Debate failed: {e}")
        return False
    finally:
        # Cleanup
        await shared_engine.cleanup()


if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        print("\n🎉 2-agent debate test PASSED!")
    else:
        print("\n💥 2-agent debate test FAILED!")
        sys.exit(1)
