#!/usr/bin/env python3
"""
Final test script for CodeConductor Multi-Agent Debate

Tests the complete debate system with proper timeouts and error handling.
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
    """Test the complete multi-agent debate system"""

    print("🧪 Testing CodeConductor Multi-Agent Debate System")
    print("=" * 60)

    # Create shared engine
    print("🚀 Initializing shared model engine...")
    shared_engine = SingleModelEngine("meta-llama-3.1-8b-instruct")
    await shared_engine.initialize()
    print("✅ Shared model engine initialized successfully")

    # Define agent personas (same as your OpenAI version)
    personas = [
        (
            "PortfolioBuilder",
            "You are PortfolioBuilder – an AI assistant that helps students create project ideas that will look impressive in their GitHub portfolios. You prioritize visual impact, modern frontend/backend tools, and high 'wow'-factor. You don't worry about deadlines or simplicity.",
        ),
        (
            "GradeOptimizer",
            "You are GradeOptimizer – an AI assistant that helps students create project ideas that maximize their chances of getting the highest possible grade. You focus on technical depth, advanced methods, and fulfilling grading criteria. You don't worry about job market or visual appeal.",
        ),
        (
            "CareerCoach",
            "You are CareerCoach – an AI assistant that helps students create project ideas that maximize their employability and relevance for the job market. You focus on modern tools, frameworks, and skills that are in demand by employers. You don't worry about academic depth or visual wow-factor.",
        ),
        (
            "RealismExpert",
            "You are RealismExpert – an AI assistant that helps students create project ideas that are realistic, feasible, and deliverable within a limited timeframe. You focus on MVP (Minimum Viable Product), simplicity, and ensuring the project can be completed on time. You don't worry about trendiness or maximum technical depth.",
        ),
    ]

    # Create agents
    agents = []
    for name, persona in personas:
        agent = LocalAIAgent(name, persona)
        agent.set_shared_engine(shared_engine)
        agents.append(agent)

    # Create debate manager
    debate = LocalDebateManager(agents)
    debate.set_shared_engine(shared_engine)

    # Test prompt
    user_prompt = "Create a simple REST API endpoint that accepts a JSON payload and returns a response. Use Python with Flask."
    print(f"📝 Using prompt: {user_prompt}")

    try:
        # Conduct the debate with very long timeouts
        print("🎭 Starting multi-agent debate...")
        debate_responses = await asyncio.wait_for(
            debate.conduct_debate(
                user_prompt, timeout_per_turn=300.0
            ),  # 5 minutes per turn
            timeout=1800.0,  # 30 minutes total timeout
        )

        # Save transcript
        with open("final_debate_transcript.yaml", "w", encoding="utf-8") as f:
            yaml.dump(debate_responses, f, default_flow_style=False, allow_unicode=True)

        with open("final_debate_transcript.json", "w", encoding="utf-8") as f:
            json.dump(debate_responses, f, indent=2, ensure_ascii=False)

        print(
            "Debate transcript saved to final_debate_transcript.yaml and final_debate_transcript.json"
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
        print("\n🎉 Multi-agent debate test PASSED!")
    else:
        print("\n💥 Multi-agent debate test FAILED!")
        sys.exit(1)
