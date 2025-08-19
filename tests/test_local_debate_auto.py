#!/usr/bin/env python3
"""
Automated test script for CodeConductor Local Multi-Agent Debate

Tests the multi-agent debate system using local models, following the same structure as ai-project-advisor.
This version runs automatically without requiring manual input.
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
    """Main function following the same structure as ai-project-advisor/src/main.py"""

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

    # Create agents (same structure as OpenAI version)
    agents = [LocalAIAgent(name, persona) for name, persona in personas]

    # Create shared engine
    shared_engine = SingleModelEngine("meta-llama-3.1-8b-instruct")

    # Initialize the shared engine
    print("🚀 Initializing shared model engine...")
    success = await shared_engine.initialize()
    if not success:
        print("❌ Failed to initialize shared model engine")
        return False

    print("✅ Shared model engine initialized successfully")

    # Create debate manager and set shared engine
    debate = LocalDebateManager(agents)
    debate.set_shared_engine(shared_engine)

    # Use a predefined prompt instead of manual input
    user_prompt = "Create a simple REST API endpoint that accepts a JSON payload and returns a response. Use Python with Flask."
    print(f"📝 Using predefined prompt: {user_prompt}")

    try:
        # Conduct the debate with longer timeout
        print("🎭 Starting multi-agent debate...")
        debate_responses = await asyncio.wait_for(
            debate.conduct_debate(
                user_prompt, timeout_per_turn=180.0
            ),  # 3 minutes per turn
            timeout=600.0,  # 10 minutes total timeout
        )

        # Save transcript
        with open("local_debate_transcript.yaml", "w", encoding="utf-8") as f:
            yaml.dump(debate_responses, f, default_flow_style=False, allow_unicode=True)

        with open("local_debate_transcript.json", "w", encoding="utf-8") as f:
            json.dump(debate_responses, f, indent=2, ensure_ascii=False)

        print(
            "Debate transcript saved to local_debate_transcript.yaml and local_debate_transcript.json"
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
        if "shared_engine" in locals():
            await shared_engine.cleanup()


if __name__ == "__main__":
    # Run the main function
    success = asyncio.run(main())

    if success:
        print("\n🎉 Local debate test PASSED!")
    else:
        print("\n💥 Local debate test FAILED!")
        sys.exit(1)
