#!/usr/bin/env python3
"""
Test CodeConductor with 3 specialized agents

Tests the debate system with 3 agents designed for CodeConductor AI development assistance.
"""

import asyncio
import sys
import yaml
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from codeconductor.debate.local_ai_agent import LocalAIAgent, LocalDebateManager
from codeconductor.ensemble.single_model_engine import SingleModelEngine


async def main():
    """Test CodeConductor with 3 specialized agents"""

    print("🧪 Testing CodeConductor - 3 Specialized Agents")
    print("=" * 55)

    # Create shared engine
    print("🚀 Initializing shared model engine...")
    shared_engine = SingleModelEngine("meta-llama-3.1-8b-instruct")
    await shared_engine.initialize()
    print("✅ Shared model engine initialized successfully")

    # Create 3 specialized agents for CodeConductor
    agents = [
        LocalAIAgent(
            "Architect",
            "You are Architect – an AI development expert who focuses on system design, architecture patterns, and scalable solutions. You prioritize clean code structure, maintainability, and best practices. You think in terms of components, interfaces, and system integration. You don't worry about implementation details or specific frameworks.",
        ),
        LocalAIAgent(
            "Coder",
            "You are Coder – an AI development expert who focuses on practical implementation, working code, and getting things done. You prioritize functionality, performance, and user experience. You think in terms of features, bugs, and deployment. You don't worry about theoretical perfection or over-engineering.",
        ),
        LocalAIAgent(
            "Optimizer",
            "You are Optimizer – an AI development expert who focuses on performance, efficiency, and resource management. You prioritize speed, memory usage, and scalability. You think in terms of bottlenecks, algorithms, and optimization techniques. You don't worry about code readability or development speed.",
        ),
    ]

    # Set shared engine for all agents
    for agent in agents:
        agent.set_shared_engine(shared_engine)

    # Create debate manager
    debate = LocalDebateManager(agents)
    debate.set_shared_engine(shared_engine)

    # Test prompt for CodeConductor
    user_prompt = "How should we implement a real-time code analysis system that can detect potential bugs and suggest improvements while the user is typing?"
    print(f"📝 Using prompt: {user_prompt}")

    try:
        # Conduct the debate with shorter timeouts
        print("🎭 Starting CodeConductor debate...")
        debate_responses = await asyncio.wait_for(
            debate.conduct_debate(
                user_prompt, timeout_per_turn=120.0
            ),  # 2 minutes per turn
            timeout=600.0,  # 10 minutes total timeout
        )

        # Save transcript
        with open("codeconductor_debate.yaml", "w", encoding="utf-8") as f:
            yaml.dump(debate_responses, f, default_flow_style=False, allow_unicode=True)

        with open("codeconductor_debate.json", "w", encoding="utf-8") as f:
            json.dump(debate_responses, f, indent=2, ensure_ascii=False)

        print(
            "Debate transcript saved to codeconductor_debate.yaml and codeconductor_debate.json"
        )

        # Print summary
        print("\n📊 CodeConductor Debate Summary:")
        for response in debate_responses:
            print(
                f"  {response['agent']} ({response['turn']}): {response['content'][:100]}..."
            )

        return True

    except asyncio.TimeoutError:
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
        print("\n🎉 CodeConductor 3-agent debate test PASSED!")
    else:
        print("\n💥 CodeConductor 3-agent debate test FAILED!")
        sys.exit(1)
