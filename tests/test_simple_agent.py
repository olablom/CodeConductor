#!/usr/bin/env python3
"""
Very simple test script for CodeConductor

Tests basic agent functionality without timeouts.
"""

import asyncio
import json
import sys
from pathlib import Path

import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from codeconductor.debate.local_ai_agent import LocalAIAgent
from codeconductor.ensemble.single_model_engine import SingleModelEngine


async def main():
    """Test a single agent without timeouts"""

    print("🧪 Testing CodeConductor Simple Agent")
    print("=" * 50)

    # Create shared engine
    print("🚀 Initializing shared model engine...")
    shared_engine = SingleModelEngine("meta-llama-3.1-8b-instruct")
    await shared_engine.initialize()
    print("✅ Shared model engine initialized successfully")

    # Create single agent
    agent = LocalAIAgent(
        "PortfolioBuilder",
        "You are PortfolioBuilder – an AI assistant that helps students create project ideas that will look impressive in their GitHub portfolios. You prioritize visual impact, modern frontend/backend tools, and high 'wow'-factor. You don't worry about deadlines or simplicity.",
    )
    agent.set_shared_engine(shared_engine)

    # Test prompt
    user_prompt = "Create a simple REST API endpoint that accepts a JSON payload and returns a response. Use Python with Flask."
    print(f"📝 Using prompt: {user_prompt}")

    try:
        # Generate response without timeout
        print("🤖 Generating response...")
        response = await agent.generate_response(
            user_prompt, timeout=300.0
        )  # 5 minutes

        print("\n✅ Response generated successfully!")
        print(f"📝 Response: {response[:500]}...")

        # Save response
        result = {"agent": agent.name, "prompt": user_prompt, "response": response}

        with open("simple_agent_test.yaml", "w", encoding="utf-8") as f:
            yaml.dump(result, f, default_flow_style=False, allow_unicode=True)

        with open("simple_agent_test.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print("📁 Response saved to simple_agent_test.yaml and simple_agent_test.json")
        return True

    except Exception as e:
        print(f"❌ Agent failed: {e}")
        return False
    finally:
        # Cleanup
        await shared_engine.cleanup()


if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        print("\n🎉 Simple agent test PASSED!")
    else:
        print("\n💥 Simple agent test FAILED!")
        sys.exit(1)
