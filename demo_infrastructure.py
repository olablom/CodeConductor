#!/usr/bin/env python3
"""
Demo script for CodeConductor v2.0 Infrastructure

This script demonstrates the BaseAgent and LLMClient working together
to show the foundation we've built for the AI Tech Lead project.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.base_agent import BaseAgent
from integrations.llm_client import LLMClient, create_llm_client
from typing import Dict, Any


class DemoAgent(BaseAgent):
    """
    Demo agent that implements the BaseAgent interface.

    This agent demonstrates how to use the LLM client for analysis,
    proposal generation, and code review.
    """

    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the context using LLM."""
        if not self.llm_client:
            return {"error": "No LLM client configured"}

        prompt = f"Analyze this context and provide insights: {context}"
        response = self.llm_client.complete(prompt)

        return {
            "insights": [response],
            "complexity": "medium",
            "estimated_time": "2 hours",
            "agent": self.name,
        }

    def propose(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a proposal based on analysis."""
        if not self.llm_client:
            return {"error": "No LLM client configured"}

        prompt = f"Based on this analysis, propose a solution: {analysis}"
        response = self.llm_client.complete(prompt)

        return {
            "solution": response,
            "approach": "LLM-driven approach",
            "confidence": 0.85,
            "agent": self.name,
        }

    def review(self, code: str) -> Dict[str, Any]:
        """Review code for quality and safety."""
        if not self.llm_client:
            return {"error": "No LLM client configured"}

        prompt = f"Review this code for quality and safety: {code}"
        response = self.llm_client.complete(prompt)

        return {
            "quality_score": 0.9,
            "issues": [],
            "recommendations": [response],
            "agent": self.name,
        }


def demo_agent_workflow():
    """Demonstrate a complete agent workflow."""
    print("🚀 CodeConductor v2.0 - Infrastructure Demo")
    print("=" * 50)

    # Create LLM client
    print("1. Creating LLM client...")
    llm_client = create_llm_client("mock")
    print(f"   ✅ LLM Client: {llm_client}")

    # Create demo agent
    print("\n2. Creating demo agent...")
    agent = DemoAgent("demo_agent", {"temperature": 0.7})
    agent.set_llm_client(llm_client)
    print(f"   ✅ Agent: {agent}")
    print(f"   📊 Status: {agent.get_status()}")

    # Demo workflow
    print("\n3. Running complete workflow...")

    # Step 1: Analyze context
    print("\n   📋 Step 1: Analyzing context...")
    context = {
        "task": "Create a REST API for user management",
        "language": "Python",
        "framework": "FastAPI",
        "requirements": ["CRUD operations", "Authentication", "Validation"],
    }
    analysis = agent.analyze(context)
    print(f"   📊 Analysis: {analysis}")

    # Step 2: Propose solution
    print("\n   💡 Step 2: Proposing solution...")
    proposal = agent.propose(analysis)
    print(f"   🎯 Proposal: {proposal}")

    # Step 3: Review code
    print("\n   🔍 Step 3: Reviewing code...")
    sample_code = """
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class User(BaseModel):
    id: int
    name: str
    email: str

@app.get("/users/{user_id}")
def get_user(user_id: int):
    return {"user_id": user_id, "name": "John Doe"}
    """
    review = agent.review(sample_code)
    print(f"   ✅ Review: {review}")

    print("\n🎉 Demo completed successfully!")
    return True


def demo_llm_client_features():
    """Demonstrate LLM client features."""
    print("\n🔧 LLM Client Features Demo")
    print("=" * 30)

    # Create client
    client = LLMClient("http://localhost:1234", "demo-model")

    # Test different prompt types
    prompts = [
        "hello world",
        "generate code for a calculator",
        "analyze this algorithm",
        "review this function",
    ]

    print("Testing different prompt types:")
    for i, prompt in enumerate(prompts, 1):
        response = client.complete(prompt)
        print(f"   {i}. '{prompt}' → {response[:60]}...")

    # Test caching
    print(f"\nCache stats: {client.get_cache_stats()}")

    # Test performance
    print("\nTesting caching performance...")
    import time

    prompt = "performance test"
    start_time = time.time()
    response1 = client.complete(prompt)
    first_call = time.time() - start_time

    start_time = time.time()
    response2 = client.complete(prompt)
    second_call = time.time() - start_time

    print(f"   First call: {first_call:.3f}s")
    print(f"   Second call: {second_call:.3f}s")
    print(f"   Cached: {second_call < first_call}")

    return True


def demo_error_handling():
    """Demonstrate error handling."""
    print("\n⚠️ Error Handling Demo")
    print("=" * 25)

    # Test BaseAgent instantiation error
    print("1. Testing BaseAgent instantiation error...")
    try:
        BaseAgent("should_fail")
        print("   ❌ Should have failed!")
    except TypeError:
        print("   ✅ Correctly prevented instantiation of abstract class")

    # Test LLM client error handling
    print("\n2. Testing LLM client error handling...")
    from integrations.llm_client import LLMError

    try:
        error = LLMError("Test error message")
        print(f"   ✅ LLMError created: {error}")
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")

    return True


def main():
    """Run the complete infrastructure demo."""
    print("🎯 CodeConductor v2.0 - Infrastructure Demo")
    print("=" * 60)

    try:
        # Run demos
        demo_agent_workflow()
        demo_llm_client_features()
        demo_error_handling()

        print("\n" + "=" * 60)
        print("✅ ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\n🎯 What we've built:")
        print("   • BaseAgent abstract class with analyze/propose/review methods")
        print("   • LLMClient with caching and mock responses")
        print("   • Complete agent workflow demonstration")
        print("   • Error handling and validation")
        print("   • Integration between agents and LLM clients")

        print("\n🚀 Next steps:")
        print("   • Implement real LLM providers (Ollama, LM Studio)")
        print("   • Create specialized agents (CodeGen, Architect, Reviewer)")
        print("   • Add message bus for inter-agent communication")
        print("   • Implement RL components for learning")

        return True

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
