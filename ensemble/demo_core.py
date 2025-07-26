#!/usr/bin/env python3
"""
Demo script for CodeConductor Core Engine
Tests ModelManager and QueryDispatcher together in a "Hello, world!" smoke test.
"""

import asyncio
import json
import sys
import os

# Add the parent directory to the path so we can import ensemble modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ensemble.model_manager import ModelManager
from ensemble.query_dispatcher import QueryDispatcher


async def main():
    """Run a complete smoke test of the core ensemble engine."""
    print("🚀 CodeConductor Core Engine - Smoke Test")
    print("=" * 50)

    # Step 1: List healthy models
    print("\n📋 Step 1: Discovering healthy models...")
    mm = ModelManager()
    try:
        healthy_models = await mm.list_healthy_models()
        print(f"✅ Found {len(healthy_models)} healthy models:")
        for model_id in healthy_models:
            print(f"   - {model_id}")
    except Exception as e:
        print(f"❌ Error discovering models: {e}")
        return

    if not healthy_models:
        print("⚠️  No healthy models found. Make sure LM Studio or Ollama is running.")
        return

    # Step 2: Dispatch prompt to healthy models
    print(
        f"\n📤 Step 2: Dispatching 'Hello, world!' to {len(healthy_models)} models..."
    )
    prompt = "Hello, world! Please respond with a simple greeting."

    dispatcher = QueryDispatcher()
    try:
        responses = await dispatcher.dispatch_to_healthy_models(prompt)
        print(f"✅ Received responses from {len(responses)} models")
    except Exception as e:
        print(f"❌ Error dispatching queries: {e}")
        return

    # Step 3: Display raw JSON responses
    print(f"\n📄 Step 3: Raw responses from models:")
    print("-" * 50)

    for model_id, response in responses.items():
        print(f"\n--- {model_id} ---")
        if isinstance(response, dict) and "error" in response:
            print(f"❌ Error: {response['error']}")
        else:
            # Try to extract the actual response content
            try:
                if "choices" in response:
                    # LM Studio format
                    content = response["choices"][0]["message"]["content"]
                    print(f"✅ Response: {content}")
                elif "response" in response:
                    # Ollama format
                    print(f"✅ Response: {response['response']}")
                else:
                    print(f"📄 Raw JSON: {json.dumps(response, indent=2)}")
            except (KeyError, IndexError) as e:
                print(f"📄 Raw JSON: {json.dumps(response, indent=2)}")

    print(f"\n🎉 Smoke test completed!")
    print(f"📊 Summary: {len(responses)}/{len(healthy_models)} models responded")


if __name__ == "__main__":
    asyncio.run(main())
