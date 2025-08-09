#!/usr/bin/env python3
"""
Test vLLM Integration in WSL2

Simple test script to verify vLLM functionality in WSL2 environment.
"""

import asyncio
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


async def test_vllm():
    """Test vLLM integration."""
    try:
        print("🚀 Testing vLLM Integration in WSL2...")

        # Import vLLM integration
        from codeconductor.vllm_integration import create_vllm_engine

        print("📦 Creating vLLM engine...")
        engine = await create_vllm_engine(
            model_name="microsoft/DialoGPT-medium", quantization="awq"
        )

        print("🔧 Engine info:")
        info = engine.get_model_info()
        for key, value in info.items():
            print(f"  {key}: {value}")

        print("\n🧪 Testing code generation...")
        prompt = "Write a Python function to calculate the factorial of a number:"
        result = await engine.generate_code(prompt)

        print(f"Generated code:\n{result}")

        print("\n🎯 Testing consensus generation...")
        consensus_result = await engine.generate_with_consensus(prompt)
        print(f"Consensus metrics: {consensus_result['consensus_metrics']}")

        await engine.cleanup()
        print("✅ vLLM test completed successfully!")

    except ImportError as e:
        print(f"❌ vLLM not available: {e}")
        print("💡 Make sure vLLM is installed in WSL2 environment")
    except Exception as e:
        print(f"❌ vLLM test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_vllm())
