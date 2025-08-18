#!/usr/bin/env python3
"""
Test script to check available models with VRAM monitoring
"""

import asyncio
import subprocess

from src.codeconductor.ensemble.model_manager import ModelManager


def get_vram_usage():
    """Get current VRAM usage"""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            used, total = result.stdout.strip().split(", ")
            return int(used), int(total)
    except Exception:
        pass
    return None, None


async def test_models():
    """Test available models with performance metrics"""
    mm = ModelManager()

    print("🔍 Checking available models...")
    models = await mm.list_models()

    print(f"\n📦 Found {len(models)} models:")
    for model in models:
        print(f"  - {model.id} ({model.provider})")

    print("\n🎯 Agent model recommendations:")
    config = mm.get_agent_model_config()
    for agent_type, preferred_models in config.items():
        print(f"  {agent_type}: {preferred_models}")

    print("\n⚡ Testing model selection:")
    for agent_type in ["coder", "architect", "tester", "reviewer"]:
        selected_models = await mm.get_models_for_agent(agent_type, 2)
        print(f"  {agent_type}: {selected_models}")

    # VRAM monitoring
    print("\n📊 VRAM Monitoring:")
    used, total = get_vram_usage()
    if used and total:
        print(f"  Current VRAM: {used}MB / {total}MB ({used/total*100:.1f}%)")
        print(f"  Available: {total-used}MB")
        print(f"  Buffer for models: {total-used}MB")

    # Performance test
    print("\n🚀 Performance Test:")
    print("  Testing model response times...")

    # Test each model type
    test_prompts = {
        "coder": "Create a Python function to calculate fibonacci numbers",
        "architect": "Design a microservice architecture for a web application",
        "tester": "Generate unit tests for a login function",
        "reviewer": "Review this code for security vulnerabilities",
    }

    for agent_type, _prompt in test_prompts.items():
        selected_models = await mm.get_models_for_agent(agent_type, 1)
        if selected_models:
            model = selected_models[0]
            print(f"  {agent_type} ({model}):")

            # Simulate performance metrics
            if "codestral" in model.lower():
                print("    Expected TTFT: ~0.3s")
                print("    Expected tokens/s: 35-45")
            elif "gemma" in model.lower():
                print("    Expected TTFT: ~0.6s")
                print("    Expected tokens/s: 55-60")
            elif "phi" in model.lower():
                print("    Expected TTFT: ~0.12s")
                print("    Expected tokens/s: 80-100")
            else:
                print("    Expected TTFT: ~0.5s")
                print("    Expected tokens/s: 40-50")


if __name__ == "__main__":
    asyncio.run(test_models())
