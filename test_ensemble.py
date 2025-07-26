#!/usr/bin/env python3
"""
Test script for CodeConductor Ensemble Engine
"""

import asyncio
import logging
import json
from ensemble import EnsembleEngine, EnsembleRequest

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


import pytest

@pytest.mark.asyncio
async def test_ensemble_engine():
    """Test the ensemble engine with a sample task."""

    print("🚀 Testing CodeConductor Ensemble Engine")
    print("=" * 50)

    # Initialize ensemble engine
    engine = EnsembleEngine(min_confidence=0.6)

    print("📡 Initializing ensemble engine...")
    success = await engine.initialize()

    if not success:
        print("❌ Failed to initialize ensemble engine")
        print("Make sure you have Ollama or LM Studio running with models loaded")
        return

    # Get model status
    status = engine.get_model_status()
    print(
        f"✅ Ensemble engine initialized with {status['online_models']} online models"
    )

    for model_id, model_info in status["models"].items():
        print(
            f"  - {model_id}: {model_info['status']} (success rate: {model_info['success_rate']:.2f})"
        )

    # Test with a sample task
    test_task = """
    Create a simple Python function that calculates the factorial of a number.
    The function should:
    1. Take an integer as input
    2. Return the factorial
    3. Handle edge cases (negative numbers, zero)
    4. Include proper error handling
    """

    print(f"\n🎯 Testing with task: {test_task[:50]}...")

    request = EnsembleRequest(
        task_description=test_task,
        context={"project_type": "python", "coding_style": "clean and documented"},
        min_models=2,
        timeout=30.0,
    )

    response = await engine.process_request(request)

    print(f"\n📊 Results:")
    print(f"  - Confidence: {response.confidence:.2f}")
    print(f"  - Execution time: {response.execution_time:.2f}s")
    print(f"  - Models used: {len(response.model_responses)}")

    if response.consensus:
        print(f"\n🎯 Consensus:")
        print(json.dumps(response.consensus, indent=2))

    if response.disagreements:
        print(f"\n⚠️  Disagreements:")
        for disagreement in response.disagreements:
            print(f"  - {disagreement}")

    # Show individual model responses
    print(f"\n🤖 Individual Model Responses:")
    for result in response.model_responses:
        status = "✅" if result.success else "❌"
        print(f"  {status} {result.model_id}: {result.response_time:.2f}s")
        if result.success and result.response:
            try:
                parsed = json.loads(result.response)
                print(f"    Approach: {parsed.get('approach', 'N/A')[:50]}...")
                print(f"    Complexity: {parsed.get('complexity', 'N/A')}")
            except:
                print(f"    Response: {result.response[:100]}...")


if __name__ == "__main__":
    asyncio.run(test_ensemble_engine())
