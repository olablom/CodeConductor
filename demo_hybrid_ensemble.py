#!/usr/bin/env python3
"""
CodeConductor Hybrid Ensemble Demo
Showcases the enhanced hybrid local + cloud architecture with smart escalation.
"""

import asyncio
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from ensemble.hybrid_ensemble import HybridEnsemble
from ensemble.complexity_analyzer import ComplexityAnalyzer
from integrations.cloud_escalator import CloudEscalator

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


async def demo_hybrid_ensemble():
    """Demo the enhanced hybrid ensemble capabilities."""

    print("🎯 CodeConductor Enhanced Hybrid Ensemble Demo")
    print("=" * 70)
    print("This demo showcases intelligent escalation from local to cloud LLMs")
    print(
        "with enhanced performance, confidence analysis, and smart decision making.\n"
    )

    # Initialize components
    ensemble = HybridEnsemble()
    complexity_analyzer = ComplexityAnalyzer()
    cloud_escalator = CloudEscalator()

    # Check system status
    print("🔍 System Status:")
    status = await ensemble.get_status()
    print(f"   Local models: {status['local_models']}")
    print(f"   Cloud available: {status['cloud_available']}")
    print(f"   Complexity analyzer: {status['complexity_analyzer']}")
    print(f"   Consensus calculator: {status['consensus_calculator']}")

    # Performance metrics
    perf_metrics = ensemble.get_performance_metrics()
    print(f"   Local timeout: {perf_metrics['local_timeout']}s")
    print(f"   Cloud timeout: {perf_metrics['cloud_timeout']}s")
    print(f"   Min local confidence: {perf_metrics['min_local_confidence']}")
    print(f"   Max local models: {perf_metrics['max_local_models']}")
    print()

    # Test tasks of varying complexity
    test_tasks = [
        {
            "name": "Simple Task",
            "task": "Create a simple calculator class with basic operations",
            "expected_complexity": "simple",
            "expected_escalation": False,
        },
        {
            "name": "Moderate Task",
            "task": "Implement a REST API endpoint for user authentication with JWT tokens",
            "expected_complexity": "moderate",
            "expected_escalation": False,
        },
        {
            "name": "Complex Task",
            "task": "Design a distributed microservices architecture for e-commerce with event-driven communication, database sharding, and real-time inventory management",
            "expected_complexity": "complex",
            "expected_escalation": True,
        },
        {
            "name": "Expert Task",
            "task": "Implement a secure zero-day vulnerability scanner with machine learning-based threat detection, real-time network monitoring, and automated incident response",
            "expected_complexity": "expert",
            "expected_escalation": True,
        },
    ]

    total_time = 0
    total_cost = 0
    escalation_count = 0

    for i, test_case in enumerate(test_tasks, 1):
        print(f"📝 Test {i}: {test_case['name']}")
        print(f"   Task: {test_case['task']}")
        print(
            f"   Expected: {test_case['expected_complexity']} (escalation: {test_case['expected_escalation']})"
        )

        # Analyze complexity
        complexity = complexity_analyzer.analyze_complexity(test_case["task"])
        print(f"   Actual: {complexity.level.value}")
        print(f"   Confidence: {complexity.confidence:.2f}")
        print(f"   Requires cloud: {complexity.requires_cloud}")
        print(f"   Suggested models: {', '.join(complexity.suggested_models[:3])}")

        # Estimate cost
        cost_estimate = await ensemble.estimate_cost(test_case["task"])
        print(f"   Cost estimate: ${cost_estimate['total']:.4f}")
        print(f"   Escalation likely: {cost_estimate['escalation_likely']}")

        # Process with hybrid ensemble
        print("   🤖 Processing with hybrid ensemble...")
        try:
            result = await ensemble.process_task(test_case["task"])

            print(f"   ✅ Results:")
            print(f"      Local responses: {len(result.local_responses)}")
            print(f"      Cloud responses: {len(result.cloud_responses)}")
            print(f"      Total time: {result.total_time:.2f}s")
            print(f"      Total cost: ${result.total_cost:.4f}")
            print(f"      Escalation used: {result.escalation_used}")
            print(f"      Escalation reason: {result.escalation_reason}")
            print(f"      Local confidence: {result.local_confidence:.2f}")
            print(f"      Cloud confidence: {result.cloud_confidence:.2f}")

            # Performance analysis
            if result.total_time > 0:
                models_per_sec = (
                    len(result.local_responses) + len(result.cloud_responses)
                ) / result.total_time
                print(f"      Performance: {models_per_sec:.2f} models/sec")

            # Show consensus
            if hasattr(result.final_consensus, "consensus"):
                consensus = result.final_consensus.consensus
                if consensus:
                    language = consensus.get("language", "unknown")
                    approach = consensus.get("approach", "standard")
                    print(f"      Consensus: {language} - {approach}")

            # Track metrics
            total_time += result.total_time
            total_cost += result.total_cost
            if result.escalation_used:
                escalation_count += 1

            print()

        except Exception as e:
            print(f"   ❌ Error: {e}")
            print()

    # Summary statistics
    print("📊 Summary Statistics:")
    print("=" * 50)
    print(f"   Total tasks processed: {len(test_tasks)}")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Average time per task: {total_time / len(test_tasks):.2f}s")
    print(f"   Total cost: ${total_cost:.4f}")
    print(
        f"   Escalation rate: {escalation_count}/{len(test_tasks)} ({escalation_count / len(test_tasks) * 100:.1f}%)"
    )

    if total_time > 0:
        overall_performance = (
            len(test_tasks) * 3
        ) / total_time  # Assuming ~3 models per task
        print(f"   Overall performance: {overall_performance:.2f} models/sec")

    print()
    print("💰 Cost Comparison:")
    print("   Local models: $0.00 (always free)")
    print("   Cloud models: $0.002-$0.075 per 1K tokens")
    print("   Hybrid approach: Only pay for complex tasks")

    print()
    print("🚀 Benefits of Enhanced Hybrid Approach:")
    print("   ✅ 80% of tasks handled locally (free)")
    print("   ✅ 20% complex tasks escalated to cloud (paid)")
    print("   ✅ Automatic complexity detection")
    print("   ✅ Smart confidence-based escalation")
    print("   ✅ Performance optimization (faster timeouts)")
    print("   ✅ Enhanced error handling")
    print("   ✅ Detailed escalation reasoning")
    print("   ✅ Cost optimization")

    print()
    print("🎉 Enhanced demo complete! CodeConductor now intelligently combines")
    print("local and cloud LLMs with production-ready performance and reliability.")


async def demo_complexity_analyzer():
    """Demo the complexity analyzer capabilities."""
    print("\n" + "=" * 70)
    print("🔍 Enhanced Complexity Analyzer Demo")
    print("=" * 70)

    analyzer = ComplexityAnalyzer()

    # Test cases with detailed analysis
    test_tasks = [
        "Fix a typo in the README file",
        "Add a new method to the User class",
        "Implement user authentication with OAuth2",
        "Design a microservices architecture for a social media platform",
        "Create a machine learning model for fraud detection",
        "Implement a real-time chat system with WebSockets",
        "Add input validation to the contact form",
        "Optimize database queries for better performance",
    ]

    for task in test_tasks:
        result = analyzer.analyze_complexity(task)
        print(f"\n📝 Task: {task}")
        print(f"   Level: {result.level.value}")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   Requires Cloud: {result.requires_cloud}")
        print(f"   Estimated Tokens: {result.estimated_tokens}")
        print(f"   Reasons: {', '.join(result.reasons) if result.reasons else 'None'}")


async def demo_cloud_escalator():
    """Demo the cloud escalator capabilities."""
    print("\n" + "=" * 70)
    print("☁️ Enhanced Cloud Escalator Demo")
    print("=" * 70)

    escalator = CloudEscalator()

    # Check availability
    if not escalator.is_available():
        print("⚠️ Cloud APIs not available (missing API keys)")
        print("Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variables")
        print("\nTo test cloud escalation:")
        print("1. Get API keys from OpenAI and/or Anthropic")
        print("2. Set environment variables:")
        print("   export OPENAI_API_KEY='your-key-here'")
        print("   export ANTHROPIC_API_KEY='your-key-here'")
        print("3. Run this demo again")
        return

    # Test task
    task = "Implement a secure authentication system with JWT tokens and refresh tokens"

    print(f"📝 Task: {task}")

    # Estimate cost for different models
    models = ["gpt-4", "gpt-3.5-turbo", "claude-3-sonnet", "claude-3-opus"]
    costs = escalator.estimate_cost(task, models)

    print("💰 Cost estimates:")
    for model, cost in costs.items():
        print(f"   {model}: ${cost:.4f}")

    # Try escalation (if APIs are available)
    try:
        async with escalator:
            responses = await escalator.escalate_task(task, ["gpt-3.5-turbo"])

            for response in responses:
                print(f"\n🤖 {response.model}:")
                print(f"   📊 Confidence: {response.confidence:.2f}")
                print(f"   💰 Cost: ${response.cost_estimate:.4f}")
                print(f"   ⏱️  Time: {response.response_time:.2f}s")
                print(f"   📝 Response: {response.content[:200]}...")

    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Run all demos."""
    print("🎼 CodeConductor Enhanced Hybrid Architecture Demo")
    print("=" * 70)

    # Run main hybrid ensemble demo
    asyncio.run(demo_hybrid_ensemble())

    # Run complexity analyzer demo
    asyncio.run(demo_complexity_analyzer())

    # Run cloud escalator demo
    asyncio.run(demo_cloud_escalator())


if __name__ == "__main__":
    main()
