#!/usr/bin/env python3
"""
Demo script for RAG integration with Ensemble Engine

This script demonstrates how the RAG system enhances the ensemble pipeline
by providing relevant context from both local project files and external sources.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import ensemble components
from ensemble.ensemble_engine import EnsembleEngine, EnsembleRequest
from context.rag_system import RAGSystem


async def demo_rag_integration():
    """Demonstrate RAG integration with ensemble engine."""
    print("🔍 RAG Integration Demo")
    print("=" * 60)

    # Initialize RAG system
    print("🔧 Initializing RAG System...")
    rag_system = RAGSystem()
    print("✅ RAG System initialized successfully")

    # Test scenarios with different types of tasks
    test_scenarios = [
        {
            "name": "Simple Function Task",
            "task": "Create a function to calculate fibonacci numbers",
            "expected_context": "local code examples, documentation",
        },
        {
            "name": "API Development Task",
            "task": "Create a REST API endpoint with Flask for user authentication",
            "expected_context": "external Stack Overflow examples, local patterns",
        },
        {
            "name": "Data Processing Task",
            "task": "Implement pandas data analysis with error handling",
            "expected_context": "external documentation, local patterns",
        },
    ]

    print("\n🧪 Testing RAG Context Retrieval")
    print("-" * 60)

    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n📝 Scenario {i}: {scenario['name']}")
        print(f"   Task: {scenario['task']}")
        print(f"   Expected Context: {scenario['expected_context']}")

        # Test context retrieval
        print("   🔍 Retrieving context...")
        context_docs = rag_system.retrieve_context(scenario["task"], k=3)

        print(f"   📚 Found {len(context_docs)} context documents:")
        for j, doc in enumerate(context_docs, 1):
            source = doc.get("source", "unknown")
            score = doc.get("relevance_score", 0)
            content_preview = doc.get("content", "")[:100] + "..."
            print(f"      {j}. [{source}] (Score: {score:.3f}): {content_preview}")

        # Test prompt augmentation
        print("   ✨ Testing prompt augmentation...")
        augmented_prompt = rag_system.augment_prompt(scenario["task"])

        print(f"   📏 Original length: {len(scenario['task'])} chars")
        print(f"   📏 Augmented length: {len(augmented_prompt)} chars")
        print(
            f"   📈 Enhancement: {len(augmented_prompt) - len(scenario['task'])} chars added"
        )

        print("   " + "-" * 40)


async def demo_rag_vs_no_rag():
    """Compare ensemble performance with and without RAG."""
    print("\n🔄 RAG vs No-RAG Comparison")
    print("=" * 60)

    # Test task
    task = "Create a FastAPI endpoint for user registration with validation"

    print(f"📝 Task: {task}")

    # Test with RAG enabled
    print("\n🔍 Testing WITH RAG...")
    ensemble_rag = EnsembleEngine(use_rlhf=True, use_rag=True)
    await ensemble_rag.initialize()

    request = EnsembleRequest(
        task_description=task,
        test_results=[{"passed": True}, {"passed": True}],
        code_quality=0.8,
        user_feedback=0.9,
    )

    response_rag = await ensemble_rag.process_request(request)

    print(f"   RAG Status: {'Enabled' if ensemble_rag.use_rag else 'Disabled'}")
    print(f"   Selected Model: {response_rag.selected_model}")
    print(f"   Confidence: {response_rag.confidence:.2f}")
    print(f"   Execution Time: {response_rag.execution_time:.2f}s")

    # Test without RAG
    print("\n⚡ Testing WITHOUT RAG...")
    ensemble_no_rag = EnsembleEngine(use_rlhf=True, use_rag=False)
    await ensemble_no_rag.initialize()

    response_no_rag = await ensemble_no_rag.process_request(request)

    print(f"   RAG Status: {'Enabled' if ensemble_no_rag.use_rag else 'Disabled'}")
    print(f"   Selected Model: {response_no_rag.selected_model}")
    print(f"   Confidence: {response_no_rag.confidence:.2f}")
    print(f"   Execution Time: {response_no_rag.execution_time:.2f}s")

    # Compare results
    print("\n📊 Comparison Results:")
    print(f"   RAG Confidence: {response_rag.confidence:.2f}")
    print(f"   No-RAG Confidence: {response_no_rag.confidence:.2f}")
    print(
        f"   Confidence Difference: {response_rag.confidence - response_no_rag.confidence:+.2f}"
    )

    if response_rag.confidence > response_no_rag.confidence:
        print("   🎉 RAG improved confidence!")
    elif response_rag.confidence < response_no_rag.confidence:
        print("   📉 RAG reduced confidence (may need more context)")
    else:
        print("   ➖ No difference in confidence")


def demo_rag_capabilities():
    """Show RAG system capabilities and features."""
    print("\n📊 RAG System Capabilities")
    print("=" * 60)

    try:
        rag_system = RAGSystem()

        # Test external context fetching
        print("🌐 Testing External Context Sources:")

        # Test Stack Overflow
        print("   📚 Stack Overflow...")
        so_results = rag_system.fetch_external_context(
            "Python Flask API", source="stackoverflow", max_results=2
        )
        print(f"      Found {len(so_results)} results")
        for i, result in enumerate(so_results, 1):
            print(f"      {i}. {result[:100]}...")

        # Test GitHub (placeholder)
        print("   🐙 GitHub...")
        gh_results = rag_system.fetch_external_context(
            "Python", source="github", max_results=2
        )
        print(f"      Found {len(gh_results)} results (placeholder)")

        # Test documentation (placeholder)
        print("   📖 Documentation...")
        doc_results = rag_system.fetch_external_context(
            "Python", source="docs", max_results=2
        )
        print(f"      Found {len(doc_results)} results (placeholder)")

        # Show context summary
        print("\n📈 Context Summary for Sample Task:")
        task = "Create a web API with authentication"
        summary = rag_system.get_context_summary(task)
        print(f"   Context Available: {summary['context_available']}")
        print(f"   Context Count: {summary['context_count']}")
        print(f"   Average Relevance: {summary['avg_relevance']:.3f}")
        print(f"   Context Types: {summary['context_types']}")

    except Exception as e:
        print(f"❌ RAG system not available: {e}")
        print("   Install with: pip install langchain chromadb")


async def demo_rag_with_ensemble():
    """Demonstrate RAG integration with full ensemble pipeline."""
    print("\n🚀 RAG + Ensemble Pipeline Demo")
    print("=" * 60)

    # Initialize ensemble with RAG
    print("🔧 Initializing Ensemble Engine with RAG...")
    ensemble = EnsembleEngine(use_rlhf=True, use_rag=True)

    success = await ensemble.initialize()
    if not success:
        print("❌ Failed to initialize ensemble engine")
        return

    print("✅ Ensemble Engine initialized successfully")

    # Test with a complex task that should benefit from RAG
    complex_task = "Create a FastAPI application with JWT authentication, SQLAlchemy ORM, and user management endpoints"

    print(f"\n📝 Complex Task: {complex_task}")

    request = EnsembleRequest(
        task_description=complex_task,
        test_results=[{"passed": True}, {"passed": True}, {"passed": True}],
        code_quality=0.9,
        user_feedback=0.9,
    )

    print("🔄 Processing with RAG-enhanced ensemble...")
    response = await ensemble.process_request(request)

    print(
        f"✅ RLHF Action: {response.rlhf_action} ({response.rlhf_action_description})"
    )
    print(f"🎯 Selected Model: {response.selected_model}")
    print(f"🧠 Confidence: {response.confidence:.2f}")
    print(f"⏱️  Execution Time: {response.execution_time:.2f}s")

    if response.consensus:
        consensus_preview = str(response.consensus)[:200] + "..."
        print(f"📝 Consensus Preview: {consensus_preview}")

    print("\n🎉 RAG + Ensemble pipeline completed successfully!")


async def main():
    """Main demo function."""
    print("🎼 CodeConductor RAG Integration Demo")
    print("=" * 60)

    # Show RAG capabilities
    demo_rag_capabilities()

    # Run RAG integration demo
    await demo_rag_integration()

    # Run comparison demo
    await demo_rag_vs_no_rag()

    # Run full pipeline demo
    await demo_rag_with_ensemble()

    print("\n🎉 Demo completed!")
    print("\n💡 Next steps:")
    print("   1. Add more external sources (GitHub, documentation sites)")
    print("   2. Implement multi-agent debugging")
    print("   3. Test with real project data")
    print("   4. Optimize context retrieval performance")


if __name__ == "__main__":
    asyncio.run(main())
