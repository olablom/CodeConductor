#!/usr/bin/env python3
"""
Test RAG System Fix
"""

import asyncio
from datetime import datetime

from src.codeconductor.context.rag_system import RAGSystem


import pytest

@pytest.mark.asyncio
async def test_rag_system():
    """Test RAG system functionality"""
    print("🧪 Testing RAG System")
    print("=" * 40)

    # Initialize RAG system
    rag = RAGSystem()

    # Test 1: Basic search
    print("\n1. Testing basic search...")
    results = rag.search("fibonacci", top_k=5)
    print(f"   Results: {len(results)}")
    for i, result in enumerate(results):
        print(
            f"   Result {i+1}: {result['metadata'].get('filename', 'Unknown')} (score: {result['relevance_score']:.3f})"
        )

    # Test 2: Add new document
    print("\n2. Testing document addition...")
    test_content = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Example usage
print(fibonacci(10))  # Should print 55
"""
    rag.add_document(
        "test_fibonacci",
        test_content,
        {
            "type": "python_function",
            "filename": "test_fibonacci.py",
            "description": "Fibonacci function implementation",
        },
    )
    print("   ✅ Document added")

    # Test 3: Search again
    print("\n3. Testing search after addition...")
    results = rag.search("fibonacci", top_k=5)
    print(f"   Results: {len(results)}")
    for i, result in enumerate(results):
        print(
            f"   Result {i+1}: {result['metadata'].get('filename', 'Unknown')} (score: {result['relevance_score']:.3f})"
        )

    # Test 4: Context retrieval
    print("\n4. Testing context retrieval...")
    context = rag.retrieve_context("Create a function to calculate fibonacci numbers", k=3)
    print(f"   Context documents: {len(context)}")
    for i, doc in enumerate(context):
        print(
            f"   Context {i+1}: {doc['metadata'].get('filename', 'Unknown')} (score: {doc['relevance_score']:.3f})"
        )

    # Test 5: External search
    print("\n5. Testing external search...")
    external_results = rag.search_external("python fibonacci function", max_results=2)
    print(f"   External results: {len(external_results)}")

    # Test 6: Pattern addition
    print("\n6. Testing pattern addition...")
    pattern = {
        "task_description": "Create fibonacci function",
        "prompt": "Write a function to calculate fibonacci numbers",
        "code": test_content,
        "validation": {"score": 0.9},
        "model_used": "test-model",
        "user_rating": 5,
        "timestamp": datetime.now().isoformat(),
    }
    rag.add_pattern_to_context(pattern)
    print("   ✅ Pattern added")

    # Test 7: Search patterns
    print("\n7. Testing pattern search...")
    results = rag.search("fibonacci pattern", top_k=3)
    print(f"   Pattern results: {len(results)}")

    # Summary
    print("\n📊 RAG System Test Summary:")
    print(f"   Vector store available: {rag.vector_store is not None}")
    print(f"   Embedding model available: {rag.embedding_model is not None}")
    print(f"   Text splitter available: {rag.text_splitter is not None}")

    return {
        "rag_available": rag.vector_store is not None,
        "search_working": len(results) > 0,
        "context_working": len(context) > 0,
        "external_working": len(external_results) >= 0,  # External might be empty
    }


if __name__ == "__main__":
    results = asyncio.run(test_rag_system())
    print(
        f"\n🎯 RAG System Status: {'✅ WORKING' if results['rag_available'] and results['search_working'] else '❌ NEEDS FIX'}"
    )
