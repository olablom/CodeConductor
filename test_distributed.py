#!/usr/bin/env python3
"""
Test script for distributed execution with Celery
"""

import time
import subprocess
import sys
from pathlib import Path


def test_celery_setup():
    """Test Celery setup and basic functionality"""
    print("🧪 Testing Celery setup...")

    try:
        from integrations.celery_app import celery_app, is_celery_available, debug_task

        # Test if broker is available
        if is_celery_available():
            print("✅ Celery broker is available")

            # Test debug task
            result = debug_task.delay()
            task_result = result.get(timeout=10)
            print(f"✅ Debug task completed: {task_result}")

            return True
        else:
            print("❌ Celery broker is not available")
            print("💡 Start Redis with: redis-server")
            return False

    except Exception as e:
        print(f"❌ Celery setup test failed: {e}")
        return False


def test_distributed_orchestrator():
    """Test distributed orchestrator functionality"""
    print("\n🧪 Testing distributed orchestrator...")

    try:
        from agents.orchestrator_distributed import DistributedAgentOrchestrator

        # Create orchestrator
        orchestrator = DistributedAgentOrchestrator(enable_plugins=True)

        # Check distributed stats
        stats = orchestrator.get_distributed_stats()
        print(f"📊 Distributed stats: {stats}")

        # Test discussion
        prompt = "Create a simple hello world function in Python"
        context = {"strategy": "conservative", "test_mode": True}

        print("🤖 Running distributed discussion...")
        result = orchestrator.facilitate_discussion(prompt, context)

        print(f"✅ Discussion completed")
        print(f"  Execution mode: {result.get('execution_mode', 'unknown')}")
        print(f"  Plugin count: {result.get('plugin_count', 0)}")
        print(
            f"  Consensus confidence: {result.get('consensus', {}).get('confidence', 0)}"
        )

        return True

    except Exception as e:
        print(f"❌ Distributed orchestrator test failed: {e}")
        return False


def test_celery_tasks():
    """Test individual Celery tasks"""
    print("\n🧪 Testing Celery tasks...")

    try:
        from agents.celery_agents import (
            codegen_analyze_task,
            architect_analyze_task,
            reviewer_analyze_task,
        )

        prompt = "Create a simple calculator function"
        context = {"test_mode": True}

        # Start tasks
        print("🚀 Starting parallel tasks...")
        tasks = {
            "codegen": codegen_analyze_task.delay(prompt, context),
            "architect": architect_analyze_task.delay(prompt, context),
            "reviewer": reviewer_analyze_task.delay(prompt, context),
        }

        # Wait for results
        results = {}
        for name, task in tasks.items():
            try:
                print(f"⏳ Waiting for {name} task...")
                result = task.get(timeout=30)
                results[name] = result
                print(f"✅ {name} task completed")
            except Exception as e:
                print(f"❌ {name} task failed: {e}")
                results[name] = {"error": str(e)}

        print(f"📋 Task results: {len(results)} tasks completed")
        return True

    except Exception as e:
        print(f"❌ Celery tasks test failed: {e}")
        return False


def test_pipeline_integration():
    """Test pipeline integration with distributed execution"""
    print("\n🧪 Testing pipeline integration...")

    try:
        # Create test prompt
        test_prompt = Path("test_distributed_prompt.md")
        test_prompt.write_text("Create a simple hello world function")

        # Run pipeline with distributed flag
        cmd = [
            sys.executable,
            "pipeline.py",
            "--prompt",
            str(test_prompt),
            "--iters",
            "1",
            "--mock",
            "--distributed",
        ]

        print(f"🚀 Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            print("✅ Pipeline with distributed execution completed successfully")
            print("📋 Output:")
            print(result.stdout[-500:])  # Last 500 chars
        else:
            print("❌ Pipeline with distributed execution failed")
            print("📋 Error:")
            print(result.stderr)

        # Cleanup
        test_prompt.unlink(missing_ok=True)

        return result.returncode == 0

    except Exception as e:
        print(f"❌ Pipeline integration test failed: {e}")
        return False


def main():
    """Run all distributed execution tests"""
    print("🎼 CodeConductor v2.0 - Distributed Execution Test")
    print("=" * 60)

    tests = [
        ("Celery Setup", test_celery_setup),
        ("Distributed Orchestrator", test_distributed_orchestrator),
        ("Celery Tasks", test_celery_tasks),
        ("Pipeline Integration", test_pipeline_integration),
    ]

    results = {}
    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    passed = 0
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\n🎯 Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All distributed execution tests passed!")
        print("🚀 CodeConductor is ready for distributed execution!")
    else:
        print("⚠️ Some tests failed. Check Redis and Celery setup.")
        print("💡 Make sure Redis is running: redis-server")
        print(
            "💡 Start Celery workers: celery -A integrations.celery_app.celery_app worker --loglevel=info"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()
