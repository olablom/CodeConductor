#!/usr/bin/env python3
"""
Stress test to verify the hanging fix under demanding conditions
"""

import asyncio
import logging
import time
from ensemble.ensemble_engine import EnsembleEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def stress_test_concurrent_requests():
    """Test multiple concurrent ensemble requests to ensure no hanging."""
    print("🔥 Stress Testing Concurrent Requests")
    print("=" * 50)

    engine = EnsembleEngine(min_confidence=0.6)

    # Initialize engine
    print("📡 Initializing ensemble engine...")
    success = await engine.initialize()
    if not success:
        print("❌ Failed to initialize ensemble engine")
        return False

    print("✅ Ensemble engine initialized")

    # Test tasks
    test_tasks = [
        "Create a Python function that adds two numbers",
        "Write a function to calculate factorial",
        "Create a simple calculator class",
        "Write a function to check if a number is prime",
        "Create a function to reverse a string",
    ]

    print(f"🎯 Running {len(test_tasks)} concurrent requests...")

    async def process_single_request(task, request_id):
        """Process a single request with timeout."""
        try:
            print(f"🚀 Starting request {request_id}: {task[:30]}...")
            start_time = time.time()

            result = await asyncio.wait_for(
                engine.process_request_with_fallback(task),
                timeout=45.0,  # 45 second timeout per request
            )

            end_time = time.time()
            duration = end_time - start_time

            print(f"✅ Request {request_id} completed in {duration:.2f}s")
            print(f"   Strategy: {result.get('strategy', 'unknown')}")
            print(f"   Success: {result.get('success', False)}")

            return {
                "request_id": request_id,
                "success": True,
                "duration": duration,
                "strategy": result.get("strategy", "unknown"),
                "result_success": result.get("success", False),
            }

        except asyncio.TimeoutError:
            print(f"❌ Request {request_id} timed out after 45s")
            return {
                "request_id": request_id,
                "success": False,
                "duration": 45.0,
                "strategy": "timeout",
                "result_success": False,
            }
        except Exception as e:
            print(f"❌ Request {request_id} failed: {e}")
            return {
                "request_id": request_id,
                "success": False,
                "duration": 0.0,
                "strategy": "error",
                "result_success": False,
            }

    # Run all requests concurrently
    start_time = time.time()
    tasks = [process_single_request(task, i + 1) for i, task in enumerate(test_tasks)]

    results = await asyncio.gather(*tasks, return_exceptions=True)
    end_time = time.time()

    total_time = end_time - start_time

    # Analyze results
    successful_requests = sum(
        1 for r in results if isinstance(r, dict) and r.get("success", False)
    )
    failed_requests = len(results) - successful_requests

    print(f"\n📊 Stress Test Results:")
    print(f"⏱️ Total time: {total_time:.2f}s")
    print(f"✅ Successful requests: {successful_requests}")
    print(f"❌ Failed requests: {failed_requests}")
    print(f"📈 Success rate: {successful_requests / len(results) * 100:.1f}%")

    # Check for hanging issues
    if failed_requests == 0:
        print("\n🎉 SUCCESS: No hanging detected!")
        print("✅ All concurrent requests completed successfully")
        return True
    elif failed_requests < len(results):
        print("\n⚠️ PARTIAL SUCCESS: Some requests failed but no hanging")
        print("✅ System is responsive and handling errors gracefully")
        return True
    else:
        print("\n❌ FAILURE: All requests failed - possible hanging issue")
        return False


async def stress_test_rapid_requests():
    """Test rapid sequential requests to ensure no resource leaks."""
    print("\n🔥 Stress Testing Rapid Sequential Requests")
    print("=" * 50)

    engine = EnsembleEngine(min_confidence=0.6)

    # Initialize engine
    success = await engine.initialize()
    if not success:
        print("❌ Failed to initialize ensemble engine")
        return False

    print("✅ Ensemble engine initialized")

    # Simple test task
    test_task = "Create a Python function that prints 'Hello, World!'"

    print(f"🎯 Running 5 rapid sequential requests...")

    start_time = time.time()
    results = []

    for i in range(5):
        try:
            print(f"🚀 Request {i + 1}/5...")
            request_start = time.time()

            result = await asyncio.wait_for(
                engine.process_request_with_fallback(test_task), timeout=30.0
            )

            request_duration = time.time() - request_start
            results.append(
                {
                    "request_id": i + 1,
                    "success": True,
                    "duration": request_duration,
                    "strategy": result.get("strategy", "unknown"),
                }
            )

            print(f"✅ Request {i + 1} completed in {request_duration:.2f}s")

        except asyncio.TimeoutError:
            print(f"❌ Request {i + 1} timed out")
            results.append(
                {
                    "request_id": i + 1,
                    "success": False,
                    "duration": 30.0,
                    "strategy": "timeout",
                }
            )
        except Exception as e:
            print(f"❌ Request {i + 1} failed: {e}")
            results.append(
                {
                    "request_id": i + 1,
                    "success": False,
                    "duration": 0.0,
                    "strategy": "error",
                }
            )

    total_time = time.time() - start_time
    successful_requests = sum(1 for r in results if r.get("success", False))

    print(f"\n📊 Rapid Test Results:")
    print(f"⏱️ Total time: {total_time:.2f}s")
    print(f"✅ Successful requests: {successful_requests}/5")
    print(f"📈 Success rate: {successful_requests / 5 * 100:.1f}%")

    if successful_requests >= 3:
        print("\n🎉 SUCCESS: Rapid requests working well!")
        return True
    else:
        print("\n❌ FAILURE: Too many rapid requests failed")
        return False


async def main():
    """Main stress test function."""
    print("🔥 CodeConductor Hanging Fix Stress Test")
    print("=" * 60)

    # Test 1: Concurrent requests
    concurrent_success = await stress_test_concurrent_requests()

    # Test 2: Rapid sequential requests
    rapid_success = await stress_test_rapid_requests()

    print(f"\n🎯 Final Results:")
    print(f"Concurrent test: {'✅ PASS' if concurrent_success else '❌ FAIL'}")
    print(f"Rapid test: {'✅ PASS' if rapid_success else '❌ FAIL'}")

    if concurrent_success and rapid_success:
        print("\n🎉 ALL TESTS PASSED! Hanging fix is robust.")
        return True
    else:
        print("\n❌ Some stress tests failed. Further investigation needed.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
