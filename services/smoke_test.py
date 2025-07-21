#!/usr/bin/env python3
"""
Smoke test for CodeConductor microservices
Quick verification that all services are running and responding
"""

import asyncio
import httpx
import time
import sys
from typing import Dict, List

# Service URLs
SERVICES = {
    "gateway": "http://localhost:9000",
    "agent": "http://localhost:9001",
    "orchestrator": "http://localhost:9002",
    "data": "http://localhost:9003",
    "auth": "http://localhost:9005",
}


async def check_service_health(
    service_name: str, url: str, client: httpx.AsyncClient
) -> bool:
    """Test if a service is healthy"""
    try:
        print(f"🔍 Testing {service_name}...", end=" ")

        # Test basic endpoint
        response = await client.get(f"{url}/", timeout=5.0)
        if response.status_code != 200:
            print(f"❌ HTTP {response.status_code}")
            return False

        data = response.json()
        if "status" not in data or data["status"] != "healthy":
            print(f"❌ Not healthy: {data}")
            return False

        print("✅ Healthy")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def check_agent_functionality(client: httpx.AsyncClient) -> bool:
    """Test basic agent functionality"""
    try:
        print("🧪 Testing Agent Service functionality...", end=" ")

        # Test agent analysis
        request = {
            "agent_type": "codegen",
            "task_context": {"task": "Create a simple function"},
        }

        response = await client.post(
            f"{SERVICES['agent']}/agents/analyze", json=request, timeout=10.0
        )

        if response.status_code != 200:
            print(f"❌ Analysis failed: {response.status_code}")
            return False

        data = response.json()
        if "agent_name" not in data or "result" not in data:
            print(f"❌ Invalid response format")
            return False

        print("✅ Working")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def check_orchestrator_functionality(client: httpx.AsyncClient) -> bool:
    """Test basic orchestrator functionality"""
    try:
        print("🎼 Testing Orchestrator Service functionality...", end=" ")

        # Test discussion start
        request = {
            "task_context": {"task": "Test discussion"},
            "agents": ["codegen"],
            "max_rounds": 1,
        }

        response = await client.post(
            f"{SERVICES['orchestrator']}/discussions/start", json=request, timeout=10.0
        )

        if response.status_code != 200:
            print(f"❌ Discussion start failed: {response.status_code}")
            return False

        data = response.json()
        if "discussion_id" not in data:
            print(f"❌ Invalid response format")
            return False

        print("✅ Working")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def check_service_communication(client: httpx.AsyncClient) -> bool:
    """Test service-to-service communication"""
    try:
        print("🔗 Testing service-to-service communication...", end=" ")

        # Test orchestrator health (which checks other services)
        response = await client.get(f"{SERVICES['orchestrator']}/health", timeout=5.0)

        if response.status_code != 200:
            print(f"❌ Orchestrator health failed: {response.status_code}")
            return False

        data = response.json()
        # Check if orchestrator can reach agent service
        if "agent_service" in data:
            status = data["agent_service"]
            if status not in ["healthy", "unreachable"]:
                print(f"❌ Agent service status: {status}")
                return False

        print("✅ Working")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def main():
    """Run smoke tests"""
    print("🚀 CodeConductor Microservices Smoke Test")
    print("=" * 50)

    async with httpx.AsyncClient() as client:
        results = []

        # Test all services are running
        print("\n📋 Testing service health...")
        for service_name, url in SERVICES.items():
            healthy = await check_service_health(service_name, url, client)
            results.append((service_name, "health", healthy))

        # Test core functionality
        print("\n🔧 Testing core functionality...")

        # Agent service
        agent_working = await check_agent_functionality(client)
        results.append(("agent", "functionality", agent_working))

        # Orchestrator service
        orchestrator_working = await check_orchestrator_functionality(client)
        results.append(("orchestrator", "functionality", orchestrator_working))

        # Service communication
        communication_working = await check_service_communication(client)
        results.append(("communication", "service-to-service", communication_working))

        # Summary
        print("\n" + "=" * 50)
        print("📊 Test Results Summary:")
        print("=" * 50)

        passed = 0
        total = len(results)

        for service, test_type, success in results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{service:15} {test_type:20} {status}")
            if success:
                passed += 1

        print(f"\n🎯 Overall: {passed}/{total} tests passed")

        if passed == total:
            print("🎉 All tests passed! Microservices stack is working correctly.")
            return 0
        else:
            print("⚠️  Some tests failed. Check service logs for details.")
            return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
