#!/usr/bin/env python3
"""
Full Stack Integration Test

This script tests the entire CodeConductor microservices stack
to ensure all services are working together correctly.
"""

import asyncio
import httpx
import json
import time
from typing import Dict, Any


class StackTester:
    """Test the entire microservices stack."""

    def __init__(self):
        self.base_urls = {
            "gateway": "http://localhost:9000",
            "agent": "http://localhost:9001",
            "orchestrator": "http://localhost:9002",
            "auth": "http://localhost:9005",
            "data": "http://localhost:9006",
        }
        self.results = {}

    async def test_service_health(self, service_name: str, url: str) -> Dict[str, Any]:
        """Test individual service health."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{url}/health", timeout=10.0)
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "status": "✅ PASS",
                        "service": service_name,
                        "response": data,
                        "error": None,
                    }
                else:
                    return {
                        "status": "❌ FAIL",
                        "service": service_name,
                        "response": None,
                        "error": f"HTTP {response.status_code}",
                    }
        except Exception as e:
            return {
                "status": "❌ FAIL",
                "service": service_name,
                "response": None,
                "error": str(e),
            }

    async def test_gateway_routing(self) -> Dict[str, Any]:
        """Test Gateway routing to all services."""
        results = {}

        # Test routing to each service through Gateway
        routes = [
            ("agent", "/api/v1/agents/health"),
            ("orchestrator", "/api/v1/orchestrator/health"),
            ("data", "/api/v1/data/health"),
            ("auth", "/api/v1/auth/health"),
        ]

        for service_name, route in routes:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{self.base_urls['gateway']}{route}", timeout=10.0
                    )
                    if response.status_code == 200:
                        results[service_name] = {
                            "status": "✅ PASS",
                            "route": route,
                            "response": response.json(),
                        }
                    else:
                        results[service_name] = {
                            "status": "❌ FAIL",
                            "route": route,
                            "error": f"HTTP {response.status_code}",
                        }
            except Exception as e:
                results[service_name] = {
                    "status": "❌ FAIL",
                    "route": route,
                    "error": str(e),
                }

        return results

    async def test_data_service_endpoints(self) -> Dict[str, Any]:
        """Test Data Service specific endpoints."""
        results = {}

        # Test bandit endpoints
        try:
            async with httpx.AsyncClient() as client:
                # Test bandit choose
                choose_data = {
                    "arms": ["arm1", "arm2", "arm3"],
                    "features": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                    "bandit_name": "test_bandit",
                }
                response = await client.post(
                    f"{self.base_urls['data']}/bandits/choose",
                    json=choose_data,
                    timeout=10.0,
                )
                if response.status_code == 200:
                    results["bandit_choose"] = {
                        "status": "✅ PASS",
                        "response": response.json(),
                    }
                else:
                    results["bandit_choose"] = {
                        "status": "❌ FAIL",
                        "error": f"HTTP {response.status_code}",
                    }
        except Exception as e:
            results["bandit_choose"] = {"status": "❌ FAIL", "error": str(e)}

        # Test Q-learning endpoints
        try:
            async with httpx.AsyncClient() as client:
                qlearning_data = {
                    "context": {
                        "task_type": "api_creation",
                        "complexity": "medium",
                        "language": "python",
                        "agent_count": 2,
                    },
                    "agent_name": "test_qlearning",
                }
                response = await client.post(
                    f"{self.base_urls['data']}/qlearning/run",
                    json=qlearning_data,
                    timeout=10.0,
                )
                if response.status_code == 200:
                    results["qlearning_run"] = {
                        "status": "✅ PASS",
                        "response": response.json(),
                    }
                else:
                    results["qlearning_run"] = {
                        "status": "❌ FAIL",
                        "error": f"HTTP {response.status_code}",
                    }
        except Exception as e:
            results["qlearning_run"] = {"status": "❌ FAIL", "error": str(e)}

        return results

    async def test_auth_service_endpoints(self) -> Dict[str, Any]:
        """Test Auth Service specific endpoints."""
        results = {}

        # Test approval endpoint
        try:
            async with httpx.AsyncClient() as client:
                approval_data = {
                    "context": {"task": "Create a simple API"},
                    "code": "print('Hello World')",
                    "task_type": "api_creation",
                    "risk_level": "low",
                    "confidence": 0.9,
                }
                response = await client.post(
                    f"{self.base_urls['auth']}/auth/approve",
                    json=approval_data,
                    timeout=10.0,
                )
                if response.status_code == 200:
                    results["approval"] = {
                        "status": "✅ PASS",
                        "response": response.json(),
                    }
                else:
                    results["approval"] = {
                        "status": "❌ FAIL",
                        "error": f"HTTP {response.status_code}",
                    }
        except Exception as e:
            results["approval"] = {"status": "❌ FAIL", "error": str(e)}

        return results

    async def test_orchestrator_endpoints(self) -> Dict[str, Any]:
        """Test Orchestrator Service specific endpoints."""
        results = {}

        # Test discussion endpoint
        try:
            async with httpx.AsyncClient() as client:
                discussion_data = {
                    "task": "Create a REST API for user management",
                    "agents": ["codegen", "review"],
                    "strategy": "majority",
                }
                response = await client.post(
                    f"{self.base_urls['orchestrator']}/orchestrator/discuss",
                    json=discussion_data,
                    timeout=10.0,
                )
                if response.status_code == 200:
                    results["discussion"] = {
                        "status": "✅ PASS",
                        "response": response.json(),
                    }
                else:
                    results["discussion"] = {
                        "status": "❌ FAIL",
                        "error": f"HTTP {response.status_code}",
                    }
        except Exception as e:
            results["discussion"] = {"status": "❌ FAIL", "error": str(e)}

        return results

    async def run_all_tests(self):
        """Run all tests."""
        print("🚀 Starting Full Stack Integration Tests...")
        print("=" * 60)

        # Test 1: Individual service health
        print("\n📋 Test 1: Individual Service Health")
        print("-" * 40)
        for service_name, url in self.base_urls.items():
            result = await self.test_service_health(service_name, url)
            print(
                f"{result['status']} {service_name.upper()}: {result.get('error', 'OK')}"
            )
            self.results[f"health_{service_name}"] = result

        # Test 2: Gateway routing
        print("\n🌐 Test 2: Gateway Routing")
        print("-" * 40)
        routing_results = await self.test_gateway_routing()
        for service_name, result in routing_results.items():
            print(
                f"{result['status']} Gateway → {service_name.upper()}: {result.get('error', 'OK')}"
            )
            self.results[f"routing_{service_name}"] = result

        # Test 3: Data Service endpoints
        print("\n📊 Test 3: Data Service Endpoints")
        print("-" * 40)
        data_results = await self.test_data_service_endpoints()
        for endpoint, result in data_results.items():
            print(
                f"{result['status']} Data Service {endpoint}: {result.get('error', 'OK')}"
            )
            self.results[f"data_{endpoint}"] = result

        # Test 4: Auth Service endpoints
        print("\n🛡️ Test 4: Auth Service Endpoints")
        print("-" * 40)
        auth_results = await self.test_auth_service_endpoints()
        for endpoint, result in auth_results.items():
            print(
                f"{result['status']} Auth Service {endpoint}: {result.get('error', 'OK')}"
            )
            self.results[f"auth_{endpoint}"] = result

        # Test 5: Orchestrator endpoints
        print("\n🎼 Test 5: Orchestrator Service Endpoints")
        print("-" * 40)
        orchestrator_results = await self.test_orchestrator_endpoints()
        for endpoint, result in orchestrator_results.items():
            print(
                f"{result['status']} Orchestrator {endpoint}: {result.get('error', 'OK')}"
            )
            self.results[f"orchestrator_{endpoint}"] = result

        # Summary
        print("\n📈 Test Summary")
        print("=" * 60)
        total_tests = len(self.results)
        passed_tests = sum(
            1 for result in self.results.values() if result["status"] == "✅ PASS"
        )
        failed_tests = total_tests - passed_tests

        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests / total_tests) * 100:.1f}%")

        if failed_tests == 0:
            print("\n🎉 ALL TESTS PASSED! Stack is fully operational!")
        else:
            print(f"\n⚠️ {failed_tests} tests failed. Check the details above.")

        return self.results


async def main():
    """Main test runner."""
    tester = StackTester()
    results = await tester.run_all_tests()

    # Save results to file
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n📄 Detailed results saved to test_results.json")


if __name__ == "__main__":
    asyncio.run(main())
