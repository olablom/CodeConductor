#!/usr/bin/env python3
"""
Integration tests for CodeConductor microservices
Tests the entire microservices stack end-to-end
"""

import pytest
import httpx
import asyncio
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Service URLs
SERVICES = {
    "gateway": "http://localhost:9000",
    "agent": "http://localhost:9001",
    "orchestrator": "http://localhost:9002",
    "data": "http://localhost:9003",
    "auth": "http://localhost:9005",
}


class TestMicroservicesStack:
    """Test suite for microservices integration"""

    @pytest.fixture(scope="class")
    def client(self):
        """HTTP client for testing"""
        return httpx.AsyncClient(timeout=30.0)

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="Microservices not running in test environment")
    async def test_services_are_running(self, client):
        """Test that all services are running and responding"""
        logger.info("Testing that all services are running...")

        for service_name, url in SERVICES.items():
            try:
                response = await client.get(f"{url}/")
                assert response.status_code == 200, (
                    f"{service_name} returned {response.status_code}"
                )

                data = response.json()
                assert "status" in data, f"{service_name} missing status field"
                assert data["status"] == "healthy", f"{service_name} not healthy"

                logger.info(f"✅ {service_name} is running and healthy")

            except Exception as e:
                pytest.fail(f"❌ {service_name} failed: {e}")

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="Microservices not running in test environment")
    async def test_health_endpoints(self, client):
        """Test detailed health endpoints"""
        logger.info("Testing detailed health endpoints...")

        for service_name, url in SERVICES.items():
            try:
                response = await client.get(f"{url}/health")
                assert response.status_code == 200, (
                    f"{service_name} health check failed"
                )

                data = response.json()
                logger.info(f"✅ {service_name} health: {data}")

            except Exception as e:
                pytest.fail(f"❌ {service_name} health check failed: {e}")

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="Microservices not running in test environment")
    async def test_agent_service_functionality(self, client):
        """Test Agent Service core functionality"""
        logger.info("Testing Agent Service functionality...")

        # Test agent analysis
        analysis_request = {
            "agent_type": "codegen",
            "task_context": {
                "task": "Create a simple Python function",
                "requirements": ["input validation", "error handling"],
            },
        }

        try:
            response = await client.post(
                f"{SERVICES['agent']}/agents/analyze", json=analysis_request
            )
            assert response.status_code == 200, "Agent analysis failed"

            data = response.json()
            assert "agent_name" in data, "Missing agent_name in response"
            assert "result" in data, "Missing result in response"
            assert "confidence" in data, "Missing confidence in response"

            logger.info(f"✅ Agent analysis successful: {data['agent_name']}")

        except Exception as e:
            pytest.fail(f"❌ Agent analysis failed: {e}")

        # Test agent proposal
        try:
            response = await client.post(
                f"{SERVICES['agent']}/agents/propose", json=analysis_request
            )
            assert response.status_code == 200, "Agent proposal failed"

            data = response.json()
            assert "agent_name" in data, "Missing agent_name in response"
            assert "result" in data, "Missing result in response"

            logger.info(f"✅ Agent proposal successful: {data['agent_name']}")

        except Exception as e:
            pytest.fail(f"❌ Agent proposal failed: {e}")

        # Test agent status
        try:
            response = await client.get(f"{SERVICES['agent']}/agents/status")
            assert response.status_code == 200, "Agent status failed"

            data = response.json()
            assert isinstance(data, list), "Agent status should return list"

            logger.info(f"✅ Agent status successful: {len(data)} agents")

        except Exception as e:
            pytest.fail(f"❌ Agent status failed: {e}")

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="Microservices not running in test environment")
    async def test_orchestrator_service_functionality(self, client):
        """Test Orchestrator Service core functionality"""
        logger.info("Testing Orchestrator Service functionality...")

        # Test discussion start
        discussion_request = {
            "task_context": {
                "task": "Design a microservice architecture",
                "requirements": ["scalability", "fault tolerance"],
            },
            "agents": ["architect", "codegen", "review"],
            "max_rounds": 2,
            "consensus_strategy": "weighted_majority",
        }

        try:
            response = await client.post(
                f"{SERVICES['orchestrator']}/discussions/start", json=discussion_request
            )
            assert response.status_code == 200, "Discussion start failed"

            data = response.json()
            assert "discussion_id" in data, "Missing discussion_id in response"
            assert "consensus_reached" in data, "Missing consensus_reached in response"

            discussion_id = data["discussion_id"]
            logger.info(f"✅ Discussion started: {discussion_id}")

            # Wait a bit for discussion to progress
            await asyncio.sleep(2)

            # Test discussion status
            response = await client.get(
                f"{SERVICES['orchestrator']}/discussions/{discussion_id}"
            )
            assert response.status_code == 200, "Discussion status failed"

            data = response.json()
            assert "discussion_id" in data, "Missing discussion_id in status"
            assert "execution_time" in data, "Missing execution_time in status"

            logger.info(
                f"✅ Discussion status successful: {data['discussion_rounds']} rounds"
            )

        except Exception as e:
            pytest.fail(f"❌ Orchestrator functionality failed: {e}")

        # Test workflow start
        workflow_request = {
            "workflow_type": "code_generation",
            "input_data": {"language": "python", "task": "Create a REST API"},
            "steps": [
                {"type": "agent_call", "agent": "architect", "operation": "analyze"},
                {"type": "agent_call", "agent": "codegen", "operation": "propose"},
            ],
        }

        try:
            response = await client.post(
                f"{SERVICES['orchestrator']}/workflows/start", json=workflow_request
            )
            assert response.status_code == 200, "Workflow start failed"

            data = response.json()
            assert "workflow_id" in data, "Missing workflow_id in response"
            assert "status" in data, "Missing status in response"

            workflow_id = data["workflow_id"]
            logger.info(f"✅ Workflow started: {workflow_id}")

        except Exception as e:
            pytest.fail(f"❌ Workflow functionality failed: {e}")

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="Microservices not running in test environment")
    async def test_service_to_service_communication(self, client):
        """Test that services can communicate with each other"""
        logger.info("Testing service-to-service communication...")

        # Test that orchestrator can reach agent service
        try:
            # This tests the orchestrator's health check which calls other services
            response = await client.get(f"{SERVICES['orchestrator']}/health")
            assert response.status_code == 200, "Orchestrator health check failed"

            data = response.json()
            # Check that orchestrator reports other services as healthy
            if "agent_service" in data:
                assert data["agent_service"] in ["healthy", "unreachable"], (
                    f"Agent service status: {data['agent_service']}"
                )

            logger.info("✅ Service-to-service communication working")

        except Exception as e:
            pytest.fail(f"❌ Service-to-service communication failed: {e}")

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="Microservices not running in test environment")
    async def test_error_handling(self, client):
        """Test error handling in services"""
        logger.info("Testing error handling...")

        # Test invalid agent type
        invalid_request = {
            "agent_type": "invalid_agent",
            "task_context": {"task": "test"},
        }

        try:
            response = await client.post(
                f"{SERVICES['agent']}/agents/analyze", json=invalid_request
            )
            # Should handle gracefully (400, 422, or 500)
            assert response.status_code in [400, 422, 500], (
                f"Expected error status, got {response.status_code}"
            )
            logger.info("✅ Error handling working for invalid agent type")

        except Exception as e:
            pytest.fail(f"❌ Error handling test failed: {e}")

        # Test invalid discussion ID
        try:
            response = await client.get(
                f"{SERVICES['orchestrator']}/discussions/invalid_id"
            )
            assert response.status_code == 404, (
                f"Expected 404, got {response.status_code}"
            )
            logger.info("✅ Error handling working for invalid discussion ID")

        except Exception as e:
            pytest.fail(f"❌ Error handling test failed: {e}")


class TestInfrastructure:
    """Test infrastructure components"""

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="Infrastructure services not running in test environment")
    async def test_rabbitmq_connectivity(self):
        """Test RabbitMQ connectivity"""
        logger.info("Testing RabbitMQ connectivity...")

        try:
            import pika

            # Try to connect to RabbitMQ
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(
                    host="localhost",
                    port=9004,
                    credentials=pika.PlainCredentials("codeconductor", "password"),
                )
            )

            channel = connection.channel()
            channel.queue_declare(queue="test_queue")
            channel.queue_delete(queue="test_queue")

            connection.close()
            logger.info("✅ RabbitMQ connectivity working")

        except Exception as e:
            pytest.skip(f"RabbitMQ not available: {e}")

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="Infrastructure services not running in test environment")
    async def test_postgres_connectivity(self):
        """Test PostgreSQL connectivity"""
        logger.info("Testing PostgreSQL connectivity...")

        try:
            import psycopg2

            # Try to connect to PostgreSQL
            conn = psycopg2.connect(
                host="localhost",
                port=5432,
                database="codeconductor",
                user="codeconductor",
                password="password",
            )

            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()

            cursor.close()
            conn.close()

            logger.info(f"✅ PostgreSQL connectivity working: {version[0]}")

        except Exception as e:
            pytest.skip(f"PostgreSQL not available: {e}")

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="Infrastructure services not running in test environment")
    async def test_redis_connectivity(self):
        """Test Redis connectivity"""
        logger.info("Testing Redis connectivity...")

        try:
            import redis

            # Try to connect to Redis
            r = redis.Redis(host="localhost", port=6379, decode_responses=True)

            # Test basic operations
            r.set("test_key", "test_value")
            value = r.get("test_key")
            r.delete("test_key")

            assert value == "test_value", "Redis value mismatch"
            logger.info("✅ Redis connectivity working")

        except Exception as e:
            pytest.skip(f"Redis not available: {e}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
