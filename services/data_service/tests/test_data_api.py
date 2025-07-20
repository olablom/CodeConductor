"""
Data Service - Integration Tests

This module contains integration tests for the Data Service API endpoints.
"""

import pytest
import httpx
import numpy as np
from typing import Dict, Any


class TestDataServiceAPI:
    """Test suite for Data Service API endpoints."""

    @pytest.fixture
    async def client(self):
        """Create test client."""
        async with httpx.AsyncClient(base_url="http://localhost:9006") as client:
            yield client

    @pytest.mark.asyncio
    async def test_health_check(self, client):
        """Test health check endpoint."""
        response = await client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "Data Service"
        assert data["version"] == "2.0.0"
        assert data["bandits_ready"] is True
        assert data["qlearning_ready"] is True
        assert data["prompt_optimizer_ready"] is True

    @pytest.mark.asyncio
    async def test_bandit_choose_arm(self, client):
        """Test bandit arm selection."""
        request_data = {
            "arms": ["arm1", "arm2", "arm3"],
            "features": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "bandit_name": "test_bandit",
        }

        response = await client.post("/bandits/choose", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "selected_arm" in data
        assert data["selected_arm"] in ["arm1", "arm2", "arm3"]
        assert "ucb_values" in data
        assert "confidence_intervals" in data
        assert data["bandit_name"] == "test_bandit"
        assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_bandit_update(self, client):
        """Test bandit update with reward."""
        # First choose an arm
        choose_request = {
            "arms": ["arm1", "arm2"],
            "features": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "bandit_name": "test_bandit_update",
        }

        choose_response = await client.post("/bandits/choose", json=choose_request)
        assert choose_response.status_code == 200

        selected_arm = choose_response.json()["selected_arm"]

        # Update with reward
        update_request = {
            "arm": selected_arm,
            "features": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "reward": 0.8,
            "bandit_name": "test_bandit_update",
        }

        update_response = await client.post("/bandits/update", json=update_request)
        assert update_response.status_code == 200

        data = update_response.json()
        assert data["updated"] is True
        assert data["arm"] == selected_arm
        assert data["reward"] == 0.8
        assert data["bandit_name"] == "test_bandit_update"

    @pytest.mark.asyncio
    async def test_bandit_stats(self, client):
        """Test bandit statistics endpoint."""
        # First use the bandit
        choose_request = {
            "arms": ["arm1", "arm2"],
            "features": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "bandit_name": "test_bandit_stats",
        }

        await client.post("/bandits/choose", json=choose_request)

        # Get stats
        response = await client.get("/bandits/test_bandit_stats/stats")
        assert response.status_code == 200

        data = response.json()
        assert data["bandit_name"] == "test_bandit_stats"
        assert data["total_pulls"] >= 1
        assert data["arm_count"] >= 2
        assert "arms" in data
        assert "config" in data

    @pytest.mark.asyncio
    async def test_qlearning_run(self, client):
        """Test Q-learning episode."""
        request_data = {
            "context": {
                "task_type": "api_creation",
                "complexity": "medium",
                "language": "python",
                "agent_count": 2,
            },
            "agent_name": "test_qlearning",
        }

        response = await client.post("/qlearning/run", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["agent_name"] == "test_qlearning"
        assert "selected_action" in data
        assert "q_value" in data
        assert "epsilon" in data
        assert "exploration" in data
        assert "confidence" in data
        assert "reasoning" in data
        assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_qlearning_update(self, client):
        """Test Q-learning update with reward."""
        # First run Q-learning
        run_request = {
            "context": {
                "task_type": "api_creation",
                "complexity": "medium",
                "language": "python",
                "agent_count": 2,
            },
            "agent_name": "test_qlearning_update",
        }

        run_response = await client.post("/qlearning/run", json=run_request)
        assert run_response.status_code == 200

        selected_action = run_response.json()["selected_action"]

        # Update with reward
        update_request = {
            "context": {
                "task_type": "api_creation",
                "complexity": "medium",
                "language": "python",
                "agent_count": 2,
            },
            "action": selected_action,
            "reward": 0.9,
            "next_context": {
                "task_type": "api_creation",
                "complexity": "medium",
                "language": "python",
                "agent_count": 2,
            },
            "agent_name": "test_qlearning_update",
        }

        update_response = await client.post("/qlearning/update", json=update_request)
        assert update_response.status_code == 200

        data = update_response.json()
        assert data["updated"] is True
        assert data["agent_name"] == "test_qlearning_update"
        assert "old_q_value" in data
        assert "new_q_value" in data
        assert data["reward"] == 0.9

    @pytest.mark.asyncio
    async def test_qlearning_stats(self, client):
        """Test Q-learning statistics endpoint."""
        # First use the agent
        run_request = {
            "context": {
                "task_type": "api_creation",
                "complexity": "medium",
                "language": "python",
                "agent_count": 2,
            },
            "agent_name": "test_qlearning_stats",
        }

        await client.post("/qlearning/run", json=run_request)

        # Get stats
        response = await client.get("/qlearning/test_qlearning_stats/stats")
        assert response.status_code == 200

        data = response.json()
        assert data["agent_name"] == "test_qlearning_stats"
        assert "total_episodes" in data
        assert "successful_episodes" in data
        assert "success_rate" in data
        assert "epsilon" in data
        assert "q_table_size" in data
        assert "average_q_value" in data
        assert "learning_rate" in data
        assert "discount_factor" in data

    @pytest.mark.asyncio
    async def test_prompt_optimize(self, client):
        """Test prompt optimization."""
        request_data = {
            "original_prompt": "Create a simple API endpoint",
            "task_id": "test_task_123",
            "arm_prev": "standard_arm",
            "passed": True,
            "blocked": False,
            "complexity": 0.5,
            "model_source": "gpt-4",
            "agent_name": "test_prompt_optimizer",
        }

        response = await client.post("/prompt/optimize", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["agent_name"] == "test_prompt_optimizer"
        assert "selected_action" in data
        assert data["original_prompt"] == "Create a simple API endpoint"
        assert "mutated_prompt" in data
        assert "mutation" in data
        assert "q_value" in data
        assert "epsilon" in data
        assert "exploration" in data
        assert "confidence" in data
        assert "reasoning" in data
        assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_prompt_optimizer_stats(self, client):
        """Test prompt optimizer statistics endpoint."""
        # First use the optimizer
        optimize_request = {
            "original_prompt": "Create a simple API endpoint",
            "task_id": "test_task_456",
            "arm_prev": "standard_arm",
            "passed": True,
            "blocked": False,
            "complexity": 0.5,
            "model_source": "gpt-4",
            "agent_name": "test_prompt_optimizer_stats",
        }

        await client.post("/prompt/optimize", json=optimize_request)

        # Get stats
        response = await client.get("/prompt/test_prompt_optimizer_stats/stats")
        assert response.status_code == 200

        data = response.json()
        assert data["agent_name"] == "test_prompt_optimizer_stats"
        assert "total_states" in data
        assert "total_entries" in data
        assert "average_q_value" in data
        assert "epsilon" in data
        assert "total_episodes" in data
        assert "successful_episodes" in data
        assert "success_rate" in data
        assert "action_stats" in data

    @pytest.mark.asyncio
    async def test_model_state(self, client):
        """Test model state management."""
        # Get bandit state
        request_data = {"model_name": "default", "model_type": "bandit"}

        response = await client.post("/models/state", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["model_name"] == "default"
        assert data["model_type"] == "bandit"
        assert "state" in data
        assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_model_reset(self, client):
        """Test model reset."""
        # Reset a bandit
        request_data = {"model_name": "test_reset_bandit", "model_type": "bandit"}

        response = await client.post("/models/reset", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["reset"] is True
        assert data["model_name"] == "test_reset_bandit"
        assert data["model_type"] == "bandit"
        assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_error_handling(self, client):
        """Test error handling."""
        # Test invalid bandit name
        response = await client.get("/bandits/nonexistent/stats")
        assert response.status_code == 404

        # Test invalid Q-learning agent
        response = await client.get("/qlearning/nonexistent/stats")
        assert response.status_code == 404

        # Test invalid prompt optimizer
        response = await client.get("/prompt/nonexistent/stats")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, client):
        """Test complete end-to-end workflow."""
        # 1. Choose bandit arm
        choose_request = {
            "arms": ["prompt1", "prompt2", "prompt3"],
            "features": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "bandit_name": "e2e_bandit",
        }

        choose_response = await client.post("/bandits/choose", json=choose_request)
        assert choose_response.status_code == 200
        selected_arm = choose_response.json()["selected_arm"]

        # 2. Update bandit with reward
        update_request = {
            "arm": selected_arm,
            "features": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "reward": 0.85,
            "bandit_name": "e2e_bandit",
        }

        update_response = await client.post("/bandits/update", json=update_request)
        assert update_response.status_code == 200

        # 3. Run Q-learning
        qlearning_request = {
            "context": {
                "task_type": "code_generation",
                "complexity": "high",
                "language": "python",
                "agent_count": 3,
            },
            "agent_name": "e2e_qlearning",
        }

        qlearning_response = await client.post("/qlearning/run", json=qlearning_request)
        assert qlearning_response.status_code == 200
        selected_action = qlearning_response.json()["selected_action"]

        # 4. Update Q-learning
        qlearning_update_request = {
            "context": {
                "task_type": "code_generation",
                "complexity": "high",
                "language": "python",
                "agent_count": 3,
            },
            "action": selected_action,
            "reward": 0.92,
            "agent_name": "e2e_qlearning",
        }

        qlearning_update_response = await client.post(
            "/qlearning/update", json=qlearning_update_request
        )
        assert qlearning_update_response.status_code == 200

        # 5. Optimize prompt
        prompt_request = {
            "original_prompt": "Create a REST API with authentication",
            "task_id": "e2e_task",
            "arm_prev": selected_arm,
            "passed": True,
            "blocked": False,
            "complexity": 0.7,
            "model_source": "gpt-4",
            "agent_name": "e2e_prompt_optimizer",
        }

        prompt_response = await client.post("/prompt/optimize", json=prompt_request)
        assert prompt_response.status_code == 200

        # 6. Verify all services are working
        health_response = await client.get("/health")
        assert health_response.status_code == 200
        health_data = health_response.json()
        assert health_data["status"] == "healthy"
        assert "e2e_bandit" in health_data["active_bandits"]
        assert "e2e_qlearning" in health_data["active_agents"]
        assert "e2e_prompt_optimizer" in health_data["active_agents"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
