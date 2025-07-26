#!/usr/bin/env python3
"""
Unit tests for QueryDispatcher
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, patch, MagicMock
from aiohttp import ClientError, ClientTimeout
from ensemble.query_dispatcher import QueryDispatcher
from ensemble.model_manager import ModelInfo


class TestQueryDispatcher:
    """Test cases for QueryDispatcher class."""

    @pytest.fixture
    def dispatcher(self):
        """Create a QueryDispatcher instance for testing."""
        return QueryDispatcher(timeout=5)

    @pytest.fixture
    def sample_lm_studio_response(self):
        """Sample LM Studio response."""
        return {
            "choices": [{"message": {"content": "4", "role": "assistant"}}],
            "model": "mistral-7b-instruct-v0.1",
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        }

    @pytest.fixture
    def sample_ollama_response(self):
        """Sample Ollama response."""
        return {
            "model": "llama2",
            "response": "4",
            "done": True,
            "context": [],
            "total_duration": 1234567890,
            "load_duration": 123456789,
            "prompt_eval_count": 10,
            "prompt_eval_duration": 12345678,
            "eval_count": 1,
            "eval_duration": 1234567,
        }

    @pytest.fixture
    def sample_models(self):
        """Sample models for testing."""
        return [
            ModelInfo(
                id="mistral-7b-instruct-v0.1",
                name="mistral-7b-instruct-v0.1",
                provider="lm_studio",
                endpoint="http://localhost:1234/v1",
                is_available=True,
            ),
            ModelInfo(
                id="llama2",
                name="llama2",
                provider="ollama",
                endpoint="http://localhost:11434",
                is_available=True,
            ),
        ]

    @pytest.mark.asyncio
    async def test_query_lm_studio_model_success(
        self, dispatcher, sample_lm_studio_response
    ):
        """Test successful LM Studio model query."""
        model_info = ModelInfo(
            id="mistral-7b-instruct-v0.1",
            name="mistral-7b-instruct-v0.1",
            provider="lm_studio",
            endpoint="http://localhost:1234/v1",
            is_available=True,
        )

        with patch("aiohttp.ClientSession") as mock_session:
            # Mock successful response
            mock_response = AsyncMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json = AsyncMock(return_value=sample_lm_studio_response)

            # Properly mock the async context manager chain
            mock_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session.return_value
            )
            mock_session.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value.post.return_value.__aenter__ = AsyncMock(
                return_value=mock_response
            )
            mock_session.return_value.post.return_value.__aexit__ = AsyncMock(
                return_value=None
            )

            model_id, response = await dispatcher._query_lm_studio_model(
                mock_session.return_value, model_info, "What is 2 + 2?"
            )

            assert model_id == "mistral-7b-instruct-v0.1"
            assert response == sample_lm_studio_response

    @pytest.mark.asyncio
    async def test_query_lm_studio_model_timeout(self, dispatcher):
        """Test LM Studio model query timeout."""
        model_info = ModelInfo(
            id="mistral-7b-instruct-v0.1",
            name="mistral-7b-instruct-v0.1",
            provider="lm_studio",
            endpoint="http://localhost:1234/v1",
            is_available=True,
        )

        with patch("aiohttp.ClientSession") as mock_session:
            # Mock timeout
            mock_session.return_value.post.side_effect = asyncio.TimeoutError()

            model_id, response = await dispatcher._query_lm_studio_model(
                mock_session.return_value, model_info, "What is 2 + 2?"
            )

            assert model_id == "mistral-7b-instruct-v0.1"
            assert response["error"] == "timeout"
            assert response["model"] == "mistral-7b-instruct-v0.1"

    @pytest.mark.asyncio
    async def test_query_ollama_model_success(self, dispatcher, sample_ollama_response):
        """Test successful Ollama model query."""
        model_info = ModelInfo(
            id="llama2",
            name="llama2",
            provider="ollama",
            endpoint="http://localhost:11434",
            is_available=True,
        )

        with patch("aiohttp.ClientSession") as mock_session:
            # Mock successful response
            mock_response = AsyncMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json = AsyncMock(return_value=sample_ollama_response)

            # Properly mock the async context manager chain
            mock_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session.return_value
            )
            mock_session.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value.post.return_value.__aenter__ = AsyncMock(
                return_value=mock_response
            )
            mock_session.return_value.post.return_value.__aexit__ = AsyncMock(
                return_value=None
            )

            model_id, response = await dispatcher._query_ollama_model(
                mock_session.return_value, model_info, "What is 2 + 2?"
            )

            assert model_id == "llama2"
            assert response == sample_ollama_response

    @pytest.mark.asyncio
    async def test_query_ollama_model_error(self, dispatcher):
        """Test Ollama model query error."""
        model_info = ModelInfo(
            id="llama2",
            name="llama2",
            provider="ollama",
            endpoint="http://localhost:11434",
            is_available=True,
        )

        with patch("aiohttp.ClientSession") as mock_session:
            # Mock HTTP error
            mock_session.return_value.post.side_effect = ClientError(
                "Connection failed"
            )

            model_id, response = await dispatcher._query_ollama_model(
                mock_session.return_value, model_info, "What is 2 + 2?"
            )

            assert model_id == "llama2"
            assert "error" in response
            assert "Connection failed" in response["error"]

    @pytest.mark.asyncio
    async def test_query_model_unknown_provider(self, dispatcher):
        """Test query with unknown provider."""
        model_info = ModelInfo(
            id="unknown",
            name="unknown",
            provider="unknown_provider",
            endpoint="http://localhost:9999",
            is_available=True,
        )

        with patch("aiohttp.ClientSession") as mock_session:
            model_id, response = await dispatcher._query_model(
                mock_session.return_value, model_info, "What is 2 + 2?"
            )

            assert model_id == "unknown"
            assert response["error"] == "Unknown provider: unknown_provider"

    @pytest.mark.asyncio
    async def test_dispatch_success(
        self,
        dispatcher,
        sample_models,
        sample_lm_studio_response,
        sample_ollama_response,
    ):
        """Test successful dispatch to multiple models."""
        with (
            patch.object(dispatcher.model_manager, "list_models") as mock_list_models,
            patch.object(dispatcher, "_query_model") as mock_query_model,
        ):
            # Mock model discovery
            mock_list_models.return_value = sample_models

            # Mock successful responses
            mock_query_model.side_effect = [
                ("mistral-7b-instruct-v0.1", sample_lm_studio_response),
                ("llama2", sample_ollama_response),
            ]

            results = await dispatcher.dispatch("What is 2 + 2?")

            assert len(results) == 2
            assert "mistral-7b-instruct-v0.1" in results
            assert "llama2" in results
            assert results["mistral-7b-instruct-v0.1"] == sample_lm_studio_response
            assert results["llama2"] == sample_ollama_response

    @pytest.mark.asyncio
    async def test_dispatch_no_models(self, dispatcher):
        """Test dispatch when no models are available."""
        with patch.object(dispatcher.model_manager, "list_models") as mock_list_models:
            mock_list_models.return_value = []

            results = await dispatcher.dispatch("What is 2 + 2?")

            assert results == {}

    @pytest.mark.asyncio
    async def test_dispatch_with_max_models(
        self, dispatcher, sample_models, sample_lm_studio_response
    ):
        """Test dispatch with max_models limit."""
        with (
            patch.object(dispatcher.model_manager, "list_models") as mock_list_models,
            patch("aiohttp.ClientSession") as mock_session,
        ):
            # Mock model discovery
            mock_list_models.return_value = sample_models

            # Mock successful response
            mock_response = AsyncMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json = AsyncMock(return_value=sample_lm_studio_response)

            # Properly mock the async context manager chain
            mock_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session.return_value
            )
            mock_session.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value.post.return_value.__aenter__ = AsyncMock(
                return_value=mock_response
            )
            mock_session.return_value.post.return_value.__aexit__ = AsyncMock(
                return_value=None
            )

            results = await dispatcher.dispatch("What is 2 + 2?", max_models=1)

            assert len(results) == 1
            assert "mistral-7b-instruct-v0.1" in results

    @pytest.mark.asyncio
    async def test_dispatch_mixed_success_error(
        self, dispatcher, sample_models, sample_lm_studio_response
    ):
        """Test dispatch with mixed success and error responses."""
        with (
            patch.object(dispatcher.model_manager, "list_models") as mock_list_models,
            patch.object(dispatcher, "_query_model") as mock_query_model,
        ):
            # Mock model discovery
            mock_list_models.return_value = sample_models

            # Mock mixed responses: success for first, error for second
            mock_query_model.side_effect = [
                ("mistral-7b-instruct-v0.1", sample_lm_studio_response),
                ("llama2", {"error": "Connection failed", "model": "llama2"}),
            ]

            results = await dispatcher.dispatch("What is 2 + 2?")

            assert len(results) == 2
            assert "mistral-7b-instruct-v0.1" in results
            assert "llama2" in results
            assert results["mistral-7b-instruct-v0.1"] == sample_lm_studio_response
            assert "error" in results["llama2"]

    @pytest.mark.asyncio
    async def test_dispatch_to_healthy_models(
        self, dispatcher, sample_models, sample_lm_studio_response
    ):
        """Test dispatch to healthy models only."""
        with (
            patch.object(dispatcher.model_manager, "list_models") as mock_list_models,
            patch.object(dispatcher.model_manager, "check_all_health") as mock_health,
            patch.object(dispatcher, "_query_model") as mock_query_model,
        ):
            # Mock model discovery
            mock_list_models.return_value = sample_models

            # Mock health check - only first model is healthy
            mock_health.return_value = {
                "mistral-7b-instruct-v0.1": True,
                "llama2": False,
            }

            # Mock successful response
            mock_query_model.return_value = (
                "mistral-7b-instruct-v0.1",
                sample_lm_studio_response,
            )

            results = await dispatcher.dispatch_to_healthy_models("What is 2 + 2?")

            assert len(results) == 1
            assert "mistral-7b-instruct-v0.1" in results
            assert "llama2" not in results

    @pytest.mark.asyncio
    async def test_dispatch_to_healthy_models_none_healthy(
        self, dispatcher, sample_models
    ):
        """Test dispatch to healthy models when none are healthy."""
        with (
            patch.object(dispatcher.model_manager, "list_models") as mock_list_models,
            patch.object(dispatcher.model_manager, "check_all_health") as mock_health,
        ):
            # Mock model discovery
            mock_list_models.return_value = sample_models

            # Mock health check - no models are healthy
            mock_health.return_value = {
                "mistral-7b-instruct-v0.1": False,
                "llama2": False,
            }

            results = await dispatcher.dispatch_to_healthy_models("What is 2 + 2?")

            assert results == {}


class TestQueryDispatcherIntegration:
    """Integration tests for QueryDispatcher (requires actual services)."""

    @pytest.mark.asyncio
    async def test_real_dispatch(self):
        """Test dispatch with real services (if available)."""
        dispatcher = QueryDispatcher(timeout=10)

        # Test with a simple prompt
        results = await dispatcher.dispatch("What is 2 + 2?", max_models=1)

        # Should not crash even if services are not available
        assert isinstance(results, dict)

        if results:
            # If we got results, verify structure
            for model_id, response in results.items():
                assert isinstance(model_id, str)
                assert isinstance(response, dict)
                if "error" not in response:
                    # Should have either choices (LM Studio) or response (Ollama)
                    assert "choices" in response or "response" in response


if __name__ == "__main__":
    pytest.main([__file__])
