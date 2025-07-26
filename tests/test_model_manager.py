#!/usr/bin/env python3
"""
Unit tests for ModelManager
"""

import pytest
import asyncio
import aiohttp
from unittest.mock import AsyncMock, patch, MagicMock
from ensemble.model_manager import ModelManager, ModelInfo


class TestModelManager:
    """Test cases for ModelManager class."""

    @pytest.fixture
    def model_manager(self):
        """Create a ModelManager instance for testing."""
        return ModelManager()

    @pytest.fixture
    def sample_lm_studio_response(self):
        """Sample LM Studio models response."""
        return {
            "data": [
                {
                    "id": "mistral-7b-instruct-v0.1",
                    "object": "model",
                    "owned_by": "organization_owner",
                },
                {
                    "id": "codellama-7b-instruct",
                    "object": "model",
                    "owned_by": "organization_owner",
                },
            ],
            "object": "list",
        }

    @pytest.fixture
    def sample_ollama_response(self):
        """Sample Ollama models response."""
        return {
            "models": [
                {
                    "name": "llama2",
                    "modified_at": "2024-01-01T00:00:00Z",
                    "size": 1234567890,
                },
                {
                    "name": "codellama",
                    "modified_at": "2024-01-01T00:00:00Z",
                    "size": 9876543210,
                },
            ]
        }

    @pytest.mark.asyncio
    async def test_discover_lm_studio_models_success(
        self, model_manager, sample_lm_studio_response
    ):
        """Test successful LM Studio model discovery."""
        with patch("aiohttp.ClientSession") as mock_session:
            # Mock successful response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=sample_lm_studio_response)

            # Properly mock the async context manager chain
            mock_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session.return_value
            )
            mock_session.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value.get.return_value.__aenter__ = AsyncMock(
                return_value=mock_response
            )
            mock_session.return_value.get.return_value.__aexit__ = AsyncMock(
                return_value=None
            )

            models = await model_manager._discover_lm_studio_models()

            assert len(models) == 2
            assert models[0].id == "mistral-7b-instruct-v0.1"
            assert models[0].provider == "lm_studio"
            assert models[0].is_available is True
            assert models[1].id == "codellama-7b-instruct"

    @pytest.mark.asyncio
    async def test_discover_lm_studio_models_failure(self, model_manager):
        """Test LM Studio model discovery failure."""
        with patch("aiohttp.ClientSession") as mock_session:
            # Mock failed response
            mock_response = AsyncMock()
            mock_response.status = 500

            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response

            models = await model_manager._discover_lm_studio_models()

            assert len(models) == 0

    @pytest.mark.asyncio
    async def test_discover_ollama_models_success(
        self, model_manager, sample_ollama_response
    ):
        """Test successful Ollama model discovery."""
        with patch("aiohttp.ClientSession") as mock_session:
            # Mock successful response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=sample_ollama_response)

            # Properly mock the async context manager chain
            mock_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session.return_value
            )
            mock_session.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value.get.return_value.__aenter__ = AsyncMock(
                return_value=mock_response
            )
            mock_session.return_value.get.return_value.__aexit__ = AsyncMock(
                return_value=None
            )

            models = await model_manager._discover_ollama_models()

            assert len(models) == 2
            assert models[0].id == "llama2"
            assert models[0].provider == "ollama"
            assert models[0].is_available is True
            assert models[1].id == "codellama"

    @pytest.mark.asyncio
    async def test_discover_ollama_models_failure(self, model_manager):
        """Test Ollama model discovery failure."""
        with patch("aiohttp.ClientSession") as mock_session:
            # Mock failed response
            mock_response = AsyncMock()
            mock_response.status = 404

            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response

            models = await model_manager._discover_ollama_models()

            assert len(models) == 0

    @pytest.mark.asyncio
    async def test_list_models_success(
        self, model_manager, sample_lm_studio_response, sample_ollama_response
    ):
        """Test successful model listing with both providers."""
        with (
            patch.object(model_manager, "_discover_lm_studio_models") as mock_lm,
            patch.object(model_manager, "_discover_ollama_models") as mock_ollama,
        ):
            # Mock successful discoveries
            mock_lm.return_value = [
                ModelInfo(
                    id="mistral",
                    name="mistral",
                    provider="lm_studio",
                    endpoint="http://localhost:1234/v1",
                    is_available=True,
                )
            ]
            mock_ollama.return_value = [
                ModelInfo(
                    id="llama2",
                    name="llama2",
                    provider="ollama",
                    endpoint="http://localhost:11434",
                    is_available=True,
                )
            ]

            models = await model_manager.list_models()

            assert len(models) == 2
            assert any(m.provider == "lm_studio" for m in models)
            assert any(m.provider == "ollama" for m in models)

    @pytest.mark.asyncio
    async def test_list_models_partial_failure(
        self, model_manager, sample_lm_studio_response
    ):
        """Test model listing when one provider fails."""
        with (
            patch.object(model_manager, "_discover_lm_studio_models") as mock_lm,
            patch.object(model_manager, "_discover_ollama_models") as mock_ollama,
        ):
            # Mock LM Studio success, Ollama failure
            mock_lm.return_value = [
                ModelInfo(
                    id="mistral",
                    name="mistral",
                    provider="lm_studio",
                    endpoint="http://localhost:1234/v1",
                    is_available=True,
                )
            ]
            mock_ollama.side_effect = Exception("Connection failed")

            models = await model_manager.list_models()

            assert len(models) == 1
            assert models[0].provider == "lm_studio"

    @pytest.mark.asyncio
    async def test_check_lm_studio_health_success(self, model_manager):
        """Test successful LM Studio health check."""
        model_info = ModelInfo(
            id="mistral-7b-instruct-v0.1",
            name="mistral-7b-instruct-v0.1",
            provider="lm_studio",
            endpoint="http://localhost:1234/v1",
            is_available=True,
        )

        with patch("aiohttp.ClientSession") as mock_session:
            # Mock successful health check
            mock_response = AsyncMock()
            mock_response.status = 200

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

            is_healthy = await model_manager._check_lm_studio_health(model_info)

            assert is_healthy is True

    @pytest.mark.asyncio
    async def test_check_lm_studio_health_failure(self, model_manager):
        """Test failed LM Studio health check."""
        model_info = ModelInfo(
            id="mistral-7b-instruct-v0.1",
            name="mistral-7b-instruct-v0.1",
            provider="lm_studio",
            endpoint="http://localhost:1234/v1",
            is_available=True,
        )

        with patch("aiohttp.ClientSession") as mock_session:
            # Mock failed health check
            mock_response = AsyncMock()
            mock_response.status = 500

            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response

            is_healthy = await model_manager._check_lm_studio_health(model_info)

            assert is_healthy is False

    @pytest.mark.asyncio
    async def test_check_ollama_health_success(self, model_manager):
        """Test successful Ollama health check."""
        model_info = ModelInfo(
            id="llama2",
            name="llama2",
            provider="ollama",
            endpoint="http://localhost:11434",
            is_available=True,
        )

        with patch("aiohttp.ClientSession") as mock_session:
            # Mock successful health check
            mock_response = AsyncMock()
            mock_response.status = 200

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

            is_healthy = await model_manager._check_ollama_health(model_info)

            assert is_healthy is True

    @pytest.mark.asyncio
    async def test_check_ollama_health_failure(self, model_manager):
        """Test failed Ollama health check."""
        model_info = ModelInfo(
            id="llama2",
            name="llama2",
            provider="ollama",
            endpoint="http://localhost:11434",
            is_available=True,
        )

        with patch("aiohttp.ClientSession") as mock_session:
            # Mock failed health check
            mock_response = AsyncMock()
            mock_response.status = 404

            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response

            is_healthy = await model_manager._check_ollama_health(model_info)

            assert is_healthy is False

    @pytest.mark.asyncio
    async def test_check_health_unknown_provider(self, model_manager):
        """Test health check with unknown provider."""
        model_info = ModelInfo(
            id="unknown",
            name="unknown",
            provider="unknown_provider",
            endpoint="http://localhost:9999",
            is_available=True,
        )

        is_healthy = await model_manager.check_health(model_info)

        assert is_healthy is False

    @pytest.mark.asyncio
    async def test_check_all_health(self, model_manager):
        """Test health checking all models."""
        models = [
            ModelInfo(
                id="model1",
                name="model1",
                provider="lm_studio",
                endpoint="http://localhost:1234/v1",
                is_available=True,
            ),
            ModelInfo(
                id="model2",
                name="model2",
                provider="ollama",
                endpoint="http://localhost:11434",
                is_available=True,
            ),
        ]

        with patch.object(model_manager, "check_health") as mock_check:
            # Mock health check results
            mock_check.side_effect = [True, False]

            health_status = await model_manager.check_all_health(models)

            assert len(health_status) == 2
            assert health_status["model1"] is True
            assert health_status["model2"] is False
            assert mock_check.call_count == 2


class TestModelInfo:
    """Test cases for ModelInfo dataclass."""

    def test_model_info_creation(self):
        """Test ModelInfo object creation."""
        model_info = ModelInfo(
            id="test-model",
            name="Test Model",
            provider="lm_studio",
            endpoint="http://localhost:1234/v1",
            is_available=True,
            metadata={"version": "1.0"},
        )

        assert model_info.id == "test-model"
        assert model_info.name == "Test Model"
        assert model_info.provider == "lm_studio"
        assert model_info.endpoint == "http://localhost:1234/v1"
        assert model_info.is_available is True
        assert model_info.metadata["version"] == "1.0"

    def test_model_info_default_metadata(self):
        """Test ModelInfo with default metadata."""
        model_info = ModelInfo(
            id="test-model",
            name="Test Model",
            provider="ollama",
            endpoint="http://localhost:11434",
            is_available=False,
        )

        assert model_info.metadata is None


# Integration test (requires actual services)
@pytest.mark.integration
class TestModelManagerIntegration:
    """Integration tests for ModelManager (requires actual LM Studio/Ollama)."""

    @pytest.mark.asyncio
    async def test_real_discovery(self):
        """Test discovery with real services (if available)."""
        manager = ModelManager()
        models = await manager.list_models()

        # Should not crash even if services are not available
        assert isinstance(models, list)

        if models:
            # If models are found, test health checking
            health_status = await manager.check_all_health(models)
            assert isinstance(health_status, dict)
            assert len(health_status) == len(models)


if __name__ == "__main__":
    pytest.main([__file__])
