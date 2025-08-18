#!/usr/bin/env python3
# Filename: tests/test_vllm_integration.py
"""
Test vLLM Integration - Only runs when vLLM is available
"""

import platform
from unittest.mock import Mock, patch

import pytest


def _is_wsl() -> bool:
    rel = platform.release().lower()
    ver = getattr(platform, "version", lambda: "")().lower() if hasattr(platform, "version") else ""
    return ("microsoft" in rel) or ("microsoft" in ver)


def _vllm_available():
    """Check if vLLM is available for testing"""
    system = platform.system()

    # Native Windows: vLLM not supported
    if system == "Windows" and not _is_wsl():
        return False

    # WSL2/Linux/macOS: try to import vLLM
    try:
        import vllm  # noqa: F401

        return True
    except ImportError:
        return False


# Skip all tests if vLLM not available
pytestmark = pytest.mark.skipif(not _vllm_available(), reason="vLLM not available on this platform")

try:
    from codeconductor.vllm_integration import VLLMEngine, create_vllm_engine
except ImportError:
    # Skip all tests if vLLM is not available
    pytestmark = pytest.mark.skip("vLLM not available")


class TestVLLMEngine:
    """Test cases for VLLMEngine."""

    @pytest.fixture
    async def engine(self):
        """Create a test vLLM engine."""
        engine = VLLMEngine(
            model_name="microsoft/DialoGPT-medium",
            quantization="awq",
            max_model_len=4096,
        )
        yield engine
        await engine.cleanup()

    @pytest.mark.asyncio
    async def test_engine_initialization(self, engine):
        """Test that the engine initializes correctly."""
        assert not engine._initialized
        assert engine.llm is None

        await engine.initialize()

        assert engine._initialized
        assert engine.llm is not None
        assert engine.sampling_params is not None

    @pytest.mark.asyncio
    async def test_generate_code(self, engine):
        """Test code generation functionality."""
        await engine.initialize()

        prompt = "Write a Python function to add two numbers:"

        # Mock the LLM generate method
        with patch.object(engine.llm, "generate") as mock_generate:
            mock_output = Mock()
            mock_output.outputs = [Mock()]
            mock_output.outputs[0].text = "def add(a, b):\n    return a + b"
            mock_generate.return_value = [mock_output]

            result = await engine.generate_code(prompt)

            assert "def add" in result
            assert "return a + b" in result
            mock_generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_with_consensus(self, engine):
        """Test consensus generation with multiple temperatures."""
        await engine.initialize()

        prompt = "Write a simple Python function:"

        # Mock the LLM generate method
        with patch.object(engine.llm, "generate") as mock_generate:
            mock_output = Mock()
            mock_output.outputs = [Mock()]
            mock_output.outputs[0].text = "def simple():\n    pass"
            mock_generate.return_value = [mock_output]

            result = await engine.generate_with_consensus(prompt)

            assert "generations" in result
            assert "consensus_metrics" in result
            assert "best_generation" in result
            assert len(result["generations"]) > 0

    def test_calculate_similarity(self, engine):
        """Test similarity calculation between code snippets."""
        code1 = "def add(a, b): return a + b"
        code2 = "def add(x, y): return x + y"
        code3 = "def multiply(a, b): return a * b"

        sim1 = engine._calculate_similarity(code1, code2)
        sim2 = engine._calculate_similarity(code1, code3)

        assert 0 <= sim1 <= 1
        assert 0 <= sim2 <= 1
        assert sim1 > sim2  # Similar functions should be more similar

    def test_calculate_variance(self, engine):
        """Test variance calculation."""
        values = [1, 2, 3, 4, 5]
        variance = engine._calculate_variance(values)

        assert variance == 2.0  # Expected variance for [1,2,3,4,5]

    def test_select_best_generation(self, engine):
        """Test best generation selection."""
        results = [
            {"temperature": 0.1, "code": "short", "length": 100},
            {"temperature": 0.2, "code": "medium", "length": 500},
            {"temperature": 0.5, "code": "long", "length": 1000},
        ]

        best = engine._select_best_generation(results)

        assert best is not None
        assert "temperature" in best
        assert "code" in best
        assert "length" in best

    @pytest.mark.asyncio
    async def test_get_model_info(self, engine):
        """Test model information retrieval."""
        await engine.initialize()

        info = engine.get_model_info()

        assert "model_name" in info
        assert "quantization" in info
        assert "max_model_len" in info
        assert info["model_name"] == "microsoft/DialoGPT-medium"
        assert info["quantization"] == "awq"


class TestVLLMFactory:
    """Test cases for vLLM factory functions."""

    @pytest.mark.asyncio
    async def test_create_vllm_engine(self):
        """Test vLLM engine creation."""
        engine = await create_vllm_engine(
            model_name="microsoft/DialoGPT-medium", quantization="awq"
        )

        assert isinstance(engine, VLLMEngine)
        assert engine._initialized
        assert engine.model_name == "microsoft/DialoGPT-medium"
        assert engine.quantization == "awq"

        await engine.cleanup()


# Integration test for actual vLLM functionality
@pytest.mark.integration
class TestVLLMIntegration:
    """Integration tests for actual vLLM functionality."""

    @pytest.mark.asyncio
    async def test_real_vllm_generation(self):
        """Test actual vLLM code generation (requires vLLM to be installed)."""
        try:
            engine = await create_vllm_engine(
                model_name="microsoft/DialoGPT-medium", quantization="awq"
            )

            prompt = "Write a Python function to calculate the sum of a list:"
            result = await engine.generate_code(prompt)

            assert len(result) > 0
            assert "def" in result.lower() or "function" in result.lower()

            await engine.cleanup()

        except ImportError:
            pytest.skip("vLLM not available for integration test")
        except Exception as e:
            pytest.skip(f"vLLM integration test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
