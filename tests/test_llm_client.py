"""
Tests for LLMClient class.

This module tests the LLMClient class and ensures it properly handles
mock responses, caching, and error conditions.
"""

import pytest
import time
from integrations.llm_client import LLMClient, LLMError, create_llm_client


class TestLLMClient:
    """Test cases for LLMClient class."""

    def test_llm_client_initialization(self):
        """Test LLMClient initialization."""
        client = LLMClient("http://localhost:1234", "test-model")

        assert client.endpoint == "http://localhost:1234"
        assert client.model == "test-model"
        assert client.cache == {}
        assert client.session is not None

    def test_llm_client_with_config(self):
        """Test LLMClient initialization with configuration."""
        config = {"temperature": 0.5, "max_tokens": 1024}
        client = LLMClient("http://localhost:1234", "test-model", **config)

        assert client.config == config
        assert client.default_params["temperature"] == 0.5
        assert client.default_params["max_tokens"] == 1024

    def test_complete_hello_prompt(self):
        """Test completion with 'hello' prompt."""
        client = LLMClient("http://localhost:1234")

        response = client.complete("hello world")

        assert "Hello!" in response
        assert "mock LLM response" in response

    def test_complete_code_prompt(self):
        """Test completion with 'code' prompt."""
        client = LLMClient("http://localhost:1234")

        response = client.complete("generate code for hello world")

        assert "```python" in response
        assert "def hello_world()" in response
        assert "print('Hello, World!')" in response

    def test_complete_analyze_prompt(self):
        """Test completion with 'analyze' prompt."""
        client = LLMClient("http://localhost:1234")

        response = client.complete("analyze this code")

        assert "Analysis:" in response
        assert "Mock insights" in response

    def test_complete_review_prompt(self):
        """Test completion with 'review' prompt."""
        client = LLMClient("http://localhost:1234")

        response = client.complete("review this code")

        assert "Review:" in response
        assert "Mock code review" in response

    def test_complete_generic_prompt(self):
        """Test completion with generic prompt."""
        client = LLMClient("http://localhost:1234")

        response = client.complete("This is a generic prompt")

        assert "Mock response for prompt:" in response
        assert "temperature:" in response

    def test_complete_with_parameters(self):
        """Test completion with custom parameters."""
        client = LLMClient("http://localhost:1234")

        response = client.complete("test prompt", temperature=0.3)

        assert "temperature: 0.3" in response

    def test_caching(self):
        """Test that responses are cached."""
        client = LLMClient("http://localhost:1234")

        # First call should not be cached
        response1 = client.complete("cached prompt")
        assert len(client.cache) == 1

        # Second call should be cached
        response2 = client.complete("cached prompt")
        assert len(client.cache) == 1  # Same cache key
        assert response1 == response2

    def test_cache_with_different_parameters(self):
        """Test that different parameters create different cache entries."""
        client = LLMClient("http://localhost:1234")

        # Same prompt, different parameters
        client.complete("test prompt", temperature=0.7)
        client.complete("test prompt", temperature=0.3)

        assert len(client.cache) == 2

    def test_clear_cache(self):
        """Test cache clearing."""
        client = LLMClient("http://localhost:1234")

        # Add some responses to cache
        client.complete("prompt 1")
        client.complete("prompt 2")
        assert len(client.cache) == 2

        # Clear cache
        client.clear_cache()
        assert len(client.cache) == 0

    def test_get_cache_stats(self):
        """Test getting cache statistics."""
        client = LLMClient("http://localhost:1234", "test-model")

        # Add a response to cache
        client.complete("test prompt")

        stats = client.get_cache_stats()

        assert stats["cache_size"] == 1
        assert stats["endpoint"] == "http://localhost:1234"
        assert stats["model"] == "test-model"

    def test_test_connection(self):
        """Test connection testing."""
        client = LLMClient("http://localhost:1234")

        # Mock client should always return True
        assert client.test_connection() is True

    def test_string_representation(self):
        """Test string representation."""
        client = LLMClient("http://localhost:1234", "test-model")

        expected = "LLMClient(endpoint='http://localhost:1234', model='test-model')"
        assert str(client) == expected

    def test_repr_representation(self):
        """Test detailed string representation."""
        config = {"temperature": 0.7}
        client = LLMClient("http://localhost:1234", "test-model", **config)

        expected = "LLMClient(endpoint='http://localhost:1234', model='test-model', config={'temperature': 0.7})"
        assert repr(client) == expected

    def test_endpoint_normalization(self):
        """Test that endpoint is normalized (trailing slash removed)."""
        client = LLMClient("http://localhost:1234/", "test-model")

        assert client.endpoint == "http://localhost:1234"


class TestLLMClientErrorHandling:
    """Test error handling in LLMClient."""

    def test_llm_error_exception(self):
        """Test that LLMError exception can be raised."""
        error = LLMError("Test error message")
        assert str(error) == "Test error message"


class TestCreateLLMClient:
    """Test the create_llm_client convenience function."""

    def test_create_mock_client(self):
        """Test creating a mock client."""
        client = create_llm_client("mock")

        assert isinstance(client, LLMClient)
        assert client.endpoint == "http://localhost:1234"
        assert client.model == "mock-model"

    def test_create_ollama_client(self):
        """Test creating an Ollama client."""
        client = create_llm_client("ollama", endpoint="http://localhost:11434", model="codellama:7b")

        assert isinstance(client, LLMClient)
        assert client.endpoint == "http://localhost:11434"
        assert client.model == "codellama:7b"

    def test_create_lm_studio_client(self):
        """Test creating an LM Studio client."""
        client = create_llm_client("lm_studio", endpoint="http://localhost:1234", model="local-model")

        assert isinstance(client, LLMClient)
        assert client.endpoint == "http://localhost:1234"
        assert client.model == "local-model"

    def test_create_unknown_provider(self):
        """Test creating client with unknown provider."""
        with pytest.raises(ValueError, match="Unknown provider: unknown"):
            create_llm_client("unknown")


class TestLLMClientIntegration:
    """Integration tests for LLMClient."""

    def test_client_with_agent_integration(self):
        """Test LLMClient integration with BaseAgent."""
        from agents.base_agent import BaseAgent

        class TestAgent(BaseAgent):
            def analyze(self, context):
                return {"insights": ["Test insight"]}

            def propose(self, analysis):
                return {"solution": "Test solution"}

            def review(self, code):
                return {"quality_score": 0.8}

        # Create agent and client
        agent = TestAgent("test_agent")
        client = LLMClient("http://localhost:1234")

        # Connect them
        agent.set_llm_client(client)

        # Verify connection
        assert agent.llm_client is client
        status = agent.get_status()
        assert status["has_llm_client"] is True

    def test_multiple_completions(self):
        """Test multiple completions with different prompts."""
        client = LLMClient("http://localhost:1234")

        prompts = ["hello world", "generate code", "analyze this", "review code"]

        responses = []
        for prompt in prompts:
            response = client.complete(prompt)
            responses.append(response)

        # Verify all responses are different
        assert len(set(responses)) == len(responses)

        # Verify cache size
        assert len(client.cache) == len(prompts)

    def test_performance_with_caching(self):
        """Test that caching improves performance."""
        client = LLMClient("http://localhost:1234")

        prompt = "performance test prompt"

        # First call (should be slower due to processing)
        start_time = time.time()
        response1 = client.complete(prompt)
        first_call_time = time.time() - start_time

        # Second call (should be faster due to caching)
        start_time = time.time()
        response2 = client.complete(prompt)
        second_call_time = time.time() - start_time

        # Verify responses are identical
        assert response1 == response2

        # Verify second call is faster (cached)
        assert second_call_time < first_call_time
