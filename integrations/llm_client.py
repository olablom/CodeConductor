"""
LLMClient - Client for interacting with Language Models

This module provides a unified interface for communicating with different
LLM providers (Ollama, LM Studio, etc.) with caching and retry logic.
"""

import json
import time
import logging
from typing import Dict, Any, Optional
import requests

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Client for interacting with Language Models.

    Provides a unified interface for different LLM providers with
    caching, retry logic, and error handling.
    """

    def __init__(self, endpoint: str, model: str = "codellama:7b", **kwargs):
        """
        Initialize the LLM client.

        Args:
            endpoint: Base URL for the LLM service
            model: Model name to use for generation
            **kwargs: Additional configuration options
        """
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.config = kwargs
        self.cache = {}
        self.session = requests.Session()

        # Default configuration
        self.default_params = {
            "temperature": 0.7,
            "max_tokens": 2048,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }
        self.default_params.update(kwargs)

        logger.info(f"Initialized LLM client: {endpoint} with model {model}")

    def complete(self, prompt: str, **kwargs) -> str:
        """
        Generate a completion for the given prompt.

        Args:
            prompt: Input prompt for the model
            **kwargs: Override default parameters

        Returns:
            Generated text response

        Raises:
            LLMError: If the request fails
        """
        # Check cache first
        cache_key = self._get_cache_key(prompt, kwargs)
        if cache_key in self.cache:
            logger.debug(f"Cache hit for prompt: {prompt[:50]}...")
            return self.cache[cache_key]

        # Prepare parameters
        params = self.default_params.copy()
        params.update(kwargs)

        try:
            # For now, return mock response
            # TODO: Implement actual LLM API calls
            response = self._mock_complete(prompt, params)

            # Cache the response
            self.cache[cache_key] = response

            logger.debug(f"Generated response for prompt: {prompt[:50]}...")
            return response

        except Exception as e:
            logger.error(f"Error generating completion: {e}")
            raise LLMError(f"Failed to generate completion: {e}")

    def _mock_complete(self, prompt: str, params: Dict[str, Any]) -> str:
        """
        Mock completion for development/testing.

        Args:
            prompt: Input prompt
            params: Generation parameters

        Returns:
            Mock response
        """
        # Simulate processing time
        time.sleep(0.1)

        # Generate different responses based on prompt content
        prompt_lower = prompt.lower()

        # Check for code generation first (more specific)
        if "generate code" in prompt_lower or (
            "code" in prompt_lower and "generate" in prompt_lower
        ):
            return "```python\n# Mock code generation\ndef hello_world():\n    print('Hello, World!')\n```"
        elif "analyze" in prompt_lower:
            return "Analysis: This appears to be a request for analysis. Mock insights provided."
        elif "review" in prompt_lower:
            return "Review: Mock code review completed. All checks passed."
        elif "hello" in prompt_lower:
            return "Hello! I'm a mock LLM response. How can I help you today?"
        else:
            return f"Mock response for prompt: {prompt[:50]}... (temperature: {params.get('temperature', 0.7)})"

    def _get_cache_key(self, prompt: str, params: Dict[str, Any]) -> str:
        """
        Generate a cache key for the prompt and parameters.

        Args:
            prompt: Input prompt
            params: Generation parameters

        Returns:
            Cache key string
        """
        # Create a deterministic key
        key_data = {
            "prompt": prompt,
            "model": self.model,
            "params": sorted(params.items()),
        }
        return json.dumps(key_data, sort_keys=True)

    def clear_cache(self):
        """Clear the response cache."""
        self.cache.clear()
        logger.info("LLM client cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "cache_size": len(self.cache),
            "endpoint": self.endpoint,
            "model": self.model,
        }

    def test_connection(self) -> bool:
        """
        Test the connection to the LLM service.

        Returns:
            True if connection is successful, False otherwise
        """
        try:
            # For mock client, always return True
            # TODO: Implement actual connection test
            logger.info("Connection test successful (mock)")
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def __str__(self) -> str:
        """String representation of the client."""
        return f"LLMClient(endpoint='{self.endpoint}', model='{self.model}')"

    def __repr__(self) -> str:
        """Detailed string representation of the client."""
        return f"LLMClient(endpoint='{self.endpoint}', model='{self.model}', config={self.config})"


class LLMError(Exception):
    """Exception raised for LLM-related errors."""

    pass


# Convenience function for creating clients
def create_llm_client(provider: str = "mock", **kwargs) -> LLMClient:
    """
    Create an LLM client for the specified provider.

    Args:
        provider: Provider name ("mock", "ollama", "lm_studio")
        **kwargs: Provider-specific configuration

    Returns:
        Configured LLMClient instance
    """
    if provider == "mock":
        return LLMClient("http://localhost:1234", "mock-model", **kwargs)
    elif provider == "ollama":
        endpoint = kwargs.pop("endpoint", "http://localhost:11434")
        model = kwargs.pop("model", "codellama:7b")
        return LLMClient(endpoint, model, **kwargs)
    elif provider == "lm_studio":
        endpoint = kwargs.pop("endpoint", "http://localhost:1234")
        model = kwargs.pop("model", "local-model")
        return LLMClient(endpoint, model, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")
