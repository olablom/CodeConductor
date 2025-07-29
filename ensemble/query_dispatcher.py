#!/usr/bin/env python3
"""
Query Dispatcher for CodeConductor MVP
Dispatches prompts to multiple LLM models in parallel with timeout and error handling.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Tuple, Optional
from aiohttp import ClientSession, ClientError, ClientTimeout
from .model_manager import ModelManager, ModelInfo

# Configure logging
logger = logging.getLogger(__name__)

# Timeout configuration - different timeouts for different providers
FAST_TIMEOUT = 30  # seconds - for fast models like Ollama
SLOW_TIMEOUT = 120  # seconds - for slower models like LM Studio

# Model-specific configurations for optimal performance
BASE_MODEL_CONFIGS = {
    "phi3": {
        "temperature": 0.3,  # Slightly higher for more creative code
        "max_tokens": 2048,  # Double the tokens for better code generation
        "top_p": 0.9,  # Nucleus sampling for better quality
        "frequency_penalty": 0.1,  # Reduce repetition
        "presence_penalty": 0.1,  # Encourage diverse outputs
        "timeout": 45,  # Slightly longer timeout for phi3
    },
    "codellama": {
        "temperature": 0.2,  # Lower for more deterministic code
        "max_tokens": 3072,  # More tokens for complex code
        "top_p": 0.95,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "timeout": 60,
    },
    "mistral": {
        "temperature": 0.25,
        "max_tokens": 2048,
        "top_p": 0.9,
        "frequency_penalty": 0.05,
        "presence_penalty": 0.05,
        "timeout": 60,
    },
    "deepseek": {
        "temperature": 0.2,
        "max_tokens": 2048,
        "top_p": 0.9,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "timeout": 60,
    },
    "gemma": {
        "temperature": 0.2,
        "max_tokens": 2048,
        "top_p": 0.9,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "timeout": 60,
    },
    "llama": {
        "temperature": 0.2,
        "max_tokens": 2048,
        "top_p": 0.9,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "timeout": 60,
    },
}

# Default configuration for unknown models
DEFAULT_CONFIG = {
    "temperature": 0.1,
    "max_tokens": 1024,
    "top_p": 0.9,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "timeout": FAST_TIMEOUT,
}


def get_model_config(model_id: str) -> dict:
    """
    Dynamically match model ID to base configuration using pattern matching.
    Falls back to default config for unknown models.
    """
    model_id_lower = model_id.lower()

    # Try to match against base model patterns
    for base_model, config in BASE_MODEL_CONFIGS.items():
        if base_model in model_id_lower:
            logger.info(f"✅ Matched {model_id} to {base_model} config")
            return config

    # Fallback to default
    logger.info(f"⚠️ No specific config for {model_id}, using default")
    return DEFAULT_CONFIG


class QueryDispatcher:
    """
    Dispatches prompts to multiple LLM models in parallel, handles timeouts and errors.
    """

    def __init__(self, model_manager: ModelManager = None, timeout: int = None) -> None:
        self.timeout = timeout  # Will be set per-request based on provider
        self.model_manager = model_manager or ModelManager()
        self.session = None

    async def __aenter__(self):
        """Initialize async context manager."""
        self.session = ClientSession()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """Clean up async context manager."""
        if self.session:
            await self.session.close()

    async def _query_lm_studio_model(
        self, session: ClientSession, model_info: ModelInfo, prompt: str
    ) -> Tuple[str, Any]:
        """
        Send prompt to a single LM Studio model and return (model_id, response_json).
        """
        url = f"{model_info.endpoint}/chat/completions"

        # Get model-specific configuration
        config = get_model_config(model_info.id)

        payload = {
            "model": model_info.id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": config["max_tokens"],
            "temperature": config["temperature"],
            "top_p": config["top_p"],
            "frequency_penalty": config["frequency_penalty"],
            "presence_penalty": config["presence_penalty"],
        }

        # DEBUG: Log the request
        logger.info(f"🐛 DEBUG: Sending request to {model_info.id}")
        logger.info(f"🐛 DEBUG: URL: {url}")
        logger.info(f"🐛 DEBUG: Payload: {payload}")

        # Use model-specific timeout
        timeout = config["timeout"]
        try:
            async with asyncio.timeout(timeout):
                async with session.post(
                    url, json=payload, timeout=ClientTimeout(total=timeout)
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()

                    # DEBUG: Log the response
                    logger.info(f"🐛 DEBUG: Response status: {resp.status}")
                    logger.info(
                        f"🐛 DEBUG: Response data keys: {list(data.keys()) if isinstance(data, dict) else 'not dict'}"
                    )

                    if isinstance(data, dict) and "choices" in data:
                        choices = data["choices"]
                        if choices and len(choices) > 0:
                            content = choices[0].get("message", {}).get("content", "")
                            logger.info(f"🐛 DEBUG: Content length: {len(content)}")
                            logger.info(
                                f"🐛 DEBUG: Content preview: {content[:200]}..."
                            )
                        else:
                            logger.warning("🐛 DEBUG: No choices in response")
                    else:
                        logger.warning(
                            f"🐛 DEBUG: Unexpected response format: {type(data)}"
                        )

                    logger.info(
                        f"✅ Successfully queried {model_info.id} with optimized config"
                    )
                    return model_info.id, data
        except asyncio.TimeoutError:
            logger.warning(f"⏰ Timeout for {model_info.id} after {timeout}s")
            return model_info.id, {"error": "timeout", "model": model_info.id}
        except ClientError as e:
            logger.error(f"❌ HTTP error for {model_info.id}: {e}")
            return model_info.id, {"error": str(e), "model": model_info.id}
        except Exception as e:
            logger.error(f"❌ Unexpected error for {model_info.id}: {e}")
            return model_info.id, {"error": str(e), "model": model_info.id}

    async def _query_ollama_model(
        self, session: ClientSession, model_info: ModelInfo, prompt: str
    ) -> Tuple[str, Any]:
        """
        Send prompt to a single Ollama model and return (model_id, response_json).
        """
        url = f"{model_info.endpoint}/api/generate"

        # Get model-specific configuration
        config = get_model_config(model_info.id)

        payload = {
            "model": model_info.id,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": config["temperature"],
                "top_p": config["top_p"],
                "frequency_penalty": config["frequency_penalty"],
                "presence_penalty": config["presence_penalty"],
                "num_predict": config[
                    "max_tokens"
                ],  # Ollama uses num_predict instead of max_tokens
            },
        }

        # Use model-specific timeout
        timeout = config["timeout"]
        try:
            async with asyncio.timeout(timeout):
                async with session.post(
                    url, json=payload, timeout=ClientTimeout(total=timeout)
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    logger.info(
                        f"✅ Successfully queried {model_info.id} with optimized config"
                    )
                    return model_info.id, data
        except asyncio.TimeoutError:
            logger.warning(f"⏰ Timeout for {model_info.id} after {timeout}s")
            return model_info.id, {"error": "timeout", "model": model_info.id}
        except ClientError as e:
            logger.error(f"❌ HTTP error for {model_info.id}: {e}")
            return model_info.id, {"error": str(e), "model": model_info.id}
        except Exception as e:
            logger.error(f"❌ Unexpected error for {model_info.id}: {e}")
            return model_info.id, {"error": str(e), "model": model_info.id}

    async def _query_model(
        self, session: ClientSession, model_info: ModelInfo, prompt: str
    ) -> Tuple[str, Any]:
        """
        Send prompt to a single model based on its provider.
        """
        if model_info.provider == "lm_studio":
            return await self._query_lm_studio_model(session, model_info, prompt)
        elif model_info.provider == "ollama":
            return await self._query_ollama_model(session, model_info, prompt)
        else:
            logger.error(f"❌ Unknown provider: {model_info.provider}")
            return model_info.id, {
                "error": f"Unknown provider: {model_info.provider}",
                "model": model_info.id,
            }

    async def dispatch(
        self, prompt: str, max_models: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Dispatch the prompt to healthy models and collect raw responses.

        Args:
            prompt: The prompt to send to models
            max_models: Maximum number of models to query (None = all)

        Returns:
            Dict mapping model IDs to their JSON responses or errors.
        """
        logger.info(f"🚀 Dispatching prompt to models...")
        logger.info(f"📝 Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

        # Get available models
        models = await self.model_manager.list_models()
        if not models:
            logger.warning("⚠️ No models available")
            return {}

        # Limit number of models if specified
        if max_models:
            models = models[:max_models]
            logger.info(f"📊 Limiting to {max_models} models")

        logger.info(f"🎯 Querying {len(models)} models: {[m.id for m in models]}")

        # Query all models in parallel using async context manager
        async with self:
            tasks = [self._query_model(self.session, model, prompt) for model in models]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        results = {}
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                model_id = models[i].id if i < len(models) else f"unknown_{i}"
                logger.error(f"❌ Exception for {model_id}: {response}")
                results[model_id] = {"error": str(response), "model": model_id}
            else:
                model_id, data = response
                results[model_id] = data

        # Log summary
        success_count = sum(1 for r in results.values() if "error" not in r)
        error_count = len(results) - success_count
        logger.info(
            f"📊 Dispatch complete: {success_count} success, {error_count} errors"
        )

        return results

    async def dispatch_to_healthy_models(self, prompt: str) -> Dict[str, Any]:
        """
        Dispatch only to models that pass health checks.
        """
        logger.info("🏥 Checking model health before dispatch...")

        # Get all models
        models = await self.model_manager.list_models()
        if not models:
            logger.warning("⚠️ No models available")
            return {}

        # Check health of all models
        health_status = await self.model_manager.check_all_health(models)

        # Filter to healthy models only
        healthy_models = [
            model for model in models if health_status.get(model.id, False)
        ]

        if not healthy_models:
            logger.warning("⚠️ No healthy models available")
            return {}

        logger.info(f"✅ Found {len(healthy_models)} healthy models")

        # Dispatch to healthy models
        return await self.dispatch_to_models(healthy_models, prompt)

    async def dispatch_to_best_models(
        self, prompt: str, min_models: int = 2, max_models: int = 5
    ) -> Dict[str, Any]:
        """
        Dispatch to the best available models based on health and performance.
        """
        logger.info(f"🎯 Dispatching to best models ({min_models}-{max_models})...")

        # Get best models
        best_models = await self.model_manager.get_best_models(min_models, max_models)

        if not best_models:
            logger.warning("⚠️ No best models available, falling back to all models")
            return await self.dispatch(prompt)

        if len(best_models) < min_models:
            logger.warning(
                f"⚠️ Only {len(best_models)} best models available (need {min_models})"
            )
            # Continue with what we have

        logger.info(
            f"✅ Using {len(best_models)} best models: {[m.id for m in best_models]}"
        )

        # Dispatch to best models
        return await self.dispatch_to_models(best_models, prompt)

    async def dispatch_to_models(
        self, models: List[ModelInfo], prompt: str
    ) -> Dict[str, Any]:
        """
        Dispatch to specific models.
        """
        logger.info(f"🚀 Dispatching to {len(models)} specific models...")

        # Query all models in parallel using async context manager
        async with self:
            tasks = [self._query_model(self.session, model, prompt) for model in models]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        results = {}
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                model_id = models[i].id if i < len(models) else f"unknown_{i}"
                logger.error(f"❌ Exception for {model_id}: {response}")
                results[model_id] = {"error": str(response), "model": model_id}
            else:
                model_id, data = response
                results[model_id] = data

        # Log summary
        success_count = sum(1 for r in results.values() if "error" not in r)
        error_count = len(results) - success_count
        logger.info(
            f"📊 Dispatch complete: {success_count} success, {error_count} errors"
        )

        return results

    async def dispatch_with_fallback(
        self, prompt: str, min_models: int = 2
    ) -> Dict[str, Any]:
        """
        Dispatch with intelligent fallback strategies.
        """
        logger.info(f"🔄 Dispatching with fallback (min: {min_models})...")

        # Try best models first
        results = await self.dispatch_to_best_models(prompt, min_models)

        success_count = sum(1 for r in results.values() if "error" not in r)

        if success_count >= min_models:
            logger.info(
                f"✅ Got {success_count} successful responses, no fallback needed"
            )
            return results

        logger.warning(
            f"⚠️ Only {success_count} successful responses, trying fallback..."
        )

        # Fallback 1: Try all models
        all_results = await self.dispatch(prompt)
        all_success_count = sum(1 for r in all_results.values() if "error" not in r)

        if all_success_count > success_count:
            logger.info(
                f"✅ Fallback improved results: {all_success_count} vs {success_count}"
            )
            return all_results

        # Fallback 2: Try with longer timeout
        logger.info("⏰ Trying with extended timeout...")
        original_timeout = self.timeout
        self.timeout = min(60, original_timeout * 2)  # Double timeout, max 60s

        try:
            extended_results = await self.dispatch(prompt)
            extended_success_count = sum(
                1 for r in extended_results.values() if "error" not in r
            )

            if extended_success_count > success_count:
                logger.info(
                    f"✅ Extended timeout helped: {extended_success_count} vs {success_count}"
                )
                return extended_results
        finally:
            self.timeout = original_timeout

        # Return best available results
        logger.warning(
            f"⚠️ All fallbacks exhausted, returning best available: {success_count} responses"
        )
        return results

    async def dispatch_parallel(
        self,
        models: List[ModelInfo],
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Dispatch prompts to multiple models in parallel - interface expected by ensemble engine.

        Args:
            models: List ofModelInfo objects to query
            prompt: The prompt to send to all models
            context: Optional context information (not used in current implementation)

        Returns:
            Dict mapping model IDs to their responses
        """
        logger.info(f"🚀 Dispatching parallel to {len(models)} models...")
        logger.info(f"🔍 Models: {[m.id for m in models]}")
        logger.info(f"🔍 Session available: {self.session is not None}")

        # Query all models in parallel (session is already managed by caller)
        if not self.session:
            logger.error("❌ No session available for dispatch_parallel")
            return {}

        try:
            tasks = [self._query_model(self.session, model, prompt) for model in models]
            logger.info(f"🔍 Created {len(tasks)} tasks")

            # Add timeout protection to prevent hanging
            # Use the maximum timeout from the models (SLOW_TIMEOUT = 120s)
            timeout_seconds = (
                SLOW_TIMEOUT
                if self.timeout is None
                else max(self.timeout, SLOW_TIMEOUT)
            )

            responses = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True), timeout=timeout_seconds
            )
            logger.info(f"🔍 Got {len(responses)} responses")

            # Process results
            results = {}
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    model_id = models[i].id if i < len(models) else f"unknown_{i}"
                    logger.error(f"❌ Exception for {model_id}: {response}")
                    results[model_id] = {"error": str(response), "model": model_id}
                else:
                    model_id, data = response
                    results[model_id] = data
                    logger.info(f"🔍 Processed response for {model_id}")

            # Log summary
            success_count = sum(1 for r in results.values() if "error" not in r)
            error_count = len(results) - success_count
            logger.info(
                f"📊 Parallel dispatch complete: {success_count} success, {error_count} errors"
            )
            logger.info(f"🔍 Returning results: {results}")

            return results

        except asyncio.TimeoutError:
            logger.error(f"⏰ dispatch_parallel timed out after {timeout_seconds}s")
            # Return timeout errors for all models
            results = {}
            for model in models:
                results[model.id] = {"error": "timeout", "model": model.id}
            return results
        except Exception as e:
            logger.error(f"❌ Exception in dispatch_parallel: {e}")
            return {}


# Convenience functions
async def dispatch_prompt(
    prompt: str, max_models: Optional[int] = None
) -> Dict[str, Any]:
    """Convenience function to dispatch a prompt."""
    dispatcher = QueryDispatcher()
    return await dispatcher.dispatch(prompt, max_models)


async def main():
    """Demo function to test QueryDispatcher."""
    print("🚀 CodeConductor Query Dispatcher Demo")
    print("=" * 50)

    dispatcher = QueryDispatcher(timeout=10)  # Shorter timeout for demo

    # Test prompt
    test_prompt = "What is 2 + 2? Please respond with just the number."

    print(f"📝 Sending prompt: {test_prompt}")
    print()

    # Dispatch to all models
    results = await dispatcher.dispatch(
        test_prompt, max_models=2
    )  # Limit to 2 for demo

    print("📊 Results:")
    for model_id, response in results.items():
        print(f"\n🎯 {model_id}:")
        if "error" in response:
            print(f"  ❌ Error: {response['error']}")
        else:
            # Extract content from response
            if "choices" in response:
                content = response["choices"][0]["message"]["content"]
                print(f"  ✅ Response: {content}")
            elif "response" in response:
                print(f"  ✅ Response: {response['response']}")
            else:
                print(f"  ✅ Raw response: {json.dumps(response, indent=2)}")

    print(f"\n📈 Summary: {len(results)} models queried")


if __name__ == "__main__":
    asyncio.run(main())
