#!/usr/bin/env python3
"""
Query Dispatcher for CodeConductor MVP
Dispatches prompts to multiple LLM models in parallel with timeout and error handling.
"""

import asyncio
import json
import logging
from typing import Any

from aiohttp import ClientError, ClientSession, ClientTimeout

from codeconductor.telemetry import get_logger

from .breakers import get_manager as get_breaker_manager
from .model_manager import ModelInfo, ModelManager

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


def _env_number(name: str) -> float | None:
    import os as _os

    val = (_os.getenv(name) or "").strip()
    if not val:
        return None
    try:
        return float(val)
    except Exception:
        return None


def _apply_sampling_overrides(config: dict) -> dict:
    """Apply env-based sampling overrides onto a copy of the config.

    Env variables honored:
      - CC_TEMP or TEMP (float)
      - CC_TOP_P or TOP_P (float)
      - MAX_TOKENS (int/float)
    """
    new_cfg = dict(config)
    temp = _env_number("CC_TEMP") or _env_number("TEMP")
    top_p = _env_number("CC_TOP_P") or _env_number("TOP_P")
    max_tokens = _env_number("MAX_TOKENS")
    if temp is not None:
        new_cfg["temperature"] = float(temp)
    if top_p is not None:
        new_cfg["top_p"] = float(top_p)
    if max_tokens is not None and max_tokens > 0:
        try:
            new_cfg["max_tokens"] = int(max_tokens)
        except Exception:
            new_cfg["max_tokens"] = int(float(max_tokens))
    return new_cfg


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
    ) -> tuple[str, Any]:
        """
        Send prompt to a single LM Studio model and return (model_id, response_json).
        """
        url = f"{model_info.endpoint}/chat/completions"

        # Get model-specific configuration
        base_cfg = get_model_config(model_info.id)
        config = _apply_sampling_overrides(base_cfg)

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
                            logger.info(f"🐛 DEBUG: Content preview: {content[:200]}...")
                            # If LM Studio returns an empty message content, mark explicitly
                            if not content:
                                logger.warning(
                                    f"🐛 DEBUG: Empty content from model {model_info.id}"
                                )
                                # Return a structured marker so upstream can save a diagnostic artifact
                                return model_info.id, {
                                    "empty_content": True,
                                    "model": model_info.id,
                                    "prompt_preview": (prompt or "")[:50],
                                    "raw": data,
                                }
                        else:
                            logger.warning("🐛 DEBUG: No choices in response")
                    else:
                        logger.warning(f"🐛 DEBUG: Unexpected response format: {type(data)}")

                    logger.info(f"✅ Successfully queried {model_info.id} with optimized config")
                    return model_info.id, data
        except TimeoutError:
            logger.warning(f"⏰ Timeout for {model_info.id} after {timeout}s")
            # Check if we actually got a response despite timeout
            try:
                async with session.post(url, json=payload, timeout=ClientTimeout(total=30)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        logger.info(f"✅ Got response from {model_info.id} despite timeout")
                        return model_info.id, data
            except Exception:
                pass
            return model_info.id, {"error": "timeout", "model": model_info.id}

    async def _query_ollama_model(
        self, session: ClientSession, model_info: ModelInfo, prompt: str
    ) -> tuple[str, Any]:
        """
        Send prompt to a single Ollama model and return (model_id, response_json).
        """
        url = f"{model_info.endpoint}/api/generate"

        # Get model-specific configuration
        base_cfg = get_model_config(model_info.id)
        config = _apply_sampling_overrides(base_cfg)

        payload = {
            "model": model_info.id,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": config["temperature"],
                "top_p": config["top_p"],
                "frequency_penalty": config["frequency_penalty"],
                "presence_penalty": config["presence_penalty"],
                "num_predict": config["max_tokens"],
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
                    logger.info(f"✅ Successfully queried {model_info.id} with optimized config")
                    return model_info.id, data
        except TimeoutError:
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
    ) -> tuple[str, Any]:
        """
        Send prompt to a single model based on its provider.
        """
        # Circuit breaker guard
        breaker = get_breaker_manager()
        tlog = get_logger()
        if not breaker.should_allow(model_info.id):
            tlog.log(
                "breaker_block",
                {
                    "model": model_info.id,
                    "state": breaker.get_state(model_info.id).state,
                },
            )
            return model_info.id, {"error": "breaker_open", "model": model_info.id}
        # Failure injection for tests (shadow-friendly)
        import os as _os

        inj_match = (_os.getenv("CC_MOCK_FAIL_MODEL") or "").strip().lower()
        inj_mode = (_os.getenv("CC_MOCK_FAIL_MODE") or "").strip().lower()
        if inj_match and inj_match in model_info.id.lower() and inj_mode:
            err_map = {
                "timeout": "timeout",
                "5xx": "HTTP 503",
                "reset": "ConnectionResetError",
            }
            err = err_map.get(inj_mode, "mock_error")
            start = asyncio.get_event_loop().time()
            total_ms = (asyncio.get_event_loop().time() - start) * 1000.0
            cls = "timeout" if err == "timeout" else ("5xx" if "HTTP" in err else "reset")
            breaker.update(
                model_info.id,
                success=False,
                total_ms=total_ms,
                ttft_ms=None,
                error_class=cls,
            )
            tlog.log(
                "dispatch",
                {
                    "model": model_info.id,
                    "success": False,
                    "total_ms": round(total_ms, 1),
                    "ttft_ms": None,
                    "error_class": cls,
                },
            )
            return model_info.id, {"error": err, "model": model_info.id}
        start = asyncio.get_event_loop().time()
        err_class: str | None = None
        try:
            if model_info.provider == "lm_studio":
                mid, data = await self._query_lm_studio_model(session, model_info, prompt)
            elif model_info.provider == "ollama":
                mid, data = await self._query_ollama_model(session, model_info, prompt)
            else:
                logger.error(f"❌ Unknown provider: {model_info.provider}")
                mid, data = (
                    model_info.id,
                    {
                        "error": f"Unknown provider: {model_info.provider}",
                        "model": model_info.id,
                    },
                )
        finally:
            total_ms = (asyncio.get_event_loop().time() - start) * 1000.0
            success = isinstance(locals().get("data", {}), dict) and "error" not in locals().get(
                "data", {}
            )
            if not success:
                # classify error
                err = (
                    locals().get("data", {}).get("error")
                    if isinstance(locals().get("data", {}), dict)
                    else None
                )
                if isinstance(err, str):
                    if "timeout" in err:
                        err_class = "timeout"
                    elif "breaker_open" in err:
                        err_class = "breaker_open"
                    elif "HTTP" in err or "5" in err:
                        err_class = "5xx"
                    else:
                        err_class = "other"
            # No TTFT available here without SSE stream; record None
            breaker.update(
                model_info.id,
                success=bool(success),
                total_ms=total_ms,
                ttft_ms=None,
                error_class=err_class,
            )
            tlog.log(
                "dispatch",
                {
                    "model": model_info.id,
                    "success": bool(success),
                    "total_ms": round(total_ms, 1),
                    "ttft_ms": None,
                    "error_class": err_class,
                },
            )
        return mid, data

    async def dispatch(self, prompt: str, max_models: int | None = None) -> dict[str, Any]:
        """
        Dispatch the prompt to healthy models and collect raw responses.

        Args:
            prompt: The prompt to send to models
            max_models: Maximum number of models to query (None = all)

        Returns:
            Dict mapping model IDs to their JSON responses or errors.
        """
        logger.info("🚀 Dispatching prompt to models...")
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
        logger.info(f"📊 Dispatch complete: {success_count} success, {error_count} errors")

        return results

    async def dispatch_to_healthy_models(self, prompt: str) -> dict[str, Any]:
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
        healthy_models = [model for model in models if health_status.get(model.id, False)]

        if not healthy_models:
            logger.warning("⚠️ No healthy models available")
            return {}

        logger.info(f"✅ Found {len(healthy_models)} healthy models")

        # Dispatch to healthy models
        return await self.dispatch_to_models(healthy_models, prompt)

    async def dispatch_to_best_models(
        self, prompt: str, min_models: int = 2, max_models: int = 5
    ) -> dict[str, Any]:
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
            logger.warning(f"⚠️ Only {len(best_models)} best models available (need {min_models})")
            # Continue with what we have

        logger.info(f"✅ Using {len(best_models)} best models: {[m.id for m in best_models]}")

        # Dispatch to best models
        return await self.dispatch_to_models(best_models, prompt)

    async def dispatch_to_models(self, models: list[ModelInfo], prompt: str) -> dict[str, Any]:
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
        logger.info(f"📊 Dispatch complete: {success_count} success, {error_count} errors")

        return results

    async def dispatch_with_fallback(self, prompt: str, min_models: int = 2) -> dict[str, Any]:
        """
        Dispatch with intelligent fallback strategies.
        """
        logger.info(f"🔄 Dispatching with fallback (min: {min_models})...")

        # Try best models first
        results = await self.dispatch_to_best_models(prompt, min_models)

        success_count = sum(1 for r in results.values() if "error" not in r)

        if success_count >= min_models:
            logger.info(f"✅ Got {success_count} successful responses, no fallback needed")
            return results

        logger.warning(f"⚠️ Only {success_count} successful responses, trying fallback...")

        # Fallback 1: Try all models
        all_results = await self.dispatch(prompt)
        all_success_count = sum(1 for r in all_results.values() if "error" not in r)

        if all_success_count > success_count:
            logger.info(f"✅ Fallback improved results: {all_success_count} vs {success_count}")
            return all_results

        # Fallback 2: Try with longer timeout
        logger.info("⏰ Trying with extended timeout...")
        original_timeout = self.timeout
        self.timeout = min(60, original_timeout * 2)  # Double timeout, max 60s

        try:
            extended_results = await self.dispatch(prompt)
            extended_success_count = sum(1 for r in extended_results.values() if "error" not in r)

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
        models: list[ModelInfo],
        prompt: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Dispatch a prompt to multiple models in parallel."""
        logger.info(f"🚀 Dispatching parallel to {len(models)} models...")
        logger.info(f"🔍 Models: {[model.id for model in models]}")

        if not models:
            return {"success": False, "error": "No models provided"}

        # Create tasks for each model
        tasks = []
        for model in models:
            task = self._query_model(self.session, model, prompt)
            tasks.append(task)

        logger.info(f"🔍 Created {len(tasks)} tasks")

        try:
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            logger.info("🔍 All tasks completed")

            # Process results
            successful_results = {}
            failed_results = {}

            for i, result in enumerate(results):
                model_id = models[i].id
                if isinstance(result, Exception):
                    failed_results[model_id] = str(result)
                    logger.error(f"❌ {model_id} failed: {result}")
                else:
                    model_id, response_data = result
                    successful_results[model_id] = response_data
                    logger.info(f"✅ {model_id} succeeded")

            return {
                "success": len(successful_results) > 0,
                "successful": successful_results,
                "failed": failed_results,
                "total_models": len(models),
                "successful_count": len(successful_results),
                "failed_count": len(failed_results),
            }

        except Exception as e:
            logger.error(f"❌ Parallel dispatch failed: {e}")
            return {"success": False, "error": str(e)}

    async def dispatch_single(
        self,
        model: ModelInfo,
        prompt: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Dispatch a prompt to a single model."""
        logger.info(f"🎯 Dispatching to single model: {model.id}")

        try:
            # Query the single model
            result = await self._query_model(self.session, model, prompt)

            if result:
                model_id, response_data = result
                logger.info(f"✅ Single model request succeeded: {model_id}")

                return {
                    "success": True,
                    "model": model_id,
                    "response": response_data,
                    "execution_time": 0.0,  # Could be calculated if needed
                }
            else:
                logger.error(f"❌ Single model request failed: {model.id}")
                return {
                    "success": False,
                    "error": f"Model {model.id} returned no response",
                    "model": model.id,
                }

        except Exception as e:
            logger.error(f"❌ Single model dispatch failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "model": model.id,
            }


# Convenience functions
async def dispatch_prompt(prompt: str, max_models: int | None = None) -> dict[str, Any]:
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
    results = await dispatcher.dispatch(test_prompt, max_models=2)  # Limit to 2 for demo

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
