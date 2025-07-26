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

REQUEST_TIMEOUT = 30  # seconds


class QueryDispatcher:
    """
    Dispatches prompts to multiple LLM models in parallel, handles timeouts and errors.
    """

    def __init__(self, timeout: int = REQUEST_TIMEOUT) -> None:
        self.timeout = timeout
        self.model_manager = ModelManager()

    async def _query_lm_studio_model(
        self, session: ClientSession, model_info: ModelInfo, prompt: str
    ) -> Tuple[str, Any]:
        """
        Send prompt to a single LM Studio model and return (model_id, response_json).
        """
        url = f"{model_info.endpoint}/chat/completions"
        payload = {
            "model": model_info.id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
            "temperature": 0.1,
        }

        try:
            async with session.post(
                url, json=payload, timeout=ClientTimeout(total=self.timeout)
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                logger.info(f"✅ Successfully queried {model_info.id}")
                return model_info.id, data
        except asyncio.TimeoutError:
            logger.warning(f"⏰ Timeout for {model_info.id}")
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
        payload = {
            "model": model_info.id,
            "prompt": prompt,
            "stream": False,
        }

        try:
            async with session.post(
                url, json=payload, timeout=ClientTimeout(total=self.timeout)
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                logger.info(f"✅ Successfully queried {model_info.id}")
                return model_info.id, data
        except asyncio.TimeoutError:
            logger.warning(f"⏰ Timeout for {model_info.id}")
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

        # Query all models in parallel
        async with ClientSession() as session:
            tasks = [self._query_model(session, model, prompt) for model in models]
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
        return await self.dispatch(prompt)

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

        # Query all models in parallel
        async with ClientSession() as session:
            tasks = [self._query_model(session, model, prompt) for model in models]
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
