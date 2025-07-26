#!/usr/bin/env python3
"""
Model Manager for CodeConductor MVP
Discovers and health-checks local LLM models (LM Studio & Ollama)
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a discovered model."""

    id: str
    name: str
    provider: str  # "lm_studio" or "ollama"
    endpoint: str
    is_available: bool
    metadata: Dict = None


class ModelManager:
    """Manages discovery and health-checking of local LLM models."""

    def __init__(self):
        self.lm_studio_endpoint = "http://localhost:1234/v1"
        self.ollama_endpoint = "http://localhost:11434"
        self.timeout = 5.0  # seconds

    async def list_models(self) -> List[ModelInfo]:
        """
        Discover all available models from LM Studio and Ollama.

        Returns:
            List of ModelInfo objects for all discovered models.
        """
        logger.info("🔍 Discovering local LLM models...")

        # Run discovery in parallel
        lm_studio_models, ollama_models = await asyncio.gather(
            self._discover_lm_studio_models(),
            self._discover_ollama_models(),
            return_exceptions=True,
        )

        # Handle exceptions gracefully
        if isinstance(lm_studio_models, Exception):
            logger.warning(f"LM Studio discovery failed: {lm_studio_models}")
            lm_studio_models = []

        if isinstance(ollama_models, Exception):
            logger.warning(f"Ollama discovery failed: {ollama_models}")
            ollama_models = []

        all_models = lm_studio_models + ollama_models
        logger.info(f"✅ Discovered {len(all_models)} models total")

        return all_models

    async def check_health(self, model_info: ModelInfo) -> bool:
        """
        Perform health check on a specific model.

        Args:
            model_info: ModelInfo object to check

        Returns:
            True if model is healthy, False otherwise
        """
        logger.info(f"🏥 Health checking {model_info.name} ({model_info.provider})")

        try:
            if model_info.provider == "lm_studio":
                return await self._check_lm_studio_health(model_info)
            elif model_info.provider == "ollama":
                return await self._check_ollama_health(model_info)
            else:
                logger.error(f"Unknown provider: {model_info.provider}")
                return False

        except Exception as e:
            logger.error(f"Health check failed for {model_info.name}: {e}")
            return False

    async def check_all_health(self, models: List[ModelInfo]) -> Dict[str, bool]:
        """
        Check health of all models in parallel.

        Args:
            models: List of ModelInfo objects

        Returns:
            Dict mapping model IDs to health status
        """
        logger.info(f"🏥 Health checking {len(models)} models...")

        # Run health checks in parallel
        health_tasks = [self.check_health(model) for model in models]
        health_results = await asyncio.gather(*health_tasks, return_exceptions=True)

        # Map results to model IDs
        health_status = {}
        for model, result in zip(models, health_results):
            if isinstance(result, Exception):
                logger.error(f"Health check exception for {model.id}: {result}")
                health_status[model.id] = False
            else:
                health_status[model.id] = result

        healthy_count = sum(health_status.values())
        logger.info(f"✅ {healthy_count}/{len(models)} models are healthy")

        return health_status

    async def list_healthy_models(self) -> List[str]:
        """
        Get list of healthy model IDs for use in querying.

        Returns:
            List of model IDs that are currently healthy.
        """
        # For now, return all discovered models
        # In a real implementation, this would filter by health status
        models = await self.list_models()
        return [model.id for model in models]

    async def _discover_lm_studio_models(self) -> List[ModelInfo]:
        """Discover models from LM Studio."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.lm_studio_endpoint}/models",
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    if response.status != 200:
                        logger.warning(f"LM Studio returned status {response.status}")
                        return []

                    data = await response.json()
                    models = []

                    for model_data in data.get("data", []):
                        model_info = ModelInfo(
                            id=model_data.get("id", "unknown"),
                            name=model_data.get("id", "unknown"),
                            provider="lm_studio",
                            endpoint=self.lm_studio_endpoint,
                            is_available=True,
                            metadata=model_data,
                        )
                        models.append(model_info)

                    logger.info(f"📦 Discovered {len(models)} LM Studio models")
                    return models

        except Exception as e:
            logger.error(f"LM Studio discovery error: {e}")
            return []

    async def _discover_ollama_models(self) -> List[ModelInfo]:
        """Discover models from Ollama."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.ollama_endpoint}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    if response.status != 200:
                        logger.warning(f"Ollama returned status {response.status}")
                        return []

                    data = await response.json()
                    models = []

                    for model_data in data.get("models", []):
                        model_info = ModelInfo(
                            id=model_data.get("name", "unknown"),
                            name=model_data.get("name", "unknown"),
                            provider="ollama",
                            endpoint=self.ollama_endpoint,
                            is_available=True,
                            metadata=model_data,
                        )
                        models.append(model_info)

                    logger.info(f"📦 Discovered {len(models)} Ollama models")
                    return models

        except Exception as e:
            logger.error(f"Ollama discovery error: {e}")
            return []

    async def _check_lm_studio_health(self, model_info: ModelInfo) -> bool:
        """Check if LM Studio model is healthy."""
        try:
            async with aiohttp.ClientSession() as session:
                # Try a simple completion request
                payload = {
                    "model": model_info.id,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 10,
                }

                async with session.post(
                    f"{self.lm_studio_endpoint}/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    return response.status == 200

        except Exception as e:
            logger.error(f"LM Studio health check error for {model_info.id}: {e}")
            return False

    async def _check_ollama_health(self, model_info: ModelInfo) -> bool:
        """Check if Ollama model is healthy."""
        try:
            async with aiohttp.ClientSession() as session:
                # Try a simple completion request
                payload = {"model": model_info.id, "prompt": "Hello", "stream": False}

                async with session.post(
                    f"{self.ollama_endpoint}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    return response.status == 200

        except Exception as e:
            logger.error(f"Ollama health check error for {model_info.id}: {e}")
            return False


# Convenience functions for easy usage
async def discover_models() -> List[ModelInfo]:
    """Discover all available models."""
    manager = ModelManager()
    return await manager.list_models()


async def check_model_health(model_info: ModelInfo) -> bool:
    """Check health of a specific model."""
    manager = ModelManager()
    return await manager.check_health(model_info)


async def main():
    """Demo function to test ModelManager."""
    print("🎯 CodeConductor Model Manager Demo")
    print("=" * 50)

    manager = ModelManager()

    # Discover models
    models = await manager.list_models()

    if not models:
        print("❌ No models discovered")
        return

    print(f"\n📦 Discovered {len(models)} models:")
    for model in models:
        print(f"  - {model.name} ({model.provider})")

    # Check health
    health_status = await manager.check_all_health(models)

    print(f"\n🏥 Health Status:")
    for model_id, is_healthy in health_status.items():
        status = "✅" if is_healthy else "❌"
        print(f"  {status} {model_id}")

    # Summary
    healthy_count = sum(health_status.values())
    print(f"\n📊 Summary: {healthy_count}/{len(models)} models are healthy")


if __name__ == "__main__":
    asyncio.run(main())
