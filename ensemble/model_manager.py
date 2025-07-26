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
        self.timeout = 30.0  # seconds
        self.health_timeout = 10.0  # shorter timeout for health checks

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

    async def health_check(self, model_id: str) -> bool:
        """Check health of a specific model by ID."""
        # Get all models and find the one we're looking for
        all_models = await self.list_models()
        model_info = next((m for m in all_models if m.id == model_id), None)

        if not model_info:
            logger.warning(f"⚠️ Model {model_id} not found")
            return False

        return await self.check_health(model_info)

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
        """Check health of all models in parallel."""
        logger.info(f"🏥 Checking health of {len(models)} models...")

        health_tasks = []
        for model in models:
            task = self.check_health(model)
            health_tasks.append(task)

        health_results = await asyncio.gather(*health_tasks, return_exceptions=True)

        health_status = {}
        for model, result in zip(models, health_results):
            if isinstance(result, bool):
                health_status[model.id] = result
                status = "✅" if result else "❌"
                logger.info(
                    f"{status} {model.id}: {'healthy' if result else 'unhealthy'}"
                )
            else:
                health_status[model.id] = False
                logger.warning(f"❌ {model.id}: health check failed - {result}")

        healthy_count = sum(health_status.values())
        logger.info(
            f"📊 Health check complete: {healthy_count}/{len(models)} models healthy"
        )

        return health_status

    async def get_best_models(
        self, min_models: int = 2, max_models: int = 5
    ) -> List[ModelInfo]:
        """Get the best available models based on health and performance."""
        logger.info(f"🎯 Selecting best {min_models}-{max_models} models...")

        # Get all models
        all_models = await self.list_models()
        if not all_models:
            logger.warning("⚠️ No models available")
            return []

        # Check health of all models
        health_status = await self.check_all_health(all_models)

        # Filter to healthy models
        healthy_models = [
            model for model in all_models if health_status.get(model.id, False)
        ]

        if len(healthy_models) < min_models:
            logger.warning(
                f"⚠️ Only {len(healthy_models)} healthy models available (need {min_models})"
            )
            # Return all healthy models even if below minimum
            return healthy_models

        # Sort by performance metrics (simple scoring for now)
        def model_score(model: ModelInfo) -> float:
            # Simple scoring based on provider and model size
            base_score = 0.5
            
            # Prefer larger models (they tend to be more capable)
            if "12b" in model.id.lower() or "13b" in model.id.lower():
                base_score += 0.3
            elif "7b" in model.id.lower() or "8b" in model.id.lower():
                base_score += 0.2
            elif "mini" in model.id.lower():
                base_score += 0.1
                
            # Prefer certain providers
            if model.provider == "lm_studio":
                base_score += 0.1
            elif model.provider == "ollama":
                base_score += 0.05
                
            return base_score

        # Sort by score and take top models
        sorted_models = sorted(healthy_models, key=model_score, reverse=True)
        selected_models = sorted_models[:max_models]

        logger.info(f"✅ Selected {len(selected_models)} best models:")
        for i, model in enumerate(selected_models, 1):
            score = model_score(model)
            logger.info(f"  {i}. {model.id} (score: {score:.2f})")

        return selected_models

    async def auto_recovery(self, model_id: str) -> bool:
        """Attempt to recover a failed model."""
        logger.info(f"🔄 Attempting auto-recovery for {model_id}...")

        # Get all models and find the one we're looking for
        all_models = await self.list_models()
        model_info = next((m for m in all_models if m.id == model_id), None)

        if not model_info:
            logger.warning(f"⚠️ Model {model_id} not found")
            return False

        # Try health check
        try:
            is_healthy = await self.check_health(model_info)
            if is_healthy:
                logger.info(f"✅ {model_id} recovered successfully")
                return True
        except Exception as e:
            logger.warning(f"⚠️ Health check failed during recovery: {e}")

        # Try to restart model (if supported)
        try:
            if model_info.provider == "ollama":
                # For Ollama, we could try to restart the model
                logger.info(f"🔄 Attempting to restart Ollama model {model_id}...")
                # This would require Ollama API calls to restart the model
                # For now, just log the attempt
                logger.info(f"📝 Manual restart required for {model_id}")
            elif model_info.provider == "lm_studio":
                logger.info(
                    f"📝 Manual restart required for LM Studio model {model_id}"
                )
        except Exception as e:
            logger.warning(f"⚠️ Recovery attempt failed: {e}")

        return False

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
                        model_id = model_data.get("id", "unknown")
                        # Skip embedding models (they don't support chat/completions)
                        if "embedding" in model_id.lower():
                            continue

                        model_info = ModelInfo(
                            id=model_id,
                            name=model_id,
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
                # Just check if the model is listed (simpler than completion)
                async with session.get(
                    f"{self.lm_studio_endpoint}/models",
                    timeout=aiohttp.ClientTimeout(total=self.health_timeout),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        model_ids = [m.get("id") for m in data.get("data", [])]
                        return model_info.id in model_ids
                    return False

        except Exception as e:
            logger.error(f"LM Studio health check error for {model_info.id}: {e}")
            return False

    async def _check_ollama_health(self, model_info: ModelInfo) -> bool:
        """Check if Ollama model is healthy."""
        try:
            async with aiohttp.ClientSession() as session:
                # Just check if the model is listed (simpler than completion)
                async with session.get(
                    f"{self.ollama_endpoint}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=self.health_timeout),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        model_names = [m.get("name") for m in data.get("models", [])]
                        return model_info.id in model_names
                    return False

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
