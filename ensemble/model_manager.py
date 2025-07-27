#!/usr/bin/env python3
"""
Model Manager for CodeConductor MVP
Discovers and health-checks local LLM models (LM Studio & Ollama)
"""

import asyncio
import aiohttp
import json
import logging
import subprocess
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Recommended Ollama models for CodeConductor
RECOMMENDED_MODELS = {
    "phi3:mini": {
        "description": "Fast, lightweight model for quick prototyping",
        "size": "3.8B",
        "specialty": "general",
        "download_cmd": "ollama pull phi3:mini",
    },
    "codellama:7b": {
        "description": "Specialized for code generation and understanding",
        "size": "7B",
        "specialty": "code",
        "download_cmd": "ollama pull codellama:7b",
    },
    "mistral:7b": {
        "description": "Balanced reasoning and code generation",
        "size": "7B",
        "specialty": "reasoning",
        "download_cmd": "ollama pull mistral:7b",
    },
    "deepseek-coder:6.7b": {
        "description": "Advanced code generation and analysis",
        "size": "6.7B",
        "specialty": "code",
        "download_cmd": "ollama pull deepseek-coder:6.7b",
    },
    "qwen2.5:7b": {
        "description": "Strong reasoning and instruction following",
        "size": "7B",
        "specialty": "reasoning",
        "download_cmd": "ollama pull qwen2.5:7b",
    },
}


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

        # Run discovery in parallel with individual timeouts
        try:
            lm_studio_models, ollama_models = await asyncio.gather(
                asyncio.wait_for(
                    self._discover_lm_studio_models(), timeout=self.timeout
                ),
                asyncio.wait_for(self._discover_ollama_models(), timeout=self.timeout),
                return_exceptions=True,
            )
        except asyncio.TimeoutError:
            logger.warning("⚠️ Model discovery timed out, using available models only")
            # Try to get at least Ollama models if LM Studio times out
            try:
                ollama_models = await asyncio.wait_for(
                    self._discover_ollama_models(), timeout=10.0
                )
                lm_studio_models = []
            except asyncio.TimeoutError:
                logger.error("❌ All model discovery timed out")
                return []

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
            # Add timeout to each health check
            task = asyncio.wait_for(
                self.check_health(model), timeout=self.health_timeout
            )
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
        """
        Get the best available models for ensemble.

        Args:
            min_models: Minimum number of models required
            max_models: Maximum number of models to return

        Returns:
            List of best ModelInfo objects
        """
        all_models = await self.list_models()

        if len(all_models) < min_models:
            logger.warning(
                f"⚠️ Only {len(all_models)} models available, need {min_models}"
            )
            return all_models

        # Score models based on provider, size, and specialty
        def model_score(model: ModelInfo) -> float:
            score = 0.0

            # Prefer Ollama models (faster, more reliable)
            if model.provider == "ollama":
                score += 10.0
            elif model.provider == "lm_studio":
                score += 5.0

            # Prefer smaller models for speed
            model_name = model.id.lower()
            if "mini" in model_name or "3b" in model_name:
                score += 5.0
            elif "7b" in model_name:
                score += 3.0
            elif "13b" in model_name or "14b" in model_name:
                score += 1.0

            # Prefer code-specialized models
            if any(code_word in model_name for code_word in ["code", "coder", "phi"]):
                score += 3.0

            return score

        # Sort by score and return top models
        scored_models = [(model, model_score(model)) for model in all_models]
        scored_models.sort(key=lambda x: x[1], reverse=True)

        best_models = [model for model, score in scored_models[:max_models]]
        logger.info(f"🎯 Selected {len(best_models)} best models for ensemble")

        return best_models

    def download_recommended_model(self, model_name: str) -> bool:
        """
        Download a recommended model using Ollama.

        Args:
            model_name: Name of the model to download

        Returns:
            True if download started successfully, False otherwise
        """
        if model_name not in RECOMMENDED_MODELS:
            logger.error(f"❌ Model {model_name} not in recommended list")
            return False

        model_info = RECOMMENDED_MODELS[model_name]
        download_cmd = model_info["download_cmd"]

        try:
            logger.info(f"📥 Starting download of {model_name}...")
            logger.info(f"💡 Description: {model_info['description']}")
            logger.info(f"📏 Size: {model_info['size']}")
            logger.info(f"🎯 Specialty: {model_info['specialty']}")

            # Start download in background
            process = subprocess.Popen(
                download_cmd.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            logger.info(f"✅ Download started for {model_name}")
            logger.info(f"💡 Run 'ollama list' to check progress")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to start download for {model_name}: {e}")
            return False

    def get_recommended_models(self) -> Dict[str, Dict]:
        """Get list of recommended models with their info."""
        return RECOMMENDED_MODELS

    def get_available_models(self) -> List[str]:
        """Get list of currently available models."""
        # This would need to be async, but for now return empty
        # In practice, this would call list_models()
        return []

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

    def update_model_stats(self, model_id: str, success: bool, response_time: float):
        """
        Update statistics for a model after a request.

        Args:
            model_id: ID of the model
            success: Whether the request was successful
            response_time: Response time in seconds
        """
        # This is a simplified implementation - in a real system you'd store these stats
        logger.debug(
            f"📊 Updated stats for {model_id}: success={success}, time={response_time:.2f}s"
        )

    async def test_model_response_time(
        self, model_info: ModelInfo, prompt: str, timeout: float = 10.0
    ) -> Dict[str, Any]:
        """Test response time of a specific model."""
        logger.info(f"⏱️ Testing response time for {model_info.id}")

        try:
            start_time = asyncio.get_event_loop().time()

            if model_info.provider == "lm_studio":
                result = await self._test_lm_studio_response(
                    model_info, prompt, timeout
                )
            elif model_info.provider == "ollama":
                result = await self._test_ollama_response(model_info, prompt, timeout)
            else:
                return {
                    "success": False,
                    "error": f"Unknown provider: {model_info.provider}",
                }

            end_time = asyncio.get_event_loop().time()
            response_time = end_time - start_time

            result["response_time"] = response_time
            return result

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _test_lm_studio_response(
        self, model_info: ModelInfo, prompt: str, timeout: float
    ) -> Dict[str, Any]:
        """Test LM Studio model response."""
        url = f"{model_info.endpoint}/chat/completions"
        payload = {
            "model": model_info.id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 50,
            "temperature": 0.1,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    return {"success": True, "response": data}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _test_ollama_response(
        self, model_info: ModelInfo, prompt: str, timeout: float
    ) -> Dict[str, Any]:
        """Test Ollama model response."""
        url = f"{model_info.endpoint}/api/generate"
        payload = {
            "model": model_info.id,
            "prompt": prompt,
            "stream": False,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    return {"success": True, "response": data}
        except Exception as e:
            return {"success": False, "error": str(e)}

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

        except asyncio.TimeoutError:
            logger.warning("⚠️ LM Studio discovery timed out")
            return []
        except aiohttp.ClientError as e:
            logger.warning(f"⚠️ LM Studio connection error: {e}")
            return []
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

        except asyncio.TimeoutError:
            logger.warning("⚠️ Ollama discovery timed out")
            return []
        except aiohttp.ClientError as e:
            logger.warning(f"⚠️ Ollama connection error: {e}")
            return []
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

        except asyncio.TimeoutError:
            logger.warning(f"⚠️ LM Studio health check timed out for {model_info.id}")
            return False
        except aiohttp.ClientError as e:
            logger.warning(
                f"⚠️ LM Studio health check connection error for {model_info.id}: {e}"
            )
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

        except asyncio.TimeoutError:
            logger.warning(f"⚠️ Ollama health check timed out for {model_info.id}")
            return False
        except aiohttp.ClientError as e:
            logger.warning(
                f"⚠️ Ollama health check connection error for {model_info.id}: {e}"
            )
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
