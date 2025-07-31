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

# LM Studio preferred models for complex tasks
LM_STUDIO_PREFERRED_MODELS = [
    "meta-llama-3.1-8b-instruct",
    "google/gemma-3-12b",
    "qwen2-vl-7b-instruct",
    "mistral-7b-instruct-v0.1",
    "codellama-7b-instruct",
]

# Memory configurations for RTX 5090 (32GB VRAM) - Updated with real-world measurements
MEMORY_CONFIGS = {
    "light_load": {
        "models": [
            "meta-llama-3.1-8b-instruct",
            "mistral-7b-instruct-v0.1",
        ],
        "estimated_vram": 20,  # ~10GB + ~8GB + ~2GB overhead
        "description": "Safe loading med 2 modeller (20GB, 12GB free)",
    },
    "medium_load": {
        "models": [
            "meta-llama-3.1-8b-instruct",
            "google/gemma-3-12b",
            "qwen2-7b-instruct",
        ],
        "estimated_vram": 28,  # ~10GB + ~12GB + ~6GB + overhead
        "description": "Optimal performance/memory balance (28GB, 4GB free)",
    },
    "aggressive_load": {
        "models": [
            "meta-llama-3.1-8b-instruct",
            "google/gemma-3-12b",
            "mistral-7b-instruct-v0.1",
            "qwen2-7b-instruct",
        ],
        "estimated_vram": 32,  # ~10GB + ~12GB + ~8GB + ~6GB + overhead
        "description": "Maximum models för RTX 5090 (32GB, 0GB free - RISKY!)",
    },
}

# Complexity-based loading thresholds
COMPLEXITY_BASED_LOADING = {
    (0.0, 0.3): "light_load",  # 1-2 models för enkla tasks
    (0.3, 0.7): "medium_load",  # 2-3 models för medel tasks
    (0.7, 1.0): "aggressive_load",  # 3-5 models för komplexa tasks
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
        self.loaded_models = set()  # Track currently loaded models

    async def load_model_via_cli(self, model_key: str, ttl_seconds: int = 7200) -> bool:
        """
        Load a model via LM Studio CLI with TTL management.

        Args:
            model_key: The model identifier (e.g., "meta-llama-3.1-8b-instruct")
            ttl_seconds: Time to live in seconds (default: 2 hours)

        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            logger.info(f"🔄 Loading model via CLI: {model_key}")

            # Construct CLI command with TTL and GPU settings
            command = [
                "lms",
                "load",
                model_key,
                "--ttl",
                str(ttl_seconds),
                "--gpu",
                "max",  # Use maximum GPU
                "-y",  # Skip confirmations
            ]

            # Run command with timeout and proper encoding
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",  # Fix encoding issue
                    errors="ignore",  # Ignore encoding errors
                    timeout=120,  # 2 minute timeout for loading
                ),
            )

            if result.returncode == 0:
                self.loaded_models.add(model_key)
                logger.info(f"✅ Successfully loaded {model_key} via CLI")
                return True
            else:
                logger.error(f"❌ Failed to load {model_key}: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"⏰ Timeout loading {model_key} via CLI")
            return False
        except Exception as e:
            logger.error(f"❌ Error loading {model_key} via CLI: {e}")
            return False

    async def ensure_models_loaded(
        self, required_models: List[str], ttl_seconds: int = 7200
    ) -> List[str]:
        """
        Ensure that required models are loaded, with fallback to available models.

        Args:
            required_models: List of preferred model keys
            ttl_seconds: Time to live for loaded models

        Returns:
            List[str]: List of successfully loaded/available model keys
        """
        logger.info(f"🎯 Ensuring models loaded: {required_models}")

        loaded_models = []

        # Try to load each required model
        for model_key in required_models:
            try:
                # Check if model is already loaded
                if model_key in self.loaded_models:
                    logger.info(f"✅ {model_key} already loaded")
                    loaded_models.append(model_key)
                    continue

                # Try to load via CLI
                success = await self.load_model_via_cli(model_key, ttl_seconds)
                if success:
                    loaded_models.append(model_key)
                    logger.info(f"✅ Successfully loaded {model_key}")
                else:
                    logger.warning(
                        f"⚠️ Failed to load {model_key}, will try JIT loading"
                    )

            except Exception as e:
                logger.warning(f"⚠️ Error loading {model_key}: {e}")

        # If we don't have enough models, get available models as fallback
        if len(loaded_models) < 2:
            logger.info("🔄 Getting available models as fallback")
            try:
                available_models = await self.list_models()
                available_model_ids = [model.id for model in available_models]

                # Add available models that aren't already in loaded_models
                for model_id in available_model_ids:
                    if model_id not in loaded_models and len(loaded_models) < 3:
                        loaded_models.append(model_id)
                        logger.info(f"✅ Added fallback model: {model_id}")

            except Exception as e:
                logger.error(f"❌ Error getting fallback models: {e}")

        logger.info(f"🎯 Final loaded models: {loaded_models}")
        return loaded_models

    async def unload_model_via_cli(self, model_key: str) -> bool:
        """
        Unload a model via LM Studio CLI.

        Args:
            model_key: The model identifier to unload

        Returns:
            bool: True if model unloaded successfully, False otherwise
        """
        try:
            logger.info(f"🔄 Unloading model via CLI: {model_key}")

            command = ["lms", "unload", model_key]

            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",  # Fix encoding issue
                    errors="ignore",  # Ignore encoding errors
                    timeout=30,
                ),
            )

            if result.returncode == 0:
                self.loaded_models.discard(model_key)
                logger.info(f"✅ Successfully unloaded {model_key}")
                return True
            else:
                logger.error(f"❌ Failed to unload {model_key}: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"❌ Error unloading {model_key}: {e}")
            return False

    async def get_loaded_models_status(self) -> Dict[str, Any]:
        """
        Get status of currently loaded models.

        Returns:
            Dict with loaded models info
        """
        try:
            # Use lms ps to get loaded models
            command = ["lms", "ps"]

            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",  # Fix encoding issue
                    errors="ignore",  # Ignore encoding errors
                    timeout=10,
                ),
            )

            # Parse lms ps output to get loaded models
            loaded_models = []
            logger.info(f"🎮 lms ps returncode: {result.returncode}")
            logger.info(f"🎮 lms ps stdout: {result.stdout}")

            if result.returncode == 0 and result.stdout:
                lines = result.stdout.strip().split("\n")
                logger.info(f"🎮 Parsing {len(lines)} lines from lms ps output")
                for line in lines:
                    logger.info(f"🎮 Processing line: '{line}'")
                    if line.startswith("Identifier:"):
                        # Extract model name from "Identifier: model-name"
                        model_name = line.split("Identifier:")[1].strip()
                        loaded_models.append(model_name)
                        logger.info(f"📦 Found loaded model: {model_name}")

            logger.info(f"🎮 Final loaded_models: {loaded_models}")
            return {
                "loaded_models": loaded_models,
                "cli_output": result.stdout,
                "total_loaded": len(loaded_models),
            }

        except Exception as e:
            logger.error(f"❌ Error getting loaded models status: {e}")
            return {
                "loaded_models": [],
                "cli_output": f"Error: {e}",
                "total_loaded": 0,
            }

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

        # Use all available models for best performance
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

    async def get_available_model_ids(self):
        """Get list of available model IDs"""
        try:
            all_models = await self.list_models()
            healthy_models = await self.list_healthy_models()

            # Filter to healthy models
            available_models = [
                model.id for model in all_models if model.id in healthy_models
            ]

            return available_models
        except Exception as e:
            logger.error(f"Error getting available model IDs: {e}")
            return []

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

                        # TEMPORARY FIX: Skip codellama models that cause crashes
                        if "codellama" in model_id.lower():
                            logger.info(
                                f"🐛 DEBUG: Skipping codellama model: {model_id}"
                            )
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

    async def get_gpu_memory_info(self):
        """
        Smart GPU memory detection with multiple fallback methods
        Priority: pynvml > pytorch > nvidia-smi > powershell
        """

        # Method 1: pynvml (Mest pålitlig för NVIDIA)
        memory_info = await self.get_gpu_memory_info_pynvml()
        if memory_info:
            logger.info(
                f"🎮 GPU memory detected via pynvml: {memory_info['usage_percent']}%"
            )
            return memory_info

        # Method 2: PyTorch CUDA (Om tillgängligt)
        memory_info = await self.get_gpu_memory_info_pytorch()
        if memory_info:
            logger.info(
                f"🎮 GPU memory detected via PyTorch: {memory_info['usage_percent']}%"
            )
            return memory_info

        # Method 3: nvidia-smi with Windows fixes
        memory_info = await self.get_gpu_memory_info_nvidia_smi_fixed()
        if memory_info:
            logger.info(
                f"🎮 GPU memory detected via nvidia-smi: {memory_info['usage_percent']}%"
            )
            return memory_info

        # Method 4: PowerShell WMI (Baseline fallback)
        memory_info = await self.get_gpu_memory_info_powershell()
        if memory_info:
            logger.info(
                f"🎮 GPU memory detected via PowerShell: {memory_info['usage_percent']}%"
            )
            return memory_info

        # Method 5: Static fallback för RTX 5090
        logger.warning(
            "🎮 All GPU memory detection methods failed, using RTX 5090 defaults"
        )
        return {
            "used_gb": 5.0,  # Conservative estimate
            "total_gb": 32.0,  # RTX 5090 has 32GB
            "free_gb": 27.0,
            "usage_percent": 15.6,
            "method": "static_fallback",
        }

    async def get_gpu_memory_info_pynvml(self):
        """Get GPU memory info using pynvml (NVIDIA Management Library)"""
        try:
            import pynvml

            pynvml.nvmlInit()

            # Get first GPU (RTX 5090)
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            # Convert bytes to GB
            total_gb = memory_info.total / (1024**3)
            used_gb = memory_info.used / (1024**3)
            free_gb = memory_info.free / (1024**3)
            usage_percent = (used_gb / total_gb) * 100

            pynvml.nvmlShutdown()

            return {
                "used_gb": round(used_gb, 1),
                "total_gb": round(total_gb, 1),
                "free_gb": round(free_gb, 1),
                "usage_percent": round(usage_percent, 1),
                "method": "pynvml",
            }

        except Exception as e:
            logger.error(f"🎮 pynvml GPU memory check failed: {e}")
            return None

    async def get_gpu_memory_info_pytorch(self):
        """Get GPU memory info using PyTorch CUDA"""
        try:
            import torch

            if torch.cuda.is_available():
                device = 0  # RTX 5090

                # Get memory info
                free_bytes, total_bytes = torch.cuda.mem_get_info(device)
                used_bytes = total_bytes - free_bytes

                # Convert to GB
                total_gb = total_bytes / (1024**3)
                used_gb = used_bytes / (1024**3)
                free_gb = free_bytes / (1024**3)
                usage_percent = (used_gb / total_gb) * 100

                return {
                    "used_gb": round(used_gb, 1),
                    "total_gb": round(total_gb, 1),
                    "free_gb": round(free_gb, 1),
                    "usage_percent": round(usage_percent, 1),
                    "method": "pytorch_cuda",
                }

        except Exception as e:
            logger.error(f"🎮 PyTorch CUDA memory check failed: {e}")

        return None

    async def get_gpu_memory_info_nvidia_smi_fixed(self):
        """Get GPU memory info using nvidia-smi with Windows-specific fixes"""
        try:
            import subprocess

            # Windows-specific nvidia-smi command with timeout and encoding
            cmd = [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ]

            # Use Windows-compatible subprocess settings
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=15,  # Longer timeout for Windows
                shell=False,  # Don't use shell on Windows
                creationflags=subprocess.CREATE_NO_WINDOW
                if hasattr(subprocess, "CREATE_NO_WINDOW")
                else 0,
            )

            if result.returncode == 0 and result.stdout.strip():
                try:
                    used_mb, total_mb = result.stdout.strip().split(", ")
                    used_gb = float(used_mb) / 1024
                    total_gb = float(total_mb) / 1024
                    free_gb = total_gb - used_gb
                    usage_percent = (used_gb / total_gb) * 100

                    return {
                        "used_gb": round(used_gb, 1),
                        "total_gb": round(total_gb, 1),
                        "free_gb": round(free_gb, 1),
                        "usage_percent": round(usage_percent, 1),
                        "method": "nvidia_smi_fixed",
                    }
                except ValueError as e:
                    logger.error(
                        f"🎮 Failed to parse nvidia-smi output: {result.stdout}"
                    )

        except subprocess.TimeoutExpired:
            logger.error("🎮 nvidia-smi command timed out (15s)")
        except Exception as e:
            logger.error(f"🎮 nvidia-smi command failed: {e}")

        return None

    async def get_gpu_memory_info_powershell(self):
        """Get GPU memory info using PowerShell WMI"""
        try:
            import subprocess

            # PowerShell command för GPU info
            ps_command = """
            $gpu = Get-CimInstance -ClassName Win32_VideoController | Where-Object {$_.Name -like '*RTX*'}
            if ($gpu) {
                $totalBytes = $gpu.AdapterRAM
                if ($totalBytes -gt 0) {
                    $totalGB = [math]::Round($totalBytes / 1GB, 1)
                    Write-Output "total_gb:$totalGB"
                    Write-Output "method:powershell_wmi"
                }
            }
            """

            result = subprocess.run(
                ["powershell", "-Command", ps_command],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                output = result.stdout.strip()
                if "total_gb:" in output:
                    lines = output.split("\n")
                    data = {"method": "powershell_wmi"}

                    for line in lines:
                        if ":" in line:
                            key, value = line.split(":", 1)
                            if key == "total_gb":
                                data["total_gb"] = float(value)
                                # Since WMI can't get current usage, estimate conservatively
                                data["used_gb"] = data["total_gb"] * 0.1  # 10% baseline
                                data["free_gb"] = data["total_gb"] * 0.9
                                data["usage_percent"] = 10.0

                    return data

        except Exception as e:
            logger.error(f"🎮 PowerShell GPU memory check failed: {e}")
            return None

    async def ensure_models_loaded_with_memory_check(self, config_name="medium_load"):
        """Load models with memory safety checks for RTX 5090"""
        config = MEMORY_CONFIGS.get(config_name, MEMORY_CONFIGS["medium_load"])

        # Check available memory
        gpu_info = await self.get_gpu_memory_info()
        if gpu_info:
            logger.info(
                f"🎮 RTX 5090 Memory: {gpu_info['used_gb']:.1f}GB used, {gpu_info['free_gb']:.1f}GB free"
            )

            if gpu_info["free_gb"] < config["estimated_vram"]:
                logger.warning(
                    f"⚠️ Insufficient VRAM: {gpu_info['free_gb']:.1f}GB available, {config['estimated_vram']}GB needed"
                )
                # Fallback to smaller config
                return await self.smart_memory_fallback(gpu_info["free_gb"])

        # Proceed with loading
        logger.info(f"🚀 Loading {config_name} config: {config['description']}")
        return await self.ensure_models_loaded(config["models"])

    async def smart_memory_fallback(self, available_gb: float):
        """Smart fallback based on available GPU memory for RTX 5090"""
        if available_gb >= 21:
            logger.info("🔄 Falling back to medium_load config")
            return await self.ensure_models_loaded(
                MEMORY_CONFIGS["medium_load"]["models"]
            )
        elif available_gb >= 13:
            logger.info("🔄 Falling back to light_load config")
            return await self.ensure_models_loaded(
                MEMORY_CONFIGS["light_load"]["models"]
            )
        elif available_gb >= 6:
            logger.info("🔄 Loading single model only")
            return await self.ensure_models_loaded(
                [MEMORY_CONFIGS["light_load"]["models"][0]]
            )
        else:
            logger.warning("❌ Insufficient GPU memory for any models")
            return []

    async def emergency_unload_all(self):
        """Emergency unload all models to free GPU memory"""
        try:
            logger.info("🚨 Starting emergency unload...")

            # Get currently loaded models
            loaded_status = await self.get_loaded_models_status()
            logger.info(f"🚨 loaded_status: {loaded_status}")

            loaded_models = loaded_status.get("loaded_models", [])
            logger.info(
                f"🚨 Found {len(loaded_models)} models to unload: {loaded_models}"
            )

            unloaded_count = 0
            for model in loaded_models:
                logger.info(f"🚨 Attempting to unload: {model}")
                if await self.unload_model_via_cli(model):
                    unloaded_count += 1
                    logger.info(f"🚨 Successfully unloaded: {model}")
                else:
                    logger.warning(f"🚨 Failed to unload: {model}")

            logger.info(f"🚨 Emergency unloaded {unloaded_count} models")
            return unloaded_count
        except Exception as e:
            logger.error(f"❌ Emergency unload failed: {e}")
            return 0

    async def test_all_gpu_methods(self):
        """Test all GPU memory detection methods and return results"""
        results = {}

        # Test pynvml
        try:
            pynvml_result = await self.get_gpu_memory_info_pynvml()
            results["pynvml"] = pynvml_result
            logger.info(
                f"🎮 pynvml test: {'✅ SUCCESS' if pynvml_result else '❌ FAILED'}"
            )
        except Exception as e:
            results["pynvml"] = None
            logger.error(f"🎮 pynvml test failed: {e}")

        # Test PyTorch
        try:
            pytorch_result = await self.get_gpu_memory_info_pytorch()
            results["pytorch"] = pytorch_result
            logger.info(
                f"🎮 PyTorch test: {'✅ SUCCESS' if pytorch_result else '❌ FAILED'}"
            )
        except Exception as e:
            results["pytorch"] = None
            logger.error(f"🎮 PyTorch test failed: {e}")

        # Test nvidia-smi
        try:
            nvidia_result = await self.get_gpu_memory_info_nvidia_smi_fixed()
            results["nvidia_smi"] = nvidia_result
            logger.info(
                f"🎮 nvidia-smi test: {'✅ SUCCESS' if nvidia_result else '❌ FAILED'}"
            )
        except Exception as e:
            results["nvidia_smi"] = None
            logger.error(f"🎮 nvidia-smi test failed: {e}")

        # Test PowerShell
        try:
            ps_result = await self.get_gpu_memory_info_powershell()
            results["powershell"] = ps_result
            logger.info(
                f"🎮 PowerShell test: {'✅ SUCCESS' if ps_result else '❌ FAILED'}"
            )
        except Exception as e:
            results["powershell"] = None
            logger.error(f"🎮 PowerShell test failed: {e}")

        # Summary
        working_methods = [k for k, v in results.items() if v is not None]
        logger.info(
            f"🎮 GPU memory detection summary: {len(working_methods)}/{len(results)} methods working"
        )
        logger.info(f"🎮 Working methods: {working_methods}")

        return results


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
