#!/usr/bin/env python3
"""
Model Manager for CodeConductor MVP
Discovers and health-checks local LLM models (LM Studio & Ollama)
"""

import os
import sys
import time
import asyncio
import logging
import subprocess
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import re

import aiohttp

# Safety check for testing environment
TESTING_MODE = os.getenv("CC_TESTING_MODE", "0") == "1"
GPU_DISABLED = os.getenv("CC_GPU_DISABLED", "0") == "1"

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
        "estimated_vram": 12,  # Reduced from 20GB to 12GB
        "description": "Safe loading med 2 modeller (12GB, 20GB free)",
        "max_vram_percent": 40,  # Max 40% VRAM usage
    },
    "medium_load": {
        "models": [
            "meta-llama-3.1-8b-instruct",
            "google/gemma-3-12b",
            "mistral-7b-instruct-v0.1",
        ],
        "estimated_vram": 18,  # Reduced from 28GB to 18GB
        "description": "Optimal performance/memory balance (18GB, 14GB free)",
        "max_vram_percent": 60,  # Max 60% VRAM usage
    },
    "aggressive_load": {
        "models": [
            "meta-llama-3.1-8b-instruct",
            "google/gemma-3-12b",
            "mistral-7b-instruct-v0.1",
            "deepseek-r1-distill-qwen-7b",
        ],
        "estimated_vram": 24,  # Reduced from 32GB to 24GB
        "description": "Maximum models för RTX 5090 (24GB, 8GB free)",
        "max_vram_percent": 80,  # Max 80% VRAM usage
    },
}

# Emergency VRAM thresholds
EMERGENCY_VRAM_THRESHOLD = 85  # Unload all if >85% VRAM
WARNING_VRAM_THRESHOLD = 75  # Start unloading if >75% VRAM

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
    metadata: dict = None


class ModelManager:
    """Manages discovery and health-checking of local LLM models."""

    def __init__(self):
        self._quick = os.getenv("CC_QUICK_CI") == "1"
        self.lm_studio_endpoint = "http://localhost:1234/v1"
        self.ollama_endpoint = "http://localhost:11434"
        self.timeout = 30.0  # seconds
        self.health_timeout = 10.0  # shorter timeout for health checks
        self.loaded_models = set()  # Track currently loaded models
        # Simple VRAM-aware LRU scheduler metadata
        self._lru_order: list[str] = []
        self._max_vram_gb: int = 28  # configurable upper bound for eviction
        # Backend gating
        backends_raw = (os.getenv("ENGINE_BACKENDS") or "").strip().lower()
        if backends_raw:
            self._engine_backends = {
                b.strip() for b in backends_raw.split(",") if b.strip()
            }
        else:
            # Default: allow both when not specified
            self._engine_backends = {"lmstudio", "ollama"}
        self._lmstudio_disable = os.getenv("LMSTUDIO_DISABLE", "0").strip() == "1"
        self._lmstudio_cli_disable = (
            os.getenv("LMSTUDIO_CLI_DISABLE", "0").strip() == "1"
        )
        # Strict selector settings
        self._strict_selector = (
            os.getenv("MODEL_SELECTOR_STRICT", "0").strip() == "1"
            or os.getenv("SELECTOR_FORCE_EXACT", "0").strip() == "1"
        )

    def _lmstudio_enabled(self) -> bool:
        return ("lmstudio" in self._engine_backends) and (not self._lmstudio_disable)

    def _lmstudio_cli_enabled(self) -> bool:
        return self._lmstudio_enabled() and (not self._lmstudio_cli_disable)

    async def load_model_via_cli(self, model_key: str, ttl_seconds: int = 7200) -> bool:
        """
        Load a model via LM Studio CLI with TTL management.

        Args:
            model_key: The model identifier (e.g., "meta-llama-3.1-8b-instruct")
            ttl_seconds: Time to live in seconds (default: 2 hours)

        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        # Respect backend/CLI gating strictly
        if not self._lmstudio_cli_enabled():
            logger.info("LM Studio CLI disabled by configuration — skipping load")
            return False

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
        self, required_models: list[str], ttl_seconds: int = 7200
    ) -> list[str]:
        """
        Ensure that required models are loaded, with fallback to available models.

        Args:
            required_models: List of preferred model keys
            ttl_seconds: Time to live for loaded models

        Returns:
            List[str]: List of successfully loaded/available model keys
        """
        # Strict guard: if strict selector is active and a forced model is provided,
        # ignore any preset load-profiles and only use the forced model.
        if self._strict_selector:
            forced = (
                os.getenv("FORCE_MODEL")
                or os.getenv("WINNER_MODEL")
                or os.getenv("ENGINE_MODEL_ALLOWLIST")
            )
            if forced:
                forced = forced.strip()
                required_models = [forced]
                logger.info(
                    f"🔒 Strict selector active — using forced model only: {forced}"
                )

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
                    # Update LRU order
                    if model_key in self._lru_order:
                        self._lru_order.remove(model_key)
                    self._lru_order.append(model_key)
                else:
                    logger.warning(
                        f"⚠️ Failed to load {model_key}, will try JIT loading"
                    )

            except Exception as e:
                logger.warning(f"⚠️ Error loading {model_key}: {e}")

        # If strict selector is active, never add fallbacks
        if not self._strict_selector and len(loaded_models) < 2:
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

    async def ensure_models_loaded_with_memory_check(
        self, loading_config: str
    ) -> list[str]:
        """
        Wrapper that loads models according to MEMORY_CONFIGS and evicts via simple LRU
        if VRAM usage is expected to exceed the configured cap.
        """
        # Strict guard: bypass preset load-profiles when forced model is set
        if self._strict_selector:
            forced = (
                os.getenv("FORCE_MODEL")
                or os.getenv("WINNER_MODEL")
                or os.getenv("ENGINE_MODEL_ALLOWLIST")
            )
            if forced:
                forced = forced.strip()
                logger.info(
                    f"🔒 Strict selector active — bypassing '{loading_config}' profile, targeting only: {forced}"
                )
                return await self.ensure_models_loaded([forced])

        desired = MEMORY_CONFIGS.get(loading_config, MEMORY_CONFIGS["medium_load"])  # type: ignore[index]
        target_models = desired.get("models", [])

        # Try to load target models
        loaded = await self.ensure_models_loaded(target_models)

        # Rough VRAM guard: if estimated exceeds cap, evict oldest until under cap
        try:
            est = desired.get("estimated_vram", 18)
            if est > self._max_vram_gb:
                # Evict oldest in LRU until below cap
                while est > self._max_vram_gb and self._lru_order:
                    oldest = self._lru_order.pop(0)
                    await self.unload_model_via_cli(oldest)
                    if oldest in loaded:
                        loaded.remove(oldest)
                    # shrink estimate pessimistically by 6GB per model
                    est -= 6
        except Exception as e:  # pragma: no cover
            logger.debug(f"LRU eviction skipped: {e}")

        return loaded

    async def emergency_unload_all(self) -> int:
        """
        Unload all currently tracked models.
        """
        count = 0
        for model in list(self.loaded_models):
            try:
                if await self.unload_model_via_cli(model):
                    count += 1
            except Exception:
                continue
        self._lru_order.clear()
        return count

    async def unload_model_via_cli(self, model_key: str) -> bool:
        """
        Unload a model via LM Studio CLI.

        Args:
            model_key: The model identifier to unload

        Returns:
            bool: True if model unloaded successfully, False otherwise
        """
        if not self._lmstudio_cli_enabled():
            logger.info("LM Studio CLI disabled by configuration — skipping unload")
            return False

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

    async def get_loaded_models_status(self) -> dict[str, Any]:
        """
        Get status of currently loaded models.

        Returns:
            Dict with loaded models info
        """
        if not self._lmstudio_cli_enabled():
            return {"loaded_models": [], "cli_output": "", "total_loaded": 0}

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

    async def list_models(self) -> list[ModelInfo]:
        """
        Discover all available models from LM Studio and Ollama.

        Returns:
            List of ModelInfo objects for all discovered models.
        """
        if self._quick:
            logger.info("[MOCK] CC_QUICK_CI=1 active — returning stub models")
            return [
                ModelInfo(
                    id="mock-model-1",
                    name="mock-model-1",
                    provider="mock",
                    endpoint="mock://",
                    is_available=True,
                    metadata={"mock": True},
                ),
                ModelInfo(
                    id="mock-model-2",
                    name="mock-model-2",
                    provider="mock",
                    endpoint="mock://",
                    is_available=True,
                    metadata={"mock": True},
                ),
            ]

        # Discovery guard: allow disabling discovery and stick to forced/allowlist
        discovery_disabled = os.getenv("DISCOVERY_DISABLE", "0").strip() == "1" or (
            os.getenv("ENGINE_DISCOVERY_MODE", "").strip().lower() == "preloaded_only"
        )

        if discovery_disabled or self._strict_selector:
            forced = (
                os.getenv("FORCE_MODEL")
                or os.getenv("WINNER_MODEL")
                or os.getenv("ENGINE_MODEL_ALLOWLIST")
            )
            if forced:
                forced = forced.strip()
                logger.info(
                    f"🔒 Discovery disabled/strict — using forced model only: {forced}"
                )
                return [
                    ModelInfo(
                        id=forced,
                        name=forced,
                        provider="lm_studio",
                        endpoint=self.lm_studio_endpoint,
                        is_available=True,
                        metadata={},
                    )
                ]

        logger.info("🔍 Discovering local LLM models...")

        # Run discovery in parallel with backend gating and individual timeouts
        lm_studio_models: list[ModelInfo] | Exception | None = []
        ollama_models: list[ModelInfo] | Exception | None = []
        tasks = []
        if self._lmstudio_enabled():
            tasks.append(
                (
                    "lmstudio",
                    asyncio.wait_for(
                        self._discover_lm_studio_models(), timeout=self.timeout
                    ),
                )
            )
        if "ollama" in self._engine_backends:
            tasks.append(
                (
                    "ollama",
                    asyncio.wait_for(
                        self._discover_ollama_models(), timeout=self.timeout
                    ),
                )
            )

        if tasks:
            try:
                results = await asyncio.gather(
                    *[t[1] for t in tasks], return_exceptions=True
                )
                for (name, _), res in zip(tasks, results, strict=False):
                    if name == "lmstudio":
                        lm_studio_models = res
                    elif name == "ollama":
                        ollama_models = res
            except TimeoutError:
                logger.warning(
                    "⚠️ Model discovery timed out, using available models only"
                )
        else:
            logger.info(
                "No backends enabled via ENGINE_BACKENDS; returning empty model list"
            )
            return []

        # Handle exceptions gracefully
        if isinstance(lm_studio_models, Exception):
            logger.warning(f"LM Studio discovery failed: {lm_studio_models}")
            lm_studio_models = []

        if isinstance(ollama_models, Exception):
            logger.warning(f"Ollama discovery failed: {ollama_models}")
            ollama_models = []

        all_models = lm_studio_models + ollama_models
        # Apply allow/deny filters if provided
        allow_raw = os.getenv("ENGINE_MODEL_ALLOWLIST", "").strip()
        deny_raw = os.getenv("ENGINE_MODEL_DENYLIST", "").strip()
        if allow_raw:
            allow = {m.strip() for m in allow_raw.split(",") if m.strip()}
            all_models = [m for m in all_models if m.id in allow]
        if deny_raw:
            deny = {m.strip() for m in deny_raw.split(",") if m.strip()}
            all_models = [m for m in all_models if m.id not in deny]
        # Strict: keep only forced
        if self._strict_selector:
            forced = (
                os.getenv("FORCE_MODEL")
                or os.getenv("WINNER_MODEL")
                or os.getenv("ENGINE_MODEL_ALLOWLIST")
            )
            if forced:
                forced = forced.strip()
                all_models = [m for m in all_models if m.id == forced]
        logger.info(f"✅ Discovered {len(all_models)} models total")

        return all_models

    async def health_check(self, model_id: str) -> bool:
        """Check health of a specific model by ID."""
        if self._quick:
            return True
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

    async def check_all_health(self, models: list[ModelInfo]) -> dict[str, bool]:
        """Check health of all models in parallel."""
        if self._quick:
            return {m.id: True for m in models}
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
        for model, result in zip(models, health_results, strict=False):
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
    ) -> list[ModelInfo]:
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
            subprocess.Popen(
                download_cmd.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            logger.info(f"✅ Download started for {model_name}")
            logger.info("💡 Run 'ollama list' to check progress")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to start download for {model_name}: {e}")
            return False

    def get_recommended_models(self) -> dict[str, dict]:
        """Get list of recommended models with their info."""
        return RECOMMENDED_MODELS

    def get_available_models(self) -> list[str]:
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

    async def list_healthy_models(self) -> list[str]:
        """
        Get list of healthy model IDs for use in querying.

        Returns:
            List of model IDs that are currently healthy.
        """
        if self._quick:
            return ["mock-model-1", "mock-model-2"]
        # For now, return all discovered models
        # In a real implementation, this would filter by health status
        models = await self.list_models()
        return [model.id for model in models]

    async def get_available_model_ids(self):
        """Get list of available model IDs"""
        try:
            if self._quick:
                return ["mock-model-1", "mock-model-2"]
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
    ) -> dict[str, Any]:
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
    ) -> dict[str, Any]:
        """Test LM Studio model response."""
        if not self._lmstudio_enabled():
            return {"success": False, "error": "lmstudio_disabled"}
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
    ) -> dict[str, Any]:
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

    async def _discover_lm_studio_models(self) -> list[ModelInfo]:
        """Discover models from LM Studio."""
        if not self._lmstudio_enabled():
            logger.info(
                "LM Studio backend disabled by configuration — skipping discovery"
            )
            return []
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

                        # Skip duplicate models with numbered suffixes (e.g., :2, :3, :4, :5)
                        if ":" in model_id and model_id.split(":")[-1].isdigit():
                            logger.info(
                                f"🐛 DEBUG: Skipping duplicate model: {model_id}"
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

        except TimeoutError:
            logger.warning("⚠️ LM Studio discovery timed out")
            return []
        except aiohttp.ClientError as e:
            logger.warning(f"⚠️ LM Studio connection error: {e}")
            return []
        except Exception as e:
            logger.error(f"LM Studio discovery error: {e}")
            return []

    async def _discover_ollama_models(self) -> list[ModelInfo]:
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

        except TimeoutError:
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
        if not self._lmstudio_enabled():
            return False
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

        except TimeoutError:
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

        except TimeoutError:
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
        
        # Safety check - disable GPU operations during testing
        if TESTING_MODE or GPU_DISABLED:
            logger.info("🎮 GPU operations disabled during testing")
            return {
                "total_gb": 32.0,
                "used_gb": 0.0,
                "free_gb": 32.0,
                "usage_percent": 0.0,
                "method": "mock"
            }

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
                creationflags=(
                    subprocess.CREATE_NO_WINDOW
                    if hasattr(subprocess, "CREATE_NO_WINDOW")
                    else 0
                ),
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
                except ValueError:
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

    async def smart_memory_cleanup(self, target_vram_percent: float = 60) -> int:
        """
        Smart memory cleanup that unloads models until VRAM usage is below target.

        Args:
            target_vram_percent: Target VRAM usage percentage

        Returns:
            int: Number of models unloaded
        """
        try:
            logger.info(
                f"🧹 Starting smart memory cleanup (target: {target_vram_percent}%)"
            )

            # Get current VRAM usage
            gpu_info = await self.get_gpu_memory_info()
            if not gpu_info:
                logger.warning("⚠️ Could not get GPU memory info, skipping cleanup")
                return 0

            current_vram_percent = gpu_info["usage_percent"]
            logger.info(f"🧹 Current VRAM usage: {current_vram_percent:.1f}%")

            if current_vram_percent <= target_vram_percent:
                logger.info(
                    f"🧹 VRAM usage ({current_vram_percent:.1f}%) is already below target ({target_vram_percent}%)"
                )
                return 0

            # Get currently loaded models
            loaded_status = await self.get_loaded_models_status()
            loaded_models = loaded_status.get("loaded_models", [])

            if not loaded_models:
                logger.info("🧹 No models currently loaded")
                return 0

            logger.info(f"🧹 Found {len(loaded_models)} loaded models: {loaded_models}")

            # Unload models until we reach target VRAM usage
            unloaded_count = 0
            for model in loaded_models:
                if current_vram_percent <= target_vram_percent:
                    break

                logger.info(f"🧹 Unloading {model} to reduce VRAM usage")
                if await self.unload_model_via_cli(model):
                    unloaded_count += 1
                    logger.info(f"🧹 Successfully unloaded: {model}")

                    # Check VRAM usage after unloading
                    await asyncio.sleep(2)  # Give GPU time to free memory
                    gpu_info = await self.get_gpu_memory_info()
                    if gpu_info:
                        current_vram_percent = gpu_info["usage_percent"]
                        logger.info(
                            f"🧹 VRAM usage after unloading {model}: {current_vram_percent:.1f}%"
                        )
                else:
                    logger.warning(f"🧹 Failed to unload: {model}")

            logger.info(f"🧹 Smart cleanup completed: unloaded {unloaded_count} models")
            return unloaded_count

        except Exception as e:
            logger.error(f"❌ Smart memory cleanup failed: {e}")
            return 0

    async def check_and_cleanup_memory(self, config_name: str = "medium_load") -> bool:
        """
        Check VRAM usage and perform cleanup if necessary.

        Args:
            config_name: Memory configuration to check against

        Returns:
            bool: True if cleanup was performed, False otherwise
        """
        try:
            config = MEMORY_CONFIGS.get(config_name, MEMORY_CONFIGS["medium_load"])
            max_vram_percent = config.get("max_vram_percent", 60)

            # Get current VRAM usage
            gpu_info = await self.get_gpu_memory_info()
            if not gpu_info:
                logger.warning("⚠️ Could not get GPU memory info")
                return False

            current_vram_percent = gpu_info["usage_percent"]
            logger.info(
                f"🎮 Current VRAM usage: {current_vram_percent:.1f}% (max: {max_vram_percent}%)"
            )

            # Emergency cleanup if VRAM usage is too high
            if current_vram_percent > EMERGENCY_VRAM_THRESHOLD:
                logger.warning(
                    f"🚨 Emergency VRAM threshold exceeded: {current_vram_percent:.1f}% > {EMERGENCY_VRAM_THRESHOLD}%"
                )
                await self.emergency_unload_all()
                return True

            # Warning cleanup if VRAM usage is high
            elif current_vram_percent > WARNING_VRAM_THRESHOLD:
                logger.warning(
                    f"⚠️ Warning VRAM threshold exceeded: {current_vram_percent:.1f}% > {WARNING_VRAM_THRESHOLD}%"
                )
                await self.smart_memory_cleanup(max_vram_percent)
                return True

            # Normal cleanup if VRAM usage exceeds config limit
            elif current_vram_percent > max_vram_percent:
                logger.info(
                    f"🧹 VRAM usage ({current_vram_percent:.1f}%) exceeds config limit ({max_vram_percent}%)"
                )
                await self.smart_memory_cleanup(max_vram_percent)
                return True

            return False

        except Exception as e:
            logger.error(f"❌ Memory check failed: {e}")
            return False

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

    async def test_all_gpu_methods(self):
        """Test all GPU memory detection methods and return results"""
        # Safety check - disable GPU operations during testing
        if TESTING_MODE or GPU_DISABLED:
            logger.info("🎮 GPU testing disabled during testing mode")
            return {
                "pynvml": {"method": "mock", "usage_percent": 0.0},
                "pytorch": {"method": "mock", "usage_percent": 0.0},
                "nvidia_smi": {"method": "mock", "usage_percent": 0.0},
                "powershell": {"method": "mock", "usage_percent": 0.0}
            }
            
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

    def get_agent_model_config(self) -> dict[str, list[str]]:
        """
        Get model configuration for different agent types.
        Updated August 2025 with latest model recommendations.
        Optimized for available models.

        Returns:
            Dict mapping agent type to preferred models
        """
        return {
            "coder": [
                "mistralai/codestral-22b-v0.1",  # Best code generation (13GB)
                "mistral-7b-instruct-v0.1",  # Fast code generation (4.1GB)
                "mistral:latest",  # Ollama version (fast)
            ],
            "architect": [
                "google/gemma-3-12b",  # Strong reasoning (8GB, 128k context)
                "mistralai/codestral-22b-v0.1",  # Good reasoning + code understanding
                "mistral-7b-instruct-v0.1",  # Fast reasoning (4.1GB)
            ],
            "tester": [
                "phi3:mini",  # Fastest testing (2GB, 0.12s TTFT)
                "mistral:latest",  # Fast Ollama version
                "mistral-7b-instruct-v0.1",  # Code understanding (4.1GB)
            ],
            "reviewer": [
                "mistralai/codestral-22b-v0.1",  # High precision code review (13GB)
                "google/gemma-3-12b",  # Security and performance analysis (8GB)
                "mistral-7b-instruct-v0.1",  # Fast review (4.1GB)
            ],
        }

    async def get_models_for_agent(
        self, agent_type: str, num_models: int = 2
    ) -> list[str]:
        """
        Get best models for a specific agent type.

        Args:
            agent_type: Type of agent (coder, architect, tester, reviewer)
            num_models: Number of models to return

        Returns:
            List of model IDs best suited for this agent type
        """
        config = self.get_agent_model_config()
        preferred_models = config.get(agent_type, config["coder"])  # Default to coder

        # Get available models
        available_models = await self.list_models()
        available_ids = [model.id for model in available_models]

        # Filter to only available models
        available_preferred = [
            model for model in preferred_models if model in available_ids
        ]

        # If not enough preferred models, add any available
        if len(available_preferred) < num_models:
            other_available = [
                model for model in available_ids if model not in available_preferred
            ]
            available_preferred.extend(
                other_available[: num_models - len(available_preferred)]
            )

        return available_preferred[:num_models]


# Convenience functions for easy usage
async def discover_models() -> list[ModelInfo]:
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

    print("\n🏥 Health Status:")
    for model_id, is_healthy in health_status.items():
        status = "✅" if is_healthy else "❌"
        print(f"  {status} {model_id}")

    # Summary
    healthy_count = sum(health_status.values())
    print(f"\n📊 Summary: {healthy_count}/{len(models)} models are healthy")


if __name__ == "__main__":
    asyncio.run(main())
