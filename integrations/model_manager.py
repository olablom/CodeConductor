"""
Model Manager for CodeConductor v2.0

Manages local LLM models via LM Studio CLI and integrates with existing LM Studio client.
"""

import requests
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from omegaconf import OmegaConf

logger = logging.getLogger("ModelManager")


class ModelManager:
    """Manages local LLM models for CodeConductor"""

    def __init__(self, config_path: str = "config/base.yaml"):
        self.config_path = Path(config_path)
        self.server_url = "http://localhost:1234"
        self.available_models = []
        self.default_model = None

        # Load configuration
        self._load_config()

        # Initialize LM Studio CLI check
        self._check_lm_studio_cli()

    def _load_config(self) -> None:
        """Load model configuration from YAML"""
        try:
            if self.config_path.exists():
                config = OmegaConf.load(self.config_path)

                # Get LLM configuration
                llm_config = config.get("llm", {})
                self.server_url = llm_config.get("server_url", "http://localhost:1234")
                self.available_models = llm_config.get("available_models", [])
                self.default_model = llm_config.get("default_model", None)

                logger.info(f"Loaded {len(self.available_models)} models from config")
            else:
                logger.warning(
                    f"Config file {self.config_path} not found, using defaults"
                )

        except Exception as e:
            logger.error(f"Failed to load config: {e}")

    def _check_lm_studio_cli(self) -> bool:
        """Check if LM Studio CLI is available"""
        try:
            result = subprocess.run(
                ["lms", "version"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                logger.info("LM Studio CLI is available")
                return True
            else:
                logger.warning("LM Studio CLI not found or not working")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("LM Studio CLI not found in PATH")
            return False

    def list_available_models(self) -> List[str]:
        """List all available models from LM Studio CLI"""
        try:
            result = subprocess.run(
                ["lms", "get"], capture_output=True, text=True, timeout=30
            )

            if result.returncode == 0:
                # Parse the output to extract model IDs
                lines = result.stdout.strip().split("\n")
                models = []

                for line in lines:
                    # Look for model IDs in the output
                    if "/" in line or line.strip().endswith("b"):
                        # Extract potential model ID
                        parts = line.strip().split()
                        if parts:
                            models.append(parts[0])

                logger.info(f"Found {len(models)} available models via CLI")
                return models
            else:
                logger.error(f"Failed to list models: {result.stderr}")
                return []

        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    def list_local_models(self) -> List[str]:
        """List currently loaded models in LM Studio server"""
        try:
            response = requests.get(f"{self.server_url}/v1/models", timeout=5)
            response.raise_for_status()

            data = response.json()
            if "data" in data:
                model_ids = [model["id"] for model in data["data"]]
                logger.info(f"Found {len(model_ids)} models loaded in server")
                return model_ids
            else:
                logger.warning("No models found in server response")
                return []

        except requests.RequestException as e:
            logger.error(f"Failed to get local models: {e}")
            return []

    def is_model_installed(self, model_id: str) -> bool:
        """Check if a model is available for installation"""
        available_models = self.list_available_models()
        return model_id in available_models

    def is_model_loaded(self, model_id: str) -> bool:
        """Check if a model is currently loaded in LM Studio server"""
        local_models = self.list_local_models()
        return model_id in local_models

    def install_model(self, model_id: str) -> bool:
        """Install a model via LM Studio CLI"""
        try:
            logger.info(f"Installing model: {model_id}")

            result = subprocess.run(
                ["lms", "get", model_id, "--yes"],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes timeout for large models
            )

            if result.returncode == 0:
                logger.info(f"✅ Successfully installed {model_id}")
                return True
            else:
                logger.error(f"❌ Failed to install {model_id}: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"❌ Installation timeout for {model_id}")
            return False
        except Exception as e:
            logger.error(f"❌ Error installing {model_id}: {e}")
            return False

    def load_model(self, model_id: str) -> bool:
        """Load a model in LM Studio server"""
        try:
            logger.info(f"Loading model: {model_id}")

            # This would require LM Studio API support for model loading
            # For now, we'll just check if it's available
            if self.is_model_loaded(model_id):
                logger.info(f"✅ Model {model_id} is already loaded")
                return True
            else:
                logger.warning(
                    f"⚠️ Model {model_id} not loaded (manual loading required)"
                )
                return False

        except Exception as e:
            logger.error(f"❌ Error loading {model_id}: {e}")
            return False

    def ensure_models(self, model_ids: Optional[List[str]] = None) -> Dict[str, bool]:
        """Ensure all specified models are installed and available"""
        if model_ids is None:
            model_ids = self.available_models

        results = {}

        for model_id in model_ids:
            logger.info(f"Checking model: {model_id}")

            # Check if model is available for installation
            if not self.is_model_installed(model_id):
                logger.warning(f"⚠️ Model {model_id} not available for installation")
                results[model_id] = False
                continue

            # Check if model is loaded
            if self.is_model_loaded(model_id):
                logger.info(f"✅ Model {model_id} is ready")
                results[model_id] = True
            else:
                logger.info(f"📥 Model {model_id} needs to be loaded")
                # Try to load the model
                results[model_id] = self.load_model(model_id)

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        return {
            "server_url": self.server_url,
            "default_model": self.default_model,
            "configured_models": self.available_models,
            "available_for_installation": self.list_available_models(),
            "currently_loaded": self.list_local_models(),
            "lm_studio_cli_available": self._check_lm_studio_cli(),
        }

    def update_config(
        self, models: List[str], default_model: Optional[str] = None
    ) -> bool:
        """Update the configuration with new models"""
        try:
            # Load existing config
            if self.config_path.exists():
                config = OmegaConf.load(self.config_path)
            else:
                config = OmegaConf.create({})

            # Update LLM section
            if "llm" not in config:
                config.llm = {}

            config.llm.available_models = models
            if default_model:
                config.llm.default_model = default_model
            config.llm.server_url = self.server_url

            # Save updated config
            OmegaConf.save(config, self.config_path)

            # Update instance variables
            self.available_models = models
            if default_model:
                self.default_model = default_model

            logger.info(f"✅ Updated config with {len(models)} models")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to update config: {e}")
            return False

    def recommend_models(self) -> List[str]:
        """Recommend good models for code generation"""
        recommended = [
            "meta-llama/Llama-3-7b",
            "hf://coder-llama/CodeLlama-7b-instruct",
            "mistralai/Mistral-7B-Instruct",
            "vicuna-13b-q4f16",
            "mosaic-ml/Gemma-7B-v0",
            "microsoft/DialoGPT-medium",
            "gpt2",
        ]

        # Filter to only available models
        available = self.list_available_models()
        return [model for model in recommended if model in available]

    def setup_default_models(self) -> bool:
        """Set up a default set of models for CodeConductor"""
        logger.info("Setting up default models for CodeConductor...")

        # Get recommended models that are available
        recommended = self.recommend_models()

        if not recommended:
            logger.warning("No recommended models available")
            return False

        # Update configuration
        success = self.update_config(
            models=recommended, default_model=recommended[0] if recommended else None
        )

        if success:
            logger.info(f"✅ Set up {len(recommended)} default models")
            logger.info(f"Default model: {self.default_model}")

        return success
