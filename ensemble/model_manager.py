"""
Model Manager for LLM Ensemble

Handles model discovery, health checks, and load balancing.
"""

import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ModelStatus(Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    ERROR = "error"

@dataclass
class ModelInfo:
    name: str
    endpoint: str
    status: ModelStatus
    response_time: float
    last_used: float
    success_rate: float
    capabilities: List[str]

class ModelManager:
    """Manages multiple LLM models for ensemble operations."""
    
    def __init__(self):
        self.models: Dict[str, ModelInfo] = {}
        self.default_endpoints = {
            "ollama": "http://localhost:11434",
            "lm_studio": "http://localhost:1234"
        }
        
    async def discover_models(self) -> Dict[str, ModelInfo]:
        """Discover available models from Ollama and LM Studio."""
        discovered_models = {}
        
        # Check Ollama
        ollama_models = await self._check_ollama_models()
        discovered_models.update(ollama_models)
        
        # Check LM Studio
        lm_studio_models = await self._check_lm_studio_models()
        discovered_models.update(lm_studio_models)
        
        self.models = discovered_models
        logger.info(f"Discovered {len(discovered_models)} models")
        return discovered_models
    
    async def _check_ollama_models(self) -> Dict[str, ModelInfo]:
        """Check for available Ollama models."""
        models = {}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.default_endpoints['ollama']}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        for model in data.get('models', []):
                            model_name = model['name']
                            models[f"ollama:{model_name}"] = ModelInfo(
                                name=model_name,
                                endpoint=f"{self.default_endpoints['ollama']}/api/generate",
                                status=ModelStatus.ONLINE,
                                response_time=0.0,
                                last_used=0.0,
                                success_rate=1.0,
                                capabilities=["text-generation", "code-generation"]
                            )
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
        
        return models
    
    async def _check_lm_studio_models(self) -> Dict[str, ModelInfo]:
        """Check for available LM Studio models."""
        models = {}
        try:
            async with aiohttp.ClientSession() as session:
                # LM Studio typically has one model loaded at a time
                async with session.get(f"{self.default_endpoints['lm_studio']}/v1/models") as response:
                    if response.status == 200:
                        data = await response.json()
                        for model in data.get('data', []):
                            model_name = model['id']
                            models[f"lm_studio:{model_name}"] = ModelInfo(
                                name=model_name,
                                endpoint=f"{self.default_endpoints['lm_studio']}/v1/chat/completions",
                                status=ModelStatus.ONLINE,
                                response_time=0.0,
                                last_used=0.0,
                                success_rate=1.0,
                                capabilities=["text-generation", "code-generation"]
                            )
        except Exception as e:
            logger.warning(f"LM Studio not available: {e}")
        
        return models
    
    async def health_check(self, model_id: str) -> bool:
        """Perform health check on specific model."""
        if model_id not in self.models:
            return False
            
        model = self.models[model_id]
        try:
            start_time = asyncio.get_event_loop().time()
            
            if "ollama" in model_id:
                healthy = await self._check_ollama_health(model)
            elif "lm_studio" in model_id:
                healthy = await self._check_lm_studio_health(model)
            else:
                return False
                
            if healthy:
                model.status = ModelStatus.ONLINE
                model.response_time = asyncio.get_event_loop().time() - start_time
            else:
                model.status = ModelStatus.ERROR
                
            return healthy
            
        except Exception as e:
            logger.error(f"Health check failed for {model_id}: {e}")
            model.status = ModelStatus.ERROR
            return False
    
    async def _check_ollama_health(self, model: ModelInfo) -> bool:
        """Check Ollama model health."""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": model.name,
                    "prompt": "test",
                    "stream": False
                }
                async with session.post(model.endpoint, json=payload, timeout=5) as response:
                    return response.status == 200
        except:
            return False
    
    async def _check_lm_studio_health(self, model: ModelInfo) -> bool:
        """Check LM Studio model health."""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": model.name,
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 10
                }
                async with session.post(model.endpoint, json=payload, timeout=5) as response:
                    return response.status == 200
        except:
            return False
    
    def get_available_models(self, min_models: int = 2) -> List[str]:
        """Get list of available models, prioritizing by success rate."""
        available = [
            model_id for model_id, model in self.models.items()
            if model.status == ModelStatus.ONLINE
        ]
        
        # Sort by success rate and response time
        available.sort(key=lambda x: (
            -self.models[x].success_rate,
            self.models[x].response_time
        ))
        
        if len(available) < min_models:
            logger.warning(f"Only {len(available)} models available, need {min_models}")
            
        return available[:min_models]
    
    def update_model_stats(self, model_id: str, success: bool, response_time: float):
        """Update model statistics after use."""
        if model_id in self.models:
            model = self.models[model_id]
            model.last_used = asyncio.get_event_loop().time()
            model.response_time = response_time
            
            # Update success rate (simple moving average)
            if success:
                model.success_rate = min(1.0, model.success_rate + 0.1)
            else:
                model.success_rate = max(0.0, model.success_rate - 0.1) 