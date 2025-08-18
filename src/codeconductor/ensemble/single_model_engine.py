"""
Single Model Engine for CodeConductor

A simplified engine that uses only one model, similar to OpenAI API approach.
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any

from .model_manager import ModelManager
from .query_dispatcher import QueryDispatcher

logger = logging.getLogger(__name__)


@dataclass
class SingleModelRequest:
    """Request for single model processing."""

    task_description: str
    context: dict[str, Any] | None = None
    timeout: float = 30.0


@dataclass
class SingleModelResponse:
    """Response from single model processing."""

    content: str
    model_used: str
    execution_time: float


class SingleModelEngine:
    """
    Simplified engine that uses only one model.

    This is similar to how OpenAI API works - one model, multiple agents.
    """

    def __init__(self, preferred_model: str = "meta-llama-3.1-8b-instruct"):
        self._quick = os.getenv("CC_QUICK_CI") == "1"
        self._seed = int(os.getenv("CC_QUICK_CI_SEED", "42"))
        self.model_manager = ModelManager()
        self.preferred_model = preferred_model
        self.query_dispatcher = QueryDispatcher()

    async def initialize(self) -> bool:
        """Initialize the single model engine."""
        try:
            logger.info("🚀 Initializing Single Model Engine...")

            # Kontrollera GPU_DISABLED först
            if os.getenv("CC_GPU_DISABLED", "0") == "1":
                logger.info(
                    "[MOCK] CC_GPU_DISABLED=1 active — skipping model loading "
                    "(SingleModelEngine.initialize)"
                )
                return True

            if self._quick:
                logger.info(
                    "[MOCK] CC_QUICK_CI=1 active — skipping model loading "
                    "(SingleModelEngine.initialize)"
                )
                return True

            # Initialize model manager
            logger.info("✅ Model manager initialized")

            # Load the preferred model
            loaded_models = (
                await (
                    self.model_manager.ensure_models_loaded_with_memory_check(
                        "light_load"
                    )
                )
            )
            if not loaded_models:
                logger.warning("⚠️ Could not load preferred model, trying fallback")
                # Try to load any available model
                available_models = await self.model_manager.get_available_model_ids()
                if available_models:
                    loaded_models = await self.model_manager.ensure_models_loaded(
                        [available_models[0]]
                    )

            if loaded_models:
                logger.info(
                    f"✅ Single model engine initialized with: {loaded_models[0]}"
                )
                return True
            else:
                logger.error("❌ No models could be loaded")
                return False

        except Exception as e:
            logger.error(f"❌ Single model engine initialization failed: {e}")
            return False

    async def process_request(self, request: SingleModelRequest) -> SingleModelResponse:
        """Process a request using a single model."""
        start_time = asyncio.get_event_loop().time()

        # Kontrollera GPU_DISABLED först
        if os.getenv("CC_GPU_DISABLED", "0") == "1":
            logger.info(
                "[MOCK] CC_GPU_DISABLED=1 active — returning mock content "
                "(SingleModelEngine.process_request)"
            )
            content = self._mock_content_for(request.task_description)
            execution_time = asyncio.get_event_loop().time() - start_time
            return SingleModelResponse(
                content=content,
                model_used=f"{self.preferred_model}-mock",
                execution_time=execution_time,
            )

        try:
            logger.info(
                f"Processing single model request: "
                f"{request.task_description[:50]}..."
            )
            if self._quick:
                # Deterministic mock path for CI
                logger.info(
                    "[MOCK] CC_QUICK_CI=1 active — returning mock content "
                    "(SingleModelEngine.process_request)"
                )
                content = self._mock_content_for(request.task_description)
                execution_time = asyncio.get_event_loop().time() - start_time
                return SingleModelResponse(
                    content=content,
                    model_used=f"{self.preferred_model}-mock",
                    execution_time=execution_time,
                )

            # Get available models
            available_models = await self.model_manager.list_models()
            if not available_models:
                raise Exception("❌ No models available")

            # Use the first available model
            model = available_models[0]
            logger.info(f"🎯 Using single model: {model.id}")

            # Send request to the model
            async with QueryDispatcher(timeout=request.timeout) as dispatcher:
                result = await dispatcher.dispatch_single(
                    model, request.task_description, request.context
                )

                if result.get("success"):
                    content = self._extract_response_content(result["response"])
                    execution_time = asyncio.get_event_loop().time() - start_time

                    logger.info(
                        f"✅ Single model request completed in "
                        f"{execution_time:.2f}s"
                    )

                    return SingleModelResponse(
                        content=content,
                        model_used=model.id,
                        execution_time=execution_time,
                    )
                else:
                    raise Exception(
                        f"Model request failed: {result.get('error', 'Unknown error')}"
                    )

        except Exception as e:
            logger.error(f"❌ Single model request failed: {e}")
            raise

    def _mock_content_for(self, prompt: str) -> str:
        """Return deterministic mock Python code used for quick CI runs."""
        p = (prompt or "").lower()
        # Bug fix / divide by zero
        if any(k in p for k in ["bug", "fix", "divide", "division by zero"]):
            return (
                "def divide(a, b):\n"
                "    try:\n"
                "        return a / b\n"
                "    except ZeroDivisionError:\n"
                "        return float('inf')\n"
            )
        # REST API
        if any(k in p for k in ["flask", "rest", "/health", "fastapi", "login"]):
            return (
                "from fastapi import FastAPI\n"
                "app = FastAPI()\n\n"
                "@app.get('/health')\n"
                "def health():\n"
                "    return {'ok': True}\n\n"
                "@app.post('/login')\n"
                "def login(username: str, password: str):\n"
                "    return {'status': 'ok'}\n"
            )
        # SQL query embedded as Python string
        if any(k in p for k in ["sql", "select ", " with ", "customers", "orders"]):
            return (
                "sql = '''\n"
                "SELECT customer_id, SUM(amount) AS total_amount\n"
                "FROM orders\n"
                "GROUP BY customer_id\n"
                "ORDER BY total_amount DESC\n"
                "LIMIT 5;\n"
                "'''\n"
            )
        # React hook example embedded as Python string
        if any(k in p for k in ["react", "hook", "usestate", "todo"]):
            return (
                "jsx = '\n"
                "import { useState } from \\'react\\';\n"
                "export function Todo(){const [items,setItems]=useState([]);"
                "return <div/>}\n"
                "'\n"
            )
        # Binary search implementation
        if "binary search" in p:
            return (
                "def binary_search(arr, target):\n"
                "    left, right = 0, len(arr) - 1\n"
                "    while left <= right:\n"
                "        mid = (left + right) // 2\n"
                "        if arr[mid] == target:\n"
                "            return mid\n"
                "        if arr[mid] < target:\n"
                "            left = mid + 1\n"
                "        else:\n"
                "            right = mid - 1\n"
                "    return -1\n"
            )
        # Fibonacci implementation
        if "fibonacci" in p:
            return (
                "def fibonacci(n):\n"
                "    if n <= 1:\n"
                "        return n\n"
                "    a, b = 0, 1\n"
                "    for _ in range(n - 1):\n"
                "        a, b = b, a + b\n"
                "    return b\n"
            )
        # Default minimal valid python
        return "print('ok')\n"

    def _extract_response_content(self, response_data: dict) -> str:
        """Extract content from model response."""
        try:
            if "choices" in response_data and response_data["choices"]:
                return response_data["choices"][0]["message"]["content"]
            elif "content" in response_data:
                return response_data["content"]
            else:
                return str(response_data)
        except Exception as e:
            logger.warning(f"⚠️ Could not extract content from response: {e}")
            return str(response_data)

    async def cleanup(self):
        """Clean up resources."""
        try:
            logger.info("🧹 Cleaning up Single Model Engine...")

            # Emergency unload all models
            try:
                if hasattr(self, "model_manager") and self.model_manager:
                    unloaded_count = await self.model_manager.emergency_unload_all()
                    logger.info(f"✅ Emergency unloaded {unloaded_count} models")
            except Exception as e:
                logger.warning(f"⚠️ Failed to unload models: {e}")

            logger.info("✅ Single Model Engine cleanup completed")

        except Exception as e:
            logger.error(f"❌ Single Model Engine cleanup failed: {e}")
