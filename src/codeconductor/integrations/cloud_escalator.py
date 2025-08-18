#!/usr/bin/env python3
"""
Cloud Escalator for CodeConductor MVP
Handles secure escalation to cloud LLM APIs with cost protection.
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

from aiohttp import ClientSession, ClientTimeout

logger = logging.getLogger(__name__)

# Cost protection constants
MAX_TOKENS_PER_REQUEST = 1000
MAX_REQUESTS_PER_HOUR = 10
MAX_COST_PER_REQUEST = 0.01  # $0.01 max per request
ESCALATION_THRESHOLD = 0.7  # Only escalate if local confidence < 0.7


@dataclass
class CloudResponse:
    """Response from cloud LLM."""

    model: str
    content: str
    tokens_used: int
    cost: float
    response_time: float
    success: bool
    error: str | None = None


class CloudEscalator:
    """Handles secure escalation to cloud LLM APIs."""

    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

        # Cost tracking
        self.total_cost = 0.0
        self.total_requests = 0
        self.request_history = []

        # Rate limiting
        self.last_request_time = 0
        self.requests_this_hour = 0
        self.hour_start_time = time.time()

        # Model configurations with cost estimates
        self.models = {
            "gpt-4": {
                "endpoint": "https://api.openai.com/v1/chat/completions",
                "max_tokens": MAX_TOKENS_PER_REQUEST,
                "cost_per_1k_tokens": 0.03,
                "provider": "openai",
            },
            "gpt-4-turbo": {
                "endpoint": "https://api.openai.com/v1/chat/completions",
                "max_tokens": MAX_TOKENS_PER_REQUEST,
                "cost_per_1k_tokens": 0.01,
                "provider": "openai",
            },
            "claude-3-sonnet": {
                "endpoint": "https://api.anthropic.com/v1/messages",
                "max_tokens": MAX_TOKENS_PER_REQUEST,
                "cost_per_1k_tokens": 0.015,
                "provider": "anthropic",
            },
            "claude-3-haiku": {
                "endpoint": "https://api.anthropic.com/v1/messages",
                "max_tokens": MAX_TOKENS_PER_REQUEST,
                "cost_per_1k_tokens": 0.0025,
                "provider": "anthropic",
            },
        }

    def _check_cost_limits(self, estimated_tokens: int, model: str) -> bool:
        """Check if request would exceed cost limits."""
        if model not in self.models:
            return False

        estimated_cost = (estimated_tokens / 1000) * self.models[model]["cost_per_1k_tokens"]

        if estimated_cost > MAX_COST_PER_REQUEST:
            logger.warning(
                f"⚠️ Estimated cost ${estimated_cost:.4f} exceeds limit ${MAX_COST_PER_REQUEST}"
            )
            return False

        if self.total_cost + estimated_cost > 0.05:  # $0.05 daily limit
            logger.warning(f"⚠️ Would exceed daily cost limit (current: ${self.total_cost:.4f})")
            return False

        return True

    def _check_rate_limits(self) -> bool:
        """Check rate limiting."""
        current_time = time.time()

        # Reset hourly counter if needed
        if current_time - self.hour_start_time > 3600:
            self.requests_this_hour = 0
            self.hour_start_time = current_time

        if self.requests_this_hour >= MAX_REQUESTS_PER_HOUR:
            logger.warning(
                f"⚠️ Rate limit exceeded: {self.requests_this_hour}/{MAX_REQUESTS_PER_HOUR} requests this hour"
            )
            return False

        # Minimum time between requests
        if current_time - self.last_request_time < 1.0:  # 1 second minimum
            logger.warning("⚠️ Rate limiting: minimum 1 second between requests")
            return False

        return True

    async def escalate_task(
        self,
        task: str,
        local_confidence: float = 0.0,
        preferred_model: str = "gpt-4-turbo",
    ) -> list[CloudResponse]:
        """
        Escalate task to cloud LLMs with safety checks.

        Args:
            task: The development task
            local_confidence: Confidence from local models (0.0-1.0)
            preferred_model: Preferred cloud model to use

        Returns:
            List of CloudResponse objects
        """
        # Safety check: only escalate if local confidence is low
        if local_confidence >= ESCALATION_THRESHOLD:
            logger.info(
                f"🏠 Local confidence {local_confidence:.2f} >= {ESCALATION_THRESHOLD}, skipping cloud escalation"
            )
            return []

        # Check API availability
        if not self.openai_api_key and not self.anthropic_api_key:
            logger.warning("⚠️ No cloud API keys available")
            return []

        # Estimate tokens
        estimated_tokens = len(task.split()) * 1.5  # Rough estimate

        # Check cost and rate limits
        if not self._check_cost_limits(estimated_tokens, preferred_model):
            logger.warning("⚠️ Cost or rate limits exceeded, skipping cloud escalation")
            return []

        if not self._check_rate_limits():
            logger.warning("⚠️ Rate limits exceeded, skipping cloud escalation")
            return []

        logger.info(
            f"☁️ Escalating to cloud: {preferred_model} (confidence: {local_confidence:.2f})"
        )

        # Select models to query
        models_to_query = [preferred_model]

        # Add backup model if available
        if preferred_model.startswith("gpt-") and "claude-3-haiku" in self.models:
            models_to_query.append("claude-3-haiku")
        elif preferred_model.startswith("claude-") and "gpt-4-turbo" in self.models:
            models_to_query.append("gpt-4-turbo")

        # Query models
        responses = []
        for model in models_to_query:
            try:
                response = await self._query_model(model, task)
                if response.success:
                    responses.append(response)
                    # Update tracking
                    self.total_cost += response.cost
                    self.total_requests += 1
                    self.requests_this_hour += 1
                    self.last_request_time = time.time()
                    self.request_history.append(
                        {
                            "time": time.time(),
                            "model": model,
                            "cost": response.cost,
                            "tokens": response.tokens_used,
                        }
                    )

                    logger.info(
                        f"✅ Cloud response from {model}: ${response.cost:.4f}, {response.tokens_used} tokens"
                    )
                else:
                    logger.warning(f"❌ Cloud request failed for {model}: {response.error}")

            except Exception as e:
                logger.error(f"❌ Exception querying {model}: {e}")

        return responses

    async def _query_model(self, model: str, task: str) -> CloudResponse:
        """Query a specific cloud model."""
        if model not in self.models:
            return CloudResponse(
                model=model,
                content="",
                tokens_used=0,
                cost=0.0,
                response_time=0.0,
                success=False,
                error="Unknown model",
            )

        model_config = self.models[model]
        start_time = time.time()

        try:
            if model_config["provider"] == "openai":
                return await self._query_openai(model, task, model_config)
            elif model_config["provider"] == "anthropic":
                return await self._query_anthropic(model, task, model_config)
            else:
                return CloudResponse(
                    model=model,
                    content="",
                    tokens_used=0,
                    cost=0.0,
                    response_time=time.time() - start_time,
                    success=False,
                    error=f"Unknown provider: {model_config['provider']}",
                )

        except Exception as e:
            return CloudResponse(
                model=model,
                content="",
                tokens_used=0,
                cost=0.0,
                response_time=time.time() - start_time,
                success=False,
                error=str(e),
            )

    async def _query_openai(self, model: str, task: str, config: dict) -> CloudResponse:
        """Query OpenAI API."""
        if not self.openai_api_key:
            return CloudResponse(
                model=model,
                content="",
                tokens_used=0,
                cost=0.0,
                response_time=0.0,
                success=False,
                error="No OpenAI API key",
            )

        start_time = time.time()

        async with ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": model,
                "messages": [{"role": "user", "content": task}],
                "max_tokens": config["max_tokens"],
                "temperature": 0.1,
            }

            try:
                async with session.post(
                    config["endpoint"],
                    json=payload,
                    headers=headers,
                    timeout=ClientTimeout(total=30),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()

                    content = data["choices"][0]["message"]["content"]
                    tokens_used = data["usage"]["total_tokens"]
                    cost = (tokens_used / 1000) * config["cost_per_1k_tokens"]

                    return CloudResponse(
                        model=model,
                        content=content,
                        tokens_used=tokens_used,
                        cost=cost,
                        response_time=time.time() - start_time,
                        success=True,
                    )

            except Exception as e:
                return CloudResponse(
                    model=model,
                    content="",
                    tokens_used=0,
                    cost=0.0,
                    response_time=time.time() - start_time,
                    success=False,
                    error=str(e),
                )

    async def _query_anthropic(self, model: str, task: str, config: dict) -> CloudResponse:
        """Query Anthropic API."""
        if not self.anthropic_api_key:
            return CloudResponse(
                model=model,
                content="",
                tokens_used=0,
                cost=0.0,
                response_time=0.0,
                success=False,
                error="No Anthropic API key",
            )

        start_time = time.time()

        async with ClientSession() as session:
            headers = {
                "x-api-key": self.anthropic_api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            }

            payload = {
                "model": model,
                "max_tokens": config["max_tokens"],
                "messages": [{"role": "user", "content": task}],
            }

            try:
                async with session.post(
                    config["endpoint"],
                    json=payload,
                    headers=headers,
                    timeout=ClientTimeout(total=30),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()

                    content = data["content"][0]["text"]
                    tokens_used = data["usage"]["input_tokens"] + data["usage"]["output_tokens"]
                    cost = (tokens_used / 1000) * config["cost_per_1k_tokens"]

                    return CloudResponse(
                        model=model,
                        content=content,
                        tokens_used=tokens_used,
                        cost=cost,
                        response_time=time.time() - start_time,
                        success=True,
                    )

            except Exception as e:
                return CloudResponse(
                    model=model,
                    content="",
                    tokens_used=0,
                    cost=0.0,
                    response_time=time.time() - start_time,
                    success=False,
                    error=str(e),
                )

    def get_cost_summary(self) -> dict[str, Any]:
        """Get cost and usage summary."""
        return {
            "total_cost": self.total_cost,
            "total_requests": self.total_requests,
            "requests_this_hour": self.requests_this_hour,
            "average_cost_per_request": self.total_cost / max(self.total_requests, 1),
            "cost_limit_remaining": max(0.05 - self.total_cost, 0),
            "rate_limit_remaining": max(MAX_REQUESTS_PER_HOUR - self.requests_this_hour, 0),
        }

    def is_available(self) -> bool:
        """Check if cloud APIs are available."""
        return bool(self.openai_api_key or self.anthropic_api_key)


# Convenience functions
async def escalate_to_cloud(task: str, local_confidence: float = 0.0) -> list[CloudResponse]:
    """Convenience function to escalate task to cloud."""
    escalator = CloudEscalator()
    return await escalator.escalate_task(task, local_confidence)


async def get_cloud_cost_summary() -> dict[str, Any]:
    """Get cloud cost summary."""
    escalator = CloudEscalator()
    return escalator.get_cost_summary()


if __name__ == "__main__":
    # Demo
    async def demo():
        escalator = CloudEscalator()

        # Check availability
        if not escalator.is_available():
            print("⚠️ Cloud APIs not available (missing API keys)")
            print("Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variables")
            return

        # Test task
        task = "Implement a secure authentication system with JWT tokens and refresh tokens"

        print("☁️ CodeConductor Cloud Escalator Demo")
        print("=" * 50)
        print(f"📝 Task: {task}")

        # Estimate cost
        # costs = escalator.estimate_cost(task) # This line is removed as per the new_code
        # print(f"💰 Estimated costs: {costs}")

        # Escalate (if APIs are available)
        try:
            # async with escalator: # This line is removed as per the new_code
            responses = await escalator.escalate_task(
                task, local_confidence=0.5
            )  # Changed to local_confidence

            for response in responses:
                print(f"\n🤖 {response.model}:")
                print(f"📊 Confidence: {response.success}")  # Changed to success
                print(f"💰 Cost: ${response.cost:.4f}")
                print(f"⏱️  Time: {response.response_time:.2f}s")
                print(f"📝 Response: {response.content[:200]}...")

        except Exception as e:
            print(f"❌ Error: {e}")

    asyncio.run(demo())
