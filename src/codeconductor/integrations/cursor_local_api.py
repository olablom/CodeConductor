#!/usr/bin/env python3
"""
Cursor Local API Integration for CodeConductor MVP.

Uses Cursor's local API on port 3000 instead of cloud service.
Adds resilient retry/backoff and proxy-avoidance for localhost to mitigate
ECONNRESET and corporate proxy/TLS interception issues.
"""

import asyncio
import os
import random
from contextlib import asynccontextmanager
import aiohttp
import logging
from typing import Optional, Dict, Any, Callable

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CursorLocalAPI:
    """Integration with Cursor's local API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        backoff_base_seconds: float = 0.25,
        backoff_max_seconds: float = 2.0,
        trust_env: Optional[bool] = None,
    ) -> None:
        # Allow override via env; default to localhost
        env_base = os.getenv("CURSOR_BASE_URL", "").strip()
        self.base_url = (base_url or env_base or "http://localhost:3000").rstrip("/")
        self.api_url = f"{self.base_url}/api"
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, int(max_retries))
        self.backoff_base_seconds = max(0.01, float(backoff_base_seconds))
        self.backoff_max_seconds = max(
            self.backoff_base_seconds, float(backoff_max_seconds)
        )
        # trust_env controls whether aiohttp uses system proxy vars (HTTP_PROXY/HTTPS_PROXY)
        # If not explicitly provided, decide based on URL and CURSOR_DISABLE_PROXY
        self._trust_env = (
            bool(trust_env)
            if trust_env is not None
            else self._decide_trust_env(self.base_url)
        )

    # ---------- Helper decision functions (unit-testable) ----------
    @staticmethod
    def _decide_trust_env(base_url: str) -> bool:
        disable_proxy = os.getenv("CURSOR_DISABLE_PROXY", "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        if disable_proxy:
            return False
        # Avoid proxy for localhost and 127.0.0.1 by default
        lowered = (base_url or "").lower()
        if lowered.startswith("http://localhost") or lowered.startswith(
            "http://127.0.0.1"
        ):
            return False
        return True

    @staticmethod
    def _should_retry(exc: BaseException) -> bool:
        retryable_types = (
            aiohttp.ServerDisconnectedError,
            aiohttp.ClientConnectorError,
            aiohttp.ClientOSError,
            aiohttp.ClientPayloadError,
            asyncio.TimeoutError,
            ConnectionResetError,
        )
        return isinstance(exc, retryable_types)

    def _compute_backoff(self, attempt_index_zero_based: int) -> float:
        # Exponential backoff with jitter
        exp = min(
            self.backoff_max_seconds,
            self.backoff_base_seconds * (2**attempt_index_zero_based),
        )
        jitter = random.uniform(0.0, self.backoff_base_seconds)
        return min(self.backoff_max_seconds, exp + jitter)

    # ---------- Session and request helpers ----------
    @asynccontextmanager
    async def _session(self):
        # trust_env False prevents system proxy use which often causes ECONNRESET for localhost
        async with aiohttp.ClientSession(trust_env=self._trust_env) as session:
            yield session

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_payload: Optional[Dict[str, Any]] = None,
        timeout_total: Optional[float] = None,
        expect_json: bool = True,
    ) -> Optional[Dict[str, Any]]:
        url = f"{self.api_url}{path}"
        timeout = aiohttp.ClientTimeout(total=timeout_total or self.timeout_seconds)

        last_error: Optional[BaseException] = None
        for attempt in range(self.max_retries + 1):
            try:
                async with self._session() as session:
                    http_call: Callable[..., Any] = getattr(session, method.lower())
                    async with http_call(
                        url, json=json_payload, timeout=timeout
                    ) as response:
                        if response.status == 200:
                            if expect_json:
                                return await response.json()
                            return {"status": response.status}
                        # Auth error: do not retry
                        if response.status in (401, 403):
                            logger.error("Cursor local API authentication required")
                            return None
                        # For other status codes, capture and retry (if configured)
                        text_snippet = ""
                        try:
                            text_snippet = (await response.text())[:200]
                        except Exception:
                            pass
                        last_error = RuntimeError(
                            f"HTTP {response.status} from {url}: {text_snippet}"
                        )
                        logger.warning(
                            f"[CURSOR][HTTP {response.status}] attempt {attempt + 1}/{self.max_retries + 1}"
                        )
            except Exception as exc:  # Network/TLS/timeout errors
                last_error = exc
                is_retryable = self._should_retry(exc)
                logger.warning(
                    f"[CURSOR][RETRY] {type(exc).__name__}: {exc} (attempt {attempt + 1}/{self.max_retries + 1})"
                )
                if not is_retryable or attempt >= self.max_retries:
                    break
                await asyncio.sleep(self._compute_backoff(attempt))

        if last_error is not None:
            logger.error(f"Cursor local API request failed: {last_error}")
        return None

    # ---------- Public API ----------
    async def check_health(self) -> bool:
        """Check if Cursor's local API is available."""
        data = await self._request(
            "get", "/health", expect_json=True, timeout_total=5.0
        )
        if not data:
            return False
        version = data.get("version", "unknown")
        logger.info(f"Cursor local API healthy: {version}")
        return True

    async def get_auth_status(self) -> Dict[str, Any]:
        """Check authentication status."""
        data = await self._request(
            "get", "/auth/status", expect_json=True, timeout_total=5.0
        )
        if data is None:
            return {"authenticated": False, "error": "request_failed"}
        if "authenticated" not in data:
            data["authenticated"] = False
        return data

    async def send_chat_completion(
        self, prompt: str, model: str = "default"
    ) -> Optional[str]:
        """Send a chat completion request to Cursor's local API."""
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "model": model,
            "stream": False,
        }
        data = await self._request(
            "post",
            "/chat/completions",
            json_payload=payload,
            expect_json=True,
            timeout_total=self.timeout_seconds,
        )
        if not data:
            return None
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")

    async def get_workspace_info(self) -> Dict[str, Any]:
        """Get current workspace information."""
        data = await self._request(
            "get", "/workspace", expect_json=True, timeout_total=5.0
        )
        return data if data is not None else {"error": "request_failed"}

    async def is_available(self) -> bool:
        """Lightweight probe used by auto-fallback selectors."""
        try:
            return await self.check_health()
        except Exception:
            return False


async def main():
    """Demo function to test Cursor Local API integration."""
    print("Cursor Local API Integration Demo")
    print("=" * 50)

    api = CursorLocalAPI()

    # Check health
    print("Checking Cursor local API health...")
    is_healthy = await api.check_health()

    if not is_healthy:
        print("Cursor local API is not available")
        return

    # Check auth status
    print("Checking authentication status...")
    auth_status = await api.get_auth_status()
    print(f"   Auth status: {auth_status}")

    # Get workspace info
    print("Getting workspace info...")
    workspace = await api.get_workspace_info()
    print(f"   Workspace: {workspace}")

    # Test chat completion (if authenticated)
    if auth_status.get("authenticated", False):
        print("Testing chat completion...")
        response = await api.send_chat_completion("Hello, can you help me with Python?")
        if response:
            print(f"   Response: {response[:100]}...")
        else:
            print("   No response received")
    else:
        print("Skipping chat completion (not authenticated)")

    print("\nDemo complete!")


if __name__ == "__main__":
    asyncio.run(main())
