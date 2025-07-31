#!/usr/bin/env python3
"""
Cursor Local API Integration for CodeConductor MVP.

Uses Cursor's local API on port 3000 instead of cloud service.
"""

import asyncio
import aiohttp
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CursorLocalAPI:
    """Integration with Cursor's local API."""

    def __init__(self):
        self.base_url = "http://localhost:3000"
        self.api_url = f"{self.base_url}/api"
        self.timeout = 30.0

    async def check_health(self) -> bool:
        """Check if Cursor's local API is available."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.api_url}/health", timeout=aiohttp.ClientTimeout(total=5.0)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(
                            f"✅ Cursor local API healthy: {data.get('version', 'unknown')}"
                        )
                        return True
                    else:
                        logger.warning(f"⚠️ Cursor API returned {response.status}")
                        return False
        except Exception as e:
            logger.error(f"❌ Cursor local API health check failed: {e}")
            return False

    async def get_auth_status(self) -> Dict[str, Any]:
        """Check authentication status."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.api_url}/auth/status",
                    timeout=aiohttp.ClientTimeout(total=5.0),
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {"authenticated": False, "status": response.status}
        except Exception as e:
            logger.error(f"❌ Auth status check failed: {e}")
            return {"authenticated": False, "error": str(e)}

    async def send_chat_completion(
        self, prompt: str, model: str = "default"
    ) -> Optional[str]:
        """Send a chat completion request to Cursor's local API."""
        try:
            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "model": model,
                "stream": False,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return (
                            data.get("choices", [{}])[0]
                            .get("message", {})
                            .get("content", "")
                        )
                    elif response.status == 401:
                        logger.error("❌ Cursor API requires authentication")
                        return None
                    else:
                        logger.error(f"❌ Cursor API error: {response.status}")
                        return None

        except Exception as e:
            logger.error(f"❌ Chat completion failed: {e}")
            return None

    async def get_workspace_info(self) -> Dict[str, Any]:
        """Get current workspace information."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.api_url}/workspace",
                    timeout=aiohttp.ClientTimeout(total=5.0),
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {"error": f"Status {response.status}"}
        except Exception as e:
            logger.error(f"❌ Workspace info failed: {e}")
            return {"error": str(e)}


async def main():
    """Demo function to test Cursor Local API integration."""
    print("🎯 Cursor Local API Integration Demo")
    print("=" * 50)

    api = CursorLocalAPI()

    # Check health
    print("1️⃣ Checking Cursor local API health...")
    is_healthy = await api.check_health()

    if not is_healthy:
        print("❌ Cursor local API is not available")
        return

    # Check auth status
    print("2️⃣ Checking authentication status...")
    auth_status = await api.get_auth_status()
    print(f"   Auth status: {auth_status}")

    # Get workspace info
    print("3️⃣ Getting workspace info...")
    workspace = await api.get_workspace_info()
    print(f"   Workspace: {workspace}")

    # Test chat completion (if authenticated)
    if auth_status.get("authenticated", False):
        print("4️⃣ Testing chat completion...")
        response = await api.send_chat_completion("Hello, can you help me with Python?")
        if response:
            print(f"   Response: {response[:100]}...")
        else:
            print("   ❌ No response received")
    else:
        print("4️⃣ Skipping chat completion (not authenticated)")

    print("\n🎉 Demo complete!")


if __name__ == "__main__":
    asyncio.run(main())
