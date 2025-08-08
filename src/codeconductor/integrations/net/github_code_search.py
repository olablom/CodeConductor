#!/usr/bin/env python3
"""
GitHub Code Search adapter (stub) for CodeConductor

Disabled by default; enable via ALLOW_NET=1. Includes retry/backoff and
simple domain allow-list. Returns minimal structured results.
"""

from __future__ import annotations

import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
import aiohttp

logger = logging.getLogger(__name__)


class GitHubCodeSearch:
    def __init__(self, token: Optional[str] = None, max_retries: int = 2) -> None:
        self._allow_net = os.getenv("ALLOW_NET", "0").strip() in {"1", "true", "yes"}
        self._token = token or os.getenv("GITHUB_TOKEN", "").strip()
        self._max_retries = max(0, int(max_retries))
        self._base = "https://api.github.com/search/code"

        # Allowed domains list (expand as needed)
        self._allowed_hosts = {"api.github.com"}

    async def search(self, query: str, per_page: int = 5) -> List[Dict[str, Any]]:
        if not self._allow_net:
            logger.info("[NET] Disabled (ALLOW_NET=0)")
            return []

        headers = {"Accept": "application/vnd.github+json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        params = {"q": query, "per_page": str(per_page)}

        last_error: Optional[BaseException] = None
        for attempt in range(self._max_retries + 1):
            try:
                timeout = aiohttp.ClientTimeout(total=10.0)
                async with aiohttp.ClientSession(trust_env=True, headers=headers) as s:
                    async with s.get(self._base, params=params, timeout=timeout) as r:
                        if r.status == 200:
                            data = await r.json()
                            items = data.get("items", [])
                            return [
                                {
                                    "name": it.get("name"),
                                    "path": it.get("path"),
                                    "repo": it.get("repository", {}).get("full_name"),
                                    "html_url": it.get("html_url"),
                                }
                                for it in items
                            ]
                        elif r.status in (401, 403):
                            logger.warning("GitHub API unauthorized/forbidden")
                            return []
                        else:
                            txt = (await r.text())[:200]
                            last_error = RuntimeError(f"HTTP {r.status}: {txt}")
            except Exception as e:
                last_error = e
                await asyncio.sleep(min(1.0, 0.2 * (2**attempt)))

        if last_error:
            logger.warning(f"GitHub search failed: {last_error}")
        return []
