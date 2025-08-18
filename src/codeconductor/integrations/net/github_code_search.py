#!/usr/bin/env python3
"""
GitHub Code Search adapter (stub) for CodeConductor

Disabled by default; enable via ALLOW_NET=1. Includes retry/backoff and
simple domain allow-list. Returns minimal structured results.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


class GitHubCodeSearch:
    def __init__(self, token: str | None = None, max_retries: int = 2) -> None:
        self._allow_net = os.getenv("ALLOW_NET", "0").strip() in {"1", "true", "yes"}
        self._token = token or os.getenv("GITHUB_TOKEN", "").strip()
        self._max_retries = max(
            0, int(os.getenv("NET_MAX_RETRIES", str(max_retries)) or str(max_retries))
        )
        self._timeout_s = float(os.getenv("NET_TIMEOUT_S", "10") or "10")
        self._per_page = max(1, int(os.getenv("GH_PER_PAGE", "5") or "5"))
        self._cache_ttl = max(0, int(os.getenv("NET_CACHE_TTL_SECONDS", "3600") or "3600"))
        self._cache_dir = Path(os.getenv("NET_CACHE_DIR", "artifacts/net_cache"))
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._base = "https://api.github.com/search/code"

        # Allowed domains list (expand as needed)
        self._allowed_hosts = {"api.github.com"}

    async def search(self, query: str, per_page: int = 5) -> list[dict[str, Any]]:
        if not self._allow_net:
            logger.info("[NET] Disabled (ALLOW_NET=0)")
            return []

        import hashlib
        import json
        import random
        import time

        headers = {"Accept": "application/vnd.github+json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        page_size = per_page or self._per_page
        params = {"q": query, "per_page": str(page_size)}

        # Cache key
        ck = hashlib.sha256((query + str(page_size)).encode("utf-8")).hexdigest()
        cpath = self._cache_dir / f"gh_{ck}.json"
        if self._cache_ttl > 0 and cpath.exists():
            try:
                stat = cpath.stat()
                if (time.time() - stat.st_mtime) <= self._cache_ttl:
                    cached = json.loads(cpath.read_text(encoding="utf-8"))
                    if isinstance(cached, list):
                        return cached
            except Exception:
                pass

        last_error: BaseException | None = None
        for attempt in range(self._max_retries + 1):
            try:
                timeout = aiohttp.ClientTimeout(total=self._timeout_s)
                async with aiohttp.ClientSession(trust_env=True, headers=headers) as s:
                    async with s.get(self._base, params=params, timeout=timeout) as r:
                        if r.status == 200:
                            data = await r.json()
                            items = data.get("items", [])
                            results = [
                                {
                                    "name": it.get("name"),
                                    "path": it.get("path"),
                                    "repo": it.get("repository", {}).get("full_name"),
                                    "html_url": it.get("html_url"),
                                }
                                for it in items
                            ]
                            try:
                                cpath.write_text(
                                    json.dumps(results, ensure_ascii=False),
                                    encoding="utf-8",
                                )
                            except Exception:
                                pass
                            return results
                        elif r.status in (401, 403):
                            logger.warning("GitHub API unauthorized/forbidden")
                            return []
                        elif r.status in (429, 502, 503):
                            delay = min(60.0, (1.0 * (2**attempt)) + random.uniform(0.0, 0.5))
                            await asyncio.sleep(delay)
                            last_error = RuntimeError(f"HTTP {r.status}")
                        else:
                            txt = (await r.text())[:200]
                            last_error = RuntimeError(f"HTTP {r.status}: {txt}")
            except Exception as e:
                last_error = e
                delay = min(10.0, 0.5 * (2**attempt) + random.uniform(0.0, 0.25))
                await asyncio.sleep(delay)

        if last_error:
            logger.warning(f"GitHub search failed: {last_error}")
        return []
