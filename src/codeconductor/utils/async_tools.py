#!/usr/bin/env python3
# filename: src/codeconductor/utils/async_tools.py
from __future__ import annotations
import asyncio, inspect
from typing import Any, Awaitable, Callable

def is_awaitable(x: Any) -> bool:
    return inspect.isawaitable(x)

def ensure_async(fn: Callable[..., Any]) -> Callable[..., Awaitable[Any]]:
    if inspect.iscoroutinefunction(fn):
        return fn
    async def _wrap(*args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))
    return _wrap

def run_sync(coro):
    """Helper to run coroutines in sync context (CLI/scripts only, not pytest)"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    else:
        # Vi är redan i en loop; skapa en task och vänta blockerat
        # ENDAST för CLI/scripts, INTE för pytest
        return loop.run_until_complete(coro)
