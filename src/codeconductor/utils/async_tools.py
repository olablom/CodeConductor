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
