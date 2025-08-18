#!/usr/bin/env python3
"""
FastAPI server exposing health endpoint and an OpenAI-compatible
/v1/chat/completions backed by CodeConductor's SingleModelEngine.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from codeconductor.ensemble.single_model_engine import (
    SingleModelEngine,
    SingleModelRequest,
)

# Optional Prometheus metrics (opt-in)
_ENABLE_METRICS = os.getenv("ENABLE_METRICS", "").strip().lower() in {
    "1",
    "true",
    "yes",
}
try:
    if _ENABLE_METRICS:
        from prometheus_fastapi_instrumentator import Instrumentator
        from prometheus_fastapi_instrumentator.metrics import (
            default,
        )
    else:  # pragma: no cover
        Instrumentator = None  # type: ignore
except Exception:  # pragma: no cover
    Instrumentator = None  # type: ignore

app = FastAPI(title="CodeConductor API", version="0.1.0")

# Mount /metrics if enabled and dependency available
if _ENABLE_METRICS and Instrumentator is not None:  # pragma: no cover
    try:
        labels = {
            "app": "codeconductor",
            "version": os.getenv("APP_VERSION", "0.1.0"),
            "commit": os.getenv("GIT_COMMIT", "dev"),
        }
        instr = Instrumentator(
            should_group_status_codes=True,
            should_ignore_untemplated=True,
            should_respect_env_var=False,
            excluded_handlers=set(),
            inprogress_granularity=True,
            grouped_status_codes=True,
        )
        # Use defaults (requests, latency, etc.) and attach constant labels
        instr.add(default(info=labels))
        instr.instrument(app).expose(app, include_in_schema=False)
    except Exception:
        pass


# ---- Models ----
class Message(BaseModel):
    role: str = Field(..., description="user|assistant|system")
    content: str


class CompletionIn(BaseModel):
    model: str | None = Field(default=None, description="Preferred model id")
    messages: list[Message]
    max_tokens: int | None = 256
    temperature: float | None = 0.2


# ---- Engine init (lazy singleton) ----
_engine_lock = asyncio.Lock()
_engine: SingleModelEngine | None = None


async def get_engine(preferred_model: str | None) -> SingleModelEngine:
    global _engine
    if _engine is None:
        async with _engine_lock:
            if _engine is None:
                _engine = SingleModelEngine(
                    preferred_model=preferred_model or "meta-llama-3.1-8b-instruct"
                )
                await _engine.initialize()
    return _engine


# ---- Routes ----
@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "timestamp": int(time.time())}


@app.post("/v1/chat/completions")
async def completions(req: CompletionIn) -> dict:
    engine = await get_engine(req.model)

    # Simple prompt construction: join all messages; prioritize latest user message
    # for code generation tasks.
    if req.messages:
        user_contents = [m.content for m in req.messages if m.role.lower() == "user"]
        prompt = user_contents[-1] if user_contents else req.messages[-1].content
    else:
        prompt = ""

    response = await engine.process_request(
        SingleModelRequest(
            task_description=prompt,
            context={
                "temperature": req.temperature,
                "max_tokens": req.max_tokens,
            },
        )
    )

    return {
        "id": "cc-" + uuid.uuid4().hex,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or response.model_used,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response.content},
                "finish_reason": "stop",
            }
        ],
    }


@app.get("/stream")
async def stream(request_id: str | None = None, prompt: str | None = None):
    """SSE endpoint streaming tokens and basic metrics.

    Query params:
    - request_id: optional client id
    - prompt: optional prompt for demo (fallback if no body is provided)
    """

    async def event_gen():
        # Minimal demo stream: simulate tokenization of content from engine
        start = time.perf_counter()
        seq = 0
        backend = "single"
        model = "local"
        last_heartbeat = start

        text = prompt or "Hello from CodeConductor streaming API."
        # Tokenize naively by words to simulate streaming
        tokens = text.split()
        for tok in tokens:
            await asyncio.sleep(0.02)  # 20ms per token demo
            seq += 1
            ttft_ms = int((time.perf_counter() - start) * 1000) if seq == 1 else None
            payload = {
                "token": tok + " ",
                "seq": seq,
                "ttft_ms": ttft_ms,
                "tps": None,
                "done": False,
                "backend": backend,
                "model": model,
            }
            # Flush event (JSON)
            yield f"data: {json.dumps(payload)}\n\n"

            # Heartbeat every ~15s to keep proxies alive
            now = time.perf_counter()
            if now - last_heartbeat > 15.0:
                yield ":keep-alive\n\n"
                last_heartbeat = now

        # done event
        payload_done = {
            "token": "",
            "seq": seq + 1,
            "ttft_ms": None,
            "tps": None,
            "done": True,
            "backend": backend,
            "model": model,
        }
        yield f"data: {json.dumps(payload_done)}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
