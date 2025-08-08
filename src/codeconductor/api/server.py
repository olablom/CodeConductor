#!/usr/bin/env python3
"""
FastAPI server exposing health endpoint and an OpenAI-compatible
/v1/chat/completions backed by CodeConductor's SingleModelEngine.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import List, Optional
import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from codeconductor.ensemble.single_model_engine import (
    SingleModelEngine,
    SingleModelRequest,
)

app = FastAPI(title="CodeConductor API", version="0.1.0")


# ---- Models ----
class Message(BaseModel):
    role: str = Field(..., description="user|assistant|system")
    content: str


class CompletionIn(BaseModel):
    model: Optional[str] = Field(default=None, description="Preferred model id")
    messages: List[Message]
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.2


# ---- Engine init (lazy singleton) ----
_engine_lock = asyncio.Lock()
_engine: Optional[SingleModelEngine] = None


async def get_engine(preferred_model: Optional[str]) -> SingleModelEngine:
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
async def stream(request_id: Optional[str] = None, prompt: Optional[str] = None):
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
