#!/usr/bin/env python3
"""
Streaming smoke test for SSE endpoint /stream.

Usage:
  python scripts/stream_smoke.py --host 127.0.0.1 --port 8000 --prompt "a b c" --timeout 5
  python scripts/stream_smoke.py --start-server --prompt "hello streaming world"

Exits 0 on success, 1 on failure.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import requests


def start_server_in_thread(host: str, port: int) -> threading.Thread:
    import uvicorn  # type: ignore

    # Ensure local 'src' is importable for codeconductor.*
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from codeconductor.api.server import app  # type: ignore

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        workers=1,
        log_level="info",
        loop="asyncio",
    )
    server = uvicorn.Server(config)

    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    # Wait a bit for server to boot
    time.sleep(0.8)
    return t


def run_smoke(host: str, port: int, prompt: str, timeout: float) -> int:
    url = f"http://{host}:{port}/stream"
    params = {"prompt": prompt}
    ttft_ms: int | None = None
    seq_last = 0
    tokens = []
    t_start = time.perf_counter()

    try:
        with requests.get(url, params=params, stream=True, timeout=(2, timeout)) as r:
            r.raise_for_status()
            done = False
            for raw in r.iter_lines(decode_unicode=True):
                if raw is None:
                    continue
                line = raw.strip()
                if not line:
                    continue
                if line.startswith(":"):
                    # heartbeat/comment
                    continue
                if not line.startswith("data: "):
                    continue
                payload = line[len("data: ") :]
                try:
                    evt = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if isinstance(evt, dict) and evt.get("done"):
                    done = True
                    break
                if not isinstance(evt, dict):
                    continue
                seq = int(evt.get("seq", 0))
                if seq <= seq_last:
                    print(f"Sequence not increasing: {seq} <= {seq_last}")
                    return 1
                seq_last = seq
                if ttft_ms is None and evt.get("ttft_ms") is not None:
                    ttft_ms = int(evt.get("ttft_ms") or 0)
                tok = evt.get("token")
                if isinstance(tok, str):
                    tokens.append(tok)

            if not done:
                print("Stream did not send done=true within timeout")
                return 1
    except Exception as e:
        print(f"Stream error: {e}")
        return 1

    t_elapsed = time.perf_counter() - t_start
    token_text = "".join(tokens)
    tps = len(tokens) / max(0.001, t_elapsed)

    # Write metrics file
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    out_dir = Path("artifacts/latency")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"stream_smoke_{ts}.json"
    summary = {
        "timestamp": ts,
        "host": host,
        "port": port,
        "prompt": prompt,
        "ttft_ms": ttft_ms,
        "tokens": len(tokens),
        "tps": round(tps, 2),
        "sample": token_text[:80],
    }
    out_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("SSE smoke: OK")
    print(f"  TTFT: {ttft_ms if ttft_ms is not None else 'n/a'} ms")
    print(f"  Tokens: {len(tokens)}  Tok/s: {tps:.1f}")
    print(f"  Output sample: {token_text[:80]!r}")
    print(f"  Metrics written: {out_path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="SSE streaming smoke test")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--prompt",
        action="append",
        default=["Hello from streaming test"],
        help="Prompt(s) to stream; can be repeated",
    )
    p.add_argument(
        "--timeout", type=float, default=5.0, help="overall read timeout (s)"
    )
    p.add_argument(
        "--start-server", action="store_true", help="start uvicorn server in background"
    )
    args = p.parse_args()

    if args.start_server:
        try:
            start_server_in_thread(args.host, args.port)
        except Exception as e:
            print(f"Failed to start server: {e}")
            return 1

    overall = 0
    for pr in args.prompt:
        code = run_smoke(args.host, args.port, pr, args.timeout)
        if code != 0:
            overall = code
    # Server thread is daemon=True; it will exit with the process
    return overall


if __name__ == "__main__":
    sys.exit(main())
