#!/usr/bin/env python3
"""
Warm-up smoke: run 5 prompts against the SSE /stream endpoint, record TTFTs,
compute median, and write artifacts/latency/warmup_<ts>.json.

Usage:
  python scripts/warmup_smoke.py --host 127.0.0.1 --port 8000 --timeout 5
  python scripts/warmup_smoke.py --start-server

Exits 0 on success, 1 on failure.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from statistics import median

import requests


def start_server_in_thread(host: str, port: int) -> threading.Thread:
    import uvicorn  # type: ignore

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root / "src") not in sys.path:
        sys.path.insert(0, str(repo_root / "src"))
    from codeconductor.api.server import app  # type: ignore

    config = uvicorn.Config(
        app, host=host, port=port, workers=1, log_level="warning", loop="asyncio"
    )
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    time.sleep(0.8)
    return t


def run_once(host: str, port: int, prompt: str, timeout: float) -> int | None:
    url = f"http://{host}:{port}/stream"
    params = {"prompt": prompt}
    ttft_ms: int | None = None
    try:
        with requests.get(url, params=params, stream=True, timeout=(2, timeout)) as r:
            r.raise_for_status()
            for raw in r.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                line = raw.strip()
                if not line or not line.startswith("data: "):
                    continue
                try:
                    evt = json.loads(line[len("data: ") :])
                except json.JSONDecodeError:
                    continue
                if isinstance(evt, dict) and evt.get("ttft_ms") is not None:
                    if ttft_ms is None:
                        ttft_ms = int(evt.get("ttft_ms") or 0)
                if isinstance(evt, dict) and evt.get("done"):
                    break
    except Exception:
        return None
    return ttft_ms


def main() -> int:
    p = argparse.ArgumentParser(description="Warm-up latency smoke")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--timeout", type=float, default=5.0)
    p.add_argument("--start-server", action="store_true")
    p.add_argument("--prompts", type=int, default=5, help="number of prompts")
    args = p.parse_args()

    if args.start_server:
        try:
            start_server_in_thread(args.host, args.port)
        except Exception as e:
            print(f"Failed to start server: {e}")
            return 1

    prompts: list[str] = [
        "warmup: quick fox",
        "warmup: hello world",
        "warmup: fibonacci 10",
        "warmup: rest api ping",
        "warmup: sql select 1",
    ]
    if args.prompts > 0 and args.prompts != len(prompts):
        prompts = (prompts * ((args.prompts + len(prompts) - 1) // len(prompts)))[: args.prompts]

    ttfts: list[int] = []
    for pr in prompts:
        ms = run_once(args.host, args.port, pr, args.timeout)
        if ms is not None:
            ttfts.append(ms)

    if not ttfts:
        print("No TTFT measurements recorded")
        return 1

    med = int(median(ttfts))
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    out_dir = Path("artifacts/latency")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"warmup_{ts}.json"
    summary = {
        "timestamp": ts,
        "host": args.host,
        "port": args.port,
        "count": len(ttfts),
        "ttft_ms": ttfts,
        "median_ttft_ms": med,
    }
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"median_ttft_ms": med, "path": str(out_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
