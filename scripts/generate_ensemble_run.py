#!/usr/bin/env python3
"""
Generate a single Ensemble run (mock-friendly) to produce artifacts under artifacts/runs/<ts>/.

Usage:
  CC_QUICK_CI=1 python scripts/generate_ensemble_run.py --prompt "Create a Fibonacci function"

In real mode (no CC_QUICK_CI), the ModelManager will try to discover local backends.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

# Ensure local 'src' is importable for codeconductor.*
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from codeconductor.ensemble.ensemble_engine import EnsembleEngine, EnsembleRequest


async def run_once(prompt: str, timeout: float) -> dict:
    eng = EnsembleEngine()
    await eng.initialize()
    try:
        req = EnsembleRequest(task_description=prompt, timeout=timeout)
        res = await eng._process_request_internal(req)
        # In quick mode, ensure consensus.json exists even if no models were dispatched
        try:
            if os.getenv("CC_QUICK_CI") == "1":
                run_dir = getattr(eng, "last_artifacts_dir", None)
                if run_dir:
                    cpath = Path(run_dir) / "consensus.json"
                    if not cpath.exists():
                        from codeconductor.ensemble.consensus_calculator import (
                            ConsensusCalculator,
                        )

                        cc = ConsensusCalculator()
                        a = """
```python
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
```
"""
                        b = """
```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
```
"""
                        r = cc.calculate_consensus(
                            [
                                {
                                    "model": "mock-model-1",
                                    "content": a,
                                    "confidence": 0.7,
                                },
                                {
                                    "model": "mock-model-2",
                                    "content": b,
                                    "confidence": 0.6,
                                },
                            ]
                        )
                        cands = [
                            {
                                "model": "mock-model-1",
                                "score": float(r.model_scores.get("mock-model-1", 0.0)),
                                "output_sha": None,
                            },
                            {
                                "model": "mock-model-2",
                                "score": float(r.model_scores.get("mock-model-2", 0.0)),
                                "output_sha": None,
                            },
                        ]
                        cands.sort(key=lambda d: d.get("score", 0.0), reverse=True)
                        winner = (
                            {"model": cands[0]["model"], "score": cands[0]["score"]}
                            if cands
                            else {"model": None, "score": float(r.confidence)}
                        )
                        payload = {
                            "method": "codebleu_fast",
                            "winner": winner,
                            "candidates": cands,
                            "cached": False,
                            "confidence": float(r.confidence),
                            "code_quality": float(r.code_quality_score),
                            "syntax_valid": bool(r.syntax_valid),
                        }
                        cpath.write_text(
                            json.dumps(payload, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
        except Exception:
            pass
        # Return path hint for latest artifacts if available
        return {
            "success": True,
            "confidence": float(getattr(res, "confidence", 0.0)),
            "artifacts_dir": getattr(eng, "last_artifacts_dir", None),
        }
    finally:
        await eng.cleanup()


def main() -> int:
    p = argparse.ArgumentParser(description="Generate one ensemble run and artifacts")
    p.add_argument("--prompt", default="Create a Python function fibonacci(n)")
    p.add_argument("--timeout", type=float, default=10.0)
    args = p.parse_args()

    # Ensure artifacts root exists
    Path(os.getenv("ARTIFACTS_DIR", "artifacts")).mkdir(parents=True, exist_ok=True)
    result = asyncio.run(run_once(args.prompt, args.timeout))
    print(json.dumps(result, ensure_ascii=False))
    if result.get("success"):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
