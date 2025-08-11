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
    # Hard-disable heavy subsystems and cap parallelism for safety
    os.environ.setdefault("RLHF_DISABLE", "1")
    os.environ.setdefault("DISCOVERY_DISABLE", "1")
    os.environ.setdefault("MODEL_SELECTOR_STRICT", "1")
    os.environ.setdefault("MAX_PARALLEL_MODELS", "1")
    # Prefer a small single model if caller didn't force one
    os.environ.setdefault(
        "FORCE_MODEL", os.getenv("FORCE_MODEL", "meta-llama-3.1-8b-instruct")
    )

    eng = EnsembleEngine(use_rlhf=False)
    await eng.initialize()
    try:
        req = EnsembleRequest(task_description=prompt, timeout=timeout)
        res = await eng._process_request_internal(req)
        # If dispatcher flagged empty content, persist a diagnostic consensus
        try:
            run_dir = getattr(eng, "last_artifacts_dir", None)
            if run_dir and isinstance(res, object):
                # Best-effort: look for a structured empty content marker in last consensus/candidates
                # Here we just write a diagnostic file if consensus is empty
                import json as _json

                cpath = Path(run_dir) / "consensus.json"
                if cpath.exists():
                    data = _json.loads(cpath.read_text(encoding="utf-8"))
                    if not data or not data.get("consensus"):
                        diag = {
                            "status": "empty_content",
                            "model": os.getenv("FORCE_MODEL") or "unknown",
                            "prompt_preview": (prompt or "")[:50],
                        }
                        cpath.write_text(
                            _json.dumps(diag, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                    else:
                        # Materialize consensus code into after/generated.py
                        try:
                            cons = data.get("consensus") or ""
                            if isinstance(cons, str) and cons.strip():
                                # Extract fenced python code if present and normalize
                                from codeconductor.utils.extract import (
                                    extract_code,
                                    normalize_python,
                                )

                                extracted = extract_code(cons, lang_hint="python")
                                cleaned = normalize_python(extracted)
                                # Heuristic trim: keep up to last code-like line
                                lines = cleaned.splitlines()

                                def _is_code_like(s: str) -> bool:
                                    st = s.strip()
                                    if not st:
                                        return False
                                    if st.startswith("#"):
                                        return True
                                    if any(
                                        st.startswith(k)
                                        for k in (
                                            "def ",
                                            "class ",
                                            "if ",
                                            "for ",
                                            "while ",
                                            "import ",
                                            "from ",
                                            "print",
                                        )
                                    ):
                                        return True
                                    # Symbols indicative of code (exclude comma/period to avoid plain text)
                                    return any(c in st for c in "()=:_'\"[]{}")

                                last = -1
                                for i, ln in enumerate(lines):
                                    if _is_code_like(ln):
                                        last = i
                                final_code = (
                                    "\n".join(lines[: last + 1]).rstrip()
                                    if last >= 0
                                    else cleaned
                                )
                                after_dir = Path(run_dir) / "after"
                                after_dir.mkdir(parents=True, exist_ok=True)
                                # Size cap (200 KB) to avoid dumping giant chat text
                                try:
                                    cap_bytes = int(
                                        os.getenv("MATERIALIZE_MAX_BYTES", "204800")
                                    )
                                except Exception:
                                    cap_bytes = 204800
                                encoded = final_code.encode("utf-8")
                                if len(encoded) > cap_bytes:
                                    final_code = (
                                        encoded[:cap_bytes]
                                        .decode("utf-8", errors="ignore")
                                        .rstrip()
                                    )

                                # Auto-enable doctest when present
                                enable_doctest = (
                                    os.getenv("MATERIALIZE_ENABLE_DOCTEST", "1").strip()
                                    == "1"
                                )
                                if (
                                    enable_doctest
                                    and (
                                        "\n>>>" in final_code
                                        or final_code.strip().startswith(">>>")
                                    )
                                    and "doctest.testmod()" not in final_code
                                ):
                                    final_code = (
                                        final_code.rstrip()
                                        + '\n\nif __name__ == "__main__":\n    import doctest\n    doctest.testmod()\n'
                                    )

                                # Sanitize duplicate module-level docstrings: keep one, prefer containing doctest
                                try:
                                    import re as _re

                                    lines_hdr = final_code.splitlines()
                                    first_code_idx = next(
                                        (
                                            i
                                            for i, ln in enumerate(lines_hdr)
                                            if ln.strip().startswith(("def ", "class "))
                                        ),
                                        None,
                                    )
                                    if (
                                        first_code_idx is not None
                                        and first_code_idx > 0
                                    ):
                                        header = "\n".join(lines_hdr[:first_code_idx])
                                        body = "\n".join(lines_hdr[first_code_idx:])
                                        blocks = _re.findall(
                                            r"([\"\']{3}[\s\S]*?[\"\']{3})", header
                                        )
                                        if len(blocks) > 1:
                                            chosen = None
                                            for b in blocks[::-1]:
                                                if ">>>" in b:
                                                    chosen = b
                                                    break
                                            if chosen is None:
                                                chosen = blocks[-1]
                                            final_code = chosen.strip() + "\n\n" + body
                                except Exception:
                                    pass

                                gen_path = after_dir / "generated.py"
                                gen_path.write_text(final_code + "\n", encoding="utf-8")

                                # Validate Python syntax with ast and py_compile; attempt auto-close for triple quotes
                                import ast, py_compile

                                def _ast_ok(txt: str) -> bool:
                                    try:
                                        ast.parse(txt)
                                        return True
                                    except Exception:
                                        return False

                                def _compile_ok(p: Path) -> tuple[bool, str]:
                                    try:
                                        py_compile.compile(str(p), doraise=True)
                                        return True, ""
                                    except Exception as e:
                                        return False, str(e)

                                strict = (
                                    os.getenv("MATERIALIZE_STRICT", "1").strip() == "1"
                                )
                                ok_ast = _ast_ok(final_code)
                                ok_comp, err_comp = _compile_ok(gen_path)
                                if strict and (not ok_ast or not ok_comp):
                                    try:
                                        txt = gen_path.read_text(encoding="utf-8")
                                        # Trim trailing non-code lines again (best-effort)
                                        tlines = txt.splitlines()
                                        last_idx = -1
                                        for i, ln in enumerate(tlines):
                                            if _is_code_like(ln):
                                                last_idx = i
                                        if last_idx >= 0:
                                            txt = (
                                                "\n".join(
                                                    tlines[: last_idx + 1]
                                                ).rstrip()
                                                + "\n"
                                            )
                                        # Close unterminated triple quotes
                                        for delim in ('"""', "'''"):
                                            if txt.count(delim) % 2 != 0:
                                                txt = txt.rstrip() + f"\n{delim}\n"
                                        gen_path.write_text(txt, encoding="utf-8")
                                    except Exception:
                                        pass
                                    # Re-validate
                                    ok_ast2 = _ast_ok(
                                        gen_path.read_text(encoding="utf-8")
                                    )
                                    ok_comp2, err_comp2 = _compile_ok(gen_path)
                                    if not (ok_ast2 and ok_comp2):
                                        # Write raw for inspection and mark syntax invalid in consensus
                                        try:
                                            (after_dir / "logs").mkdir(
                                                parents=True, exist_ok=True
                                            )
                                            (after_dir / "logs" / "raw.txt").write_text(
                                                cons, encoding="utf-8"
                                            )
                                        except Exception:
                                            pass
                                        try:
                                            cdata = _json.loads(
                                                Path(
                                                    run_dir, "consensus.json"
                                                ).read_text(encoding="utf-8")
                                            )
                                            if isinstance(cdata, dict):
                                                cdata["syntax_valid"] = False
                                                Path(
                                                    run_dir, "consensus.json"
                                                ).write_text(
                                                    _json.dumps(
                                                        cdata,
                                                        ensure_ascii=False,
                                                        indent=2,
                                                    ),
                                                    encoding="utf-8",
                                                )
                                        except Exception:
                                            pass
                        except Exception:
                            pass
        except Exception:
            pass
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
