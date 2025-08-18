#!/usr/bin/env python3
"""
KPI utilities: build and write KPI JSON artifacts for a run.

Schema reference:
- src/codeconductor/utils/schemas/kpi.schema.json

This module is intentionally dependency-light and best-effort. It should never
break the main flow; on any failure it logs and returns gracefully.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _parse_codebleu_weights(env_val: str | None) -> tuple[float, float, float]:
    if not env_val:
        return (0.5, 0.3, 0.2)
    try:
        parts = [float(x) for x in env_val.split(",")]
        if len(parts) != 3:
            return (0.5, 0.3, 0.2)
        s = sum(parts)
        if s <= 0:
            return (0.5, 0.3, 0.2)
        return (parts[0] / s, parts[1] / s, parts[2] / s)
    except Exception:
        return (0.5, 0.3, 0.2)


def calc_pass_rate(passed: int, failed: int) -> float | None:
    denom = passed + failed
    if denom <= 0:
        return None
    return passed / denom


def build_config_digest(keys: list[str]) -> str:
    rows = []
    for k in sorted(keys):
        v = os.getenv(k)
        rows.append(f"{k}={'' if v is None else v}")
    h = hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()
    return f"sha256:{h}"


@dataclass
class TestSummary:
    suite_name: str
    total: int
    passed: int
    failed: int
    skipped: int


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_kpi(
    *,
    run_id: str,
    artifacts_dir: Path,
    t_start_iso: str,
    t_first_green_iso: str | None,
    ttft_ms: int,
    tests_before: TestSummary,
    tests_after: TestSummary,
    winner_model: str,
    winner_score: float,
    consensus_method: str,
    sampling: dict[str, Any],
    codebleu_weights_env: str | None,
    codebleu_lang_env: str | None,
    exit_status: dict[str, Any],
    agent_config: list[dict] | None = None,
    models: list[dict] | None = None,
    engine: dict | None = None,
    hardware: dict | None = None,
    latency_ms: dict | None = None,
    token_usage: dict | None = None,
    safety: dict | None = None,
    dataset_tags: list[str] | None = None,
) -> dict[str, Any]:
    # Normalize sampling to schema keys
    sampling_out: dict[str, Any] = {
        "temperature": sampling.get("temperature"),
        "top_p": sampling.get("top_p"),
        "top_k": sampling.get("top_k", None),
        "max_tokens": sampling.get("max_tokens", None),
        "presence_penalty": sampling.get("presence_penalty", None),
        "frequency_penalty": sampling.get("frequency_penalty", None),
    }

    codebleu_weights = list(_parse_codebleu_weights(codebleu_weights_env))
    codebleu_lang = (codebleu_lang_env or "").strip().lower() or "python"

    pr_before = calc_pass_rate(tests_before.passed, tests_before.failed)
    pr_after = calc_pass_rate(tests_after.passed, tests_after.failed)

    kpi: dict[str, Any] = {
        "kpi_schema_version": "1.0.0",
        "run_id": run_id,
        "ts": _utc_now_iso(),
        "artifacts_dir": str(artifacts_dir),
        "ttft_ms": int(max(0, ttft_ms)),
        "t_start": t_start_iso,
        "t_first_green": t_first_green_iso or t_start_iso,
        "first_prompt_success": bool((tests_after.failed == 0) and (tests_after.total > 0)),
        "tests_before": {
            "suite_name": tests_before.suite_name,
            "total": tests_before.total,
            "passed": tests_before.passed,
            "failed": tests_before.failed,
            "skipped": tests_before.skipped,
        },
        "tests_after": {
            "suite_name": tests_after.suite_name,
            "total": tests_after.total,
            "passed": tests_after.passed,
            "failed": tests_after.failed,
            "skipped": tests_after.skipped,
        },
        "pass_rate_before": pr_before,
        "pass_rate_after": pr_after,
        "winner_model": winner_model,
        "winner_score": float(winner_score),
        "consensus_method": consensus_method,
        "codebleu_weights": codebleu_weights,
        "codebleu_lang": codebleu_lang,
        "sampling": sampling_out,
        "config_digest": build_config_digest(
            [
                "ENGINE_BACKENDS",
                "ALLOW_NET",
                "CODEBLEU_WEIGHTS",
                "CODEBLEU_LANG",
                "SELECTOR_POLICY",
                "EXPORT_PRIVACY_LEVEL",
                "EXPORT_SIZE_LIMIT_MB",
                "EXPORT_INCLUDE_RAW",
            ]
        ),
        "exit_status": exit_status,
    }

    if agent_config:
        kpi["agent_config"] = agent_config
    if models:
        kpi["models"] = models
    if engine:
        kpi["engine"] = engine
    if hardware:
        kpi["hardware"] = hardware
    if latency_ms:
        kpi["latency_ms"] = latency_ms
    if token_usage is not None:
        kpi["token_usage"] = token_usage
    if safety:
        kpi["safety"] = safety
    if dataset_tags:
        kpi["dataset_tags"] = dataset_tags

    return kpi
