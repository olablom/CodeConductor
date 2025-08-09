#!/usr/bin/env python3
"""
Model Selector for CodeConductor

Deterministic policy-based selection among available models:
- latency: minimize expected latency (prefer p95, else average, else recent)
- context: maximize context fit (prompt_len/max_ctx) and capacity
- quality: combine historical codebleu avg, failure rate, retry rate

Always returns: primary model id, ordered fallbacks, and sampling params.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import os
import logging

from .model_manager import ModelInfo
from .breakers import get_manager as get_breaker_manager
from codeconductor.telemetry import get_logger


logger = logging.getLogger(__name__)


@dataclass
class SelectionInput:
    models: List[ModelInfo]
    prompt_len: int
    policy: str = "latency"  # latency | context | quality
    # Optional hints/overrides
    max_candidates: int = 3
    temperature: Optional[float] = None
    top_p: Optional[float] = None


@dataclass
class SelectionOutput:
    selected_model: Optional[str]
    fallbacks: List[str]
    policy: str
    scores: Dict[str, float]
    sampling: Dict[str, float]
    why: Dict[str, Any]


class ModelSelector:
    def __init__(self):
        self.default_temperature = float(os.getenv("CC_TEMP", "0.2"))
        self.default_top_p = float(os.getenv("CC_TOP_P", "0.95"))

    def select(self, sin: SelectionInput) -> SelectionOutput:
        policy = (sin.policy or "latency").lower()
        # Env override
        forced = os.getenv("FORCE_MODEL") or os.getenv("SELECTOR_FORCE")
        if forced:
            candidates = {m.id: m for m in sin.models}
            chosen = None
            if forced in candidates:
                chosen = forced
            else:
                for mid in candidates:
                    if mid.lower().endswith(forced.lower()):
                        chosen = mid
                        break
            if not chosen and sin.models:
                chosen = sin.models[0].id
            logger.info(f"ModelSelector forced by env: {forced} => {chosen}")
            sampling = {
                "temperature": sin.temperature
                if sin.temperature is not None
                else self.default_temperature,
                "top_p": sin.top_p if sin.top_p is not None else self.default_top_p,
            }
            return SelectionOutput(
                selected_model=chosen,
                fallbacks=[m.id for m in sin.models if m.id != chosen][
                    : max(0, sin.max_candidates - 1)
                ],
                policy=policy,
                scores={chosen: 1.0} if chosen else {},
                sampling=sampling,
                why={"reason": "forced_by_env", "forced_value": forced},
            )
        scoring_func = {
            "latency": self._score_latency,
            "context": self._score_context,
            "quality": self._score_quality,
        }.get(policy, self._score_latency)

        scores: Dict[str, float] = {}
        breaker = get_breaker_manager()
        tlog = get_logger()
        for m in sin.models:
            st = breaker.get_state(m.id)
            if st.state == "Open" and not breaker.shadow:
                # Skip open models
                tlog.log(
                    "selector_skip_open",
                    {"model": m.id, "reason": st.reason, "state": st.state},
                )
                continue
            scores[m.id] = scoring_func(m, sin)

        # Order by score desc
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        selected = ordered[0][0] if ordered else None
        fallbacks = [mid for mid, _ in ordered[1 : 1 + max(0, sin.max_candidates - 1)]]

        sampling = {
            "temperature": sin.temperature
            if sin.temperature is not None
            else self.default_temperature,
            "top_p": sin.top_p if sin.top_p is not None else self.default_top_p,
        }

        # Why chosen
        why = {
            "reason": "policy_scoring",
            "policy": policy,
            "top_candidates": ordered[:2],
        }
        # Telemetry log
        logger.info(
            f"ModelSelector policy={policy} selected={selected} fallbacks={fallbacks} scores={scores} sampling={sampling}"
        )

        return SelectionOutput(
            selected_model=selected,
            fallbacks=fallbacks,
            policy=policy,
            scores=scores,
            sampling=sampling,
            why=why,
        )

    # --- scoring policies ---
    def _score_latency(self, model: ModelInfo, sin: SelectionInput) -> float:
        md = model.metadata or {}
        # Lower latency better → invert into score in [0,1]
        lat = (
            float(md.get("latency_p95", 0))
            or float(md.get("latency_avg", 0))
            or float(md.get("latency_recent", 0))
        )
        if lat <= 0:
            return 0.5  # unknown latency → neutral default
        # Map: 0.2s → ~1.0, 5s → ~0.0 (clamped)
        score = max(0.0, min(1.0, 1.0 - (lat / 5.0)))
        return score

    def _score_context(self, model: ModelInfo, sin: SelectionInput) -> float:
        md = model.metadata or {}
        max_ctx = int(md.get("context_window", md.get("max_ctx", 4096)))
        if max_ctx <= 0:
            return 0.0
        ratio = sin.prompt_len / float(max_ctx)
        if ratio > 1.0:
            # overflow → penalize
            return max(0.0, 0.5 - min(1.0, ratio - 1.0))
        # Higher capacity, lower ratio → higher score
        base = 1.0 - ratio
        capacity_bonus = min(max_ctx / 131072.0, 0.3)  # +up to 0.3 for very large ctx
        return max(0.0, min(1.0, base + capacity_bonus))

    def _score_quality(self, model: ModelInfo, sin: SelectionInput) -> float:
        md = model.metadata or {}
        codebleu = float(md.get("codebleu_avg", 0.6))  # default mid
        fail_rate = float(md.get("fail_rate", 0.05))  # lower better
        retry_rate = float(md.get("retry_rate", 0.05))  # lower better
        # Weighted: CodeBLEU 60%, fail 25%, retry 15% (inverted)
        score = (
            0.6 * codebleu
            + 0.25 * (1.0 - min(1.0, fail_rate))
            + 0.15 * (1.0 - min(1.0, retry_rate))
        )
        return max(0.0, min(1.0, score))
