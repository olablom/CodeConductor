from __future__ import annotations

import os
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from codeconductor.telemetry import get_logger


@dataclass
class Observation:
    ts: float
    success: bool
    total_ms: float | None = None
    ttft_ms: float | None = None
    error_class: str | None = None  # timeout|5xx|reset|breaker_open|other


@dataclass
class ModelState:
    state: str = "Closed"  # Closed|Open|HalfOpen
    reason: str | None = None
    last_change_ts: float = field(default_factory=lambda: time.monotonic())
    next_probe_at: float | None = None
    halfopen_remaining_probes: int = 0


class BreakerManager:
    """
    Per-model circuit breaker with rolling window aggregation and hysteresis.

    - Window: time-based (WINDOW_SEC) and count-based (WINDOW_COUNT), whichever closes first
    - Thresholds: error rate, TTFT p95, total latency p99, consecutive fails
    - Hysteresis: require two consecutive windows to violate TTFT before Open
    - Half-open: allow N probes before closing or re-opening
    - Cooldown jitter: ±20%
    - Shadow mode: observe-only (no blocking)
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.obs: dict[str, deque[Observation]] = {}
        self.states: dict[str, ModelState] = {}
        self._last_p95_breach: dict[str, int] = {}  # count consecutive TTFT breaches
        self._obs_count: dict[str, int] = {}

        # Config (env-overridable)
        self.err_rate = float(os.getenv("CC_BREAKER_ERRRATE", "0.2"))
        self.ttft_p95_lat_ms = int(os.getenv("CC_BREAKER_TTFT_P95_MS_LAT", "600"))
        self.total_p99_ms = int(os.getenv("CC_BREAKER_LAT_P99_MS", "5000"))
        self.consec_fails = int(os.getenv("CC_BREAKER_CONSEC_FAILS", "3"))
        self.cooldown_sec = int(os.getenv("CC_BREAKER_COOLDOWN_SEC", "30"))
        self.window_sec = int(os.getenv("CC_BREAKER_WINDOW_SEC", "60"))
        self.window_count = int(os.getenv("CC_BREAKER_WINDOW_COUNT", "50"))
        self.halfopen_probes = int(os.getenv("CC_BREAKER_HALFOPEN_PROBES", "2"))
        self.shadow = os.getenv("CC_BREAKER_SHADOW", "0") == "1"

    # --- public API ---
    def should_allow(self, model_id: str) -> bool:
        with self.lock:
            st = self.states.get(model_id)
            now = time.monotonic()
            if not st:
                return True
            if st.state == "Open":
                if st.next_probe_at and now >= st.next_probe_at:
                    prev = st.state
                    st.state = "HalfOpen"
                    st.halfopen_remaining_probes = max(1, self.halfopen_probes)
                    st.last_change_ts = now
                    get_logger().log(
                        "breaker_event",
                        {
                            "model": model_id,
                            "from": prev,
                            "to": "HalfOpen",
                            "reason": st.reason,
                        },
                    )
                    return True
                return True if self.shadow else False
            if st.state == "HalfOpen":
                return True
            return True

    def update(
        self,
        model_id: str,
        *,
        success: bool,
        total_ms: float | None,
        ttft_ms: float | None,
        error_class: str | None,
    ) -> None:
        with self.lock:
            buf = self.obs.setdefault(model_id, deque(maxlen=self.window_count))
            now = time.monotonic()
            buf.append(
                Observation(
                    ts=now,
                    success=success,
                    total_ms=total_ms,
                    ttft_ms=ttft_ms,
                    error_class=error_class,
                )
            )
            self._evict_older(buf, now)
            self._obs_count[model_id] = self._obs_count.get(model_id, 0) + 1

            st = self.states.setdefault(model_id, ModelState())

            # HalfOpen handling: consume probe budget
            if st.state == "HalfOpen":
                if success:
                    st.halfopen_remaining_probes -= 1
                    if st.halfopen_remaining_probes <= 0:
                        prev = st.state
                        st.state = "Closed"
                        st.reason = None
                        st.last_change_ts = now
                        st.next_probe_at = None
                        get_logger().log(
                            "breaker_event",
                            {
                                "model": model_id,
                                "from": prev,
                                "to": "Closed",
                                "reason": "probe_success",
                            },
                        )
                        return
                else:
                    # immediate reopen
                    self._open(
                        model_id, st, reason=error_class or "HALFOPEN_FAIL", now=now
                    )
                    return

            # Closed evaluation
            if st.state == "Closed":
                reason = self._evaluate_open_reason(buf)
                if reason is not None:
                    # hysteresis for TTFT p95
                    if reason == "TTFT_P95":
                        # Only count breach at fixed observation windows (mod window_count == 0)
                        if self.window_count > 0 and (
                            self._obs_count.get(model_id, 0) % self.window_count == 0
                        ):
                            c = self._last_p95_breach.get(model_id, 0) + 1
                            self._last_p95_breach[model_id] = c
                        else:
                            # do not change breach counter within the same window
                            c = self._last_p95_breach.get(model_id, 0)
                        if c < 2:
                            return
                    else:
                        self._last_p95_breach[model_id] = 0
                    # Telemetry metrics snapshot
                    err_rate, p95_ttft, p99_total, consec = self._compute_metrics(buf)
                    self._open(
                        model_id,
                        st,
                        reason=reason,
                        now=now,
                        metrics={
                            "err_rate": err_rate,
                            "p95_ttft_ms": p95_ttft,
                            "p99_total_ms": p99_total,
                            "consec_fails": consec,
                        },
                    )
            elif st.state == "Open":
                # keep open until cooldown triggers HalfOpen in should_allow
                pass

    def get_state(self, model_id: str) -> ModelState:
        with self.lock:
            return self.states.get(model_id, ModelState())

    # --- internal helpers ---
    def _open(
        self,
        model_id: str,
        st: ModelState,
        *,
        reason: str,
        now: float,
        metrics: dict[str, float] | None = None,
    ) -> None:
        prev = st.state
        st.state = "Open"
        st.reason = reason
        st.last_change_ts = now
        jitter = 0.8 + random.random() * 0.4  # 0.8..1.2
        st.next_probe_at = now + int(self.cooldown_sec * jitter)
        st.halfopen_remaining_probes = 0
        payload = {
            "model": model_id,
            "from": prev,
            "to": "Open",
            "reason": reason,
            "cooldown_s": self.cooldown_sec,
        }
        if metrics:
            payload.update(metrics)
        get_logger().log("breaker_event", payload)

    def _evict_older(self, buf: deque[Observation], now: float) -> None:
        # Evict older than window_sec
        cutoff = now - self.window_sec
        while buf and buf[0].ts < cutoff:
            buf.popleft()

    def _evaluate_open_reason(self, buf: deque[Observation]) -> str | None:
        if not buf:
            return None
        n = len(buf)
        fails = sum(1 for o in buf if not o.success)
        err_rate = fails / n
        # Consecutive fails
        consec = 0
        for o in reversed(buf):
            if not o.success:
                consec += 1
            else:
                break
        if consec >= self.consec_fails:
            return "CONSEC_FAILS"
        # Error rate per class (worst wins)
        if err_rate > self.err_rate:
            # pick dominant error class
            classes: dict[str, int] = {}
            for o in buf:
                if not o.success and o.error_class:
                    classes[o.error_class] = classes.get(o.error_class, 0) + 1
            worst = max(classes, key=classes.get) if classes else "ERR_RATE"
            return worst
        # TTFT p95 (if present) and total p99
        ttfts = sorted([o.ttft_ms for o in buf if o.ttft_ms is not None])
        if ttfts:
            idx = max(0, int(0.95 * len(ttfts)) - 1)
            p95 = ttfts[idx]
            if p95 is not None and p95 > self.ttft_p95_lat_ms:
                return "TTFT_P95"
        totals = sorted([o.total_ms for o in buf if o.total_ms is not None])
        if totals:
            idx = max(0, int(0.99 * len(totals)) - 1)
            p99 = totals[idx]
            if p99 is not None and p99 > self.total_p99_ms:
                return "LAT_P99"
        return None

    def _compute_metrics(
        self, buf: deque[Observation]
    ) -> tuple[float, float | None, float | None, int]:
        n = len(buf)
        fails = sum(1 for o in buf if not o.success)
        err_rate = fails / n if n else 0.0
        consec = 0
        for o in reversed(buf):
            if not o.success:
                consec += 1
            else:
                break
        ttfts = sorted([o.ttft_ms for o in buf if o.ttft_ms is not None])
        p95 = None
        if ttfts:
            idx = max(0, int(0.95 * len(ttfts)) - 1)
            p95 = ttfts[idx]
        totals = sorted([o.total_ms for o in buf if o.total_ms is not None])
        p99 = None
        if totals:
            idx = max(0, int(0.99 * len(totals)) - 1)
            p99 = totals[idx]
        return err_rate, p95, p99, consec


_GLOBAL_MANAGER: BreakerManager | None = None


def get_manager() -> BreakerManager:
    global _GLOBAL_MANAGER
    if _GLOBAL_MANAGER is None:
        _GLOBAL_MANAGER = BreakerManager()
    return _GLOBAL_MANAGER
