from __future__ import annotations

from codeconductor.ensemble.breakers import BreakerManager


def make_mgr() -> BreakerManager:
    return BreakerManager()


def test_closed_to_open_on_consec_fails(monkeypatch):
    mgr = make_mgr()
    mgr.consec_fails = 3
    mid = "m1"
    assert mgr.should_allow(mid) is True
    for _ in range(3):
        mgr.update(
            mid, success=False, total_ms=1000, ttft_ms=None, error_class="timeout"
        )
    st = mgr.get_state(mid)
    assert st.state == "Open"
    assert st.reason in ("CONSEC_FAILS", "timeout")


def test_err_rate_trigger(monkeypatch):
    mgr = make_mgr()
    mgr.err_rate = 0.2
    mid = "m2"
    # 5 obs: 2 fails -> 40%
    mgr.update(mid, success=False, total_ms=10, ttft_ms=None, error_class="timeout")
    mgr.update(mid, success=True, total_ms=10, ttft_ms=None, error_class=None)
    mgr.update(mid, success=False, total_ms=10, ttft_ms=None, error_class="5xx")
    mgr.update(mid, success=True, total_ms=10, ttft_ms=None, error_class=None)
    mgr.update(mid, success=True, total_ms=10, ttft_ms=None, error_class=None)
    st = mgr.get_state(mid)
    assert st.state == "Open"


def test_ttft_hysteresis_two_windows(monkeypatch):
    mgr = make_mgr()
    mgr.ttft_p95_lat_ms = 100
    mgr.window_sec = 99999  # disable time eviction; use count window
    mgr.window_count = 10
    mid = "m3"
    # First window breach: should not open yet due to hysteresis
    for _ in range(10):
        mgr.update(mid, success=True, total_ms=50, ttft_ms=200, error_class=None)
    assert mgr.get_state(mid).state == "Closed"
    # Second breach
    for _ in range(10):
        mgr.update(mid, success=True, total_ms=50, ttft_ms=200, error_class=None)
    assert mgr.get_state(mid).state == "Open"


def test_halfopen_probes(monkeypatch):
    mgr = make_mgr()
    mgr.consec_fails = 1
    mgr.cooldown_sec = 0
    mgr.halfopen_probes = 2
    mid = "m4"
    # Open immediately by 1 fail
    mgr.update(mid, success=False, total_ms=10, ttft_ms=None, error_class="timeout")
    st = mgr.get_state(mid)
    assert st.state == "Open"
    # should_allow moves to HalfOpen
    assert mgr.should_allow(mid) is True
    assert mgr.get_state(mid).state == "HalfOpen"
    # Two successes close it
    mgr.update(mid, success=True, total_ms=10, ttft_ms=None, error_class=None)
    assert mgr.get_state(mid).state == "HalfOpen"
    mgr.update(mid, success=True, total_ms=10, ttft_ms=None, error_class=None)
    assert mgr.get_state(mid).state == "Closed"
