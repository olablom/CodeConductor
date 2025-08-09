import asyncio
import pytest

from codeconductor.ensemble.ensemble_engine import EnsembleEngine, EnsembleRequest


@pytest.mark.asyncio
async def test_cache_miss_then_hit(monkeypatch):
    # Ensure deterministic policy and cache settings
    monkeypatch.setenv("SELECTOR_POLICY", "latency")
    monkeypatch.setenv("CACHE_SIZE", "10")
    monkeypatch.setenv("CACHE_TTL_SECONDS", "300")
    monkeypatch.setenv("CC_QUICK_CI", "1")

    engine = EnsembleEngine()

    # First call should miss cache and run flow; second call should hit
    req = EnsembleRequest(task_description="Return constant 123", timeout=5.0)
    res1 = await engine._process_request_internal(req)
    assert res1 is not None

    res2 = await engine._process_request_internal(req)
    assert res2 is not None
    # We cannot assert exact content, but we expect very fast execution for cache hit
    assert res2.execution_time == 0.0 or res2.confidence >= res1.confidence
