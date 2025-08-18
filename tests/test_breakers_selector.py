from __future__ import annotations

from codeconductor.ensemble.breakers import get_manager
from codeconductor.ensemble.model_manager import ModelInfo
from codeconductor.ensemble.model_selector import ModelSelector, SelectionInput


def test_selector_skips_open_models(monkeypatch):
    # Prepare 3 models
    models = [
        ModelInfo(
            id="m1",
            name="m1",
            provider="mock",
            endpoint="mock://",
            is_available=True,
            metadata={},
        ),
        ModelInfo(
            id="m2",
            name="m2",
            provider="mock",
            endpoint="mock://",
            is_available=True,
            metadata={},
        ),
        ModelInfo(
            id="m3",
            name="m3",
            provider="mock",
            endpoint="mock://",
            is_available=True,
            metadata={},
        ),
    ]
    mgr = get_manager()
    # Force m2 open
    mgr.update("m2", success=False, total_ms=10, ttft_ms=None, error_class="timeout")
    mgr.consec_fails = 1
    mgr.update("m2", success=False, total_ms=10, ttft_ms=None, error_class="timeout")

    sel = ModelSelector()
    inp = SelectionInput(models=models, prompt_len=10, policy="latency")
    out = sel.select(inp)
    # Ensure selected is not m2 since breaker should skip it
    assert out.selected_model in ("m1", "m3")
