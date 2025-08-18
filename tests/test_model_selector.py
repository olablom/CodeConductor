import sys
from pathlib import Path

# Ensure local package path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from codeconductor.ensemble.model_manager import ModelInfo
from codeconductor.ensemble.model_selector import ModelSelector, SelectionInput


def make_model_simple(id: str, **md):
    return ModelInfo(
        id=id,
        name=id,
        provider="mock",
        endpoint="mock://",
        is_available=True,
        metadata=md,
    )  # type: ignore[arg-type]


def test_selector_latency_policy_prefers_lower_p95(monkeypatch):
    sel = ModelSelector()
    a = make_model_simple("A", latency_p95=0.4)
    b = make_model_simple("B", latency_p95=0.2)
    out = sel.select(SelectionInput(models=[a, b], prompt_len=100, policy="latency"))
    assert out.selected_model == "B"
    assert out.why["policy"] == "latency"


def test_selector_context_policy_fits_prompt_len():
    sel = ModelSelector()
    a = make_model_simple("A", context_window=2048)
    b = make_model_simple("B", context_window=8192)
    out = sel.select(SelectionInput(models=[a, b], prompt_len=5000, policy="context"))
    assert out.selected_model == "B"


def test_selector_quality_policy_prefers_higher_codebleu():
    sel = ModelSelector()
    a = make_model_simple("A", codebleu_avg=0.7, fail_rate=0.05, retry_rate=0.05)
    b = make_model_simple("B", codebleu_avg=0.8, fail_rate=0.04, retry_rate=0.04)
    out = sel.select(SelectionInput(models=[a, b], prompt_len=100, policy="quality"))
    assert out.selected_model == "B"


def test_selector_forced_env(monkeypatch):
    sel = ModelSelector()
    a = make_model_simple("deepseek-v3", latency_p95=1.0)
    b = make_model_simple("mixtral-8x7b", latency_p95=0.2)
    monkeypatch.setenv("FORCE_MODEL", "deepseek-v3")
    try:
        out = sel.select(SelectionInput(models=[a, b], prompt_len=100, policy="latency"))
    finally:
        monkeypatch.delenv("FORCE_MODEL", raising=False)
    assert out.selected_model == "deepseek-v3"
    assert out.why.get("reason") == "forced_by_env"


def make_model(id: str, provider: str, meta: dict) -> ModelInfo:
    return ModelInfo(
        id=id,
        name=id,
        provider=provider,
        endpoint="mock://",
        is_available=True,
        metadata=meta,
    )


def test_latency_policy_picks_lowest_latency():
    models = [
        make_model("m_fast", "mock", {"latency_p95": 0.3}),
        make_model("m_slow", "mock", {"latency_p95": 2.0}),
    ]
    sel = ModelSelector()
    out = sel.select(SelectionInput(models=models, prompt_len=100, policy="latency"))
    assert out.selected_model == "m_fast"
    assert out.fallbacks == ["m_slow"]
    assert 0.0 <= out.scores["m_fast"] <= 1.0


def test_context_policy_prefers_larger_ctx_when_needed():
    models = [
        make_model("m_small", "mock", {"context_window": 2048}),
        make_model("m_big", "mock", {"context_window": 32768}),
    ]
    sel = ModelSelector()
    out = sel.select(SelectionInput(models=models, prompt_len=8000, policy="context"))
    assert out.selected_model == "m_big"


def test_quality_policy_prefers_higher_codebleu_lower_fail_retry():
    models = [
        make_model(
            "m_q1",
            "mock",
            {"codebleu_avg": 0.75, "fail_rate": 0.10, "retry_rate": 0.05},
        ),
        make_model(
            "m_q2",
            "mock",
            {"codebleu_avg": 0.85, "fail_rate": 0.02, "retry_rate": 0.02},
        ),
    ]
    sel = ModelSelector()
    out = sel.select(SelectionInput(models=models, prompt_len=100, policy="quality"))
    assert out.selected_model == "m_q2"
