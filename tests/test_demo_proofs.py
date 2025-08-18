import sys
from pathlib import Path

import pytest

# Ensure local src is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def test_sampling_override_applied_env(monkeypatch):
    from codeconductor.ensemble.query_dispatcher import (
        BASE_MODEL_CONFIGS,
        _apply_sampling_overrides,
    )

    # Set env overrides
    monkeypatch.setenv("CC_TEMP", "0.05")
    monkeypatch.setenv("CC_TOP_P", "0.85")
    monkeypatch.setenv("MAX_TOKENS", "160")

    base_cfg = BASE_MODEL_CONFIGS["llama"].copy()
    new_cfg = _apply_sampling_overrides(base_cfg)

    assert new_cfg["temperature"] == pytest.approx(0.05)
    assert new_cfg["top_p"] == pytest.approx(0.85)
    assert new_cfg["max_tokens"] == 160


def test_model_manager_strict_forced_single(monkeypatch):
    from codeconductor.ensemble.model_manager import ModelInfo, ModelManager

    # Strict locking env
    monkeypatch.setenv("MODEL_SELECTOR_STRICT", "1")
    monkeypatch.setenv("FORCE_MODEL", "only-this-model")
    monkeypatch.setenv("ENGINE_BACKENDS", "lmstudio")
    monkeypatch.setenv("LMSTUDIO_DISABLE", "0")
    monkeypatch.setenv("LMSTUDIO_CLI_DISABLE", "1")  # prevent CLI loads
    monkeypatch.setenv("DISCOVERY_DISABLE", "1")

    mm = ModelManager()

    # Track if discovery would be called (it should NOT in strict path)
    called = {"list_models": 0}

    async def _fake_list_models():  # pragma: no cover - executed if bug
        called["list_models"] += 1
        return [
            ModelInfo(
                id="only-this-model",
                name="only-this-model",
                provider="lm_studio",
                endpoint="http://localhost:1234/v1",
                is_available=True,
                metadata={},
            )
        ]

    async def _fake_load_via_cli(_model_key: str, ttl_seconds: int = 7200) -> bool:
        # CLI is disabled; ensure we don't try to actually load
        return False

    monkeypatch.setattr(mm, "list_models", _fake_list_models, raising=True)
    monkeypatch.setattr(mm, "load_model_via_cli", _fake_load_via_cli, raising=True)

    # Under strict guard, ensure_models_loaded_with_memory_check should bypass profiles
    # and attempt only the forced model, without adding fallbacks.
    # Run the async function synchronously for test
    import asyncio

    loaded = asyncio.get_event_loop().run_until_complete(
        mm.ensure_models_loaded_with_memory_check("medium_load")
    )

    # Since CLI loading is disabled, we expect no models actually loaded,
    # but critically: there must be no extra/fallback models added.
    assert isinstance(loaded, list)
    assert loaded == [] or loaded == ["only-this-model"]
    assert called["list_models"] == 0  # strict path must not discover


def test_kpi_first_prompt_success_logic():
    from pathlib import Path

    from codeconductor.utils.kpi import TestSummary, build_kpi

    before = TestSummary(suite_name="pytest", total=0, passed=0, failed=0, skipped=0)

    # Case 1: After has tests and none failed -> success True
    after_ok = TestSummary(suite_name="pytest", total=3, passed=3, failed=0, skipped=0)
    kpi_ok = build_kpi(
        run_id="r1",
        artifacts_dir=Path("artifacts/runs/r1"),
        t_start_iso="2025-01-01T00:00:00Z",
        t_first_green_iso="2025-01-01T00:00:01Z",
        ttft_ms=1234,
        tests_before=before,
        tests_after=after_ok,
        winner_model="m",
        winner_score=0.9,
        consensus_method="codebleu",
        sampling={"temperature": 0.1, "top_p": 0.9, "max_tokens": 128},
        codebleu_weights_env="0.2,0.6,0.2",
        codebleu_lang_env="python",
        exit_status={
            "patched": True,
            "tests_passed": True,
            "tests_missing": False,
            "error": None,
        },
    )
    assert kpi_ok["first_prompt_success"] is True

    # Case 2: After has no tests -> success False
    after_none = TestSummary(suite_name="pytest", total=0, passed=0, failed=0, skipped=0)
    kpi_no = build_kpi(
        run_id="r2",
        artifacts_dir=Path("artifacts/runs/r2"),
        t_start_iso="2025-01-01T00:00:00Z",
        t_first_green_iso=None,
        ttft_ms=2222,
        tests_before=before,
        tests_after=after_none,
        winner_model="m",
        winner_score=0.7,
        consensus_method="codebleu",
        sampling={"temperature": 0.1, "top_p": 0.9, "max_tokens": 128},
        codebleu_weights_env="0.2,0.6,0.2",
        codebleu_lang_env="python",
        exit_status={
            "patched": True,
            "tests_passed": False,
            "tests_missing": True,
            "error": None,
        },
    )
    assert kpi_no["first_prompt_success"] is False
