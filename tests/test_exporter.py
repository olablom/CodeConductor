from pathlib import Path

from codeconductor.utils.exporter import export_latest_run, verify_manifest


def test_export_latest_run_basic(tmp_path: Path, monkeypatch):
    # Arrange a fake run dir with minimal files
    artifacts = tmp_path / "artifacts"
    run_dir = artifacts / "runs" / "20250101_120000"
    run_dir.mkdir(parents=True)
    (run_dir / "run_config.json").write_text(
        '{"env": {"OPENAI_API_KEY": "X", "PATH": "C:/Users/me"}}', encoding="utf-8"
    )
    (run_dir / "selector_decision.json").write_text("{}", encoding="utf-8")
    (run_dir / "consensus.json").write_text("{}", encoding="utf-8")

    # Act
    zip_path, manifest = export_latest_run(
        artifacts_dir=str(artifacts),
        include_raw=False,
        redact_env=True,
        size_limit_mb=5,
        policy="latency",
        selected_model="m1",
        cache_hit=True,
    )

    # Assert
    assert Path(zip_path).exists()
    assert manifest.get("policy") == "latency"
    assert manifest.get("selected_model") == "m1"
    assert manifest.get("cache") == "HIT"

    # Verify manifest integrity
    verification = verify_manifest(zip_path)
    assert verification.get("verified") is True
