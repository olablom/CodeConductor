#!/usr/bin/env python3
"""
Export the latest run artifacts into a zip bundle and verify the manifest.

Usage:
  python scripts/export_latest.py

Outputs a single JSON line with fields {"zip": <path>, "verified": true|false}.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    # Ensure local 'src' is importable for codeconductor.*
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from codeconductor.utils.exporter import export_case_bundle

    # Defaults for PR-3 (public_safe v1)
    privacy = os.getenv("EXPORT_PRIVACY_LEVEL", "public_safe")
    size_mb = int(os.getenv("EXPORT_SIZE_LIMIT_MB", "5"))

    zip_path, manifest = export_case_bundle(
        artifacts_dir=os.getenv("ARTIFACTS_DIR", "artifacts"),
        privacy_level=privacy,
        size_limit_mb=size_mb,
    )

    # Optional schema validation if jsonschema is available
    verified = True
    details = {}
    try:
        import jsonschema  # type: ignore
        from pathlib import Path as _Path

        # Load schemas
        base = (
            _Path(__file__).resolve().parents[1]
            / "src"
            / "codeconductor"
            / "utils"
            / "schemas"
        )
        man_schema = json.loads(
            (base / "manifest.schema.json").read_text(encoding="utf-8")
        )
        kpi_schema = json.loads((base / "kpi.schema.json").read_text(encoding="utf-8"))

        # Validate manifest
        jsonschema.validate(instance=manifest, schema=man_schema)

        # Validate KPI from latest run dir
        # best-effort: read artifacts/runs/<latest>/kpi.json
        runs = sorted(
            (_Path(os.getenv("ARTIFACTS_DIR", "artifacts")) / "runs").glob("*")
        )
        if runs:
            kpi_path = runs[-1] / "kpi.json"
            if kpi_path.exists():
                kpi = json.loads(kpi_path.read_text(encoding="utf-8"))
                jsonschema.validate(instance=kpi, schema=kpi_schema)
    except Exception as e:  # pragma: no cover
        verified = False
        details = {"error": str(e)}

    # Ensure UTF-8 capable stdout on Windows terminals
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    print(
        json.dumps(
            {"zip": zip_path, "verified": verified, **details}, ensure_ascii=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
