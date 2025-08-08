from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import time
import zipfile


SENSITIVE_KEYS = (
    "API",
    "KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _latest_run_dir(artifacts_dir: Path) -> Path | None:
    runs = sorted(
        (artifacts_dir / "runs").glob("*"), key=lambda p: p.name, reverse=True
    )
    return runs[0] if runs else None


def _redact_env(env: Dict[str, Any]) -> Dict[str, Any]:
    redacted: Dict[str, Any] = {}
    for k, v in (env or {}).items():
        if any(s in k.upper() for s in SENSITIVE_KEYS):
            redacted[k] = "***REDACTED***"
        elif (
            isinstance(v, str)
            and ("\\" in v or "/" in v)
            and ("HOME" in k.upper() or "PATH" in k.upper())
        ):
            redacted[k] = "***REDACTED_PATH***"
        else:
            redacted[k] = v
    return redacted


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _manifest_for_files(
    files: List[Path], overlays: Dict[str, bytes] | None = None
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for p in files:
        try:
            if overlays and p.name in overlays:
                content = overlays[p.name]
                sha = hashlib.sha256(content).hexdigest()
                size = len(content)
            else:
                sha = _sha256_file(p)
                size = p.stat().st_size
            entries.append(
                {
                    "file": str(p.name),
                    "size": size,
                    "sha256": sha,
                    "mtime": int(p.stat().st_mtime),
                }
            )
        except Exception:
            continue
    return entries


def export_latest_run(
    artifacts_dir: str = "artifacts",
    include_raw: bool = False,
    redact_env: bool = True,
    size_limit_mb: int = 50,
    retention: int = 20,
    policy: str | None = None,
    selected_model: str | None = None,
    cache_hit: bool | None = None,
    app_version: str | None = None,
    git_commit: str | None = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Create a zip bundle for the latest run directory and return (zip_path, manifest).
    """
    artifacts = Path(artifacts_dir)
    run_dir = _latest_run_dir(artifacts)
    if run_dir is None:
        raise FileNotFoundError("No runs found under artifacts/runs")

    # Gather candidate files
    candidates = list(run_dir.glob("*.json")) + list(run_dir.glob("*.txt"))

    # Optionally exclude raw-like files when include_raw is False
    def is_raw(p: Path) -> bool:
        return p.suffix.lower() in {".log", ".txt", ".md"}

    files_to_add: List[Path] = []
    excluded: List[Dict[str, Any]] = []
    size_limit_bytes = size_limit_mb * 1024 * 1024

    for p in candidates:
        if not include_raw and is_raw(p):
            excluded.append({"file": p.name, "reason": "raw_excluded"})
            continue
        try:
            if p.stat().st_size > size_limit_bytes:
                excluded.append({"file": p.name, "reason": "size_exceeded"})
                continue
        except Exception:
            continue
        files_to_add.append(p)

    # Prepare sanitized overlays
    overlays: Dict[str, bytes] = {}
    rc_path = run_dir / "run_config.json"
    if rc_path.exists():
        rc = _read_json(rc_path)
        if redact_env and isinstance(rc.get("env"), dict):
            rc["env"] = _redact_env(rc["env"])  # type: ignore[index]
        overlays["run_config.json"] = json.dumps(
            rc, ensure_ascii=False, indent=2
        ).encode("utf-8")

    # Build manifest
    file_entries = _manifest_for_files(files_to_add, overlays)
    manifest: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "created_at": int(time.time()),
        "policy": policy,
        "selected_model": selected_model,
        "cache": "HIT" if cache_hit else "MISS",
        "app_version": app_version,
        "git_commit": git_commit,
        "files": file_entries,
        "excluded": excluded,
    }

    # Zip file name and retention
    exports_dir = artifacts / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    ts = run_dir.name
    hitmiss = "hit" if cache_hit else "miss"
    safe_policy = (policy or "").lower() or "latency"
    zip_name = f"codeconductor_run_{ts}_{safe_policy}_{hitmiss}.zip"
    zip_path = exports_dir / zip_name

    with zipfile.ZipFile(
        zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as z:
        # Add files (with optional overlays)
        for p in files_to_add:
            arcname = p.name
            if arcname in overlays:
                z.writestr(arcname, overlays[arcname])
            else:
                z.write(p, arcname=arcname)
        # Add manifest
        z.writestr("MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))

    # Retention policy
    zips = sorted(
        exports_dir.glob("codeconductor_run_*.zip"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for old in zips[retention:]:
        try:
            old.unlink()
        except Exception:
            pass

    return str(zip_path), manifest


def verify_manifest(zip_path: str) -> Dict[str, Any]:
    """
    Verify MANIFEST.json inside the zip by recomputing SHA256 for included files.

    Returns: { verified: bool, mismatches: [ {file, expected, actual} ] }
    """
    mismatches: List[Dict[str, str]] = []
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            data = z.read("MANIFEST.json")
            manifest = json.loads(data.decode("utf-8"))
            files = manifest.get("files", [])
            for entry in files:
                fname = entry.get("file")
                expected = entry.get("sha256")
                try:
                    content = z.read(fname)
                    actual = hashlib.sha256(content).hexdigest()
                    if actual != expected:
                        mismatches.append(
                            {
                                "file": fname or "",
                                "expected": expected or "",
                                "actual": actual,
                            }
                        )
                except KeyError:
                    mismatches.append(
                        {
                            "file": fname or "",
                            "expected": expected or "",
                            "actual": "<missing>",
                        }
                    )
            return {"verified": len(mismatches) == 0, "mismatches": mismatches}
    except Exception as e:  # pragma: no cover
        return {"verified": False, "error": str(e), "mismatches": mismatches}
