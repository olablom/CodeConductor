from __future__ import annotations

import difflib
import hashlib
import json
import os
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

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


def _redact_env(env: dict[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
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


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _manifest_for_files(
    files: list[Path], overlays: dict[str, bytes] | None = None
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
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
) -> tuple[str, dict[str, Any]]:
    """
    Create a zip bundle for the latest run directory and return (zip_path, manifest).
    """
    artifacts = Path(artifacts_dir)
    run_dir = _latest_run_dir(artifacts)
    if run_dir is None:
        raise FileNotFoundError("No runs found under artifacts/runs")

    # Gather candidate files (top-level JSON/TXT) + optionally include generated code
    candidates = list(run_dir.glob("*.json")) + list(run_dir.glob("*.txt"))

    # Optionally exclude raw-like files when include_raw is False
    def is_raw(p: Path) -> bool:
        return p.suffix.lower() in {".log", ".txt", ".md"}

    files_to_add: list[Path] = []
    excluded: list[dict[str, Any]] = []
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

    # Optionally include generated code file from after/
    include_code = os.getenv("EXPORT_INCLUDE_CODE", "1").strip() in {"1", "true", "yes"}
    gen_path = run_dir / "after" / "generated.py"
    if include_code and gen_path.exists():
        # We do not add to candidates (since z.write needs explicit arcname), handle later
        pass

    # Prepare sanitized overlays
    overlays: dict[str, bytes] = {}
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
    manifest: dict[str, Any] = {
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

    # Write atomically via temporary file then replace
    tmp_path = zip_path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(
        tmp_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as z:
        # Add files (with optional overlays)
        for p in files_to_add:
            arcname = p.name
            if arcname in overlays:
                z.writestr(arcname, overlays[arcname])
            else:
                z.write(p, arcname=arcname)
        # Add generated code if requested
        if include_code and gen_path.exists():
            try:
                z.write(gen_path, arcname=str(Path("after") / gen_path.name))
            except Exception:
                pass
        # Add manifest
        z.writestr("MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))

    try:
        os.replace(tmp_path, zip_path)
    except Exception:
        # Best-effort cleanup
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass

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


def verify_manifest(zip_path: str) -> dict[str, Any]:
    """
    Verify MANIFEST.json inside the zip by recomputing SHA256 for included files.

    Returns: { verified: bool, mismatches: [ {file, expected, actual} ] }
    """
    mismatches: list[dict[str, str]] = []
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


# -------------------- Case bundle (PR-3, public_safe v1) --------------------


def _read_text_safe(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _write_temp(z: zipfile.ZipFile, arcname: str, content: str) -> None:
    z.writestr(arcname, content.encode("utf-8"))


def _unified_diff(
    a: str, b: str, fromfile: str = "before", tofile: str = "after"
) -> str:
    a_lines = a.splitlines(keepends=True)
    b_lines = b.splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(a_lines, b_lines, fromfile=fromfile, tofile=tofile)
    )


def _safe_json_load(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _build_readme_case(
    kpi: dict[str, Any],
    *,
    privacy_level: str,
    raw: bool,
    size_cap_mb: int,
    bundle_truncated: bool,
    redactions: list[str],
) -> str:
    ttft = kpi.get("ttft_ms")
    fps = kpi.get("first_prompt_success")
    prb = kpi.get("pass_rate_before")
    pra = kpi.get("pass_rate_after")
    wm = kpi.get("winner_model")
    ws = kpi.get("winner_score")
    cm = kpi.get("consensus_method")
    samp = kpi.get("sampling", {})
    lines = []
    lines.append("# Case: 30s local fix\n")
    lines.append("\n## KPI\n")
    lines.append("| Metric | Value |\n|---|---|\n")
    lines.append(f"| ttft_ms | {ttft} |\n")
    lines.append(f"| first_prompt_success | {fps} |\n")
    lines.append(f"| pass_rate_before | {prb} |\n")
    lines.append(f"| pass_rate_after | {pra} |\n")
    lines.append(f"| winner_model | {wm} |\n")
    lines.append(f"| winner_score | {ws} |\n")
    lines.append(f"| consensus_method | {cm} |\n")
    lines.append(f"| winner_sampling | {json.dumps(samp)} |\n")
    lines.append("\n## Policy\n")
    lines.append("| Field | Value |\n|---|---|\n")
    lines.append(f"| privacy_level | {privacy_level} |\n")
    lines.append(f"| RAW | {int(bool(raw))} |\n")
    lines.append(f"| size_cap_MB | {size_cap_mb} |\n")
    lines.append(f"| bundle_truncated | {bundle_truncated} |\n")

    if redactions:
        lines.append("\n## Redactions\n")
        for item in redactions:
            lines.append(f"- {item}\n")

    lines.append("\n## Timeline\n")
    lines.append("- prompt_in → planning → debate → selection → patch → tests green\n")
    lines.append("\n## Notes\n")
    lines.append("Bundle generated by CodeConductor export pipeline.\n")
    return "".join(lines)


def export_case_bundle(
    *,
    artifacts_dir: str = "artifacts",
    privacy_level: str = "public_safe",
    size_limit_mb: int = 5,
    include_raw: bool | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Build a case bundle zip according to manifest schema (public_safe v1).

    Layout:
    - manifest.json, kpi.json, consensus.json, selector_decision.json
    - diffs/, before/, after/, tests/, logs/, README_case.md
    """
    artifacts = Path(artifacts_dir)
    run_dir = _latest_run_dir(artifacts)
    if run_dir is None:
        raise FileNotFoundError("No runs found under artifacts/runs")

    # Inputs
    kpi = _safe_json_load(run_dir / "kpi.json")
    consensus = _safe_json_load(run_dir / "consensus.json")
    selector = _safe_json_load(run_dir / "selector_decision.json")

    run_id = kpi.get("run_id") or run_dir.name
    exports_dir = artifacts / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    zip_path = exports_dir / f"codeconductor_case_{run_id}.zip"

    # Resolve policy and defaults
    level = (privacy_level or "public_safe").strip().lower()
    warnings: list[str] = []
    valid_levels = {"public_safe", "team_safe", "full_internal"}
    if level not in valid_levels:
        warnings.append(
            f"Unknown privacy_level '{privacy_level}', falling back to public_safe"
        )
        level = "public_safe"

    if include_raw is None:
        include_raw = os.getenv("EXPORT_INCLUDE_RAW", "0").strip() in {
            "1",
            "true",
            "yes",
        }

    # Redactions list
    redactions: list[str] = []
    if level != "full_internal":
        redactions.extend(["raw", "logs/model", "before/src/**"])

    # Prepare components
    readme_case = _build_readme_case(
        kpi,
        privacy_level=level,
        raw=bool(include_raw),
        size_cap_mb=size_limit_mb,
        bundle_truncated=False,
        redactions=redactions,
    )

    # For v1, we may not have real before/after files; create placeholders
    before_files: list[tuple[str, bytes]] = []
    after_files: list[tuple[str, bytes]] = []
    diffs_files: list[tuple[str, bytes]] = []
    logs_files: list[tuple[str, bytes]] = [
        ("pipeline.txt", b"pipeline log N/A in v1\n")
    ]
    tests_dir = run_dir / "tests"
    tests_files: list[Path] = []
    if tests_dir.exists():
        tests_files = list(tests_dir.glob("*.json"))

    # Attempt to derive a single diff from consensus code if present
    cons_code = None
    try:
        cons_code = consensus.get("code") or consensus.get("consensus") or ""
        if isinstance(consensus, dict) and isinstance(consensus.get("consensus"), str):
            cons_code = consensus["consensus"]
    except Exception:
        cons_code = ""
    if cons_code:
        diff = _unified_diff(
            "", cons_code, fromfile="before/generated.py", tofile="after/generated.py"
        )
        diffs_files.append(("generated_code.diff", diff.encode("utf-8")))
        after_files.append(("generated.py", cons_code.encode("utf-8")))

    size_limit_bytes = size_limit_mb * 1024 * 1024
    files_index: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    code_hashes: dict[str, str] = {}

    with zipfile.ZipFile(
        zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as z:
        # Add zip comment for quick sanity check
        z.comment = (
            f"run_id={run_id};privacy={level};schema=kpi@1.0.0,manifest@1.0.0".encode()
        )
        # Core JSONs
        core_files = [
            ("kpi.json", json.dumps(kpi, ensure_ascii=False, indent=2).encode("utf-8"))
        ]
        if consensus:
            core_files.append(
                (
                    "consensus.json",
                    json.dumps(consensus, ensure_ascii=False, indent=2).encode("utf-8"),
                )
            )
        if selector:
            core_files.append(
                (
                    "selector_decision.json",
                    json.dumps(selector, ensure_ascii=False, indent=2).encode("utf-8"),
                )
            )

        total_size = 0
        for name, data in core_files:
            if total_size + len(data) > size_limit_bytes:
                break
            z.writestr(name, data)
            total_size += len(data)
            hashes[name] = hashlib.sha256(data).hexdigest()
            files_index.append({"path": name, "size_bytes": len(data)})

        # README_case.md
        data = readme_case.encode("utf-8")
        if total_size + len(data) <= size_limit_bytes:
            _write_temp(z, "README_case.md", readme_case)
            total_size += len(data)
            hashes["README_case.md"] = hashlib.sha256(data).hexdigest()
            files_index.append({"path": "README_case.md", "size_bytes": len(data)})

        # tests/
        for p in tests_files:
            data = p.read_bytes()
            arc = f"tests/{p.name}"
            if total_size + len(data) > size_limit_bytes:
                continue
            z.writestr(arc, data)
            total_size += len(data)
            hashes[arc] = hashlib.sha256(data).hexdigest()
            files_index.append({"path": arc, "size_bytes": len(data)})

        # logs/
        for name, data in logs_files:
            arc = f"logs/{name}"
            if total_size + len(data) > size_limit_bytes:
                continue
            z.writestr(arc, data)
            total_size += len(data)
            hashes[arc] = hashlib.sha256(data).hexdigest()
            files_index.append({"path": arc, "size_bytes": len(data)})

        # diffs/
        for name, data in diffs_files:
            arc = f"diffs/{name}"
            if total_size + len(data) > size_limit_bytes:
                continue
            z.writestr(arc, data)
            total_size += len(data)
            hashes[arc] = hashlib.sha256(data).hexdigest()
            files_index.append({"path": arc, "size_bytes": len(data)})

        # before/ (placeholders in v1)
        for name, data in before_files:
            arc = f"before/{name}"
            if total_size + len(data) > size_limit_bytes:
                continue
            z.writestr(arc, data)
            total_size += len(data)
            hashes[arc] = hashlib.sha256(data).hexdigest()
            files_index.append({"path": arc, "size_bytes": len(data)})

        # after/
        for name, data in after_files:
            arc = f"after/{name}"
            if total_size + len(data) > size_limit_bytes:
                continue
            z.writestr(arc, data)
            total_size += len(data)
            sh = hashlib.sha256(data).hexdigest()
            hashes[arc] = sh
            files_index.append({"path": arc, "size_bytes": len(data)})
            code_hashes[f"after/{name}"] = sh

        # manifest.json (schema-compatible minimal)
        manifest = {
            "manifest_schema_version": "1.0.0",
            "run_id": run_id,
            "ts": datetime.utcfromtimestamp(time.time()).isoformat() + "Z",
            "version": os.getenv("APP_VERSION", "early-alpha"),
            "privacy_level": level,
            "paths": {
                "kpi": "kpi.json",
                "consensus": "consensus.json",
                "selector_decision": "selector_decision.json",
                "before_dir": "before/",
                "after_dir": "after/",
                "diffs_dir": "diffs/",
                "tests_dir": "tests/",
                "logs_dir": "logs/",
                "readme_case": "README_case.md",
            },
            "files": files_index,
            "hashes": hashes,
            "code_hashes": code_hashes,
            "bundle_truncated": False,
            "redactions": redactions,
            "warnings": warnings,
            "generated_by": {
                "commit": os.getenv("GIT_COMMIT", "dev"),
                "config_digest": "",  # optional here; in KPI
            },
            "engine": kpi.get("engine", {"backend": "other", "version": "unknown"}),
            "models": kpi.get("models", []),
            "agents": kpi.get("agent_config", []),
        }
        data = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
        z.writestr("manifest.json", data)

    return str(zip_path), manifest
