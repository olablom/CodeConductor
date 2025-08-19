#!/usr/bin/env python3
"""
Cleanup utility for artifacts/runs.

Features:
- Delete run directories older than N days
- Optionally keep only the most recent K runs regardless of age
- Dry-run mode to preview deletions

Usage examples:
  python scripts/cleanup_runs.py                # default: --days 7 --keep 50
  python scripts/cleanup_runs.py --days 3       # delete older than 3 days
  python scripts/cleanup_runs.py --keep 20      # keep only latest 20 runs
  python scripts/cleanup_runs.py --days 10 --keep 30 --dry-run

Notes:
- A "run" is any directory under artifacts/runs/ with name like YYYYMMDD_HHMMSS
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import time
import zipfile
from pathlib import Path


def _is_run_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    name = path.name
    # Expect format YYYYMMDD_HHMMSS (length 15 with underscore)
    if len(name) != 15 or "_" not in name:
        return False
    try:
        _dt.datetime.strptime(name, "%Y%m%d_%H%M%S")
        return True
    except Exception:
        return False


def _list_runs(runs_root: Path) -> list[Path]:
    if not runs_root.exists():
        return []
    runs = [p for p in runs_root.iterdir() if _is_run_dir(p)]
    # Sort newest first by directory name timestamp (lexicographic on run-id)
    runs.sort(key=lambda p: p.name, reverse=True)
    return runs


def _partition_by_age(runs: list[Path], days: int) -> tuple[list[Path], list[Path]]:
    if days <= 0:
        return runs, []
    cutoff = _dt.datetime.utcnow() - _dt.timedelta(days=days)
    keep: list[Path] = []
    old: list[Path] = []
    for r in runs:
        try:
            ts = _dt.datetime.strptime(r.name, "%Y%m%d_%H%M%S")
        except Exception:
            # If unexpected name format, err on the side of keeping
            keep.append(r)
            continue
        if ts < cutoff:
            old.append(r)
        else:
            keep.append(r)
    return keep, old


def cleanup_runs(
    *,
    artifacts_dir: Path,
    days: int,
    keep: int,
    dry_run: bool,
) -> dict:
    runs_root = artifacts_dir / "runs"
    # Locking to avoid concurrent prunes
    lock_dir = artifacts_dir / ".prune.lock"
    if lock_dir.exists():
        age = 0
        try:
            age = time.time() - lock_dir.stat().st_mtime
        except Exception:
            pass
        if age <= 600:
            return {
                "artifacts_dir": str(artifacts_dir),
                "runs_root": str(runs_root),
                "total_runs": 0,
                "to_delete": [],
                "kept": [],
                "deleted": [],
                "dry_run": True,
                "locked": True,
            }
        else:
            shutil.rmtree(lock_dir, ignore_errors=True)
    try:
        lock_dir.mkdir(parents=True, exist_ok=False)
    except Exception:
        return {
            "artifacts_dir": str(artifacts_dir),
            "runs_root": str(runs_root),
            "total_runs": 0,
            "to_delete": [],
            "kept": [],
            "deleted": [],
            "dry_run": True,
            "locked": True,
        }
    runs = _list_runs(runs_root)

    result = {
        "artifacts_dir": str(artifacts_dir),
        "runs_root": str(runs_root),
        "total_runs": len(runs),
        "to_delete": [],
        "kept": [],
        "deleted": [],
        "dry_run": bool(dry_run),
    }

    if not runs:
        return result

    # Age-based partition
    recent, old = _partition_by_age(runs, days)

    # OR logic: keep if index < keep OR age <= days
    keep_cap = max(keep, 0)
    to_keep: list[Path] = []
    overflow: list[Path] = []
    for idx, r in enumerate(recent):
        if idx < keep_cap:
            to_keep.append(r)
        else:
            overflow.append(r)

    # To delete = old by age + overflow beyond keep cap
    to_delete = old + overflow

    # PIN_RUNS support
    pinned_runs = set(filter(None, os.getenv("PIN_RUNS", "").split(";")))
    to_delete = [p for p in to_delete if p.name not in pinned_runs]

    # Reference integrity: skip runs referenced by kept exports
    def _get_referenced_runs(exports_dir: Path) -> set[str]:
        refs: set[str] = set()
        for zpath in exports_dir.glob("*.zip"):
            try:
                with zipfile.ZipFile(zpath, "r") as zf:
                    name = (
                        "manifest.json"
                        if "manifest.json" in zf.namelist()
                        else (
                            "MANIFEST.json"
                            if "MANIFEST.json" in zf.namelist()
                            else None
                        )
                    )
                    if not name:
                        continue
                    try:
                        manifest = json.loads(
                            zf.read(name).decode("utf-8", errors="ignore")
                        )
                        rid = manifest.get("run_id")
                        if isinstance(rid, str) and rid:
                            refs.add(rid)
                    except Exception:
                        continue
            except Exception:
                continue
        return refs

    referenced = _get_referenced_runs(artifacts_dir / "exports")
    to_delete = [p for p in to_delete if p.name not in referenced]

    result["to_delete"] = [str(p) for p in to_delete]
    result["kept"] = [str(p) for p in to_keep]

    # Optional size cap for entire artifacts
    def _dir_size_bytes(p: Path) -> int:
        total = 0
        for root, _dirs, files in os.walk(p):
            if any(skip in root for skip in (".trash", ".prune.lock", "_unzipped")):
                continue
            for fn in files:
                try:
                    total += (Path(root) / fn).stat().st_size
                except Exception:
                    pass
        return total

    cap_gb_str = os.getenv("ARTIFACTS_SIZE_CAP_GB", "").strip()
    before_bytes = _dir_size_bytes(artifacts_dir)
    extra_delete: list[Path] = []
    if cap_gb_str:
        try:
            cap_bytes = int(float(cap_gb_str) * (1024**3))
            if before_bytes > cap_bytes:
                # Build referenced runs from kept exports (optional simple pass)
                referenced: set[str] = set()
                exports_dir = artifacts_dir / "exports"
                try:
                    for z in exports_dir.glob("codeconductor_case_*.zip"):
                        # Prefer not to parse zip here for speed;
                        # fallback to run_id in filename
                        try:
                            rid = z.stem.split("_")[-1]
                            if rid:
                                referenced.add(rid)
                        except Exception:
                            pass
                except Exception:
                    pass
                # Candidates: oldest lex first among to_delete,
                # skipping pinned and referenced
                pin_runs = set([x for x in os.getenv("PIN_RUNS", "").split(";") if x])
                cands = [
                    p
                    for p in to_delete
                    if p.name not in pin_runs and p.name not in referenced
                ]
                cands.sort(key=lambda p: p.name)  # oldest first
                remaining = before_bytes
                for p in cands:
                    if remaining <= cap_bytes:
                        break
                    # Approximate size of this run dir
                    sz = _dir_size_bytes(p)
                    remaining -= sz
                    extra_delete.append(p)
        except Exception:
            pass

    if dry_run:
        # Release lock
        try:
            shutil.rmtree(lock_dir, ignore_errors=True)
        except Exception:
            pass
        return result

    # Perform deletion
    for d in to_delete + extra_delete:
        try:
            for child in d.rglob("*"):
                if child.is_file() or child.is_symlink():
                    try:
                        child.unlink()
                    except Exception:
                        pass
            # Remove directories bottom-up
            for child in sorted(d.rglob("*"), reverse=True):
                if child.is_dir():
                    try:
                        child.rmdir()
                    except Exception:
                        pass
            # Finally remove the run dir itself
            d.rmdir()
            result["deleted"].append(str(d))
        except Exception:
            # Best-effort: ignore errors
            pass

    # Release lock
    try:
        shutil.rmtree(lock_dir, ignore_errors=True)
    except Exception:
        pass

    # Cleanup old trash in exports/.trash if present (24h default)
    try:
        trash_dir = artifacts_dir / "exports" / ".trash"
        cutoff = time.time() - 24 * 3600
        if trash_dir.exists():
            for entry in os.scandir(trash_dir):
                try:
                    if entry.is_file() and entry.stat().st_mtime < cutoff:
                        os.remove(entry.path)
                    elif entry.is_dir() and entry.stat().st_mtime < cutoff:
                        shutil.rmtree(entry.path, ignore_errors=True)
                except Exception:
                    continue
    except Exception:
        pass
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Prune artifacts/runs directories")
    parser.add_argument(
        "--artifacts",
        default=os.getenv("ARTIFACTS_DIR", "artifacts"),
        help="Artifacts root directory (default: env ARTIFACTS_DIR or 'artifacts')",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Delete runs older than N days (default: 7). Use 0 to disable age pruning.",
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=50,
        help="Keep only the most recent K runs (default: 50). Use 0 to disable.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without deleting",
    )
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts).resolve()
    res = cleanup_runs(
        artifacts_dir=artifacts_dir,
        days=args.days,
        keep=args.keep,
        dry_run=args.dry_run,
    )

    # Human-readable summary
    print("Artifacts root:", res["artifacts_dir"])
    print("Runs root:", res["runs_root"])
    print("Total runs:", res["total_runs"])
    print("Kept:", len(res["kept"]))
    print("To delete:", len(res["to_delete"]))
    if res["dry_run"]:
        print("(dry-run) Nothing deleted")
    else:
        print("Deleted:", len(res["deleted"]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
