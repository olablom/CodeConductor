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
import os
from pathlib import Path
from typing import List, Tuple


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


def _list_runs(runs_root: Path) -> List[Path]:
    if not runs_root.exists():
        return []
    runs = [p for p in runs_root.iterdir() if _is_run_dir(p)]
    # Sort newest first by directory name timestamp
    runs.sort(key=lambda p: p.name, reverse=True)
    return runs


def _partition_by_age(runs: List[Path], days: int) -> Tuple[List[Path], List[Path]]:
    if days <= 0:
        return runs, []
    cutoff = _dt.datetime.utcnow() - _dt.timedelta(days=days)
    keep: List[Path] = []
    old: List[Path] = []
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

    # Keep only latest N among "recent" according to --keep
    to_keep = recent[: max(keep, 0)] if keep > 0 else recent
    overflow = recent[max(keep, 0) :] if keep > 0 else []

    # To delete = old by age + overflow beyond keep cap
    to_delete = old + overflow

    result["to_delete"] = [str(p) for p in to_delete]
    result["kept"] = [str(p) for p in to_keep]

    if dry_run:
        return result

    # Perform deletion
    for d in to_delete:
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
