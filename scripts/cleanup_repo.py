#!/usr/bin/env python3
"""
Cleanup utility for CodeConductor workspace.

Removes common artifact/log/temp directories that are safe to delete.
Dry-run by default; pass --apply to actually delete.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

CANDIDATES = [
    "__pycache__",
    ".pytest_cache",
    "htmlcov",
    "logs",
    "artifacts",
    "validation_logs",
    "cache",
    ".cache",
    "dist",
    "build",
    "*.log",
    "*_test_results.json",
]


def list_targets(root: Path) -> list[Path]:
    targets: list[Path] = []
    for pat in CANDIDATES:
        targets.extend(root.glob(f"**/{pat}"))
    # De-duplicate and keep within repo
    uniq: list[Path] = []
    seen = set()
    for t in targets:
        p = t.resolve()
        if p in seen:
            continue
        if root in p.parents or p == root:
            uniq.append(p)
            seen.add(p)
    return uniq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually delete files")
    args = ap.parse_args()

    root = Path.cwd()
    targets = list_targets(root)
    if not targets:
        print("No cleanup targets found.")
        return 0

    print(f"Found {len(targets)} targets:")
    for t in targets:
        print(f" - {t}")

    if not args.apply:
        print("\nDry run. Use --apply to delete.")
        return 0

    for t in targets:
        try:
            if t.is_dir():
                shutil.rmtree(t, ignore_errors=True)
            elif t.exists():
                t.unlink(missing_ok=True)
        except Exception as e:
            print(f"Failed to delete {t}: {e}")
    print("Cleanup complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
