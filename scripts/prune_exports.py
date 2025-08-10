#!/usr/bin/env python3
"""
Prune artifacts/exports/*.zip bundles.

Rules:
- Classify bundles as "full" if manifest.json has a 'paths' object that includes
  at least: kpi, consensus, selector_decision and the referenced files exist in the zip.
- "minimal" if manifest.json missing or lacks required paths/files.
- Keep the N most recent full bundles; optionally delete all minimal bundles.

Usage:
  python scripts/prune_exports.py --keep-full 20 --delete-minimal --dry-run

Output: human-readable summary and a compact JSON line at the end.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import zipfile


@dataclass
class BundleInfo:
    path: Path
    is_full: bool
    reason: str
    run_id: Optional[str]


def _read_manifest(z: zipfile.ZipFile) -> Tuple[Optional[dict], str]:
    # Try lowercase first (our current exporter), then uppercase (older flow)
    for name in ("manifest.json", "MANIFEST.json"):
        try:
            with z.open(name) as f:
                data = json.loads(f.read().decode("utf-8", errors="ignore"))
                return data, "ok"
        except KeyError:
            continue
        except Exception as e:
            return None, f"manifest_parse_error: {e}"
    return None, "manifest_missing"


def _zip_has(z: zipfile.ZipFile, member: str) -> bool:
    try:
        z.getinfo(member)
        return True
    except KeyError:
        return False


def classify_bundle(zip_path: Path) -> BundleInfo:
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            manifest, status = _read_manifest(z)
            if manifest is None:
                return BundleInfo(zip_path, is_full=False, reason=status, run_id=None)

            paths = manifest.get("paths") or {}
            required_keys = ["kpi", "consensus", "selector_decision"]
            missing_keys = [k for k in required_keys if not paths.get(k)]
            if missing_keys:
                return BundleInfo(
                    zip_path,
                    is_full=False,
                    reason=f"missing_paths:{','.join(missing_keys)}",
                    run_id=manifest.get("run_id"),
                )

            # Verify files exist inside the zip
            missing_files: List[str] = []
            for k in required_keys:
                p = str(paths.get(k))
                if not p or not _zip_has(z, p):
                    missing_files.append(p or k)
            if missing_files:
                return BundleInfo(
                    zip_path,
                    is_full=False,
                    reason=f"missing_files:{','.join(missing_files)}",
                    run_id=manifest.get("run_id"),
                )

            # Consider as full if required files exist; tests/diffs may be optional v1
            return BundleInfo(
                zip_path, is_full=True, reason="ok", run_id=manifest.get("run_id")
            )

    except Exception as e:
        return BundleInfo(zip_path, is_full=False, reason=f"zip_error:{e}", run_id=None)


def prune_exports(
    *,
    exports_dir: Path,
    keep_full: int,
    delete_minimal: bool,
    dry_run: bool,
) -> Dict[str, any]:
    zips = sorted(
        exports_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    infos: List[BundleInfo] = [classify_bundle(p) for p in zips]

    full = [b for b in infos if b.is_full]
    minimal = [b for b in infos if not b.is_full]

    to_keep_full = full[: max(keep_full, 0)] if keep_full > 0 else full
    overflow_full = full[max(keep_full, 0) :] if keep_full > 0 else []

    to_delete: List[BundleInfo] = []
    to_delete.extend(overflow_full)
    if delete_minimal:
        to_delete.extend(minimal)

    deleted: List[str] = []
    if not dry_run:
        for b in to_delete:
            try:
                b.path.unlink(missing_ok=True)  # type: ignore[arg-type]
                deleted.append(str(b.path))
            except Exception:
                pass

    summary = {
        "exports_dir": str(exports_dir),
        "total": len(infos),
        "full": len(full),
        "minimal": len(minimal),
        "kept_full": [str(b.path) for b in to_keep_full],
        "to_delete": [str(b.path) for b in to_delete],
        "deleted": deleted,
        "dry_run": dry_run,
    }

    # Human-readable
    print(f"Exports: {exports_dir}")
    print(f"Total: {len(infos)} | Full: {len(full)} | Minimal: {len(minimal)}")
    print(f"Keeping (full): {len(to_keep_full)}")
    print(f"Deleting: {len(to_delete)} (dry_run={dry_run})")
    # Final JSON line
    print(json.dumps(summary))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Prune exports bundles")
    parser.add_argument(
        "--artifacts-dir",
        default=os.getenv("ARTIFACTS_DIR", "artifacts"),
        help="Artifacts root (default: env ARTIFACTS_DIR or 'artifacts')",
    )
    parser.add_argument(
        "--keep-full",
        type=int,
        default=20,
        help="Keep latest N full bundles (default: 20)",
    )
    parser.add_argument(
        "--delete-minimal",
        action="store_true",
        help="Delete minimal/incomplete bundles as well",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show actions without deleting"
    )

    args = parser.parse_args()
    artifacts = Path(args.artifacts_dir).resolve()
    exports_dir = artifacts / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)

    prune_exports(
        exports_dir=exports_dir,
        keep_full=args.keep_full,
        delete_minimal=args.delete_minimal,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
