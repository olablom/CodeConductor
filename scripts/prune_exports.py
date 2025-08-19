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
import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BundleInfo:
    path: Path
    is_full: bool
    reason: str
    run_id: str | None


def _read_manifest(z: zipfile.ZipFile) -> tuple[dict | None, str]:
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
            missing_files: list[str] = []
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


def _acquire_lock(lock_dir: Path, ttl_seconds: int = 600) -> bool:
    """Create a lock directory atomically. Returns True if acquired.
    Removes stale locks older than ttl.
    """
    try:
        if lock_dir.exists():
            try:
                age = time.time() - lock_dir.stat().st_mtime
                if age > ttl_seconds:
                    shutil.rmtree(lock_dir, ignore_errors=True)
                else:
                    return False
            except Exception:
                return False
        lock_dir.mkdir(parents=True, exist_ok=False)
        try:
            (lock_dir / "pid.txt").write_text(str(os.getpid()), encoding="utf-8")
        except Exception:
            pass
        return True
    except FileExistsError:
        return False
    except Exception:
        return False


def prune_exports(
    *,
    exports_dir: Path,
    keep_full: int,
    delete_minimal: bool,
    dry_run: bool,
) -> dict[str, any]:
    # Acquire lock
    artifacts_root = exports_dir.parent
    lock_dir = artifacts_root / ".prune.lock"
    if not _acquire_lock(lock_dir):
        print(json.dumps({"locked": True, "where": str(lock_dir)}))
        return {"locked": True}

    zips = sorted(
        exports_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    infos: list[BundleInfo] = [classify_bundle(p) for p in zips]

    full = [b for b in infos if b.is_full]
    minimal = [b for b in infos if not b.is_full]

    to_keep_full = full[: max(keep_full, 0)] if keep_full > 0 else full
    overflow_full = full[max(keep_full, 0) :] if keep_full > 0 else []

    # Pins
    pin_str = os.getenv("PIN_EXPORTS", "").strip()
    pinned: set[str] = set([x for x in pin_str.split(";") if x]) if pin_str else set()

    to_delete: list[BundleInfo] = []
    to_delete.extend(overflow_full)
    if delete_minimal:
        to_delete.extend(minimal)
    # Remove pinned from deletion list
    to_delete = [b for b in to_delete if b.path.name not in pinned]

    # Optional size cap for exports dir
    def _dir_size_bytes(p: Path) -> int:
        total = 0
        for root, _dirs, files in os.walk(p):
            # Skip internal temp folders
            if any(skip in root for skip in (".trash", "_unzipped")):
                continue
            for fn in files:
                try:
                    total += (Path(root) / fn).stat().st_size
                except Exception:
                    pass
        return total

    cap_gb_str = os.getenv("EXPORT_SIZE_CAP_GB", "").strip()
    extra_delete: list[BundleInfo] = []
    before_bytes = _dir_size_bytes(exports_dir)
    if cap_gb_str:
        try:
            cap_bytes = int(float(cap_gb_str) * (1024**3))
            if before_bytes > cap_bytes:
                # Oldest-first candidates not already scheduled and not pinned
                planned = set(b.path for b in to_delete)
                cand = [p for p in zips if p not in planned and p.name not in pinned]
                cand.sort(key=lambda p: p.stat().st_mtime)  # oldest first
                remaining = before_bytes
                for p in cand:
                    if remaining <= cap_bytes:
                        break
                    try:
                        sz = p.stat().st_size
                    except Exception:
                        sz = 0
                    remaining -= sz
                    extra_delete.append(
                        BundleInfo(p, is_full=False, reason="size_cap", run_id=None)
                    )
        except Exception:
            pass

    # Mass-delete guard (>70%) unless FORCE
    planned_total = len(to_delete) + len(extra_delete)
    if not dry_run and zips:
        ratio = planned_total / max(1, len(zips))
        if ratio > 0.7 and os.getenv("FORCE", "0") not in {"1", "true", "yes"}:
            summary = {
                "exports_dir": str(exports_dir),
                "total": len(zips),
                "planned_delete": planned_total,
                "aborted": True,
                "reason": "mass_delete_guard",
            }
            print(json.dumps(summary))
            try:
                shutil.rmtree(lock_dir, ignore_errors=True)
            except Exception:
                pass
            return summary

    # Apply deletions (retention + cap)
    deleted: list[str] = []
    if not dry_run:
        trash_dir = exports_dir / ".trash"
        trash_dir.mkdir(parents=True, exist_ok=True)
        for b in to_delete + extra_delete:
            try:
                # Move to trash first
                target = trash_dir / b.path.name
                try:
                    os.replace(b.path, target)
                except Exception:
                    # Fallback unlink
                    b.path.unlink(missing_ok=True)  # type: ignore[arg-type]
                    deleted.append(str(b.path))
                    continue
                # Now delete from trash
                try:
                    target.unlink(missing_ok=True)
                except Exception:
                    pass
                deleted.append(str(b.path))
            except Exception:
                pass

    after_bytes = _dir_size_bytes(exports_dir)
    summary = {
        "exports_dir": str(exports_dir),
        "total": len(infos),
        "full": len(full),
        "minimal": len(minimal),
        "kept_full": [str(b.path) for b in to_keep_full],
        "to_delete": [str(b.path) for b in to_delete + extra_delete],
        "deleted": deleted,
        "dry_run": dry_run,
        "before_bytes": before_bytes,
        "after_bytes": after_bytes,
        "cap_bytes": int(float(cap_gb_str) * (1024**3)) if cap_gb_str else None,
    }

    # Human-readable
    print(f"Exports: {exports_dir}")
    print(f"Total: {len(infos)} | Full: {len(full)} | Minimal: {len(minimal)}")
    print(f"Keeping (full): {len(to_keep_full)}")
    print(f"Deleting: {len(to_delete)} (dry_run={dry_run})")
    # Final JSON line
    print(json.dumps(summary))
    # Release lock
    try:
        shutil.rmtree(lock_dir, ignore_errors=True)
    except Exception:
        pass
    # Cleanup old trash (24h)
    try:
        trash_dir = exports_dir / ".trash"
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
