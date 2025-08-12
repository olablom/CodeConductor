#!/usr/bin/env python3
"""
Quick patch to harden self-repair trigger in GUI.
Run: python fix_self_repair.py

This script ensures that:
- The first validation call passes task_input and require_trailer based on the task.
- The repair prompt is built with require_trailer_by_task when needed.
- The re-validation after repair disables trailer requirement (so trailer is removed).

It is idempotent and will only modify src/codeconductor/app.py if needed.
"""

from __future__ import annotations

import re
from pathlib import Path


APP_PATH = Path("src/codeconductor/app.py")


def main() -> int:
    if not APP_PATH.exists():
        print(f"ERROR: {APP_PATH} not found")
        return 1

    src = APP_PATH.read_text(encoding="utf-8")
    changed = False

    # 1) Ensure first validate_python_code(...) includes task_input and require_trailer
    # Pattern around the first validate call
    pat_validate1 = re.compile(
        r"report\s*=\s*validate_python_code\(\s*code_txt\s*,\s*run_doctests=True([^)]*)\)",
        re.DOTALL,
    )

    def _ensure_validate_first(m: re.Match[str]) -> str:
        inside = m.group(1)
        inserts = []
        if "task_input=" not in inside:
            inserts.append("task_input=task")
        if "require_trailer=" not in inside:
            inserts.append('require_trailer=("# SYNTAX_ERROR BELOW" in (task or ""))')
        if not inserts:
            return m.group(0)
        new_inside = inside.strip()
        if new_inside and not new_inside.strip().endswith(","):
            new_inside = new_inside + ", "
        new_inside = new_inside + ", ".join(inserts)
        return m.group(0).replace(inside, new_inside)

    src2, n1 = pat_validate1.subn(_ensure_validate_first, src, count=1)
    if n1:
        changed = changed or (src2 != src)
        src = src2

    # 2) Ensure build_repair_prompt(...) includes require_trailer_by_task argument
    pat_build = re.compile(
        r"build_repair_prompt\(\s*code_txt\s*,\s*report\s*,\s*([^)]*)\)",
        re.DOTALL,
    )

    def _ensure_build_prompt(m: re.Match[str]) -> str:
        inside = m.group(1)
        if "require_trailer_by_task=" in inside:
            return m.group(0)
        # We assume the first third argument is doctest text; append require flag
        new_inside = inside.strip()
        if new_inside and not new_inside.strip().endswith(","):
            new_inside = new_inside + ", "
        new_inside = (
            new_inside
            + 'require_trailer_by_task=("# SYNTAX_ERROR BELOW" in (task or ""))'
        )
        return m.group(0).replace(inside, new_inside)

    src2, n2 = pat_build.subn(_ensure_build_prompt, src, count=1)
    if n2:
        changed = changed or (src2 != src)
        src = src2

    # 3) Ensure re-validation uses require_trailer=False
    pat_validate2 = re.compile(
        r"report\s*=\s*validate_python_code\(\s*code_txt\s*,\s*run_doctests=True([^)]*)\)\s*\n\s*iter_count\s*\+=\s*1",
        re.DOTALL,
    )

    def _ensure_validate_second(m: re.Match[str]) -> str:
        inside = m.group(1)
        if "require_trailer=False" in inside:
            return m.group(0)
        new_inside = inside.strip()
        if new_inside and not new_inside.strip().endswith(","):
            new_inside = new_inside + ", "
        new_inside = new_inside + "require_trailer=False"
        return m.group(0).replace(inside, new_inside)

    src2, n3 = pat_validate2.subn(_ensure_validate_second, src, count=1)
    if n3:
        changed = changed or (src2 != src)
        src = src2

    if changed:
        APP_PATH.write_text(src, encoding="utf-8")
        print(f"Patched {APP_PATH}")
    else:
        print("No changes needed (already patched)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
