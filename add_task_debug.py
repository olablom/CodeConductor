#!/usr/bin/env python3
"""
Add debug logging to show the task content near the trailer-injection check.

Run: python add_task_debug.py
"""

from __future__ import annotations

import re
from pathlib import Path


APP_FILE = Path("src/codeconductor/app.py")


def add_debug() -> None:
    if not APP_FILE.exists():
        raise SystemExit(f"ERROR: {APP_FILE} not found")

    content = APP_FILE.read_text(encoding="utf-8")

    pattern = r'(requires_trailer = "# SYNTAX_ERROR BELOW" in \(task or ""\))'
    replacement = (
        'print(f"DEBUG: Task content: {task[:200] if task else "None"}")\n'
        '                            print(f"DEBUG: Looking for SYNTAX_ERROR in task: {\'# SYNTAX_ERROR BELOW\' in (task or "")}\n")\n'
        "                            \\1"
    )

    new_content, n = re.subn(pattern, replacement, content)
    if n == 0:
        print("No matching trailer check found; no changes made.")
        return

    APP_FILE.write_text(new_content, encoding="utf-8")
    print("✅ Added debug logging")


if __name__ == "__main__":
    add_debug()
