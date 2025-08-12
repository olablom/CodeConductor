#!/usr/bin/env python3
"""
Add debug logging around the self-repair loop in app.py to understand why it may not trigger.

Run: python debug_repair_loop.py
"""

from __future__ import annotations

import re
from pathlib import Path


APP = Path("src/codeconductor/app.py")


def add_repair_debug() -> None:
    if not APP.exists():
        raise SystemExit(f"ERROR: {APP} not found")

    content = APP.read_text(encoding="utf-8")

    # 1) Add debug before entering repair loop
    pat_before = re.compile(r"(iter_count\s*=\s*0)")
    content, n1 = pat_before.subn(
        (
            'print(f"DEBUG: About to enter repair loop")\n'
            '                        print(f"DEBUG: report.ok = {report.ok}")\n'
            "                        print(f\"DEBUG: report.syntax_ok = {getattr(report, 'syntax_ok', 'N/A')}\")\n"
            "                        print(f\"DEBUG: report.doctest_failures = {getattr(report, 'doctest_failures', 'N/A')}\")\n"
            "                        \\1"
        ),
        content,
        count=1,
    )

    # 2) Add debug inside repair loop
    pat_loop = re.compile(r"(while \(not report\.ok\) and iter_count < 2:\s*)")
    content, n2 = pat_loop.subn(
        (
            '\\1\n                            print(f"DEBUG: Inside repair loop, iteration {iter_count}")\n'
        ),
        content,
        count=1,
    )

    # 3) Add debug after loop exit
    pat_after = re.compile(
        r"(# Only keep file if final report is ok; otherwise do not leave a broken file)"
    )
    content, n3 = pat_after.subn(
        (
            'print(f"DEBUG: Exited repair loop. Final report.ok = {report.ok}")\n'
            "                        \\1"
        ),
        content,
        count=1,
    )

    APP.write_text(content, encoding="utf-8")
    print(
        f"✅ Added repair loop debug logging (inserted: before={bool(n1)}, inside={bool(n2)}, after={bool(n3)})"
    )


if __name__ == "__main__":
    add_repair_debug()
