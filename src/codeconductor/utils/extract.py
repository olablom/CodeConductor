"""
Utilities for extracting and normalizing code from model outputs.

This module provides robust markdown fence stripping, SQL fallbacks, and
Python normalization suitable for quick validation and compilation.
"""

from __future__ import annotations

import re
import textwrap
from typing import Optional


# Matches ```lang\n ... ``` with any (optional) language tag.
FENCE_RE = re.compile(r"```(?:\w+)?\s*([\s\S]*?)```", re.MULTILINE)


def extract_code(raw_text: str, lang_hint: Optional[str] = None) -> str:
    """
    Extract code from a model response.

    Strategy:
    - Prefer fenced blocks; pick the longest block if multiple.
    - If none found and SQL is likely, assemble a SQL snippet from lines that
      contain common SQL keywords. Ensure trailing semicolon when possible.
    - Fallback to returning the raw text trimmed.
    """

    if not raw_text:
        return ""

    # 1) Prefer fenced blocks (return the longest)
    blocks = [m.group(1).strip() for m in FENCE_RE.finditer(raw_text)]
    if blocks:
        # Ensure we return python block when multiple blocks exist and one appears to be python
        if (lang_hint or "").lower() == "python":
            # Prefer blocks that look like python by having def/class/if etc.
            def score_py(b: str) -> int:
                score = 0
                for kw in ("def ", "class ", "if ", "import ", "from "):
                    if kw in b:
                        score += 1
                return score

            blocks_sorted = sorted(
                blocks, key=lambda b: (score_py(b), len(b)), reverse=True
            )
            return blocks_sorted[0].strip()
        return max(blocks, key=len).strip()

    # 2) SQL fallback
    lower_text = raw_text.lower()
    is_sql = (lang_hint or "").lower() in {
        "sql",
        "postgres",
        "mysql",
    } or "select " in lower_text
    if is_sql:
        # Prefer taking the substring starting from the first SQL keyword.
        # Try DML/DDL first to avoid matching natural-language "with ...".
        first_kw = re.search(
            r"\b(select|insert|update|delete|create)\b", raw_text, re.I
        )
        if not first_kw:
            # Consider WITH only at line start (SQL CTE), not inside prose
            first_kw = re.search(r"(?m)^(with)\b", raw_text, re.I)
        if first_kw:
            sql_sub = raw_text[first_kw.start() :]
            # Cut at the first terminating semicolon if present
            semi = sql_sub.find(";")
            if semi != -1:
                return sql_sub[: semi + 1].strip()
            # Otherwise, try to assemble from keyword-bearing lines and ensure semicolon
            sql_lines = []
            for line in sql_sub.splitlines():
                if re.search(
                    r"\b(select|insert|update|delete|create|with|from|where|group by|order by|limit)\b",
                    line,
                    re.I,
                ):
                    sql_lines.append(line.rstrip())
            if sql_lines:
                joined = "\n".join(sql_lines)
                if not joined.rstrip().endswith(";"):
                    joined = joined.rstrip() + ";"
                return joined.strip()

    # 3) Plain fallback
    return raw_text.strip()


def normalize_python(code: str, preserve_doctest: bool = True) -> str:
    """
    Make Python code ready for compile():
    - Normalize newlines
    - Optionally preserve doctest prompts (>>>) inside docstrings
    - Dedent
    - Trim whitespace

    If preserve_doctest is True (default), we avoid stripping lines that
    begin with '>>>' so that doctest examples remain intact. This is
    important when the model returns examples embedded in docstrings.
    """

    if not code:
        return code

    code = code.replace("\r\n", "\n").replace("\r", "\n")
    if not preserve_doctest:
        code = re.sub(r"^\s*>>>\s?", "", code, flags=re.M)
    # If preserving doctest, still strip interpreter output prompts '...'
    # only when they are clearly continuation lines outside of strings would
    # require parsing; keep it simple and leave them as-is.
    code = textwrap.dedent(code)
    return code.strip()
