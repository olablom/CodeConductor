"""
Validation and repair prompt utilities for generated Python code.

Responsibilities:
- Structural validation (syntax, docstring presence/closure, doctest markers)
- Doctest execution against a saved file
- Building a concise repair prompt for the model

All helpers are intentionally lightweight and without Streamlit deps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ast
import re
import subprocess
import sys
import types
import doctest


_TRIPLE_QUOTE_RE = re.compile(r"([\"]{3}|[\']{3})")
_DOCTEST_MARKER_RE = re.compile(r"^\s*>>>\s*", flags=re.M)


@dataclass
class ValidationReport:
    syntax_ok: bool
    has_module_docstring: bool
    docstring_closed: bool
    has_doctest_markers: bool
    doctest_tests: int
    doctest_failures: int
    doctest_passed: bool
    ok: bool
    errors: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "syntax_ok": self.syntax_ok,
            "has_module_docstring": self.has_module_docstring,
            "docstring_closed": self.docstring_closed,
            "has_doctest_markers": self.has_doctest_markers,
            "doctest_tests": self.doctest_tests,
            "doctest_failures": self.doctest_failures,
            "doctest_passed": self.doctest_passed,
            "ok": self.ok,
            "errors": list(self.errors),
        }


def _has_balanced_triple_quotes(code: str) -> bool:
    if not code:
        return True
    # Count occurrences of each delimiter separately to reduce false positives
    for delim in ('"""', "'''"):
        if code.count(delim) % 2 != 0:
            return False
    return True


def _module_docstring_present(code: str) -> bool:
    try:
        module = ast.parse(code)
        return ast.get_docstring(module) is not None
    except Exception:
        # If it does not parse, approximate by checking first non-empty, non-comment block
        # for a triple-quoted string
        lines = [ln.rstrip() for ln in (code or "").splitlines()]
        # Skip shebang and encoding cookies and comments/blank
        i = 0
        while i < len(lines) and (
            lines[i].strip() == ""
            or lines[i].lstrip().startswith("#")
            or lines[i].lstrip().startswith("#!/")
        ):
            i += 1
        if i < len(lines):
            return lines[i].lstrip().startswith(('"""', "'''"))
        return False


def run_doctest_on_code(code: str) -> tuple[int, int, str]:
    """Execute doctest on code string. Returns (tests, failures, output)."""
    try:
        module = types.ModuleType("_cc_generated")
        exec(compile(code, "<generated>", "exec"), module.__dict__)
        runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
        finder = doctest.DocTestFinder(exclude_empty=False)
        for test in finder.find(module):
            runner.run(test)
        results = runner.summarize(verbose=False)
        tests_run = results.attempted
        failures = results.failed
        out_text = f"doctest: attempted={tests_run} failed={failures}"
        return tests_run, failures, out_text
    except Exception as e:
        return 0, 1, f"doctest error: {e}"


def validate_python_code(code: str, run_doctests: bool = True) -> ValidationReport:
    errors: List[str] = []

    # Syntax
    syntax_ok = True
    try:
        ast.parse(code or "")
    except Exception as e:
        syntax_ok = False
        errors.append(f"ast.parse: {e}")

    # Docstring
    has_module_docstring = _module_docstring_present(code or "")
    if not has_module_docstring:
        errors.append("missing module docstring")

    docstring_closed = _has_balanced_triple_quotes(code or "")
    if not docstring_closed:
        errors.append("unterminated triple-quoted string (docstring likely open)")

    # Doctest markers
    has_doctest_markers = bool(_DOCTEST_MARKER_RE.search(code or ""))
    doctest_tests = 0
    doctest_failures = 0
    doctest_passed = False
    if not has_doctest_markers:
        errors.append("no doctest markers (>>> not found)")

    if run_doctests and syntax_ok:
        tests, fails, _ = run_doctest_on_code(code or "")
        doctest_tests, doctest_failures = tests, fails
        doctest_passed = (fails == 0) and (tests >= 1)
        if tests == 0:
            errors.append(
                "No doctests executed (test_count=0). Ensure >>> doctest examples are present in the module docstring."
            )
        if fails > 0:
            errors.append(f"Doctest failures: {fails}")

    ok = (
        syntax_ok
        and has_module_docstring
        and docstring_closed
        and has_doctest_markers
        and doctest_tests >= 1
        and doctest_failures == 0
    )

    return ValidationReport(
        syntax_ok=syntax_ok,
        has_module_docstring=has_module_docstring,
        docstring_closed=docstring_closed,
        has_doctest_markers=has_doctest_markers,
        doctest_tests=doctest_tests,
        doctest_failures=doctest_failures,
        doctest_passed=doctest_passed,
        ok=ok,
        errors=errors,
    )


def run_doctest_on_file(path: Path, timeout_seconds: int = 15) -> Tuple[bool, str]:
    """Run doctest against a saved Python file; return (ok, output)."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "doctest", "-v", str(path)],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        ok = proc.returncode == 0
        out = (proc.stdout or "") + (proc.stderr or "")
        return ok, out
    except Exception as e:
        return False, f"doctest error: {e}"


def build_repair_prompt(
    original_code: str,
    issues: ValidationReport | Dict[str, object],
    doctest_output: Optional[str] = None,
    filename_hint: str = "generated.py",
) -> str:
    """
    Construct a concise repair prompt instructing the model to fix issues and
    return only a single fenced python block.
    """
    if isinstance(issues, ValidationReport):
        issue_list = issues.errors
    else:
        issue_list = [str(x) for x in (issues.get("errors") or [])]

    checklist = "\n".join(f"- {it}" for it in issue_list) or "- syntax/structure issues"
    emphasis = ""
    if isinstance(issues, ValidationReport):
        if (not issues.has_doctest_markers) or issues.doctest_tests == 0:
            emphasis = (
                "\n\nYour previous output contained no doctests. Add doctests inside the module docstring at the very top,"
                " using lines starting with >>> and exact expected outputs. Close the triple-quoted docstring properly."
            )
    doctest_note = (
        "\n\nDoctest failures:\n" + doctest_output.strip() if doctest_output else ""
    )
    return (
        "You are fixing a Python module. Return ONLY one fenced python code block for the file "
        f"named {filename_hint}. Preserve public function names and signatures when possible.\n\n"
        "Fix these issues:\n" + checklist + emphasis + doctest_note + "\n\n"
        "Requirements:\n"
        "- Include a top-level module docstring that contains at least one doctest example (>>>).\n"
        "- Ensure all triple-quoted strings are closed.\n"
        "- Ensure valid Python (passes ast.parse).\n"
        "- If doctests are present, include an if __name__ == '__main__': block that runs doctest.testmod().\n"
        "- No extra prose. No explanations. Only the code block.\n\n"
        "Current code:\n```python\n" + original_code.strip() + "\n```\n"
    )
