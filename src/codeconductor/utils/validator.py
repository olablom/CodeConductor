"""
Validation and repair prompt utilities for generated Python code.

Responsibilities:
- Structural validation (syntax, docstring presence/closure, doctest markers)
- Doctest execution against a saved file
- Building a concise repair prompt for the model

All helpers are intentionally lightweight and without Streamlit deps.
"""

from __future__ import annotations

import ast
import doctest
import re
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path

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
    # Policy checks
    header_ok: bool
    trailer_required: bool
    trailer_ok: bool
    ok: bool
    errors: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "syntax_ok": self.syntax_ok,
            "has_module_docstring": self.has_module_docstring,
            "docstring_closed": self.docstring_closed,
            "has_doctest_markers": self.has_doctest_markers,
            "doctest_tests": self.doctest_tests,
            "doctest_failures": self.doctest_failures,
            "doctest_passed": self.doctest_passed,
            "header_ok": self.header_ok,
            "trailer_required": self.trailer_required,
            "trailer_ok": self.trailer_ok,
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


def _check_exact_header(code: str) -> tuple[bool, list[str]]:
    """Return (header_ok, header_errors). Enforces first three lines exactly and
    that nothing precedes the shebang (no BOM/blank/comment)."""
    required = [
        "#!/usr/bin/env python3",
        "# -*- coding: utf-8 -*-",
        "# generated.py",
    ]
    errs: list[str] = []

    # Detect BOM or any leading whitespace/comment before shebang
    if code.startswith("\ufeff"):
        errs.append("BOM detected before shebang; remove BOM")

    lines = (code or "").splitlines()
    if not lines:
        errs.append("empty file (missing header)")
        return False, errs

    # Anything before shebang? i.e., first non-empty character must start the shebang
    if not code.startswith(required[0]):
        errs.append(
            "invalid header: line 1 must be '#!/usr/bin/env python3' with nothing before it"
        )

    # Check next two lines exactly
    if len(lines) < 3:
        errs.append("invalid header: file must have at least three lines for header")
        return False, errs
    if lines[:3] != required:
        # Produce specific diagnostics
        exp = " | ".join(required)
        got = " | ".join(lines[:3])
        errs.append(f"invalid header: expected first three lines: {exp}; got: {got}")
        return False, errs

    return True, []


def _check_required_trailer(code: str, require_trailer: bool) -> tuple[bool, list[str]]:
    """If trailer is required, ensure last two lines are exactly the markers."""
    if not require_trailer:
        return True, []
    # Ignore trailing empty lines
    lines = (code or "").splitlines()
    while lines and lines[-1].strip() == "":
        lines.pop()
    if len(lines) < 2:
        return False, [
            "invalid trailer: missing required last two lines '# SYNTAX_ERROR BELOW' and '('",
        ]
    if lines[-2] != "# SYNTAX_ERROR BELOW" or lines[-1] != "(":
        return False, [
            "invalid trailer: expected last two lines to be exactly '# SYNTAX_ERROR BELOW' and '('",
        ]
    return True, []


def validate_python_code(
    code: str,
    run_doctests: bool = True,
    *,
    enforce_header: bool = True,
    require_trailer: bool | None = None,
    task_input: str | None = None,
) -> ValidationReport:
    errors: list[str] = []

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

    # Policy: exact header
    header_ok = True
    if enforce_header:
        header_ok, header_errors = _check_exact_header(code or "")
        errors.extend(header_errors)

    # Policy: conditional trailer
    trailer_needed = bool(require_trailer)
    if require_trailer is None and task_input is not None:
        trailer_needed = "# SYNTAX_ERROR BELOW" in task_input
    trailer_ok = True
    if trailer_needed:
        trailer_ok, trailer_errors = _check_required_trailer(code or "", True)
        errors.extend(trailer_errors)
    else:
        # When trailer is NOT required (e.g., post-repair), ensure marker is absent
        if "# SYNTAX_ERROR BELOW" in (code or ""):
            trailer_ok = False
            errors.append(
                "trailer present when not required: remove '# SYNTAX_ERROR BELOW' and '(' from EOF"
            )

    ok = (
        syntax_ok
        and has_module_docstring
        and docstring_closed
        and has_doctest_markers
        and doctest_tests >= 1
        and doctest_failures == 0
        and header_ok
        and (trailer_ok if trailer_needed else True)
    )

    return ValidationReport(
        syntax_ok=syntax_ok,
        has_module_docstring=has_module_docstring,
        docstring_closed=docstring_closed,
        has_doctest_markers=has_doctest_markers,
        doctest_tests=doctest_tests,
        doctest_failures=doctest_failures,
        doctest_passed=doctest_passed,
        header_ok=header_ok,
        trailer_required=trailer_needed,
        trailer_ok=trailer_ok,
        ok=ok,
        errors=errors,
    )


def run_doctest_on_file(path: Path, timeout_seconds: int = 15) -> tuple[bool, str]:
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


def _diagnose_header_trailer(original_code: str) -> str:
    """Return short diagnostics for header/trailer presence and correctness."""
    lines = (original_code or "").splitlines()
    header_found = lines[:3]
    trailer_found = lines[-2:] if len(lines) >= 2 else lines
    diag = [
        f"Header(first 3): {' | '.join(header_found) if header_found else '(missing)'}",
        f"Trailer(last 2): {' | '.join(trailer_found) if trailer_found else '(missing)'}",
    ]
    return "\n".join(diag)


def build_repair_prompt(
    original_code: str,
    issues: ValidationReport | dict[str, object],
    doctest_output: str | None = None,
    filename_hint: str = "generated.py",
    *,
    require_trailer_by_task: bool = False,
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
    # Policy guidance
    header_policy = (
        "- Place the exact three header lines at file start; nothing before them.\n"
        "  1) #!/usr/bin/env python3\n"
        "  2) # -*- coding: utf-8 -*-\n"
        "  3) # generated.py\n"
    )
    # Decide trailer instruction: if the current code already contains the syntax-error trailer
    # we want to REMOVE it in the repaired version. Otherwise, if the task requires it, we instruct to append it.
    has_trailer_marker = "# SYNTAX_ERROR BELOW" in (original_code or "")
    if has_trailer_marker:
        trailer_policy = (
            "- Remove the two trailer lines from EOF if present (they are only for triggering a SyntaxError):\n"
            "  '# SYNTAX_ERROR BELOW' and '(' must NOT remain in the final file.\n"
        )
    elif require_trailer_by_task:
        trailer_policy = "- Append the two exact trailer lines at EOF: first '# SYNTAX_ERROR BELOW' then '(' (on its own line).\n"
    else:
        trailer_policy = ""
    doctest_note = (
        "\n\nDoctest failures:\n" + doctest_output.strip() if doctest_output else ""
    )
    diagnosis = _diagnose_header_trailer(original_code)
    return (
        "You are fixing a Python module. Return ONLY one fenced python code block for the file "
        f"named {filename_hint}. Preserve public function names and signatures when possible.\n\n"
        "Fix these issues:\n" + checklist + emphasis + doctest_note + "\n\n"
        "Requirements:\n"
        + header_policy
        + trailer_policy
        + "- Include a top-level module docstring that contains at least one doctest example (>>>).\n"
        + "- Ensure all triple-quoted strings are closed.\n"
        + "- Ensure valid Python (passes ast.parse).\n"
        + "- If doctests are present, include an if __name__ == '__main__': block that runs doctest.testmod().\n"
        + "- No extra prose. No explanations. Only the code block.\n\n"
        + "Header/Trailer diagnostics:\n"
        + diagnosis
        + "\n\n"
        + "Current code:\n```python\n"
        + original_code.strip()
        + "\n```\n"
    )
