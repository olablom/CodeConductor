#!/usr/bin/env python3
"""
Repository Map and Prompt Context Helper

Builds a lightweight JSON map of the repository using Tree-sitter (if available)
and provides helpers to retrieve top-K relevant paths/snippets to enrich prompts.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Optional import of TreeSitterAnalyzer
try:
    from codeconductor.analysis.tree_sitter_analyzer import (
        TreeSitterAnalyzer,
    )

    TS_AVAILABLE = True
except Exception as e:  # pragma: no cover
    TS_AVAILABLE = False
    logger.warning(f"Tree-sitter analyzer not available: {e}")


@dataclass
class RepoElement:
    type: str
    name: str
    file_path: str
    language: str
    line_start: int
    line_end: int


def _read_file_safe(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:  # pragma: no cover
        logger.debug(f"Failed to read {path}: {e}")
        return ""


def build_repo_map(project_root: str | Path) -> dict[str, Any]:
    """
    Build a repository map with file tree and code elements (if Tree-sitter is available).

    Returns dict:
      {
        "root": str,
        "files": ["relative/path.py", ...],
        "elements": [{type,name,file_path,language,line_start,line_end}, ...],
        "stats": {...}
      }
    """
    root = Path(project_root).resolve()
    files: list[str] = []
    elements: list[dict[str, Any]] = []

    exclude_dirs = {
        ".git",
        "node_modules",
        "venv",
        ".venv",
        "env",
        "__pycache__",
        "dist",
        "build",
        ".pytest_cache",
    }

    if TS_AVAILABLE:
        try:
            analyzer = TreeSitterAnalyzer()
        except Exception as e:  # pragma: no cover
            logger.warning(f"Could not initialize TreeSitterAnalyzer: {e}")
            analyzer = None
    else:
        analyzer = None

    for r, dirs, fs in os.walk(root):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for f in fs:
            p = Path(r) / f
            rel = str(p.relative_to(root))
            files.append(rel)
            # Collect elements for code files if analyzer available
            if analyzer is not None and p.suffix.lower() in {
                ".py",
                ".ts",
                ".tsx",
                ".js",
                ".jsx",
            }:
                try:
                    for el in analyzer.analyze_file(str(p)):
                        elements.append(
                            {
                                "type": el.type,
                                "name": el.name,
                                "file_path": rel,
                                "language": el.language,
                                "line_start": el.line_start,
                                "line_end": el.line_end,
                            }
                        )
                except Exception as e:  # pragma: no cover
                    logger.debug(f"Analyze failed for {p}: {e}")

    stats = {
        "total_files": len(files),
        "total_elements": len(elements),
        "tree_sitter": bool(analyzer is not None),
    }

    return {"root": str(root), "files": files, "elements": elements, "stats": stats}


def format_project_structure(repo_map: dict[str, Any], max_lines: int = 200) -> str:
    """
    Produce a markdown-like tree of the repository for prompt context.
    """
    lines: list[str] = ["Project Structure (trimmed):"]
    for path in sorted(repo_map.get("files", []))[:max_lines]:
        depth = path.count("/")
        indent = "  " * depth
        lines.append(f"{indent}- {path.split('/')[-1]}  ⟵ {path}")
    if len(repo_map.get("files", [])) > max_lines:
        lines.append(f"... (+{len(repo_map['files']) - max_lines} more)")
    return "\n".join(lines)


def _score_match(task: str, candidate: str) -> float:
    task_l = task.lower()
    cand_l = candidate.lower()
    score = 0.0
    for token in {t for t in task_l.replace("_", " ").replace("-", " ").split() if t}:
        if token in cand_l:
            score += 1.0
    # prefer shorter paths (likely closer to root)
    score += max(0.0, 2.0 - cand_l.count("/")) * 0.1
    return score


def get_top_k_paths(repo_map: dict[str, Any], task_description: str, k: int = 5) -> list[str]:
    """
    Rank file paths by naive keyword overlap with task description.
    """
    files = repo_map.get("files", [])
    ranked = sorted(files, key=lambda p: _score_match(task_description, p), reverse=True)
    return ranked[:k]


def extract_snippet(
    project_root: str | Path,
    rel_path: str,
    start: int = 1,
    end: int = 9999,
    pad: int = 5,
) -> str:
    """
    Extract a readable snippet around the specified line range; falls back to head.
    """
    root = Path(project_root)
    path = root / rel_path
    text = _read_file_safe(path)
    if not text:
        return ""
    lines = text.splitlines()
    s = max(0, min(len(lines), start - 1 - pad))
    e = min(len(lines), end - 1 + pad)
    snippet = "\n".join(lines[s:e])
    return snippet


def get_top_k_snippets(
    repo_map: dict[str, Any], task_description: str, k: int = 3, max_chars: int = 1500
) -> list[dict[str, Any]]:
    """
    Retrieve up to k code snippets relevant to the task by scoring elements and paths.
    """
    root = repo_map.get("root", ".")
    elements = repo_map.get("elements", [])

    scored_elements: list[tuple[float, dict[str, Any]]] = []
    for el in elements:
        key = f"{el.get('name', '')} {el.get('file_path', '')} {el.get('type', '')}"
        score = _score_match(task_description, key)
        if score > 0:
            scored_elements.append((score, el))

    top: list[dict[str, Any]] = []
    for _, el in sorted(scored_elements, key=lambda p: p[0], reverse=True)[:k]:
        snippet = extract_snippet(
            root, el["file_path"], el.get("line_start", 1), el.get("line_end", 1)
        )
        if snippet:
            snippet = snippet[:max_chars]
            top.append(
                {
                    "file": el["file_path"],
                    "name": el.get("name", ""),
                    "type": el.get("type", ""),
                    "language": el.get("language", ""),
                    "snippet": snippet,
                }
            )

    # Fallback: use top paths if no elements matched
    if not top:
        for rel in get_top_k_paths(repo_map, task_description, k=k):
            snip = extract_snippet(root, rel)
            if snip:
                top.append(
                    {
                        "file": rel,
                        "snippet": snip[:max_chars],
                        "type": "file",
                        "language": "",
                    }
                )

    return top


def build_prompt_context(
    repo_map: dict[str, Any],
    task_description: str,
    k_paths: int = 6,
    k_snippets: int = 3,
) -> str:
    """
    Build a concise markdown context block for prompts combining tree and snippets.
    """
    structure = format_project_structure(repo_map, max_lines=200)
    top_paths = get_top_k_paths(repo_map, task_description, k=k_paths)
    snippets = get_top_k_snippets(repo_map, task_description, k=k_snippets)

    parts: list[str] = []
    parts.append(structure)
    if top_paths:
        parts.append("\nTop relevant paths:")
        for p in top_paths:
            parts.append(f"- {p}")
    if snippets:
        parts.append("\nRelevant snippets:")
        for sn in snippets:
            header = (
                f"# {sn.get('file')} :: {sn.get('name', '').strip()} ({sn.get('type', '')})".strip()
            )
            parts.append(header)
            parts.append("```\n" + sn.get("snippet", "").strip() + "\n```")

    return "\n".join(parts)
