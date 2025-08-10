"""
CodeConductor - Personal AI Development Platform

Minimal package initializer.

Avoid importing heavy UI modules (Streamlit/Tauri) at import time so that
CLI tools and backends can import `codeconductor.*` without side effects.
"""

from __future__ import annotations

import os

# Enforce local-first privacy: disable anonymous telemetry where supported
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")  # Chroma
os.environ.setdefault("POSTHOG_DISABLED", "1")  # PostHog client
os.environ.setdefault("LANGCHAIN_ENDPOINT", "")  # LangSmith off by default
os.environ.setdefault("LANGSMITH_TRACING", "false")
os.environ.setdefault("OPENAI_API_KEY", "")  # prevent accidental usage

__version__ = "0.1.0"
__author__ = "Ola Blom"
__email__ = "olablom@github.com"

# Export nothing heavy by default; submodules should be imported explicitly.
__all__: list[str] = []
