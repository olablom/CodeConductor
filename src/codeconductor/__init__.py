"""
CodeConductor - Personal AI Development Platform

Minimal package initializer.

Avoid importing heavy UI modules (Streamlit/Tauri) at import time so that
CLI tools and backends can import `codeconductor.*` without side effects.
"""

from __future__ import annotations

import os

# --- HARD CPU-ONLY GUARD (måste ligga allra först) ---
if os.getenv("CC_HARD_CPU_ONLY", "0") == "1":
    # Dölj alla CUDA-enheter för alla libbar (PyTorch, TensorFlow, vLLM, etc.)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("VLLM_NO_CUDA", "1")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")

    # För PyTorch: förhindra att CUDA initieras i efterhand
    os.environ.setdefault("TORCH_USE_CUDA_DISABLED", "0")  # defensivt
    os.environ.setdefault("GPU_FORCE_64BIT_PTR", "1")  # defensivt

    # Säker mock: tala om för vår kod att absolut inte röra GPU
    os.environ["CC_GPU_DISABLED"] = "1"
    os.environ["CC_TESTING_MODE"] = "1"
# ------------------------------------------------------

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
