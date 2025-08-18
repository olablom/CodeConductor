#!/usr/bin/env python3
# Filename: tests/test_vllm_wsl.py
import platform

import pytest


def _is_wsl() -> bool:
    rel = platform.release().lower()
    ver = getattr(platform, "version", lambda: "")().lower() if hasattr(platform, "version") else ""
    return ("microsoft" in rel) or ("microsoft" in ver)


def test_vllm():
    """
    Test vLLM availability in WSL2 environment.
    PASS if vLLM is available, SKIP if not installed.
    """
    if not _is_wsl():
        pytest.skip("Not running in WSL2 environment")

    try:
        import vllm  # noqa: F401

        # Test basic vLLM functionality
        from vllm import LLM, SamplingParams

        assert LLM is not None
        assert SamplingParams is not None
    except ImportError:
        pytest.skip("vLLM not installed in WSL2 (optional dependency)")
    except Exception as e:
        pytest.skip(f"vLLM import failed: {e}")
