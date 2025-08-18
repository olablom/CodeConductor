#!/usr/bin/env python3
# Filename: tests/test_vllm_windows.py
import platform

import pytest


def _is_wsl() -> bool:
    rel = platform.release().lower()
    ver = getattr(platform, "version", lambda: "")().lower() if hasattr(platform, "version") else ""
    return ("microsoft" in rel) or ("microsoft" in ver)


def test_vllm_availability():
    """
    vLLM stöds inte på *native* Windows. I WSL2/Linux/macOS är det valfritt.
    Testet PASSAR om vLLM finns; annars SKIP med tydlig orsak.
    Inga vilseledande prints.
    """
    system = platform.system()
    in_wsl = _is_wsl()

    # Native Windows (ej WSL): alltid skip
    if system == "Windows" and not in_wsl:
        pytest.skip("vLLM not supported on native Windows (use LM Studio/Ollama instead)")

    # WSL2: försök importera; om ej installerat → skip (optional dep)
    if in_wsl:
        try:
            import vllm  # noqa: F401
        except Exception:
            pytest.skip("vLLM not installed in WSL2 (optional dependency)")
        return  # import ok ⇒ test pass

    # Linux/macOS: försök importera; om ej installerat → skip (optional)
    try:
        import vllm  # noqa: F401
    except Exception:
        pytest.skip("vLLM not installed on this platform (optional dependency)")
    # import ok ⇒ test pass
