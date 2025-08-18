#!/usr/bin/env python3
"""
Minimal example to demonstrate mock vs GPU-marked tests.

- test_mock_default: runs in mock mode (no marker)
- test_real_gpu_marked: runs only when selecting -m gpu
"""

import os
import pytest
import asyncio


def test_mock_default():
    """Default mock-mode: CC_GPU_DISABLED should be '1' and engine returns mocked content."""
    from codeconductor.ensemble.ensemble_engine import EnsembleEngine

    # Mock is enforced by conftest for unmarked tests
    assert os.getenv("CC_GPU_DISABLED") == "1"

    engine = EnsembleEngine()
    coro = engine.process_request(task_description="hello", timeout=5.0)
    assert hasattr(coro, "__await__")

    result = asyncio.run(coro)

    assert "generated_code" in result
    assert "[MOCKED]" in result["generated_code"]
    assert result["model_used"] == "ensemble-mock"


@pytest.mark.gpu
def test_real_gpu_marked():
    """GPU-marked: only runs when selecting -m gpu and CC_GPU_DISABLED is 0."""
    assert os.getenv("CC_GPU_DISABLED") == "0", "GPU tests must run with CC_GPU_DISABLED=0"

    # Optional: if torch is available, assert CUDA visibility; otherwise skip gracefully
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available on this system")
    except Exception:
        pytest.skip("PyTorch not available for GPU validation")

    # If we got here, environment is GPU-enabled
    assert True
