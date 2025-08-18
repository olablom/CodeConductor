"""
Test configuration and fixtures for CodeConductor
"""

import asyncio
import gc
import logging
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import pytest and torch if available
try:
    import pytest
except ImportError:
    pytest = None

try:
    import torch
except ImportError:
    torch = None


# Register markers and set deterministic seeds
def pytest_configure(config):
    """Register custom pytest markers and set deterministic seeds"""
    # Register markers
    config.addinivalue_line(
        "markers",
        "gpu: marks tests that require real GPU hardware (run locally)",
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "vllm: marks tests that require vLLM")
    config.addinivalue_line("markers", "windows: marks tests that are Windows-specific")
    config.addinivalue_line("markers", "linux: marks tests that are Linux-specific")
    config.addinivalue_line("markers", "asyncio: marks tests as async (pytest-asyncio)")

    # Deterministic seeds
    seed = int(os.getenv("CODECONDUCTOR_SEED", "1337"))
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"🔒 Deterministic seeds set: {seed}")

    try:
        import random

        random.seed(seed)
    except Exception:
        pass

    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch as _torch

        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed(seed)
            _torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _gpu_sanitizer(request):
    """
    Enforce mock/safe default for unit/CI; real GPU tests opt-in via -m gpu
    Also handles VRAM cleanup after each test
    """
    # Check if test is explicitly marked for GPU usage
    is_gpu_test = request.node.get_closest_marker("gpu") is not None

    if is_gpu_test:
        # GPU test - allow real GPU usage but still enforce cleanup
        os.environ["CC_TESTING_MODE"] = "1"
        os.environ["CC_GPU_DISABLED"] = "0"
        logger.info("🎮 GPU test: allowing real GPU access (CC_GPU_DISABLED=0)")
    else:
        # Non-GPU test - enforce safe mock mode
        os.environ["CC_TESTING_MODE"] = "1"
        os.environ["CC_GPU_DISABLED"] = "1"
        logger.info("🔒 GPU sanitizer: enforcing safe mode (CC_GPU_DISABLED=1)")

    yield

    # Teardown: städa VRAM och Python-refs oavsett markering
    logger.info("🧹 GPU sanitizer: cleaning up after test")
    gc.collect()

    if torch and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("✅ GPU memory cleaned")
        except Exception as e:
            logger.warning(f"⚠️ GPU cleanup failed: {e}")
    else:
        logger.info("ℹ️ No CUDA available, skipping GPU cleanup")

    # Reset to safe mode after GPU test
    if is_gpu_test:
        os.environ["CC_GPU_DISABLED"] = "1"
        logger.info("🔒 GPU test: reset to safe mode (CC_GPU_DISABLED=1)")


@pytest.fixture(autouse=True)
def _hard_gpu_guard(request):
    """
    Hard guard: block all real GPU calls when CC_GPU_DISABLED=1
    This prevents any accidental real GPU usage in mock mode
    """
    # Check if test is explicitly marked for GPU usage
    is_gpu_test = request.node.get_closest_marker("gpu") is not None

    if os.getenv("CC_GPU_DISABLED") == "1" and not is_gpu_test:
        import subprocess
        import pytest

        # Blockera alla subprocess-anrop (nvidia-smi, etc.)
        def _forbid_subprocess(*args, **kwargs):
            raise AssertionError(f"subprocess called in mock mode: {args} {kwargs}")

        # Blockera torch.cuda om det finns
        try:
            import torch

            if hasattr(torch, "cuda"):
                # Mocka torch.cuda till att alltid returnera False
                torch.cuda.is_available = lambda: False
                torch.cuda.device_count = lambda: 0
                torch.cuda.current_device = lambda: -1
                torch.cuda.get_device_name = lambda device: "mock-gpu"
                torch.cuda.memory_allocated = lambda device: 0
                torch.cuda.memory_reserved = lambda device: 0
                torch.cuda.max_memory_allocated = lambda device: 0
                torch.cuda.max_memory_reserved = lambda device: 0
        except Exception:
            pass

        # Blockera vLLM om det finns
        try:
            import vllm

            # Mocka vLLM till att alltid returnera mock
            vllm.LLM = lambda *args, **kwargs: type(
                "MockLLM", (), {"generate": lambda *a, **k: [{"outputs": [{"text": "[MOCKED]"}]}]}
            )()
        except Exception:
            pass

        # Blockera transformers om det finns
        try:
            import transformers

            # Mocka AutoModelForCausalLM till att alltid returnera mock
            transformers.AutoModelForCausalLM.from_pretrained = lambda *args, **kwargs: type(
                "MockModel",
                (),
                {"generate": lambda *a, **k: type("MockOutput", (), {"sequences": [[0, 1, 2]]})()},
            )()
        except Exception:
            pass

        logger.info("🔒 Hard GPU guard: blocked all real GPU calls")
    elif is_gpu_test:
        logger.info("🎮 GPU test: allowing real GPU calls")

    yield


@pytest.fixture(autouse=True)
def cleanup_gpu():
    """Auto-cleanup GPU mellan tester"""
    yield
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


@pytest.fixture(autouse=True)
def gpu_monitor():
    """Monitor GPU usage during tests"""
    try:
        import torch

        if torch.cuda.is_available():
            initial = torch.cuda.memory_allocated()
            yield
            final = torch.cuda.memory_allocated()
            if final > initial:
                logger.warning(f"⚠️ GPU memory increased: {initial} -> {final} bytes")
        else:
            yield
    except ImportError:
        yield


@pytest.fixture(scope="session")
def event_loop():
    """Skapa en event loop för async tester"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Mocka Streamlit i tester
@pytest.fixture(autouse=True)
def mock_streamlit():
    """Mocka Streamlit för tester"""
    import sys
    from unittest.mock import MagicMock

    if "streamlit" not in sys.modules:
        mock_st = MagicMock()
        mock_st.session_state = MagicMock()
        sys.modules["streamlit"] = mock_st
        sys.modules["streamlit.session_state"] = mock_st.session_state

    yield

    # Cleanup
    if "streamlit" in sys.modules:
        del sys.modules["streamlit"]


@pytest.fixture
def tmp_artifacts(tmp_path):
    """Temporary artifacts directory för tester"""
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    return artifacts_dir
