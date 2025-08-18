"""
Pytest configuration för CodeConductor
"""

import asyncio
import os
import random
from pathlib import Path

import pytest
import gc
import logging

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import torch
except Exception:
    torch = None

# Lägg till src i Python path
repo_root = Path(__file__).resolve().parent.parent
src_path = repo_root / "src"
if str(src_path) not in str(Path.cwd()):
    import sys

    sys.path.insert(0, str(src_path))


def pytest_sessionstart(session):
    """Sätt deterministiska seeds för alla tester"""
    seed = int(os.getenv("CODECONDUCTOR_SEED", "1337"))
    random.seed(seed)

    # Sätt numpy seed om tillgängligt
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    # Sätt torch seed om tillgängligt
    set_torch_seed(seed)

    print(f"🔒 Deterministic seeds set: {seed}")


# Sätt torch seed om tillgängligt
def set_torch_seed(seed: int = 42):
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

@pytest.fixture(autouse=True)
def _gpu_sanitizer():
    """
    Enforce mock/safe default for unit/CI; real GPU tests opt-in via -m gpu
    Also handles VRAM cleanup after each test
    """
    # Force safe defaults - override only for explicit GPU tests
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
