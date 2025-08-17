"""
Pytest configuration för CodeConductor
"""

import pytest
import asyncio
import gc
import os
import random
from pathlib import Path

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
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    print(f"🔒 Deterministic seeds set: {seed}")

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

@pytest.fixture
def gpu_monitor():
    """Monitor för VRAM-användning under tester"""
    try:
        import torch
        if torch.cuda.is_available():
            initial = torch.cuda.memory_allocated()
            yield
            final = torch.cuda.memory_allocated()
            leak = final - initial
            if leak > 100_000_000:  # 100MB threshold
                pytest.warning(f"Potential VRAM leak: {leak/1024**2:.1f}MB")
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
    
    if 'streamlit' not in sys.modules:
        mock_st = MagicMock()
        mock_st.session_state = MagicMock()
        sys.modules['streamlit'] = mock_st
        sys.modules['streamlit.session_state'] = mock_st.session_state
    
    yield
    
    # Cleanup
    if 'streamlit' in sys.modules:
        del sys.modules['streamlit']

@pytest.fixture
def tmp_artifacts(tmp_path):
    """Temporary artifacts directory för tester"""
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    return artifacts_dir


