# --- HARD CPU-ONLY GUARD (måste ligga allra först) ---
import os

if os.getenv("CC_HARD_CPU_ONLY", "0") == "1":
    # Dölj alla CUDA-enheter för alla libbar (PyTorch, TensorFlow, vLLM, etc.)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("VLLM_NO_CUDA", "1")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")

    # För PyTorch: förhindra att CUDA initieras i efterhand
    os.environ.setdefault("TORCH_USE_CUDA_DSA", "0")  # defensivt
    os.environ.setdefault("GPU_FORCE_64BIT_PTR", "1")  # defensivt

    # Säker mock: tala om för vår kod att absolut inte röra GPU
    os.environ["CC_GPU_DISABLED"] = "1"
    os.environ["CC_TESTING_MODE"] = "1"
# ------------------------------------------------------

"""
Test configuration and fixtures for CodeConductor
"""

import asyncio
import gc
import logging
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
    print(f"Deterministic seeds set: {seed}")

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
        logger.info("GPU test: allowing real GPU access (CC_GPU_DISABLED=0)")
    else:
        # Non-GPU test - enforce safe mock mode
        os.environ["CC_TESTING_MODE"] = "1"
        os.environ["CC_GPU_DISABLED"] = "1"
        logger.info("GPU sanitizer: enforcing safe mode (CC_GPU_DISABLED=1)")

    yield

    # Teardown: städa VRAM och Python-refs oavsett markering
    logger.info("GPU sanitizer: cleaning up after test")
    gc.collect()

    if torch and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("GPU memory cleaned")
        except Exception as e:
            logger.warning(f"GPU cleanup failed: {e}")
    else:
        logger.info("No CUDA available, skipping GPU cleanup")

    # Reset to safe mode after GPU test
    if is_gpu_test:
        os.environ["CC_GPU_DISABLED"] = "1"
        logger.info("GPU test: reset to safe mode (CC_GPU_DISABLED=1)")


@pytest.fixture(autouse=True)
def _hard_gpu_guard(request):
    """
    Hard guard: block all real GPU calls when CC_GPU_DISABLED=1
    This prevents any accidental real GPU usage in mock mode
    """
    # Check if test is explicitly marked for GPU usage
    is_gpu_test = request.node.get_closest_marker("gpu") is not None

    if os.getenv("CC_GPU_DISABLED") == "1" and not is_gpu_test:

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
                "MockLLM",
                (),
                {"generate": lambda *a, **k: [{"outputs": [{"text": "[MOCKED]"}]}]},
            )()
        except Exception:
            pass

        # Blockera transformers om det finns
        try:
            import transformers

            # Mocka AutoModelForCausalLM till att alltid returnera mock
            transformers.AutoModelForCausalLM.from_pretrained = (
                lambda *args, **kwargs: type(
                    "MockModel",
                    (),
                    {
                        "generate": lambda *a, **k: type(
                            "MockOutput", (), {"sequences": [[0, 1, 2]]}
                        )()
                    },
                )()
            )
        except Exception:
            pass

        logger.info("Hard GPU guard: blocked all real GPU calls")
    elif is_gpu_test:
        logger.info("GPU test: allowing real GPU calls")

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
                logger.warning(f"GPU memory increased: {initial} -> {final} bytes")
        else:
            yield
    except ImportError:
        yield


@pytest.fixture(scope="function")
def event_loop():
    """Create event loop for async tests with proper Windows handling"""
    if sys.platform.startswith("win"):
        # Use WindowsSelectorEventLoopPolicy for Windows
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
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


def pytest_sessionstart(session):
    if os.getenv("CC_HARD_CPU_ONLY", "0") == "1":
        try:
            import torch

            assert (
                not torch.cuda.is_available()
            ), "CUDA blev tillgänglig i HARD CPU mode"
        except Exception:
            pass


# ---- Ultra-mock: stäng av embeddings + vector store helt i testläge ----
import os
import types

import pytest


def _dummy_embed(texts: list[str]) -> list[list[float]]:
    dim = 384  # neutral standarddim; koden ska inte bero på exakt värde
    return [[0.0] * dim for _ in texts]


class _DummyEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return _dummy_embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return _dummy_embed([text])[0]


class _DummyDoc:
    def __init__(self, page_content: str, metadata: dict | None = None):
        self.page_content = page_content
        self.metadata = metadata or {}


def _dummy_sim_search(query: str, k: int = 5, **_) -> list[tuple[_DummyDoc, float]]:
    # Liten, deterministisk, helt CPU-fri "träfflista"
    return [(_DummyDoc(f"[MOCK] {query} #{i + 1}"), 0.75) for i in range(k)]


@pytest.fixture(autouse=True)
def ultra_mock_vector_and_embeddings(monkeypatch):
    """
    Aktiveras när CC_ULTRA_MOCK=1 eller CC_HARD_CPU_ONLY=1 eller CC_GPU_DISABLED=1.
    - Mockar HuggingFace/SBERT/SentenceTransformer till en no-op CPU-embedder.
    - Mockar Chroma/FAISS/VectorStore .similarity_search* till en snabb stub.
    Robust med try/except så den inte kraschar om paket saknas.
    """
    if (
        os.getenv("CC_ULTRA_MOCK", "0") == "1"
        or os.getenv("CC_HARD_CPU_ONLY", "0") == "1"
        or os.getenv("CC_GPU_DISABLED", "0") == "1"
    ):
        # 1) Blockera alla vanliga embedder-klasser
        for mod_name, cls_name in [
            ("sentence_transformers", "SentenceTransformer"),
            ("langchain.embeddings.huggingface", "HuggingFaceEmbeddings"),
            ("langchain_community.embeddings.huggingface", "HuggingFaceEmbeddings"),
            (
                "langchain_community.embeddings.sentence_transformer",
                "SentenceTransformerEmbeddings",
            ),
        ]:
            try:
                mod = __import__(mod_name, fromlist=[cls_name])
                monkeypatch.setattr(
                    mod, cls_name, lambda *a, **k: _DummyEmbeddings(), raising=False
                )
            except Exception:
                pass

        # 2) Mocka typiska vector-store-klasser (Chroma/FAISS) till no-op
        for mod_name, cls_name in [
            ("langchain_community.vectorstores.chroma", "Chroma"),
            ("langchain.vectorstores.chroma", "Chroma"),
            ("langchain_community.vectorstores.faiss", "FAISS"),
            ("langchain.vectorstores.faiss", "FAISS"),
        ]:
            try:
                mod = __import__(mod_name, fromlist=[cls_name])
                vec_cls = getattr(mod, cls_name, None)
                if vec_cls:
                    # Gör alla similarity-search varianter till snabba stubs
                    for fn in [
                        "similarity_search",
                        "similarity_search_by_vector",
                        "similarity_search_with_score",
                        "similarity_search_with_relevance_scores",
                        "max_marginal_relevance_search",
                    ]:
                        if hasattr(vec_cls, fn):
                            monkeypatch.setattr(
                                vec_cls,
                                fn,
                                lambda *a, **k: _dummy_sim_search(
                                    a[1] if len(a) > 1 else k.get("query", "")
                                ),
                                raising=False,
                            )
            except Exception:
                pass

        # 3) Om något i er kod skapar "egna" embeddings eller vector stores, försök hooka generiska namn
        try:

            # Exempel: codeconductor.rag.embeddings.get_embedder -> dummy
            for path, name in [
                ("codeconductor.rag.embeddings", "get_embedder"),
                ("codeconductor.rag.vector_store", "get_vector_store"),
            ]:
                try:
                    mod = __import__(path, fromlist=[name])
                    if hasattr(mod, "get_embedder"):
                        monkeypatch.setattr(
                            mod,
                            "get_embedder",
                            lambda *a, **k: _DummyEmbeddings(),
                            raising=False,
                        )
                    # vector store kan vara fabrik; returnera enkelt objekt med sökmetoder
                    if hasattr(mod, "get_vector_store"):
                        dummy_vs = types.SimpleNamespace(
                            similarity_search=lambda query, k=5, **_: [
                                d for d, _ in _dummy_sim_search(query, k)
                            ],
                            similarity_search_with_relevance_scores=_dummy_sim_search,
                        )
                        monkeypatch.setattr(
                            mod,
                            "get_vector_store",
                            lambda *a, **k: dummy_vs,
                            raising=False,
                        )
                except Exception:
                    pass
        except Exception:
            pass


# --- hard VRAM cleanup at end of session ---
import asyncio as _asyncio
import os as _os
import sys as _sys


def _cpu_only_env():
    # Mirrors our other guards; keeps this idempotent
    _os.environ.setdefault("CC_TESTING_MODE", "1")
    _os.environ.setdefault("CC_GPU_DISABLED", "1")
    _os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    # Block LM Studio discovery during tests
    _os.environ.setdefault("ENGINE_BACKENDS", "ollama")
    _os.environ.setdefault("LMSTUDIO_DISABLE", "1")
    _os.environ.setdefault("LMSTUDIO_CLI_DISABLE", "1")
    _os.environ.setdefault("DISCOVERY_DISABLE", "1")
    _os.environ.setdefault("ENGINE_DISCOVERY_MODE", "preloaded_only")


def _unload_sync():
    from codeconductor.ensemble.model_manager import ModelManager

    async def _cleanup():
        mm = ModelManager()
        try:
            # best-effort: unload via LM Studio if enabled; otherwise noop
            n = await mm.emergency_unload_all()
        except Exception:
            n = -1
        return n

    try:
        return _asyncio.run(_cleanup())
    except RuntimeError:
        # inside existing loop (rare under pytest); fallback
        return -2


def pytest_sessionfinish(session, exitstatus):
    _cpu_only_env()
    try:
        n = _unload_sync()
        _sys.stderr.write(f"\n[pytest teardown] emergency_unload_all() -> {n}\n")
    except Exception as e:
        _sys.stderr.write(f"\n[pytest teardown] unload failed: {e}\n")
