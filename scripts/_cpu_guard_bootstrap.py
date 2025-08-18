# CPU Guard Bootstrap: runs very early via .pth to enforce CPU-only and stub GPU modules
import os, sys, types

# 1) Environment hardening
os.environ.setdefault("CC_ULTRA_MOCK", "1")
os.environ.setdefault("CC_HARD_CPU_ONLY", "1")
os.environ.setdefault("CC_GPU_DISABLED", "1")
os.environ.setdefault("CC_TESTING_MODE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TORCH_USE_CUDA_DISABLED", "1")
os.environ.setdefault("VLLM_NO_CUDA", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")

BLOCKED_PREFIXES = (
    "torch",
    "transformers",
    "sentence_transformers",
    "GPUtil",
)


def _make_stub(fullname: str) -> types.ModuleType:
    mod = types.ModuleType(fullname)
    if fullname == "torch":

        class _Cuda:
            @staticmethod
            def is_available():
                return False

            def __getattr__(self, _):
                return False

        mod.cuda = _Cuda()
        mod.device = lambda *_a, **_k: "cpu"

        def _no(*_a, **_k):
            return None

        mod.no_grad = _no
        mod.inference_mode = _no
        mod.__version__ = "2.0.0+mock"
    elif fullname == "torch.cuda":
        stub = types.ModuleType("torch.cuda")
        stub.is_available = lambda: False
        stub.device_count = lambda: 0
        stub.current_device = lambda: -1
        return stub
    elif fullname.startswith("transformers"):
        mod.__version__ = "4.0.0+mock"
        mod.AutoTokenizer = type("MockTokenizer", (), {})
        mod.AutoModel = type("MockModel", (), {})
    elif fullname.startswith("sentence_transformers"):
        mod.__version__ = "2.0.0+mock"
        mod.SentenceTransformer = type("MockSentenceTransformer", (), {})
    elif fullname.startswith("GPUtil"):
        mod.getGPUs = lambda: []
        mod.GPUtil = type("MockGPUtil", (), {})
    return mod


# Pre-populate known names
for name in ("torch", "torch.cuda", "transformers", "sentence_transformers", "GPUtil"):
    if name in sys.modules:
        del sys.modules[name]
    sys.modules[name] = _make_stub(name)


class BlockGPUFinder:
    """MetaPathFinder that blocks GPU-heavy modules and returns stubs."""

    def find_spec(self, fullname, path, target=None):
        for prefix in BLOCKED_PREFIXES:
            if fullname == prefix or fullname.startswith(prefix + "."):
                return self._spec_for(fullname)
        return None

    def _spec_for(self, fullname: str):
        from importlib.machinery import ModuleSpec

        def _loader_exec(module):
            # replace module contents with stub
            stub = _make_stub(fullname)
            module.__dict__.update(stub.__dict__)

        return ModuleSpec(fullname, type("_CPUGuardLoader", (), {"exec_module": _loader_exec}))


# Install finder at highest priority
sys.meta_path.insert(0, BlockGPUFinder())

# Minimal log (avoid emojis for PS compatibility)
print(f"CPUGuard: enabled (CUDA_VISIBLE_DEVICES='{os.environ.get('CUDA_VISIBLE_DEVICES', '')}')")
