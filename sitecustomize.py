# sitecustomize.py  — laddas automatiskt av Python om den ligger på sys.path
import os, sys, types

# 1) Hårda CPU-only flaggor (innan allt annat)
os.environ.setdefault("CC_ULTRA_MOCK", "1")
os.environ.setdefault("CC_HARD_CPU_ONLY", "1")
os.environ.setdefault("CC_GPU_DISABLED", "1")
os.environ.setdefault("CC_TESTING_MODE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")      # döljer GPU helt
os.environ.setdefault("HF_HUB_OFFLINE", "1")          # inga modelldownloads
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TORCH_USE_CUDA_DISABLED", "1")
os.environ.setdefault("VLLM_NO_CUDA", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")

# 2) Lätta stubbar som ersätter GPU-tunga paket
def _make_stub(name: str):
    mod = types.ModuleType(name)
    # Minimalt API som ofta anropas
    if name == "torch":
        class _Cuda:
            @staticmethod
            def is_available(): return False
            def __getattr__(self, _): return False
        mod.cuda = _Cuda()
        mod.device = lambda *_a, **_k: "cpu"
        def _no(*_a, **_k): return None
        mod.no_grad = _no
        mod.inference_mode = _no
        mod.__version__ = "2.0.0+mock"
    elif name == "torch.cuda":
        mod.is_available = lambda: False
        mod.device_count = lambda: 0
        mod.current_device = lambda: -1
    elif name == "transformers":
        mod.__version__ = "4.0.0+mock"
        mod.AutoTokenizer = type("MockTokenizer", (), {})
        mod.AutoModel = type("MockModel", (), {})
    elif name == "sentence_transformers":
        mod.__version__ = "2.0.0+mock"
        mod.SentenceTransformer = type("MockSentenceTransformer", (), {})
    elif name == "GPUtil":
        mod.getGPUs = lambda: []
        mod.GPUtil = type("MockGPUtil", (), {})
    
    sys.modules[name] = mod

# 3) Pre-populera sys.modules med stubbar OCH override befintliga
for pkg in ["torch", "torch.cuda", "transformers", "sentence_transformers", "GPUtil"]:
    _make_stub(pkg)  # Alltid skapa stub, även om modulen redan finns

# 4) ULTRA-AGGRESSIV: Override torch.cuda.is_available med monkey patching
try:
    # Om torch redan finns, monkey patch cuda.is_available
    if "torch" in sys.modules:
        torch_mod = sys.modules["torch"]
        if hasattr(torch_mod, "cuda") and hasattr(torch_mod.cuda, "is_available"):
            # Spara original för debugging
            original_is_available = torch_mod.cuda.is_available
            # Override med False
            torch_mod.cuda.is_available = lambda: False
            print(f"🔒 Overrode torch.cuda.is_available from {original_is_available} to False")
        else:
            print(f"🔒 torch.cuda.is_available not found, created stub")
    else:
        print(f"🔒 torch not in sys.modules yet")
except Exception as e:
    print(f"⚠️  Error overriding torch.cuda.is_available: {e}")

# 5) Logga att sitecustomize aktiverades
print(f"🔒 sitecustomize.py: GPU libraries blocked for {sys.executable}")
print(f"   CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
print(f"   CC_HARD_CPU_ONLY: {os.environ.get('CC_HARD_CPU_ONLY', 'NOT SET')}")

# 6) Verifiera att override fungerade
try:
    if "torch" in sys.modules:
        torch_mod = sys.modules["torch"]
        if hasattr(torch_mod, "cuda") and hasattr(torch_mod.cuda, "is_available"):
            result = torch_mod.cuda.is_available()
            print(f"   VERIFICATION: torch.cuda.is_available() = {result}")
        else:
            print(f"   VERIFICATION: torch.cuda.is_available not accessible")
    else:
        print(f"   VERIFICATION: torch not in sys.modules")
except Exception as e:
    print(f"   VERIFICATION ERROR: {e}")
