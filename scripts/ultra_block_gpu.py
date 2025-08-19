#!/usr/bin/env python3
"""
Ultra-aggressive GPU blocker - prevents GPU libraries from importing
"""

import builtins
import os
import sys


def ultra_block_gpu():
    """Block GPU libraries at the deepest level possible"""

    print("🚨 ULTRA-AGGRESSIVE GPU BLOCKING ACTIVATED")

    # 1. Set environment variables
    os.environ["CC_HARD_CPU_ONLY"] = "1"
    os.environ["CC_GPU_DISABLED"] = "1"
    os.environ["CC_ULTRA_MOCK"] = "1"
    os.environ["CC_TESTING_MODE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["VLLM_NO_CUDA"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TORCH_USE_CUDA_DISABLED"] = "1"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

    # 2. Block imports at builtin level
    original_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        # Block any GPU-related imports
        blocked_modules = [
            "torch",
            "torch.cuda",
            "torch.cuda.is_available",
            "transformers",
            "sentence_transformers",
            "sentence_transformers.SentenceTransformer",
            "langchain_community.embeddings.huggingface",
            "langchain_community.embeddings.huggingface.HuggingFaceEmbeddings",
            "langchain_community.vectorstores.chroma",
            "langchain_community.vectorstores.chroma.Chroma",
            "langchain_community.vectorstores.faiss",
            "langchain_community.vectorstores.faiss.FAISS",
            "cuda",
            "cudnn",
            "cublas",
            "cudart",
        ]

        for blocked in blocked_modules:
            if blocked in name:
                print(f"🚫 BLOCKED GPU IMPORT: {name}")
                # Return a mock module instead
                return create_mock_module(name)

        # Allow other imports
        return original_import(name, *args, **kwargs)

    # 3. Create mock modules for blocked imports
    def create_mock_module(name):
        class MockModule:
            def __init__(self, module_name):
                self.__name__ = module_name
                self.__file__ = f"<mocked {module_name}>"

            def __getattr__(self, attr):
                if attr == "cuda":
                    return MockModule(f"{self.__name__}.cuda")
                if attr == "is_available":
                    return lambda: False
                return MockModule(f"{self.__name__}.{attr}")

            def __call__(self, *args, **kwargs):
                return MockModule(f"{self.__name__}()")

            def __bool__(self):
                return False

            def __str__(self):
                return f"<MockModule {self.__name__}>"

        return MockModule(name)

    # 4. Replace builtin import
    builtins.__import__ = blocked_import

    # 5. Pre-populate sys.modules with mocks
    for module_name in ["torch", "transformers", "sentence_transformers"]:
        sys.modules[module_name] = create_mock_module(module_name)

    print("✅ GPU imports blocked at builtin level")
    print("✅ Mock modules created")

    return blocked_import


def test_blocking():
    """Test that GPU imports are actually blocked"""

    print("\n🧪 Testing GPU import blocking...")

    try:
        print("Testing: import torch")
        import torch

        print(f"✅ torch imported (type: {type(torch)})")
        print(f"   torch.cuda.is_available(): {getattr(torch, 'cuda', None)}")
    except Exception as e:
        print(f"❌ torch import failed: {e}")

    try:
        print("Testing: import transformers")
        import transformers

        print(f"✅ transformers imported (type: {type(transformers)})")
    except Exception as e:
        print(f"❌ transformers import failed: {e}")

    try:
        print("Testing: import sentence_transformers")
        import sentence_transformers

        print(
            f"✅ sentence_transformers imported (type: {type(sentence_transformers)})"
        )
    except Exception as e:
        print(f"❌ sentence_transformers import failed: {e}")


def main():
    print("🚨 ULTRA-AGGRESSIVE GPU BLOCKER")
    print("=" * 50)

    # Activate blocking
    blocked_import = ultra_block_gpu()

    # Test blocking
    test_blocking()

    print("\n🔒 All GPU imports should now be blocked")
    print("Try running your tests now - they should fail gracefully")


if __name__ == "__main__":
    main()
