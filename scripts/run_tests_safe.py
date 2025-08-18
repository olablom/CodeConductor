#!/usr/bin/env python3
"""
Ultra-safe test runner that blocks GPU libraries before any imports
"""

import os
import sys
import subprocess
from pathlib import Path

def block_gpu_libraries():
    """Block GPU libraries at system level before any imports"""
    
    # Set environment variables that block GPU
    os.environ["CC_HARD_CPU_ONLY"] = "1"
    os.environ["CC_GPU_DISABLED"] = "1" 
    os.environ["CC_ULTRA_MOCK"] = "1"
    os.environ["CC_TESTING_MODE"] = "1"
    
    # Block CUDA completely
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["VLLM_NO_CUDA"] = "1"
    
    # Block HuggingFace/Transformers
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Block PyTorch GPU
    os.environ["TORCH_USE_CUDA_DISABLED"] = "1"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    
    # Block other GPU libraries
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    
    print("🔒 GPU libraries blocked at system level")
    print(f"CC_HARD_CPU_ONLY: {os.environ.get('CC_HARD_CPU_ONLY')}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

def mock_gpu_modules():
    """Mock GPU modules before they can be imported"""
    
    # Create mock modules
    class MockModule:
        def __getattr__(self, name):
            return MockModule()
        def __call__(self, *args, **kwargs):
            return MockModule()
        def __bool__(self):
            return False
    
    # Mock critical GPU modules
    sys.modules['torch'] = MockModule()
    sys.modules['torch.cuda'] = MockModule()
    sys.modules['torch.cuda.is_available'] = lambda: False
    sys.modules['transformers'] = MockModule()
    sys.modules['sentence_transformers'] = MockModule()
    sys.modules['langchain_community.embeddings.huggingface'] = MockModule()
    sys.modules['langchain_community.vectorstores.chroma'] = MockModule()
    
    print("🎭 GPU modules mocked")

def run_tests_safely():
    """Run tests with all GPU protections active"""
    
    print("🚨 ULTRA-SAFE TEST MODE ACTIVATED")
    print("=" * 50)
    
    # Block GPU libraries first
    block_gpu_libraries()
    
    # Mock GPU modules
    mock_gpu_modules()
    
    # Verify environment
    print("\n🔍 Environment verification:")
    for key in ['CC_HARD_CPU_ONLY', 'CC_GPU_DISABLED', 'CUDA_VISIBLE_DEVICES']:
        print(f"  {key}: {os.environ.get(key, 'NOT SET')}")
    
    # Now run pytest
    print("\n🧪 Starting pytest with GPU protection...")
    
    # Build pytest command
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-k", "not gpu and not vllm and not master",
        "-q",
        "--tb=short",
        "-ra",
        "--maxfail=3"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run pytest as subprocess to ensure clean environment
        result = subprocess.run(cmd, env=os.environ, check=False)
        return result.returncode
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1

if __name__ == "__main__":
    exit_code = run_tests_safely()
    print(f"\n🏁 Tests completed with exit code: {exit_code}")
    sys.exit(exit_code)
