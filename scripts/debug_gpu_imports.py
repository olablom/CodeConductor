#!/usr/bin/env python3
"""
Debug script to find what's triggering GPU imports
"""

import os
import sys
import importlib
from pathlib import Path

def set_gpu_blockers():
    """Set all GPU blocking environment variables"""
    os.environ["CC_HARD_CPU_ONLY"] = "1"
    os.environ["CC_GPU_DISABLED"] = "1"
    os.environ["CC_ULTRA_MOCK"] = "1"
    os.environ["CC_TESTING_MODE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["VLLM_NO_CUDA"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TORCH_USE_CUDA_DISABLED"] = "1"
    
    print("🔒 GPU blockers set")

def monitor_imports():
    """Monitor what modules are imported"""
    
    # Track original import
    original_import = __builtins__.__import__
    
    def monitored_import(name, *args, **kwargs):
        if any(gpu_lib in name.lower() for gpu_lib in ['torch', 'transformers', 'sentence', 'cuda', 'gpu']):
            print(f"🚨 GPU LIBRARY IMPORT DETECTED: {name}")
            print(f"   Called from: {sys._getframe(1).f_code.co_name}")
            print(f"   File: {sys._getframe(1).f_code.co_filename}:{sys._getframe(1).f_lineno}")
            
        return original_import(name, *args, **kwargs)
    
    __builtins__.__import__ = monitored_import
    print("🔍 Import monitoring activated")

def test_imports():
    """Test importing various modules to see what triggers GPU"""
    
    print("\n🧪 Testing imports...")
    
    # Test basic imports
    try:
        print("Testing: import codeconductor")
        import codeconductor
        print("✅ codeconductor imported")
    except Exception as e:
        print(f"❌ codeconductor failed: {e}")
    
    try:
        print("Testing: import tests")
        import tests
        print("✅ tests imported")
    except Exception as e:
        print(f"❌ tests failed: {e}")
    
    try:
        print("Testing: import pytest")
        import pytest
        print("✅ pytest imported")
    except Exception as e:
        print(f"❌ pytest failed: {e}")

def main():
    print("🚨 GPU IMPORT DEBUGGER")
    print("=" * 50)
    
    # Set GPU blockers first
    set_gpu_blockers()
    
    # Monitor imports
    monitor_imports()
    
    # Test imports
    test_imports()
    
    print("\n🔍 Check nvidia-smi for GPU activity")
    print("If GPU usage spikes, the problematic import was detected above")

if __name__ == "__main__":
    main()
