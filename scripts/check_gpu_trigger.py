#!/usr/bin/env python3
"""
Simple GPU trigger checker - just diagnose, don't block
"""

import os
import sys


def check_environment():
    """Check current environment variables"""
    print("🔍 Environment Check:")
    print(f"  CC_HARD_CPU_ONLY: {os.getenv('CC_HARD_CPU_ONLY', 'NOT SET')}")
    print(f"  CC_GPU_DISABLED: {os.getenv('CC_GPU_DISABLED', 'NOT SET')}")
    print(f"  CUDA_VISIBLE_DEVICES: '{os.getenv('CUDA_VISIBLE_DEVICES', 'NOT SET')}'")
    print(f"  HF_HUB_OFFLINE: {os.getenv('HF_HUB_OFFLINE', 'NOT SET')}")


def check_imports():
    """Check what modules are already imported"""
    print("\n📦 Already Imported Modules:")

    gpu_related = []
    for module_name in sys.modules:
        if any(
            gpu_lib in module_name.lower()
            for gpu_lib in ["torch", "transformers", "sentence", "cuda", "gpu"]
        ):
            gpu_related.append(module_name)

    if gpu_related:
        print("  🚨 GPU-related modules already imported:")
        for module in gpu_related:
            print(f"    - {module}")
    else:
        print("  ✅ No GPU modules imported yet")


def test_simple_imports():
    """Test importing basic modules to see what happens"""
    print("\n🧪 Testing Basic Imports:")

    try:
        print("Testing: import codeconductor")

        print("  ✅ codeconductor imported successfully")
    except Exception as e:
        print(f"  ❌ codeconductor failed: {e}")

    try:
        print("Testing: import tests")

        print("  ✅ tests imported successfully")
    except Exception as e:
        print(f"  ❌ tests failed: {e}")


def main():
    print("🔍 GPU TRIGGER DIAGNOSTIC")
    print("=" * 40)

    # Check environment
    check_environment()

    # Check what's already imported
    check_imports()

    # Test basic imports
    test_simple_imports()

    print("\n💡 Next steps:")
    print("1. Check nvidia-smi for GPU activity")
    print("2. If GPU spikes, the problematic import was above")
    print("3. We can then target that specific module")


if __name__ == "__main__":
    main()
