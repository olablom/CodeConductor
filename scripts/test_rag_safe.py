#!/usr/bin/env python3
"""
Test RAG system safely to isolate GPU trigger
"""

import os
import sys
import subprocess


def set_gpu_blockers():
    """Set GPU blocking environment variables"""
    os.environ["CC_HARD_CPU_ONLY"] = "1"
    os.environ["CC_GPU_DISABLED"] = "1"
    os.environ["CC_ULTRA_MOCK"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    print("🔒 GPU blockers set")


def run_rag_test():
    """Run the RAG system test that was causing GPU issues"""

    print("🧪 Running RAG system test to isolate GPU trigger...")

    # Test the RAG system specifically
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_rag_fix.py::test_rag_system",
        "-v",
        "--tb=short",
    ]

    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, env=os.environ, check=False)
        return result.returncode
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def main():
    print("🚨 RAG TEST GPU ISOLATION")
    print("=" * 40)

    # Set GPU blockers
    set_gpu_blockers()

    # Run RAG test
    exit_code = run_rag_test()

    print(f"\n🏁 RAG test completed with exit code: {exit_code}")
    print("💡 Check nvidia-smi - if GPU spiked, RAG test triggered it")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
