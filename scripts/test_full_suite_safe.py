#!/usr/bin/env python3
"""
Test full test suite safely with GPU protection
"""

import os
import sys
import subprocess


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
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

    print("🔒 All GPU blockers set")


def run_full_suite():
    """Run the full test suite with GPU protection"""

    print("🧪 Running full test suite with GPU protection...")

    # Run the same command that was causing GPU issues
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/",
        "-k",
        "not gpu and not vllm and not master",
        "-q",
        "--tb=short",
        "-ra",
        "--maxfail=3",
    ]

    print(f"Command: {' '.join(cmd)}")
    print("⚠️  WARNING: This is the command that caused GPU issues before!")
    print("💡 Monitor nvidia-smi in another terminal")

    try:
        result = subprocess.run(cmd, env=os.environ, check=False)
        return result.returncode
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def main():
    print("🚨 FULL TEST SUITE GPU PROTECTION")
    print("=" * 50)

    # Set all GPU blockers
    set_gpu_blockers()

    # Run full suite
    exit_code = run_full_suite()

    print(f"\n🏁 Full test suite completed with exit code: {exit_code}")
    print("💡 If GPU spiked, we need stronger protection")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
