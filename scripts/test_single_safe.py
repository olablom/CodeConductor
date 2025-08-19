#!/usr/bin/env python3
"""
Test single test safely to isolate GPU trigger
"""

import os
import subprocess
import sys


def set_gpu_blockers():
    """Set GPU blocking environment variables"""
    os.environ["CC_HARD_CPU_ONLY"] = "1"
    os.environ["CC_GPU_DISABLED"] = "1"
    os.environ["CC_ULTRA_MOCK"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    print("🔒 GPU blockers set")


def run_single_test():
    """Run just one simple test to see if it triggers GPU"""

    print("🧪 Running single test to isolate GPU trigger...")

    # Try the simplest possible test first
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_breakers_unit.py::test_closed_to_open_on_consec_fails",
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
    print("🚨 SINGLE TEST GPU ISOLATION")
    print("=" * 40)

    # Set GPU blockers
    set_gpu_blockers()

    # Run single test
    exit_code = run_single_test()

    print(f"\n🏁 Test completed with exit code: {exit_code}")
    print("💡 Check nvidia-smi - if GPU spiked, the test triggered it")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
