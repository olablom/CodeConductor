#!/usr/bin/env python3
"""
Safe test script that disables GPU operations to prevent system crashes.
"""

import os
import subprocess
import sys


def main():
    """Run tests with GPU disabled"""
    print("🔒 Running tests in SAFE MODE (GPU disabled)")

    # Set environment variables to disable GPU
    env = os.environ.copy()
    env.update(
        {
            "CC_TESTING_MODE": "1",
            "CC_GPU_DISABLED": "1",
            "PYTHONHASHSEED": "0",
            "CODECONDUCTOR_SEED": "1337",
        }
    )

    # Run basic import test first
    print("📦 Testing basic imports...")
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import codeconductor; print('✅ CodeConductor imports successfully')",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            print("✅ Basic imports successful")
        else:
            print(f"❌ Import failed: {result.stderr}")
            return 1

    except subprocess.TimeoutExpired:
        print("❌ Import test timed out")
        return 1
    except Exception as e:
        print(f"❌ Import test error: {e}")
        return 1

    # Run a few simple tests
    print("🧪 Running simple tests...")
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/",
                "-v",
                "--tb=short",
                "--maxfail=3",
                "-k",
                "not vllm and not gpu",  # Skip GPU-heavy tests
            ],
            env=env,
            timeout=300,
        )  # 5 minutes max

        if result.returncode == 0:
            print("✅ All tests passed!")
            return 0
        else:
            print(f"❌ Some tests failed (exit code: {result.returncode})")
            return result.returncode

    except subprocess.TimeoutExpired:
        print("❌ Tests timed out after 5 minutes")
        return 1
    except Exception as e:
        print(f"❌ Test execution error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
