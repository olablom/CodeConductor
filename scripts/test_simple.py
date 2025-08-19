#!/usr/bin/env python3
"""
Simple test script that runs basic tests safely.
"""

import os
import sys


def main():
    """Run simple tests"""
    print("🔒 Running simple tests (GPU disabled)")

    # Set environment variables
    os.environ["CC_TESTING_MODE"] = "1"
    os.environ["CC_GPU_DISABLED"] = "1"
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["CODECONDUCTOR_SEED"] = "1337"

    print("✅ Environment variables set")

    # Test basic import
    try:
        import codeconductor

        print("✅ CodeConductor imported successfully")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return 1

    # Test basic functionality
    try:
        from codeconductor.ensemble.model_manager import ModelManager

        mm = ModelManager()
        print("✅ ModelManager created successfully")
    except Exception as e:
        print(f"❌ ModelManager creation failed: {e}")
        return 1

    print("✅ All basic tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
