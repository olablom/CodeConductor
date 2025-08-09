#!/usr/bin/env python3
"""
Test vLLM Integration Status

Checks vLLM availability and provides setup information.
"""

import sys
import os


def test_vllm_availability():
    """Test if vLLM is available and provide setup info."""
    print("🔍 Checking vLLM Integration Status...")

    # Check if we're in WSL2
    try:
        with open("/proc/version", "r") as f:
            if "microsoft" in f.read().lower():
                print("✅ Running in WSL2 environment")
                wsl2 = True
            else:
                print("❌ Not running in WSL2")
                wsl2 = False
    except FileNotFoundError:
        print("❌ Not running in WSL2 (no /proc/version)")
        wsl2 = False

    # Try to import vLLM
    try:
        import vllm

        print(f"✅ vLLM available: {vllm.__version__}")

        # Test basic vLLM functionality
        from vllm import LLM, SamplingParams

        print("✅ vLLM imports successful")

        # Check CUDA availability
        import torch

        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.version.cuda}")
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("❌ CUDA not available")

    except ImportError as e:
        print(f"❌ vLLM not available: {e}")

        if wsl2:
            print("\n💡 Setup instructions for WSL2:")
            print("1. Open WSL2 terminal:")
            print("   wsl -d Ubuntu")
            print("2. Navigate to project:")
            print("   cd /mnt/c/Users/olabl/Documents/GitHub/CodeConductor")
            print("3. Activate vLLM environment:")
            print("   source vllm_env/bin/activate")
            print("4. Test vLLM:")
            print("   python test_vllm_wsl.py")
        else:
            print("\n💡 vLLM is only available in WSL2 environment")
            print("   Use 'wsl -d Ubuntu' to access WSL2")

    # Test CodeConductor integration
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
        from codeconductor.vllm_integration import VLLMEngine

        print("✅ CodeConductor vLLM integration available")
    except ImportError as e:
        print(f"❌ CodeConductor vLLM integration not available: {e}")

    print("\n📊 Summary:")
    print(f"   WSL2 Environment: {'✅' if wsl2 else '❌'}")
    print(f"   vLLM Available: {'✅' if 'vllm' in sys.modules else '❌'}")

    # Check CUDA availability safely
    cuda_available = False
    try:
        if "torch" in sys.modules:
            import torch

            cuda_available = torch.cuda.is_available()
    except:
        pass
    print(f"   CUDA Available: {'✅' if cuda_available else '❌'}")


if __name__ == "__main__":
    test_vllm_availability()
