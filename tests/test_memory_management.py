#!/usr/bin/env python3
"""
Test script for CodeConductor Memory Management

Tests the smart memory management system to ensure VRAM doesn't overflow.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from codeconductor.ensemble.model_manager import ModelManager
from codeconductor.monitoring.memory_watchdog import (
    start_memory_watchdog,
    stop_memory_watchdog,
)


async def test_memory_management():
    """Test the memory management system"""

    print("🧪 Testing CodeConductor Memory Management")
    print("=" * 50)

    try:
        # Initialize model manager
        model_manager = ModelManager()
        print("✅ Model manager initialized")

        # Start memory watchdog
        await start_memory_watchdog(model_manager, check_interval=10.0)
        print("✅ Memory watchdog started")

        # Test memory info
        print("\n📊 Testing GPU Memory Detection:")
        gpu_info = await model_manager.get_gpu_memory_info()
        if gpu_info:
            print(f"✅ VRAM Usage: {gpu_info['usage_percent']:.1f}%")
            print(f"✅ Used: {gpu_info['used_gb']:.1f}GB")
            print(f"✅ Free: {gpu_info['free_gb']:.1f}GB")
            print(f"✅ Total: {gpu_info['total_gb']:.1f}GB")
        else:
            print("❌ Could not get GPU memory info")
            return False

        # Test memory cleanup
        print("\n🧹 Testing Memory Cleanup:")
        cleanup_performed = await model_manager.check_and_cleanup_memory("medium_load")
        if cleanup_performed:
            print("✅ Memory cleanup performed")
        else:
            print("✅ No cleanup needed")

        # Test smart cleanup
        print("\n🧹 Testing Smart Memory Cleanup:")
        unloaded_count = await model_manager.smart_memory_cleanup(60.0)
        print(f"✅ Smart cleanup unloaded {unloaded_count} models")

        # Test emergency unload
        print("\n🚨 Testing Emergency Unload:")
        unloaded_count = await model_manager.emergency_unload_all()
        print(f"✅ Emergency unload completed: {unloaded_count} models")

        # Test model loading with memory check
        print("\n🚀 Testing Model Loading with Memory Check:")
        loaded_models = await model_manager.ensure_models_loaded_with_memory_check("light_load")
        print(f"✅ Loaded {len(loaded_models)} models: {loaded_models}")

        # Wait a bit for watchdog to run
        print("\n⏳ Waiting for watchdog to run...")
        await asyncio.sleep(15)

        # Stop watchdog
        await stop_memory_watchdog()
        print("✅ Memory watchdog stopped")

        print("\n🎉 Memory management test PASSED!")
        return True

    except Exception as e:
        print(f"❌ Memory management test FAILED: {e}")
        return False


if __name__ == "__main__":
    # Run test
    success = asyncio.run(test_memory_management())

    if success:
        print("\n🎉 All tests PASSED!")
    else:
        print("\n💥 Tests FAILED!")
        sys.exit(1)
