#!/usr/bin/env python3
"""
GPU Integration Tests - Run locally on RTX 5090 with real hardware access.
These tests are NOT for CI - they require actual GPU hardware.
"""

import os
import pytest
import asyncio
import logging
from codeconductor.ensemble.model_manager import ModelManager

logger = logging.getLogger(__name__)

# Mark all tests in this file as GPU tests
pytestmark = pytest.mark.gpu

@pytest.fixture(autouse=True)
def enable_real_gpu():
    """Enable real GPU access for these tests"""
    # Override the GPU sanitizer for these specific tests
    os.environ["CC_GPU_DISABLED"] = "0"
    os.environ["CC_TESTING_MODE"] = "0"
    
    logger.info("🎮 GPU test: enabling real GPU access (CC_GPU_DISABLED=0)")
    
    yield
    
    # Reset to safe mode after test
    os.environ["CC_GPU_DISABLED"] = "1"
    os.environ["CC_TESTING_MODE"] = "1"
    
    logger.info("🔒 GPU test: reset to safe mode (CC_GPU_DISABLED=1)")

@pytest.mark.asyncio
async def test_real_gpu_memory_info():
    """Test real GPU memory detection on RTX 5090"""
    manager = ModelManager()
    
    # This should now call real GPU methods
    gpu_info = await manager.get_gpu_memory_info()
    
    assert gpu_info is not None, "GPU info should be returned"
    assert "total_gb" in gpu_info, "Should have total_gb field"
    assert "used_gb" in gpu_info, "Should have used_gb field"
    assert "free_gb" in gpu_info, "Should have free_gb field"
    assert "usage_percent" in gpu_info, "Should have usage_percent field"
    
    # RTX 5090 specific checks (allow for Windows reporting variations)
    assert gpu_info["total_gb"] >= 31.0, f"Expected at least 31GB, got {gpu_info['total_gb']}GB"
    assert gpu_info["used_gb"] >= 0, "Used GB should be non-negative"
    assert gpu_info["free_gb"] >= 0, "Free GB should be non-negative"
    assert 0 <= gpu_info["usage_percent"] <= 100, "Usage should be 0-100%"
    
    print(f"✅ Real GPU info: {gpu_info['used_gb']:.1f}GB / {gpu_info['total_gb']:.1f}GB ({gpu_info['usage_percent']:.1f}%)")

@pytest.mark.asyncio
async def test_real_gpu_methods():
    """Test all GPU detection methods on real hardware"""
    manager = ModelManager()
    
    results = await manager.test_all_gpu_methods()
    
    assert results is not None, "Should return results dict"
    
    # Check that at least one method worked
    working_methods = [k for k, v in results.items() if v is not None]
    assert len(working_methods) > 0, f"No GPU methods working: {results}"
    
    print(f"✅ Working GPU methods: {working_methods}")
    
    # Check that working methods return proper data
    for method_name in working_methods:
        method_result = results[method_name]
        assert method_result is not None, f"Method {method_name} should not be None"
        assert "method" in method_result, f"Method {method_name} should have 'method' field"
        assert method_result["method"] != "mock", f"Method {method_name} should not be mock"

@pytest.mark.asyncio
async def test_gpu_memory_cleanup():
    """Test GPU memory cleanup functionality"""
    manager = ModelManager()
    
    # Get initial memory state
    initial_info = await manager.get_gpu_memory_info()
    initial_used = initial_info["used_gb"]
    
    print(f"📊 Initial GPU memory: {initial_used:.1f}GB")
    
    # Try to clean up memory
    cleanup_result = await manager.smart_memory_cleanup(target_vram_percent=50)
    
    # Get final memory state
    final_info = await manager.get_gpu_memory_info()
    final_used = final_info["used_gb"]
    
    print(f"📊 Final GPU memory: {final_used:.1f}GB (cleanup: {cleanup_result} models unloaded)")
    
    # Memory should not have increased significantly
    assert final_used <= initial_used + 2.0, f"Memory increased too much: {initial_used:.1f}GB -> {final_used:.1f}GB"

@pytest.mark.asyncio
async def test_gpu_emergency_cleanup():
    """Test emergency GPU cleanup"""
    manager = ModelManager()
    
    # Get current memory state
    gpu_info = await manager.get_gpu_memory_info()
    current_usage = gpu_info["usage_percent"]
    
    print(f"📊 Current GPU usage: {current_usage:.1f}%")
    
    # Only test emergency cleanup if usage is high
    if current_usage > 80:
        print("🚨 High GPU usage detected, testing emergency cleanup")
        
        unloaded_count = await manager.emergency_unload_all()
        print(f"🧹 Emergency cleanup unloaded {unloaded_count} models")
        
        # Check memory after cleanup
        after_info = await manager.get_gpu_memory_info()
        after_usage = after_info["usage_percent"]
        
        print(f"📊 GPU usage after emergency cleanup: {after_usage:.1f}%")
        
        # Usage should have decreased
        assert after_usage < current_usage, "Emergency cleanup should reduce GPU usage"
    else:
        print("ℹ️ GPU usage is normal, skipping emergency cleanup test")

if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "-s"])
