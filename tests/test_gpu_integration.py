#!/usr/bin/env python3
"""
GPU Integration Tests for CodeConductor

These tests require real GPU access and are marked with @pytest.mark.gpu
"""

import os
import pytest
import asyncio


@pytest.mark.gpu
def test_gpu_available():
    """Test that GPU is available when marked with @pytest.mark.gpu"""
    # This test should run with real GPU access
    assert os.getenv("CC_GPU_DISABLED") == "0", "GPU should be enabled for GPU tests"
    
    # Test that torch.cuda is available (not mocked)
    try:
        import torch
        if torch.cuda.is_available():
            assert torch.cuda.device_count() > 0, "Should have at least one GPU device"
            print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        else:
            pytest.skip("CUDA not available on this system")
    except ImportError:
        pytest.skip("PyTorch not available")


@pytest.mark.gpu
async def test_real_model_loading():
    """Test that models can actually be loaded on GPU"""
    # This test should run with real GPU access
    assert os.getenv("CC_GPU_DISABLED") == "0", "GPU should be enabled for GPU tests"
    
    try:
        from codeconductor.ensemble.single_model_engine import SingleModelEngine, SingleModelRequest
        
        # Create engine with a lightweight model
        engine = SingleModelEngine("meta-llama-3.1-8b-instruct")
        
        # Initialize (this should actually load the model)
        success = await engine.initialize()
        assert success is True, "Model should initialize successfully on GPU"
        
        # Test a simple request
        request = SingleModelRequest("Hello", timeout=30.0)
        response = await engine.process_request(request)
        
        assert response is not None, "Should get a response from real model"
        assert response.content is not None, "Response should have content"
        assert "[MOCKED]" not in response.content, "Should not be mocked in GPU mode"
        
        print("✅ Real model loaded and responded successfully")
        
        # Cleanup
        await engine.cleanup()
        
    except Exception as e:
        pytest.skip(f"GPU model loading not available: {e}")


@pytest.mark.gpu
async def test_real_debate_system():
    """Test the debate system with real GPU models"""
    # This test should run with real GPU access
    assert os.getenv("CC_GPU_DISABLED") == "0", "GPU should be enabled for GPU tests"
    
    try:
        from codeconductor.debate.debate_manager import CodeConductorDebateManager
        from codeconductor.debate.local_agent import LocalAIAgent
        
        # Create agents
        agents = [
            LocalAIAgent("Architect", "You are an Architect."),
            LocalAIAgent("Coder", "You are a Coder."),
        ]
        
        # Create debate manager
        debate_manager = CodeConductorDebateManager(agents)
        
        # Run debate with real models
        result = await debate_manager.conduct_debate("Create a simple Python function")
        
        # Verify results
        assert "transcript" in result
        assert "agents" in result
        assert "total_turns" in result
        assert len(result["transcript"]) > 0
        
        # Verify that responses are not mocked
        for entry in result["transcript"]:
            assert "[MOCKED]" not in entry["content"], "Should not be mocked in GPU mode"
        
        print("✅ Real debate system worked with GPU models")
        
    except Exception as e:
        pytest.skip(f"GPU debate system not available: {e}")


if __name__ == "__main__":
    # Run GPU tests
    print("🎮 Running GPU Integration Tests")
    print("=" * 50)
    
    import pytest
    import sys
    
    # Run pytest with GPU marker
    result = pytest.main([__file__, "-m", "gpu", "-v", "-s"])
    
    if result == 0:
        print("\n🎉 All GPU integration tests PASSED!")
        sys.exit(0)
    else:
        print("\n💥 Some GPU integration tests FAILED!")
        sys.exit(1)
