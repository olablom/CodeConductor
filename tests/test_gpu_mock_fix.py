#!/usr/bin/env python3
"""
Test GPU Mock Fix for CodeConductor

Verifies that CC_GPU_DISABLED=1 properly mocks GPU operations.
"""

import os
import pytest

# Kör alltid i mock-läge här
@pytest.fixture(autouse=True)
def _enable_gpu_mock(monkeypatch):
    monkeypatch.setenv("CC_GPU_DISABLED", "1")
    monkeypatch.setenv("CC_TESTING_MODE", "1")

def test_model_manager_returns_mock(monkeypatch):
    """Test that model manager respects GPU mock mode"""
    # Valfritt: gör ett hårt skydd – om någon försöker kalla nvidia-smi, faila.
    import subprocess
    def _forbid(*a, **k):
        raise AssertionError("subprocess called in mock mode!")
    monkeypatch.setattr(subprocess, "run", _forbid, raising=True)

    # Testa att ModelManager kan skapas utan att krascha
    from codeconductor.ensemble.model_manager import ModelManager
    manager = ModelManager()
    
    # Verifiera att den inte försöker ladda riktiga modeller
    assert manager is not None
    
    # Testa att GPU_DISABLED respekteras
    assert os.getenv("CC_GPU_DISABLED") == "1"
    
    print("✅ ModelManager respects GPU mock mode")

def test_agents_respect_gpu_mock(monkeypatch):
    # Om agenten ändå försöker initiera en riktig backend – markera fel.
    # Exempel: blockera torch.cuda.is_available() som försök
    try:
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    except Exception:
        pass

    from codeconductor.debate.local_agent import LocalAIAgent
    a = LocalAIAgent("Mocker", "You are a test agent.")
    # Ska ge mockad sträng/dict – inte försöka ladda riktiga modeller
    out = a.propose(prompt="ping")
    # Beroende på din returtyp: stöd både sync/await
    if hasattr(out, "__await__"):
        import asyncio
        out = asyncio.run(out)
    assert "MOCK" in str(out).upper()

def test_ensemble_engine_respects_gpu_mock(monkeypatch):
    """Test that EnsembleEngine returns mock responses when GPU is disabled"""
    from codeconductor.ensemble.ensemble_engine import EnsembleEngine
    
    engine = EnsembleEngine()
    
    # Testa process_request
    response = engine.process_request(
        task_description="Create a simple API",
        timeout=30.0,
        prefer_fast_models=False,
        enable_fallback=True,
    )
    
    # Kontrollera att det är en coroutine (async)
    assert hasattr(response, "__await__"), "Response should be awaitable"
    
    # Kör den async
    import asyncio
    result = asyncio.run(response)
    
    # Verifiera mock-innehåll
    assert "generated_code" in result
    assert "[MOCKED]" in result["generated_code"]
    assert result["model_used"] == "ensemble-mock"
    assert result["execution_time"] < 0.1  # Mock ska vara snabb

def test_single_model_engine_respects_gpu_mock(monkeypatch):
    """Test that SingleModelEngine returns mock responses when GPU is disabled"""
    from codeconductor.ensemble.single_model_engine import SingleModelEngine, SingleModelRequest
    
    engine = SingleModelEngine("test-model")
    
    # Testa initialize
    import asyncio
    success = asyncio.run(engine.initialize())
    assert success is True
    
    # Testa process_request
    request = SingleModelRequest("Create a simple REST API with Flask", timeout=30.0)
    response = asyncio.run(engine.process_request(request))
    
    assert response.model_used == "test-model-mock"
    assert "FastAPI" in response.content  # Mock REST API response
    assert response.execution_time < 0.1  # Mock ska vara snabb

def test_debate_system_respects_gpu_mock(monkeypatch):
    """Test that the complete debate system works with GPU disabled"""
    from codeconductor.debate.debate_manager import CodeConductorDebateManager
    from codeconductor.debate.local_agent import LocalAIAgent
    
    # Skapa agenter
    agents = [
        LocalAIAgent("Architect", "You are an Architect."),
        LocalAIAgent("Coder", "You are a Coder."),
    ]
    
    # Skapa debate manager
    debate_manager = CodeConductorDebateManager(agents)
    
    # Kör debatt
    import asyncio
    result = asyncio.run(debate_manager.conduct_debate("Create a simple API"))
    
    # Verifiera resultat
    assert "transcript" in result
    assert "agents" in result
    assert "total_turns" in result
    assert len(result["transcript"]) > 0
    
    # Verifiera att alla svar är mockade
    for entry in result["transcript"]:
        assert "[MOCKED]" in entry["content"]

def test_no_real_gpu_calls_in_mock_mode(monkeypatch):
    """Hard test: ensure no real GPU calls are made in mock mode"""
    import subprocess
    
    # Blockera alla subprocess-anrop
    def _forbid_subprocess(*args, **kwargs):
        raise AssertionError(f"subprocess called in mock mode: {args} {kwargs}")
    
    monkeypatch.setattr(subprocess, "run", _forbid_subprocess, raising=True)
    monkeypatch.setattr(subprocess, "Popen", _forbid_subprocess, raising=True)
    monkeypatch.setattr(subprocess, "call", _forbid_subprocess, raising=True)
    
    # Blockera torch.cuda om det finns
    try:
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False, raising=True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 0, raising=True)
    except Exception:
        pass
    
    # Blockera vLLM om det finns
    try:
        import vllm
        monkeypatch.setattr(vllm, "LLM", _forbid_subprocess, raising=True)
    except Exception:
        pass
    
    # Nu ska alla tester köra utan att försöka nå GPU
    from codeconductor.debate.local_agent import LocalAIAgent
    from codeconductor.ensemble.ensemble_engine import EnsembleEngine
    
    # Skapa agenter och engine
    agent = LocalAIAgent("Test", "Test persona")
    engine = EnsembleEngine()
    
    # Testa att de inte kraschar
    assert agent.name == "Test"
    assert engine is not None
    
    print("✅ No real GPU calls detected in mock mode")

if __name__ == "__main__":
    # Kör alla tester
    print("🧪 Testing GPU Mock Fix with Hard Guards")
    print("=" * 60)
    
    import pytest
    import sys
    
    # Kör pytest med våra tester
    result = pytest.main([__file__, "-v", "-s"])
    
    if result == 0:
        print("\n🎉 All GPU mock tests PASSED!")
        sys.exit(0)
    else:
        print("\n💥 Some GPU mock tests FAILED!")
        sys.exit(1)
