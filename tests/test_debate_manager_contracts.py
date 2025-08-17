"""
Contract tests for CodeConductorDebateManager
Verifies default behavior and transcript functionality
"""

import pytest
from pathlib import Path
from codeconductor.debate.debate_manager import CodeConductorDebateManager


def test_manager_defaults():
    """Test that CodeConductorDebateManager works without arguments"""
    mgr = CodeConductorDebateManager()
    
    # Should have default agents
    assert len(mgr.agents) >= 2
    assert all(hasattr(agent, 'name') for agent in mgr.agents)
    assert all(hasattr(agent, 'persona') for agent in mgr.agents)
    
    # Should have empty transcript initially
    assert len(mgr.full_transcript) == 0


def test_manager_with_custom_agents():
    """Test that CodeConductorDebateManager works with custom agents"""
    from codeconductor.debate.local_ai_agent import LocalAIAgent
    
    custom_agents = [
        LocalAIAgent(name="Custom1", persona="You are Custom1."),
        LocalAIAgent(name="Custom2", persona="You are Custom2."),
    ]
    
    mgr = CodeConductorDebateManager(agents=custom_agents)
    
    assert len(mgr.agents) == 2
    assert mgr.agents[0].name == "Custom1"
    assert mgr.agents[1].name == "Custom2"


def test_transcript_functionality(tmp_artifacts):
    """Test transcript saving and retrieval"""
    mgr = CodeConductorDebateManager()
    
    # Add some mock transcript entries
    mgr.full_transcript = [
        {"agent": "Architect", "turn": "proposal", "content": "Test proposal"},
        {"agent": "Coder", "turn": "proposal", "content": "Test proposal 2"},
    ]
    
    # Test get_transcript
    transcript = mgr.get_transcript()
    assert len(transcript) == 2
    assert transcript[0]["agent"] == "Architect"
    assert transcript[1]["agent"] == "Coder"
    
    # Test save_transcript
    output_file = mgr.save_transcript("test_transcript.yaml")
    assert output_file.exists()
    
    # Verify YAML file was created
    yaml_file = Path("test_transcript.yaml")
    assert yaml_file.exists()
    
    # Verify JSON file was created
    json_file = Path("test_transcript.json")
    assert json_file.exists()
    
    # Cleanup
    yaml_file.unlink(missing_ok=True)
    json_file.unlink(missing_ok=True)


def test_consensus_extraction():
    """Test consensus extraction from transcript"""
    mgr = CodeConductorDebateManager()
    
    # Mock transcript with final recommendations
    mgr.full_transcript = [
        {"agent": "Architect", "turn": "proposal", "content": "Proposal 1"},
        {"agent": "Coder", "turn": "proposal", "content": "Proposal 2"},
        {"agent": "Architect", "turn": "final_recommendation", "content": "Final 1"},
        {"agent": "Coder", "turn": "final_recommendation", "content": "Final 2"},
    ]
    
    consensus = mgr.extract_consensus()
    assert "## Architect" in consensus
    assert "## Coder" in consensus
    assert "Final 1" in consensus
    assert "Final 2" in consensus


def test_agent_interface_compliance():
    """Test that default agents have required interface"""
    mgr = CodeConductorDebateManager()
    
    for agent in mgr.agents:
        # Required attributes
        assert hasattr(agent, 'name')
        assert hasattr(agent, 'persona')
        
        # Required methods
        assert hasattr(agent, 'generate_response')
        assert callable(agent.generate_response)


@pytest.mark.asyncio
async def test_debate_conduct_basic():
    """Test basic debate conduction (without actual model calls)"""
    mgr = CodeConductorDebateManager()
    
    # Mock the generate_response method to avoid actual model calls
    for agent in mgr.agents:
        agent.generate_response = lambda prompt, timeout=30.0: f"Mock response from {agent.name}"
    
    result = await mgr.conduct_debate("Test prompt")
    
    assert "transcript" in result
    assert "agents" in result
    assert "total_turns" in result
    assert len(result["transcript"]) > 0
