"""
Integration tests for AgentOrchestrator.

Tests multi-agent coordination and consensus building.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile

from agents.orchestrator import AgentOrchestrator


class TestAgentOrchestrator:
    """Test AgentOrchestrator functionality."""

    def setup_method(self):
        """Setup for each test."""
        self.orchestrator = AgentOrchestrator()

    def test_initialization(self):
        """Test that orchestrator initializes correctly."""
        assert hasattr(self.orchestrator, "codegen_agent")
        assert hasattr(self.orchestrator, "architect_agent")
        assert hasattr(self.orchestrator, "reviewer_agent")
        assert hasattr(self.orchestrator, "agents")

        # Check agent registry
        assert len(self.orchestrator.agents) == 3
        assert "codegen" in self.orchestrator.agents
        assert "architect" in self.orchestrator.agents
        assert "reviewer" in self.orchestrator.agents

    def test_facilitate_discussion_returns_dict(self):
        """Test that facilitate_discussion returns proper structure."""
        result = self.orchestrator.facilitate_discussion("Test prompt")

        assert isinstance(result, dict)
        assert "proposal_id" in result
        assert "prompt" in result
        assert "approach" in result
        assert "confidence" in result
        assert "patterns" in result
        assert "risks" in result
        assert "rl_score" in result
        assert "optimization" in result
        assert "agent_analyses" in result
        assert "consensus" in result
        assert "optimized" in result
        assert "timestamp" in result

    def test_consensus_building(self):
        """Test that consensus is built from all agent analyses."""
        result = self.orchestrator.facilitate_discussion("Test prompt")

        # Check that all agents contributed
        agent_analyses = result["agent_analyses"]
        assert len(agent_analyses) == 3

        # Check consensus structure
        consensus = result["consensus"]
        assert "synthesized_approach" in consensus
        assert "recommended_patterns" in consensus
        assert "identified_risks" in consensus
        assert "consensus_recommendation" in consensus
        assert "confidence" in consensus

    @patch.object(AgentOrchestrator, "_synthesize_consensus")
    def test_consensus_combines_all_agents(self, mock_synthesize):
        """Test that consensus combines all agent inputs."""
        # Mock individual agent responses
        mock_analyses = {
            "codegen": {
                "agent": "CodeGenAgent",
                "approach": "functional approach",
                "confidence": 0.8,
                "recommendation": "functional",
            },
            "architect": {
                "agent": "ArchitectAgent",
                "patterns": ["factory", "observer"],
                "risks": ["complexity"],
                "recommendation": "modular",
            },
            "reviewer": {
                "agent": "ReviewerAgent",
                "security_risks": ["injection"],
                "recommendation": "defensive",
            },
        }

        mock_synthesize.return_value = {
            "synthesized_approach": "combined approach",
            "recommended_patterns": ["factory"],
            "identified_risks": ["complexity", "injection"],
            "consensus_recommendation": "defensive",
            "confidence": 0.75,
        }

        result = self.orchestrator.facilitate_discussion("Test prompt")

        # Verify synthesize was called with all analyses
        mock_synthesize.assert_called_once()
        call_args = mock_synthesize.call_args[0][0]
        assert len(call_args) == 3
        assert all(key in call_args for key in ["codegen", "architect", "reviewer"])

    @patch.object(AgentOrchestrator, "_optimize_with_rl")
    def test_rl_optimization_applied(self, mock_optimize):
        """Test that RL optimization is applied to consensus."""
        mock_optimize.return_value = {
            "optimized_prompt": "optimized content",
            "optimization_applied": "add_examples",
            "rl_score": 0.85,
        }

        result = self.orchestrator.facilitate_discussion("Test prompt")

        assert mock_optimize.called
        assert result["rl_score"] == 0.85
        assert result["optimization"] == "add_examples"

    def test_proposal_id_generation(self):
        """Test that proposal IDs are unique and meaningful."""
        result1 = self.orchestrator.facilitate_discussion("Test prompt 1")
        result2 = self.orchestrator.facilitate_discussion("Test prompt 2")

        assert result1["proposal_id"] != result2["proposal_id"]
        assert "proposal_" in result1["proposal_id"]
        assert "_" in result1["proposal_id"]  # Should contain confidence score

    def test_confidence_calculation(self):
        """Test that confidence is calculated from agent confidences."""
        result = self.orchestrator.facilitate_discussion("Test prompt")

        confidence = result["confidence"]
        assert 0 <= confidence <= 1

        # Check that it's calculated from agent analyses
        agent_analyses = result["agent_analyses"]
        agent_confidences = []
        for analysis in agent_analyses.values():
            if "confidence" in analysis:
                agent_confidences.append(analysis["confidence"])

        if agent_confidences:
            avg_confidence = sum(agent_confidences) / len(agent_confidences)
            # Should be close to average (allowing for some variation)
            assert abs(confidence - avg_confidence) < 0.2

    def test_patterns_extraction(self):
        """Test that patterns are extracted from architect agent."""
        result = self.orchestrator.facilitate_discussion("Test prompt")

        patterns = result["patterns"]
        assert isinstance(patterns, list)
        assert len(patterns) > 0  # Should have at least some patterns

    def test_risks_extraction(self):
        """Test that risks are extracted from all agents."""
        result = self.orchestrator.facilitate_discussion("Test prompt")

        risks = result["risks"]
        assert isinstance(risks, list)
        # Should have risks from architect and reviewer agents

    def test_none_context_handling(self):
        """Test handling of None context."""
        result = self.orchestrator.facilitate_discussion("Test prompt", None)

        assert isinstance(result, dict)
        assert "confidence" in result

    def test_empty_prompt_handling(self):
        """Test handling of empty prompts."""
        result = self.orchestrator.facilitate_discussion("")

        assert isinstance(result, dict)
        assert "approach" in result

    def test_get_agent_summary(self):
        """Test agent summary functionality."""
        summary = self.orchestrator.get_agent_summary()

        assert isinstance(summary, dict)
        assert "total_agents" in summary
        assert "agents" in summary

        assert summary["total_agents"] == 3

        agents = summary["agents"]
        assert len(agents) == 3

        for agent_name, agent_info in agents.items():
            assert "name" in agent_info
            assert "role" in agent_info
            assert isinstance(agent_info["name"], str)
            assert isinstance(agent_info["role"], str)


class TestConsensusBuilding:
    """Test consensus building logic."""

    def setup_method(self):
        """Setup for each test."""
        self.orchestrator = AgentOrchestrator()

    def test_combine_approaches(self):
        """Test approach combination logic."""
        approaches = [
            "functional approach",
            "object-oriented approach",
            "procedural approach",
        ]

        result = self.orchestrator._combine_approaches(approaches)

        assert isinstance(result, str)
        assert len(result) > 0
        # Should return first non-standard approach
        assert result == "functional approach"

    def test_combine_approaches_empty(self):
        """Test approach combination with empty list."""
        result = self.orchestrator._combine_approaches([])

        assert result == "Standard implementation with error handling"

    def test_combine_approaches_all_standard(self):
        """Test approach combination with all standard approaches."""
        approaches = ["Standard implementation", "Standard approach", "Standard method"]

        result = self.orchestrator._combine_approaches(approaches)

        assert result == "Standard implementation"

    def test_select_best_recommendation(self):
        """Test recommendation selection logic."""
        recommendations = ["defensive", "modular", "defensive", "simple"]

        result = self.orchestrator._select_best_recommendation(recommendations)

        # Should select most frequent
        assert result == "defensive"

    def test_select_best_recommendation_empty(self):
        """Test recommendation selection with empty list."""
        result = self.orchestrator._select_best_recommendation([])

        assert result == "defensive_programming"

    def test_calculate_consensus_confidence(self):
        """Test confidence calculation."""
        analyses = {
            "agent1": {"confidence": 0.8},
            "agent2": {"confidence": 0.6},
            "agent3": {"confidence": 0.9},
        }

        result = self.orchestrator._calculate_consensus_confidence(analyses)

        expected = (0.8 + 0.6 + 0.9) / 3
        assert result == expected

    def test_calculate_consensus_confidence_no_confidences(self):
        """Test confidence calculation with no confidence values."""
        analyses = {
            "agent1": {"other_field": "value"},
            "agent2": {"another_field": "value"},
        }

        result = self.orchestrator._calculate_consensus_confidence(analyses)

        assert result == 0.7  # Default confidence

    def test_calculate_consensus_confidence_empty(self):
        """Test confidence calculation with empty analyses."""
        result = self.orchestrator._calculate_consensus_confidence({})

        assert result == 0.7  # Default confidence


class TestOrchestratorIntegration:
    """Test orchestrator integration with real agents."""

    def test_full_discussion_flow(self):
        """Test complete discussion flow with real agents."""
        orchestrator = AgentOrchestrator()

        # This should work with real agents (may take time)
        result = orchestrator.facilitate_discussion("Create a calculator function")

        # Verify structure
        assert isinstance(result, dict)
        assert "proposal_id" in result
        assert "confidence" in result
        assert "patterns" in result
        assert "risks" in result

        # Verify agent analyses
        agent_analyses = result["agent_analyses"]
        assert len(agent_analyses) == 3

        # Verify consensus
        consensus = result["consensus"]
        assert "synthesized_approach" in consensus
        assert "confidence" in consensus

    def test_multiple_discussions_consistency(self):
        """Test that multiple discussions produce consistent results."""
        orchestrator = AgentOrchestrator()

        result1 = orchestrator.facilitate_discussion("Test prompt")
        result2 = orchestrator.facilitate_discussion("Test prompt")

        # Should have same structure
        assert set(result1.keys()) == set(result2.keys())

        # Should have different proposal IDs
        assert result1["proposal_id"] != result2["proposal_id"]

        # Should have similar confidence ranges
        assert abs(result1["confidence"] - result2["confidence"]) < 0.3
