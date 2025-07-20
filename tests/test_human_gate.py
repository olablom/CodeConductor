"""
Tests for HumanGate component.

Tests human-in-the-loop approval system.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json
from datetime import datetime

from integrations.human_gate import HumanGate


class TestHumanGate:
    """Test HumanGate functionality."""

    def setup_method(self):
        """Setup for each test."""
        # Use temporary file for testing
        self.temp_log_path = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self.temp_log_path.close()
        self.human_gate = HumanGate(self.temp_log_path.name)

    def teardown_method(self):
        """Cleanup after each test."""
        # Remove temporary file
        Path(self.temp_log_path.name).unlink(missing_ok=True)

    def test_initialization(self):
        """Test that HumanGate initializes correctly."""
        assert hasattr(self.human_gate, "approval_log_path")
        assert hasattr(self.human_gate, "approval_history")
        assert isinstance(self.human_gate.approval_history, list)

    def test_approval_log_path_creation(self):
        """Test that approval log directory is created."""
        temp_dir = tempfile.mkdtemp()
        log_path = Path(temp_dir) / "nested" / "approval_log.json"

        human_gate = HumanGate(str(log_path))

        assert log_path.parent.exists()

        # Cleanup
        import shutil

        shutil.rmtree(temp_dir)

    def test_load_approval_history_empty(self):
        """Test loading approval history when file doesn't exist."""
        # Should return empty list for new file
        history = self.human_gate._load_approval_history()
        assert isinstance(history, list)
        assert len(history) == 0

    def test_load_approval_history_existing(self):
        """Test loading existing approval history."""
        # Create test data
        test_data = [
            {
                "proposal": {"test": "data"},
                "decision": {"approved": True, "timestamp": "2025-01-01T00:00:00Z"},
            }
        ]

        # Write to file
        with open(self.temp_log_path.name, "w") as f:
            json.dump(test_data, f)

        # Reload
        self.human_gate.approval_history = self.human_gate._load_approval_history()

        assert len(self.human_gate.approval_history) == 1
        assert self.human_gate.approval_history[0]["proposal"]["test"] == "data"

    @patch("builtins.input")
    def test_request_approval_approve(self, mock_input):
        """Test approval workflow."""
        mock_input.return_value = "y"

        proposal = {
            "prompt": "Test prompt",
            "approach": "Test approach",
            "confidence": 0.8,
            "rl_score": 0.75,
            "patterns": ["factory"],
            "risks": ["complexity"],
            "optimization": "add_examples",
            "agent_analyses": {"codegen": {"agent": "CodeGenAgent", "recommendation": "functional"}},
        }

        approved, final_proposal = self.human_gate.request_approval(proposal)

        assert approved is True
        assert final_proposal == proposal

        # Check that decision was logged
        assert len(self.human_gate.approval_history) == 1
        decision = self.human_gate.approval_history[0]["decision"]
        assert decision["approved"] is True
        assert decision["reason"] == "human_approved"

    @patch("builtins.input")
    def test_request_approval_reject(self, mock_input):
        """Test rejection workflow."""
        mock_input.side_effect = ["n", "Too complex"]

        proposal = {
            "prompt": "Test prompt",
            "approach": "Test approach",
            "confidence": 0.8,
        }

        approved, final_proposal = self.human_gate.request_approval(proposal)

        assert approved is False
        assert final_proposal is None

        # Check that decision was logged
        assert len(self.human_gate.approval_history) == 1
        decision = self.human_gate.approval_history[0]["decision"]
        assert decision["approved"] is False
        assert decision["reason"] == "human_rejected"
        assert decision["feedback"] == "Too complex"

    @patch("builtins.input")
    def test_request_approval_edit(self, mock_input):
        """Test edit workflow."""
        mock_input.side_effect = ["edit", "New approach"]

        proposal = {
            "prompt": "Test prompt",
            "approach": "Old approach",
            "confidence": 0.8,
        }

        approved, final_proposal = self.human_gate.request_approval(proposal)

        assert approved is True
        assert final_proposal["approach"] == "New approach"
        assert final_proposal["edited_by_human"] is True
        assert "edit_timestamp" in final_proposal

        # Check that decision was logged
        assert len(self.human_gate.approval_history) == 1
        decision = self.human_gate.approval_history[0]["decision"]
        assert decision["approved"] is True
        assert decision["reason"] == "human_edited"
        assert decision["new_approach"] == "New approach"

    @patch("builtins.input")
    def test_request_approval_explain(self, mock_input):
        """Test explain workflow."""
        mock_input.side_effect = ["explain", "y"]

        proposal = {
            "prompt": "Test prompt",
            "approach": "Test approach",
            "confidence": 0.8,
            "rl_score": 0.75,
            "optimization": "add_examples",
            "agent_analyses": {"codegen": {"agent": "CodeGenAgent"}},
            "consensus": {
                "synthesized_approach": "Combined approach",
                "consensus_recommendation": "defensive",
            },
        }

        approved, final_proposal = self.human_gate.request_approval(proposal)

        # Should still approve after explanation
        assert approved is True
        assert final_proposal == proposal

    @patch("builtins.input")
    def test_request_approval_invalid_input(self, mock_input):
        """Test handling of invalid input."""
        mock_input.side_effect = ["invalid", "y"]

        proposal = {
            "prompt": "Test prompt",
            "approach": "Test approach",
            "confidence": 0.8,
        }

        approved, final_proposal = self.human_gate.request_approval(proposal)

        # Should eventually approve after valid input
        assert approved is True
        assert final_proposal == proposal

    def test_log_decision(self):
        """Test decision logging."""
        proposal = {"test": "proposal"}
        decision = {"approved": True, "timestamp": "2025-01-01T00:00:00Z"}

        self.human_gate._log_decision(proposal, decision)

        # Check in-memory
        assert len(self.human_gate.approval_history) == 1
        assert self.human_gate.approval_history[0]["proposal"] == proposal
        assert self.human_gate.approval_history[0]["decision"] == decision

        # Check file
        with open(self.temp_log_path.name, "r") as f:
            saved_data = json.load(f)

        assert len(saved_data) == 1
        assert saved_data[0]["proposal"] == proposal
        assert saved_data[0]["decision"] == decision

    def test_get_approval_stats_empty(self):
        """Test approval statistics with no history."""
        stats = self.human_gate.get_approval_stats()

        assert stats["total_decisions"] == 0
        assert stats["approved"] == 0
        assert stats["rejected"] == 0
        assert stats["edited"] == 0
        assert stats["approval_rate"] == 0.0

    def test_get_approval_stats_with_data(self):
        """Test approval statistics with history."""
        # Add test data
        test_decisions = [
            {"approved": True, "reason": "human_approved"},
            {"approved": False, "reason": "human_rejected"},
            {"approved": True, "reason": "human_edited"},
            {"approved": True, "reason": "human_approved"},
        ]

        for decision in test_decisions:
            self.human_gate._log_decision({"test": "proposal"}, decision)

        stats = self.human_gate.get_approval_stats()

        assert stats["total_decisions"] == 4
        assert stats["approved"] == 3
        assert stats["rejected"] == 1
        assert stats["edited"] == 1
        assert stats["approval_rate"] == 0.75

    def test_explain_proposal(self):
        """Test proposal explanation."""
        proposal = {
            "agent_analyses": {"codegen": {"agent": "CodeGenAgent"}},
            "confidence": 0.8,
            "optimization": "add_examples",
            "consensus": {
                "synthesized_approach": "Combined approach",
                "consensus_recommendation": "defensive",
            },
        }

        # Should not raise any exceptions
        self.human_gate._explain_proposal(proposal)


class TestHumanGateIntegration:
    """Test HumanGate integration with real proposals."""

    def setup_method(self):
        """Setup for each test."""
        self.temp_log_path = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self.temp_log_path.close()
        self.human_gate = HumanGate(self.temp_log_path.name)

    def teardown_method(self):
        """Cleanup after each test."""
        Path(self.temp_log_path.name).unlink(missing_ok=True)

    def test_real_proposal_approval(self):
        """Test approval of a real proposal from orchestrator."""
        from agents.orchestrator import AgentOrchestrator
        from agents.codegen_agent import CodeGenAgent
        from agents.architect_agent import ArchitectAgent

        # Create agents for orchestrator
        agents = [CodeGenAgent(), ArchitectAgent()]
        orchestrator = AgentOrchestrator(agents=agents)

        # Create a simple proposal instead of running full discussion
        proposal = {
            "prompt": "Create a simple function",
            "confidence": 0.8,
            "agent_analyses": {"codegen": {"agent": "CodeGenAgent"}},
            "consensus": {"synthesized_approach": "Combined approach"},
        }

        # Mock approval
        with patch("builtins.input", return_value="y"):
            approved, final_proposal = self.human_gate.request_approval(proposal)

        assert approved is True
        assert final_proposal == proposal

        # Check statistics
        stats = self.human_gate.get_approval_stats()
        assert stats["total_decisions"] == 1
        assert stats["approved"] == 1
        assert stats["approval_rate"] == 1.0

    def test_multiple_decisions_tracking(self):
        """Test tracking of multiple decisions."""
        proposals = [
            {"prompt": "Test 1", "confidence": 0.8},
            {"prompt": "Test 2", "confidence": 0.9},
            {"prompt": "Test 3", "confidence": 0.7},
        ]

        decisions = ["y", "n", "edit"]

        for proposal, decision in zip(proposals, decisions):
            if decision == "edit":
                with patch("builtins.input", side_effect=[decision, "New approach"]):
                    self.human_gate.request_approval(proposal)
            elif decision == "n":
                with patch("builtins.input", side_effect=[decision, "Too complex"]):
                    self.human_gate.request_approval(proposal)
            else:
                with patch("builtins.input", side_effect=[decision]):
                    self.human_gate.request_approval(proposal)

        stats = self.human_gate.get_approval_stats()
        assert stats["total_decisions"] == 3
        assert stats["approved"] == 2
        assert stats["rejected"] == 1
        assert stats["edited"] == 1
        assert stats["approval_rate"] == 2 / 3

    def test_persistence_across_instances(self):
        """Test that decisions persist across HumanGate instances."""
        # First instance
        human_gate1 = HumanGate(self.temp_log_path.name)

        proposal = {"test": "proposal"}
        decision = {"approved": True, "reason": "test"}

        human_gate1._log_decision(proposal, decision)

        # Second instance should see the same data
        human_gate2 = HumanGate(self.temp_log_path.name)

        assert len(human_gate2.approval_history) == 1
        assert human_gate2.approval_history[0]["proposal"] == proposal
        assert human_gate2.approval_history[0]["decision"] == decision

        stats = human_gate2.get_approval_stats()
        assert stats["total_decisions"] == 1
        assert stats["approved"] == 1


class TestHumanGateEdgeCases:
    """Test HumanGate edge cases and error handling."""

    def setup_method(self):
        """Setup for each test."""
        self.temp_log_path = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self.temp_log_path.close()
        self.human_gate = HumanGate(self.temp_log_path.name)

    def teardown_method(self):
        """Cleanup after each test."""
        Path(self.temp_log_path.name).unlink(missing_ok=True)

    def test_empty_proposal_handling(self):
        """Test handling of empty proposals."""
        proposal = {}

        with patch("builtins.input", return_value="y"):
            approved, final_proposal = self.human_gate.request_approval(proposal)

        assert approved is True
        assert final_proposal == proposal

    def test_malformed_proposal_handling(self):
        """Test handling of malformed proposals."""
        proposal = None

        with patch("builtins.input", return_value="y"):
            approved, final_proposal = self.human_gate.request_approval(proposal)

        # Should handle gracefully
        assert approved is True

    def test_corrupted_log_file(self):
        """Test handling of corrupted log file."""
        # Write invalid JSON
        with open(self.temp_log_path.name, "w") as f:
            f.write("invalid json content")

        # Should handle gracefully
        human_gate = HumanGate(self.temp_log_path.name)
        assert isinstance(human_gate.approval_history, list)
        assert len(human_gate.approval_history) == 0

    def test_large_proposal_handling(self):
        """Test handling of large proposals."""
        large_proposal = {
            "prompt": "x" * 10000,  # Very long prompt
            "approach": "x" * 5000,  # Very long approach
            "agent_analyses": {
                "agent1": {"data": "x" * 1000},
                "agent2": {"data": "x" * 1000},
                "agent3": {"data": "x" * 1000},
            },
        }

        with patch("builtins.input", return_value="y"):
            approved, final_proposal = self.human_gate.request_approval(large_proposal)

        assert approved is True
        assert final_proposal == large_proposal
