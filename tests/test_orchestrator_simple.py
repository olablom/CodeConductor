"""
Simplified unit tests for AgentOrchestrator
"""

import unittest
from unittest.mock import Mock
import sys
import os

# Add the parent directory to the path to import the agents module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.orchestrator import AgentOrchestrator, DiscussionRound
from agents.base_agent import BaseAgent


class MockAgent(BaseAgent):
    """Mock agent for testing"""

    def __init__(
        self,
        name: str,
        analysis_result: dict = None,
        proposal_result: dict = None,
        review_result: dict = None,
    ):
        super().__init__(name)
        self.analysis_result = analysis_result or {}
        self.proposal_result = proposal_result or {}
        self.review_result = review_result or {}

    def analyze(self, context: dict) -> dict:
        return self.analysis_result

    def propose(self, analysis: dict, context: dict) -> dict:
        return self.proposal_result

    def review(self, proposal: dict, context: dict) -> dict:
        return self.review_result


class TestAgentOrchestratorSimple(unittest.TestCase):
    """Simplified test cases for AgentOrchestrator"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_agents = [
            MockAgent(
                "Agent1",
                analysis_result={"score": 0.8, "issues": ["issue1"]},
                proposal_result={"approval": "approve", "score": 0.9},
            ),
            MockAgent(
                "Agent2",
                analysis_result={"score": 0.7, "issues": ["issue2"]},
                proposal_result={"approval": "approve", "score": 0.8},
            ),
            MockAgent(
                "Agent3",
                analysis_result={"score": 0.9, "issues": ["issue3"]},
                proposal_result={"approval": "reject", "score": 0.6},
            ),
        ]

        self.orchestrator = AgentOrchestrator(
            agents=self.mock_agents,
            config={"consensus_strategy": "majority", "max_rounds": 3},
        )

        self.sample_context = {
            "requirements": {"functionality": "test"},
            "constraints": {"time": "limited"},
        }

    def test_init(self):
        """Test AgentOrchestrator initialization"""
        orchestrator = AgentOrchestrator(
            agents=self.mock_agents,
            config={"consensus_strategy": "majority", "max_rounds": 5},
        )

        self.assertEqual(len(orchestrator.agents), 3)
        self.assertEqual(orchestrator.config["consensus_strategy"], "majority")
        self.assertEqual(orchestrator.config["max_rounds"], 5)

    def test_init_with_defaults(self):
        """Test AgentOrchestrator initialization with defaults"""
        orchestrator = AgentOrchestrator(agents=self.mock_agents)

        self.assertEqual(orchestrator.config["consensus_strategy"], "weighted_majority")
        self.assertEqual(orchestrator.config["max_rounds"], 3)
        self.assertEqual(orchestrator.config["consensus_threshold"], 0.7)

    def test_run_discussion_basic(self):
        """Test basic discussion run"""
        result = self.orchestrator.run_discussion(self.sample_context)

        self.assertIn("consensus", result)
        self.assertIn("discussion_rounds", result)
        self.assertIn("consensus_reached", result)
        self.assertIn("discussion_summary", result)
        self.assertIn("metadata", result)

        self.assertIsInstance(result["discussion_rounds"], int)
        self.assertIsInstance(result["consensus_reached"], bool)
        self.assertIsInstance(result["discussion_summary"], dict)

    def test_run_discussion_with_consensus(self):
        """Test discussion that reaches consensus"""
        # All agents approve
        approving_agents = [
            MockAgent(
                "Agent1",
                analysis_result={"score": 0.8},
                proposal_result={"approval": "approve", "score": 0.9},
            ),
            MockAgent(
                "Agent2",
                analysis_result={"score": 0.7},
                proposal_result={"approval": "approve", "score": 0.8},
            ),
            MockAgent(
                "Agent3",
                analysis_result={"score": 0.9},
                proposal_result={"approval": "approve", "score": 0.7},
            ),
        ]

        orchestrator = AgentOrchestrator(agents=approving_agents)
        result = orchestrator.run_discussion(self.sample_context)

        # Should reach consensus with all agents approving
        self.assertTrue(result["consensus_reached"])
        self.assertIsNotNone(result["consensus"])

    def test_run_discussion_without_consensus(self):
        """Test discussion that doesn't reach consensus"""
        # Mixed approvals
        mixed_agents = [
            MockAgent(
                "Agent1",
                analysis_result={"score": 0.8},
                proposal_result={"approval": "approve", "score": 0.9},
            ),
            MockAgent(
                "Agent2",
                analysis_result={"score": 0.7},
                proposal_result={"approval": "reject", "score": 0.3},
            ),
            MockAgent(
                "Agent3",
                analysis_result={"score": 0.9},
                proposal_result={"approval": "approve_with_caution", "score": 0.6},
            ),
        ]

        orchestrator = AgentOrchestrator(
            agents=mixed_agents,
            config={"consensus_strategy": "unanimous", "max_rounds": 1},
        )
        result = orchestrator.run_discussion(self.sample_context)

        # Should not reach consensus with unanimous strategy
        self.assertFalse(result["consensus_reached"])
        self.assertIsNone(result["consensus"])

    def test_run_discussion_max_rounds_exceeded(self):
        """Test discussion that exceeds max rounds"""
        # Agents that never reach consensus
        disagreeing_agents = [
            MockAgent(
                "Agent1",
                analysis_result={"score": 0.8},
                proposal_result={"approval": "approve", "score": 0.9},
            ),
            MockAgent(
                "Agent2",
                analysis_result={"score": 0.7},
                proposal_result={"approval": "reject", "score": 0.3},
            ),
        ]

        orchestrator = AgentOrchestrator(
            agents=disagreeing_agents, config={"max_rounds": 1}
        )
        result = orchestrator.run_discussion(self.sample_context)

        self.assertFalse(result["consensus_reached"])
        self.assertEqual(result["discussion_rounds"], 1)

    def test_run_discussion_with_error(self):
        """Test discussion with agent error"""
        error_agent = MockAgent("ErrorAgent")
        error_agent.analyze = Mock(side_effect=Exception("Test error"))

        orchestrator = AgentOrchestrator(agents=[error_agent])
        result = orchestrator.run_discussion(self.sample_context)

        # Should handle the error gracefully
        self.assertIsNotNone(result)
        self.assertIn("consensus", result)
        self.assertIn("discussion_rounds", result)

    def test_discussion_history(self):
        """Test discussion history tracking"""
        self.orchestrator.run_discussion(self.sample_context)

        history = self.orchestrator.get_discussion_history()
        self.assertIsInstance(history, list)
        self.assertGreater(len(history), 0)

        for round_data in history:
            self.assertIsInstance(round_data, DiscussionRound)
            self.assertIsInstance(round_data.round_id, int)
            self.assertIsInstance(round_data.task_context, dict)
            self.assertIsInstance(round_data.analyses, list)
            self.assertIsInstance(round_data.proposals, list)

    def test_agent_statistics(self):
        """Test agent statistics"""
        self.orchestrator.run_discussion(self.sample_context)

        stats = self.orchestrator.get_agent_statistics()
        self.assertIsInstance(stats, dict)
        self.assertIn("total_agents", stats)
        self.assertIn("agent_names", stats)
        self.assertIn("agent_performance", stats)
        self.assertIn("agent_weights", stats)

    def test_reset_discussion(self):
        """Test discussion reset"""
        # Run a discussion first
        self.orchestrator.run_discussion(self.sample_context)

        # Verify history exists
        self.assertGreater(len(self.orchestrator.get_discussion_history()), 0)

        # Reset discussion
        self.orchestrator.reset_discussion()

        # Verify history is cleared
        self.assertEqual(len(self.orchestrator.get_discussion_history()), 0)

    def test_different_consensus_strategies(self):
        """Test different consensus strategies"""
        strategies = ["majority", "unanimous", "weighted_majority"]

        for strategy in strategies:
            orchestrator = AgentOrchestrator(
                agents=self.mock_agents, config={"consensus_strategy": strategy}
            )
            result = orchestrator.run_discussion(self.sample_context)

            self.assertIsNotNone(result)
            self.assertIn("consensus_reached", result)
            self.assertIn("consensus", result)


class TestDiscussionRound(unittest.TestCase):
    """Test cases for DiscussionRound dataclass"""

    def test_discussion_round_creation(self):
        """Test DiscussionRound creation"""
        round_obj = DiscussionRound(
            round_id=1,
            task_context={"test": "context"},
            analyses=[{"agent_name": "Agent1", "score": 0.8}],
            proposals=[{"agent_name": "Agent1", "approval": "approve"}],
            consensus={"decision": "approve"},
            metadata={"consensus_reached": True},
        )

        self.assertEqual(round_obj.round_id, 1)
        self.assertEqual(round_obj.task_context, {"test": "context"})
        self.assertEqual(len(round_obj.analyses), 1)
        self.assertEqual(len(round_obj.proposals), 1)
        self.assertEqual(round_obj.consensus["decision"], "approve")
        self.assertTrue(round_obj.metadata["consensus_reached"])

    def test_discussion_round_defaults(self):
        """Test DiscussionRound with default values"""
        round_obj = DiscussionRound(
            round_id=1, task_context={}, analyses=[], proposals=[]
        )

        self.assertEqual(round_obj.round_id, 1)
        self.assertIsNone(round_obj.consensus)
        self.assertIsNone(round_obj.metadata)


if __name__ == "__main__":
    unittest.main()
