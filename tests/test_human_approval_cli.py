"""
Tests for Human Approval CLI

Tests the CLI interface for human approval of agent consensus proposals.
"""

import unittest
from unittest.mock import Mock, patch, mock_open
import json
import sys
import tempfile
import os
from datetime import datetime
import subprocess

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cli.human_approval import HumanApprovalCLI, ApprovalResult


class TestHumanApprovalCLI(unittest.TestCase):
    """Test cases for HumanApprovalCLI class."""

    def setUp(self):
        """Set up test fixtures."""
        self.cli = HumanApprovalCLI()

        # Sample proposal for testing
        self.sample_proposal = {
            "title": "Test Proposal",
            "summary": "A test proposal for unit testing",
            "consensus_reached": True,
            "discussion_rounds": 2,
            "consensus": {
                "decision": "approve",
                "confidence": 0.85,
                "reasoning": "Test reasoning",
            },
            "discussion_summary": {
                "agent_agreements": 3,
                "key_points": ["Point 1", "Point 2"],
                "concerns": ["Concern 1"],
            },
            "metadata": {
                "timestamp": "2024-01-01T00:00:00",
                "agent_names": ["Agent1", "Agent2"],
                "consensus_strategy": "majority",
            },
        }

    def test_init(self):
        """Test CLI initialization."""
        cli = HumanApprovalCLI()
        self.assertIsNotNone(cli.editor)
        self.assertEqual(cli.history, [])

    def test_display_proposal(self):
        """Test proposal display functionality."""
        # Capture stdout to verify display
        with patch("sys.stdout") as mock_stdout:
            self.cli.display_proposal(self.sample_proposal)

            # Verify that display was called
            self.assertTrue(mock_stdout.write.called)

    def test_get_user_decision_approve(self):
        """Test user decision - approve."""
        with patch("builtins.input", return_value="A"):
            decision = self.cli.get_user_decision()
            self.assertEqual(decision, "approve")

    def test_get_user_decision_reject(self):
        """Test user decision - reject."""
        with patch("builtins.input", return_value="R"):
            decision = self.cli.get_user_decision()
            self.assertEqual(decision, "reject")

    def test_get_user_decision_edit(self):
        """Test user decision - edit."""
        with patch("builtins.input", return_value="E"):
            decision = self.cli.get_user_decision()
            self.assertEqual(decision, "edit")

    def test_get_user_decision_quit(self):
        """Test user decision - quit."""
        with patch("builtins.input", return_value="Q"):
            decision = self.cli.get_user_decision()
            self.assertEqual(decision, "quit")

    def test_get_user_decision_invalid_then_valid(self):
        """Test invalid input followed by valid input."""
        with patch("builtins.input", side_effect=["INVALID", "A"]):
            decision = self.cli.get_user_decision()
            self.assertEqual(decision, "approve")

    def test_show_help(self):
        """Test help display."""
        with patch("sys.stdout") as mock_stdout:
            self.cli.show_help()
            self.assertTrue(mock_stdout.write.called)

    @patch("subprocess.run")
    @patch("tempfile.NamedTemporaryFile")
    def test_edit_proposal_success(self, mock_temp_file, mock_subprocess):
        """Test successful proposal editing."""
        # Mock temporary file
        mock_file = Mock()
        mock_file.name = "/tmp/test.json"
        mock_temp_file.return_value.__enter__.return_value = mock_file

        # Mock subprocess success
        mock_subprocess.return_value = Mock()

        # Mock file reading
        edited_content = json.dumps(self.sample_proposal)
        with patch("builtins.open", mock_open(read_data=edited_content)):
            with patch("os.unlink"):
                result = self.cli.edit_proposal(self.sample_proposal)

        self.assertIsNotNone(result)
        self.assertEqual(result, self.sample_proposal)

    @patch("subprocess.run")
    @patch("tempfile.NamedTemporaryFile")
    def test_edit_proposal_invalid_json(self, mock_temp_file, mock_subprocess):
        """Test proposal editing with invalid JSON."""
        # Mock temporary file
        mock_file = Mock()
        mock_file.name = "/tmp/test.json"
        mock_temp_file.return_value.__enter__.return_value = mock_file

        # Mock subprocess success
        mock_subprocess.return_value = Mock()

        # Mock file reading with invalid JSON
        with patch("builtins.open", mock_open(read_data="invalid json")):
            with patch("os.unlink"):
                result = self.cli.edit_proposal(self.sample_proposal)

        self.assertIsNone(result)

    @patch("subprocess.run")
    @patch("tempfile.NamedTemporaryFile")
    def test_edit_proposal_editor_error(self, mock_temp_file, mock_subprocess):
        """Test proposal editing with editor error."""
        # Mock temporary file
        mock_file = Mock()
        mock_file.name = "/tmp/test.json"
        mock_temp_file.return_value.__enter__.return_value = mock_file

        # Mock subprocess error
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, "editor")

        with patch("os.unlink"):
            result = self.cli.edit_proposal(self.sample_proposal)

        self.assertIsNone(result)

    def test_get_comments_with_input(self):
        """Test getting comments with user input."""
        with patch("builtins.input", return_value="Test comment"):
            comments = self.cli.get_comments()
            self.assertEqual(comments, "Test comment")

    def test_get_comments_empty_input(self):
        """Test getting comments with empty input."""
        with patch("builtins.input", return_value=""):
            comments = self.cli.get_comments()
            self.assertIsNone(comments)

    @patch("cli.human_approval.HumanApprovalCLI.get_user_decision")
    @patch("cli.human_approval.HumanApprovalCLI.get_comments")
    def test_process_approval_approve(self, mock_get_comments, mock_get_decision):
        """Test approval process - approve."""
        mock_get_decision.return_value = "approve"
        mock_get_comments.return_value = "Test comment"

        result = self.cli.process_approval(self.sample_proposal)

        self.assertTrue(result.approved)
        self.assertEqual(result.user_decision, "approve")
        self.assertEqual(result.comments, "Test comment")
        self.assertEqual(result.proposal, self.sample_proposal)

    @patch("cli.human_approval.HumanApprovalCLI.get_user_decision")
    @patch("cli.human_approval.HumanApprovalCLI.get_comments")
    def test_process_approval_reject(self, mock_get_comments, mock_get_decision):
        """Test approval process - reject."""
        mock_get_decision.return_value = "reject"
        mock_get_comments.return_value = "Rejection reason"

        result = self.cli.process_approval(self.sample_proposal)

        self.assertFalse(result.approved)
        self.assertEqual(result.user_decision, "reject")
        self.assertEqual(result.comments, "Rejection reason")

    @patch("cli.human_approval.HumanApprovalCLI.get_user_decision")
    @patch("cli.human_approval.HumanApprovalCLI.edit_proposal")
    @patch("cli.human_approval.HumanApprovalCLI.get_comments")
    def test_process_approval_edit_then_approve(self, mock_get_comments, mock_edit_proposal, mock_get_decision):
        """Test approval process - edit then approve."""
        # First call returns edit, second call returns approve
        mock_get_decision.side_effect = ["edit", "approve"]
        mock_edit_proposal.return_value = self.sample_proposal
        mock_get_comments.return_value = "Edited and approved"

        result = self.cli.process_approval(self.sample_proposal)

        self.assertTrue(result.approved)
        self.assertEqual(result.user_decision, "approve_edited")
        self.assertEqual(result.comments, "Edited and approved")

    @patch("cli.human_approval.HumanApprovalCLI.get_user_decision")
    @patch("cli.human_approval.HumanApprovalCLI.edit_proposal")
    def test_process_approval_edit_failed(self, mock_edit_proposal, mock_get_decision):
        """Test approval process - edit failed."""
        mock_get_decision.return_value = "edit"
        mock_edit_proposal.return_value = None

        # Should return to decision menu, so we need to mock another decision
        with patch.object(self.cli, "process_approval") as mock_recursive:
            mock_recursive.return_value = ApprovalResult(
                approved=True,
                proposal=self.sample_proposal,
                user_decision="approve",
                timestamp=datetime.now(),
            )

            result = self.cli.process_approval(self.sample_proposal)

            # Verify recursive call was made
            mock_recursive.assert_called_once_with(self.sample_proposal)

    def test_save_approval_history(self):
        """Test saving approval history."""
        result = ApprovalResult(
            approved=True,
            proposal=self.sample_proposal,
            user_decision="approve",
            timestamp=datetime.now(),
            comments="Test comment",
        )

        self.cli.save_approval_history(result)

        self.assertEqual(len(self.cli.history), 1)
        history_entry = self.cli.history[0]
        self.assertEqual(history_entry["decision"], "approve")
        self.assertTrue(history_entry["approved"])
        self.assertEqual(history_entry["comments"], "Test comment")
        self.assertEqual(history_entry["proposal_title"], "Test Proposal")

    def test_get_approval_history(self):
        """Test getting approval history."""
        # Add some history
        result = ApprovalResult(
            approved=True,
            proposal=self.sample_proposal,
            user_decision="approve",
            timestamp=datetime.now(),
        )
        self.cli.save_approval_history(result)

        history = self.cli.get_approval_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["decision"], "approve")


class TestApprovalResult(unittest.TestCase):
    """Test cases for ApprovalResult dataclass."""

    def test_approval_result_creation(self):
        """Test ApprovalResult creation."""
        timestamp = datetime.now()
        proposal = {"title": "Test"}

        result = ApprovalResult(
            approved=True,
            proposal=proposal,
            user_decision="approve",
            timestamp=timestamp,
            comments="Test comment",
        )

        self.assertTrue(result.approved)
        self.assertEqual(result.proposal, proposal)
        self.assertEqual(result.user_decision, "approve")
        self.assertEqual(result.timestamp, timestamp)
        self.assertEqual(result.comments, "Test comment")

    def test_approval_result_default_comments(self):
        """Test ApprovalResult with default comments."""
        timestamp = datetime.now()
        proposal = {"title": "Test"}

        result = ApprovalResult(
            approved=False,
            proposal=proposal,
            user_decision="reject",
            timestamp=timestamp,
        )

        self.assertFalse(result.approved)
        self.assertEqual(result.user_decision, "reject")
        self.assertIsNone(result.comments)


class TestCLIMain(unittest.TestCase):
    """Test cases for CLI main function."""

    @patch("cli.human_approval.HumanApprovalCLI")
    @patch("sys.argv", ["human_approval.py", "--demo"])
    def test_main_demo(self, mock_cli_class):
        """Test main function with demo flag."""
        mock_cli = Mock()
        mock_cli_class.return_value = mock_cli

        # Mock the approval result
        mock_result = Mock()
        mock_result.approved = True
        mock_cli.process_approval.return_value = mock_result

        with patch("sys.exit") as mock_exit:
            from cli.human_approval import main

            main()

            # Verify CLI was called with demo proposal
            mock_cli.process_approval.assert_called_once()
            mock_cli.save_approval_history.assert_called_once_with(mock_result)
            mock_exit.assert_called_once_with(0)

    @patch("cli.human_approval.HumanApprovalCLI")
    @patch("builtins.open", mock_open(read_data='{"title": "Test"}'))
    @patch("sys.argv", ["human_approval.py", "--proposal", "test.json"])
    def test_main_proposal_file(self, mock_cli_class):
        """Test main function with proposal file."""
        mock_cli = Mock()
        mock_cli_class.return_value = mock_cli

        mock_result = Mock()
        mock_result.approved = False
        mock_cli.process_approval.return_value = mock_result

        with patch("sys.exit") as mock_exit:
            from cli.human_approval import main

            main()

            mock_exit.assert_called_once_with(1)

    @patch("sys.argv", ["human_approval.py"])
    def test_main_no_args(self):
        """Test main function with no arguments."""
        with patch("sys.exit") as mock_exit:
            from cli.human_approval import main

            main()

            mock_exit.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
