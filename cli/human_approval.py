#!/usr/bin/env python3
"""
Human Approval CLI for CodeConductor v2.0

This module provides a command-line interface for human users to review,
approve, reject, or edit agent consensus proposals before code generation.
"""

import json
import sys
import tempfile
import subprocess
import os
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ApprovalResult:
    """Result of human approval process."""

    approved: bool
    proposal: Dict[str, Any]
    user_decision: str
    timestamp: datetime
    comments: Optional[str] = None


class HumanApprovalCLI:
    """
    CLI interface for human approval of agent consensus proposals.

    Provides an interactive interface where users can:
    - Review consensus proposals
    - Approve, reject, or edit proposals
    - Add comments and feedback
    """

    def __init__(self):
        self.editor = os.environ.get("EDITOR", "notepad" if os.name == "nt" else "nano")
        self.history = []

    def display_proposal(self, proposal: Dict[str, Any]) -> None:
        """
        Display the consensus proposal in a user-friendly format.

        Args:
            proposal: The consensus proposal from agents
        """
        print("\n" + "=" * 80)
        print("🤖 AGENT CONSENSUS PROPOSAL")
        print("=" * 80)

        # Display basic info
        if "title" in proposal:
            print(f"📋 Title: {proposal['title']}")

        if "summary" in proposal:
            print(f"📝 Summary: {proposal['summary']}")

        if "consensus_reached" in proposal:
            status = "✅ REACHED" if proposal["consensus_reached"] else "❌ NOT REACHED"
            print(f"🎯 Consensus Status: {status}")

        if "discussion_rounds" in proposal:
            print(f"🔄 Discussion Rounds: {proposal['discussion_rounds']}")

        # Display consensus details
        if "consensus" in proposal and proposal["consensus"]:
            consensus = proposal["consensus"]
            print(f"\n🎯 Consensus Decision: {consensus.get('decision', 'Unknown')}")
            print(f"📊 Confidence Score: {consensus.get('confidence', 0.0):.2f}")

            if "reasoning" in consensus:
                print(f"\n💭 Reasoning:")
                print(f"   {consensus['reasoning']}")

        # Display discussion summary
        if "discussion_summary" in proposal:
            summary = proposal["discussion_summary"]
            print(f"\n📊 Discussion Summary:")

            if "agent_agreements" in summary:
                print(f"   🤝 Agent Agreements: {summary['agent_agreements']}")

            if "key_points" in summary:
                print(f"   🔑 Key Points:")
                for i, point in enumerate(summary["key_points"], 1):
                    print(f"      {i}. {point}")

            if "concerns" in summary:
                print(f"   ⚠️  Concerns:")
                for i, concern in enumerate(summary["concerns"], 1):
                    print(f"      {i}. {concern}")

        # Display metadata
        if "metadata" in proposal:
            metadata = proposal["metadata"]
            print(f"\n📋 Metadata:")
            print(f"   🕒 Timestamp: {metadata.get('timestamp', 'Unknown')}")
            print(f"   🤖 Agents: {', '.join(metadata.get('agent_names', []))}")
            print(f"   ⚙️  Strategy: {metadata.get('consensus_strategy', 'Unknown')}")

        print("=" * 80)

    def get_user_decision(self) -> str:
        """
        Get user decision through interactive CLI.

        Returns:
            User's decision: 'approve', 'reject', 'edit', or 'help'
        """
        while True:
            print("\n🤔 What would you like to do?")
            print("   [A] Approve - Accept the proposal and continue")
            print("   [R] Reject - Reject the proposal and stop")
            print("   [E] Edit - Modify the proposal")
            print("   [H] Help - Show detailed help")
            print("   [Q] Quit - Exit without decision")

            choice = input("\nEnter your choice (A/R/E/H/Q): ").strip().upper()

            if choice in ["A", "APPROVE"]:
                return "approve"
            elif choice in ["R", "REJECT"]:
                return "reject"
            elif choice in ["E", "EDIT"]:
                return "edit"
            elif choice in ["H", "HELP"]:
                self.show_help()
            elif choice in ["Q", "QUIT"]:
                return "quit"
            else:
                print("❌ Invalid choice. Please try again.")

    def show_help(self) -> None:
        """Display detailed help information."""
        print("\n" + "=" * 60)
        print("📖 HUMAN APPROVAL HELP")
        print("=" * 60)
        print("""
This interface allows you to review and approve agent consensus proposals.

OPTIONS:
  Approve (A): Accept the current proposal and continue to code generation
  Reject (R): Reject the proposal and stop the process
  Edit (E): Modify the proposal in your default text editor
  Help (H): Show this help message
  Quit (Q): Exit without making a decision

EDITING:
  When you choose to edit, the proposal will be opened in your default editor.
  The file will be in JSON format. Make your changes and save the file.
  The system will validate your changes before accepting them.

CONSENSUS PROPOSAL STRUCTURE:
  - title: Brief description of the proposal
  - summary: Detailed summary of the proposal
  - consensus: The agreed-upon decision and reasoning
  - discussion_summary: Key points and concerns from agent discussion
  - metadata: Technical details about the discussion process

TIPS:
  - Review the consensus reasoning carefully
  - Check for any concerns raised by agents
  - Consider the confidence score when making your decision
  - Use edit mode to refine proposals that are mostly good
        """)
        print("=" * 60)

    def edit_proposal(self, proposal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Allow user to edit the proposal in their default editor.

        Args:
            proposal: Original proposal to edit

        Returns:
            Edited proposal or None if editing was cancelled
        """
        print(f"\n📝 Opening proposal in {self.editor} for editing...")

        # Create temporary file with proposal
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(proposal, f, indent=2, default=str)
            temp_file = f.name

        try:
            # Open file in user's default editor
            subprocess.run([self.editor, temp_file], check=True)

            # Read back the edited file
            with open(temp_file, "r") as f:
                edited_content = f.read()

            # Parse the edited JSON
            try:
                edited_proposal = json.loads(edited_content)
                print("✅ Proposal edited successfully!")
                return edited_proposal

            except json.JSONDecodeError as e:
                print(f"❌ Invalid JSON in edited file: {e}")
                print("Please fix the JSON syntax and try again.")
                return None

        except subprocess.CalledProcessError:
            print(f"❌ Failed to open editor: {self.editor}")
            print(
                "Please set the EDITOR environment variable to your preferred editor."
            )
            return None
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file)
            except OSError:
                pass

    def get_comments(self) -> Optional[str]:
        """
        Get optional comments from the user.

        Returns:
            User comments or None
        """
        print("\n💬 Would you like to add any comments? (optional)")
        print("   Press Enter to skip, or type your comments:")

        comments = input().strip()
        return comments if comments else None

    def process_approval(self, proposal: Dict[str, Any]) -> ApprovalResult:
        """
        Process the human approval workflow.

        Args:
            proposal: The consensus proposal to review

        Returns:
            ApprovalResult with user's decision and any modifications
        """
        # Display the proposal
        self.display_proposal(proposal)

        # Get user decision
        decision = self.get_user_decision()

        if decision == "quit":
            print("\n👋 Exiting without decision.")
            sys.exit(0)

        # Handle different decisions
        if decision == "approve":
            comments = self.get_comments()
            print("\n✅ Proposal APPROVED!")
            return ApprovalResult(
                approved=True,
                proposal=proposal,
                user_decision="approve",
                timestamp=datetime.now(),
                comments=comments,
            )

        elif decision == "reject":
            comments = self.get_comments()
            print("\n❌ Proposal REJECTED!")
            return ApprovalResult(
                approved=False,
                proposal=proposal,
                user_decision="reject",
                timestamp=datetime.now(),
                comments=comments,
            )

        elif decision == "edit":
            edited_proposal = self.edit_proposal(proposal)
            if edited_proposal is None:
                print("\n🔄 Returning to decision menu...")
                return self.process_approval(proposal)  # Recursive call

            # Show the edited proposal and get final decision
            print("\n📋 Here's your edited proposal:")
            self.display_proposal(edited_proposal)

            final_decision = self.get_user_decision()
            comments = self.get_comments()

            if final_decision == "approve":
                print("\n✅ Edited proposal APPROVED!")
                return ApprovalResult(
                    approved=True,
                    proposal=edited_proposal,
                    user_decision="approve_edited",
                    timestamp=datetime.now(),
                    comments=comments,
                )
            elif final_decision == "reject":
                print("\n❌ Edited proposal REJECTED!")
                return ApprovalResult(
                    approved=False,
                    proposal=edited_proposal,
                    user_decision="reject_edited",
                    timestamp=datetime.now(),
                    comments=comments,
                )
            else:
                print("\n🔄 Returning to decision menu...")
                return self.process_approval(edited_proposal)  # Recursive call

    def save_approval_history(self, result: ApprovalResult) -> None:
        """
        Save approval result to history.

        Args:
            result: The approval result to save
        """
        self.history.append(
            {
                "timestamp": result.timestamp.isoformat(),
                "decision": result.user_decision,
                "approved": result.approved,
                "comments": result.comments,
                "proposal_title": result.proposal.get("title", "Unknown"),
            }
        )

    def get_approval_history(self) -> list:
        """
        Get approval history.

        Returns:
            List of previous approval decisions
        """
        return self.history


def main():
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Human Approval CLI for CodeConductor")
    parser.add_argument("--proposal", "-p", type=str, help="Path to proposal JSON file")
    parser.add_argument("--demo", action="store_true", help="Run with demo proposal")

    args = parser.parse_args()

    cli = HumanApprovalCLI()

    if args.demo:
        # Demo proposal
        demo_proposal = {
            "title": "Create REST API for User Management",
            "summary": "Build a FastAPI-based REST API with CRUD operations for user management",
            "consensus_reached": True,
            "discussion_rounds": 2,
            "consensus": {
                "decision": "approve",
                "confidence": 0.85,
                "reasoning": "All agents agree this is a well-defined, achievable task with clear requirements.",
            },
            "discussion_summary": {
                "agent_agreements": 3,
                "key_points": [
                    "Use FastAPI for modern, fast API development",
                    "Implement proper validation with Pydantic",
                    "Include authentication and authorization",
                    "Add comprehensive error handling",
                ],
                "concerns": [
                    "Need to define specific user fields",
                    "Consider rate limiting for production",
                ],
            },
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "agent_names": ["CodeGenAgent", "ArchitectAgent", "ReviewerAgent"],
                "consensus_strategy": "weighted_majority",
            },
        }

        result = cli.process_approval(demo_proposal)
        # Save to history
        cli.save_approval_history(result)

        # Exit with appropriate code
        if result.approved:
            print(f"\n🚀 Proceeding to code generation...")
            sys.exit(0)
        else:
            print(f"\n🛑 Process stopped due to rejection.")
            sys.exit(1)

    elif args.proposal:
        # Load proposal from file
        try:
            with open(args.proposal, "r") as f:
                proposal = json.load(f)
            result = cli.process_approval(proposal)
            # Save to history
            cli.save_approval_history(result)

            # Exit with appropriate code
            if result.approved:
                print(f"\n🚀 Proceeding to code generation...")
                sys.exit(0)
            else:
                print(f"\n🛑 Process stopped due to rejection.")
                sys.exit(1)
        except FileNotFoundError:
            print(f"❌ Proposal file not found: {args.proposal}")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON in proposal file: {args.proposal}")
            sys.exit(1)
    else:
        print("❌ Please provide either --proposal or --demo")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
