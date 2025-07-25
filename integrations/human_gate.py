"""
HumanGate - Human-in-the-Loop approval system.

Required component for Gabriel's vision of controlled deployment.
"""

from typing import Dict, Any, Tuple
import json
from datetime import datetime
from pathlib import Path


class HumanGate:
    """Hanterar human approval för AI-förslag."""

    def __init__(self, approval_log_path: str = "data/approval_log.json"):
        self.approval_log_path = Path(approval_log_path)
        self.approval_log_path.parent.mkdir(parents=True, exist_ok=True)

        # Ladda tidigare approvals
        self.approval_history = self._load_approval_history()

    def request_approval(self, proposal: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Begär human approval för ett AI-förslag.

        Args:
            proposal: AI-förslaget att godkänna

        Returns:
            Tuple av (approved, final_proposal)
        """
        # Handle None proposal
        if proposal is None:
            proposal = {}

        print("\n" + "=" * 60)
        print("🤖 AI CONSENSUS PROPOSAL")
        print("=" * 60)
        print(f"📝 Prompt: {proposal.get('prompt', 'Unknown')}")
        print(f"🎯 Approach: {proposal.get('approach', 'Unknown')}")
        print(f"📊 Confidence: {proposal.get('confidence', 0):.1%}")
        print(f"🧠 RL Score: {proposal.get('rl_score', 0):.2f}")
        print(f"🏗️ Patterns: {', '.join(proposal.get('patterns', []))}")
        print(f"⚠️ Risks: {', '.join(proposal.get('risks', []))}")
        print(f"⚡ Optimization: {proposal.get('optimization', 'none')}")
        print("=" * 60)

        # Visa agent-analyser
        print("\n🔍 Agent Analyses:")
        agent_analyses = proposal.get("agent_analyses", {})
        for agent_name, analysis in agent_analyses.items():
            agent_name_display = analysis.get("agent", agent_name)
            recommendation = analysis.get("recommendation", "analysis complete")
            print(f"  - {agent_name_display}: {recommendation}")

        print("=" * 60)

        # Begär approval
        while True:
            response = input("\n✅ Approve? (y/n/edit/explain): ").lower().strip()

            if response == "y":
                return self._approve_proposal(proposal)
            elif response == "n":
                return self._reject_proposal(proposal)
            elif response == "edit":
                return self._edit_proposal(proposal)
            elif response == "explain":
                self._explain_proposal(proposal)
            else:
                print(
                    "❌ Invalid response. Use: y (approve), n (reject), edit, or explain"
                )

    def _approve_proposal(
        self, proposal: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Godkänner förslaget."""
        approval_data = {
            "approved": True,
            "timestamp": datetime.now().isoformat(),
            "reason": "human_approved",
            "proposal_id": proposal.get("proposal_id", "unknown"),
        }

        # Logga approval
        self._log_decision(proposal, approval_data)

        print("✅ Proposal APPROVED!")
        return True, proposal

    def _reject_proposal(self, proposal: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Avvisar förslaget."""
        feedback = input("Why rejected? (optional): ").strip()

        rejection_data = {
            "approved": False,
            "timestamp": datetime.now().isoformat(),
            "reason": "human_rejected",
            "feedback": feedback,
            "proposal_id": proposal.get("proposal_id", "unknown"),
        }

        # Logga rejection
        self._log_decision(proposal, rejection_data)

        print("❌ Proposal REJECTED!")
        return False, None

    def _edit_proposal(self, proposal: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Redigerar förslaget."""
        print("\n✏️ Editing proposal...")

        # Enkel redigering - låt användaren ändra approach
        new_approach = input(
            f"Current approach: {proposal.get('approach', 'Unknown')}\nNew approach (or press Enter to keep): "
        ).strip()

        if new_approach:
            proposal["approach"] = new_approach
            proposal["edited_by_human"] = True
            proposal["edit_timestamp"] = datetime.now().isoformat()

        # Logga edit
        edit_data = {
            "approved": True,
            "timestamp": datetime.now().isoformat(),
            "reason": "human_edited",
            "original_approach": proposal.get("approach", "Unknown"),
            "new_approach": new_approach if new_approach else "unchanged",
            "proposal_id": proposal.get("proposal_id", "unknown"),
        }

        self._log_decision(proposal, edit_data)

        print("✅ Proposal EDITED and APPROVED!")
        return True, proposal

    def _explain_proposal(self, proposal: Dict[str, Any]):
        """Förklarar förslaget i detalj."""
        print("\n📚 Detailed Explanation:")
        print(
            f"  - This proposal was generated by {len(proposal.get('agent_analyses', {}))} AI agents"
        )
        print(f"  - The consensus confidence is {proposal.get('confidence', 0):.1%}")
        print(
            f"  - The RL optimization applied: {proposal.get('optimization', 'none')}"
        )

        # Visa detaljerad konsensus
        consensus = proposal.get("consensus", {})
        if consensus:
            print(
                f"  - Synthesized approach: {consensus.get('synthesized_approach', 'Unknown')}"
            )
            print(
                f"  - Consensus recommendation: {consensus.get('consensus_recommendation', 'Unknown')}"
            )

    def _log_decision(self, proposal: Dict[str, Any], decision: Dict[str, Any]):
        """Loggar human decision."""
        log_entry = {"proposal": proposal, "decision": decision}

        self.approval_history.append(log_entry)

        # Spara till fil
        with open(self.approval_log_path, "w") as f:
            json.dump(self.approval_history, f, indent=2)

    def _load_approval_history(self) -> list:
        """Laddar tidigare approval-historik."""
        if self.approval_log_path.exists():
            try:
                with open(self.approval_log_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load approval history: {e}")

        return []

    def get_approval_stats(self) -> Dict[str, Any]:
        """Returnerar approval-statistik."""
        if not self.approval_history:
            return {
                "total_decisions": 0,
                "approved": 0,
                "rejected": 0,
                "edited": 0,
                "approval_rate": 0.0,
            }

        total = len(self.approval_history)
        approved = sum(
            1 for entry in self.approval_history if entry["decision"]["approved"]
        )
        rejected = sum(
            1 for entry in self.approval_history if not entry["decision"]["approved"]
        )
        edited = sum(
            1
            for entry in self.approval_history
            if entry["decision"].get("reason") == "human_edited"
        )

        return {
            "total_decisions": total,
            "approved": approved,
            "rejected": rejected,
            "edited": edited,
            "approval_rate": approved / total if total > 0 else 0.0,
        }
