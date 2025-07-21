"""
Auth Service - Human Gate

This module contains the HumanGate migrated from the main CodeConductor
codebase to the microservices architecture.
"""

import json
import logging
from typing import Dict, Any
from datetime import datetime
from pathlib import Path


class HumanGate:
    """Handles human approval for AI proposals in microservices environment."""

    def __init__(self, approval_log_path: str = "/tmp/approval_log.json"):
        """
        Initialize the human gate.

        Args:
            approval_log_path: Path to store approval logs
        """
        self.approval_log_path = Path(approval_log_path)
        self.approval_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Load previous approvals
        self.approval_history = self._load_approval_history()
        self.logger.info(
            f"HumanGate initialized with {len(self.approval_history)} previous approvals"
        )

    async def request_approval(self, context: Dict[str, Any]) -> bool:
        """
        Request human approval for a context.

        Args:
            context: Context requiring approval

        Returns:
            True if approved, False if rejected
        """
        try:
            # Extract relevant information from context
            proposal = self._extract_proposal_from_context(context)

            # For microservices, we'll simulate approval based on context
            # In a real implementation, this would trigger a UI notification
            approved = await self._simulate_human_approval(proposal, context)

            # Log the decision
            self._log_decision(
                proposal,
                {
                    "approved": approved,
                    "timestamp": datetime.now().isoformat(),
                    "reason": "human_decision",
                    "context_type": context.get("task_type", "unknown"),
                },
            )

            self.logger.info(f"Human approval decision: {approved}")
            return approved

        except Exception as e:
            self.logger.error(f"Error in human approval request: {e}")
            return False

    def _extract_proposal_from_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract proposal information from context."""
        return {
            "prompt": context.get("task", "Unknown task"),
            "approach": context.get("approach", "Standard approach"),
            "confidence": context.get("confidence", 0.5),
            "risk_level": context.get("risk_level", "medium"),
            "code": context.get("code", ""),
            "task_type": context.get("task_type", "unknown"),
            "agent_analyses": context.get("agent_analyses", {}),
            "violations": context.get("violations", []),
            "recommendations": context.get("recommendations", []),
        }

    async def _simulate_human_approval(
        self, proposal: Dict[str, Any], context: Dict[str, Any]
    ) -> bool:
        """
        Simulate human approval decision based on context.

        In a real implementation, this would:
        1. Send notification to human operator
        2. Wait for response via UI/API
        3. Return the decision
        """
        risk_level = proposal.get("risk_level", "medium")
        confidence = proposal.get("confidence", 0.5)
        violations = proposal.get("violations", [])

        # Auto-approve low risk with high confidence
        if risk_level == "low" and confidence > 0.8:
            self.logger.info("Auto-approving low risk, high confidence proposal")
            return True

        # Auto-reject critical violations
        critical_violations = [v for v in violations if v.get("severity") == "critical"]
        if critical_violations:
            self.logger.warning("Auto-rejecting proposal with critical violations")
            return False

        # For medium/high risk, simulate human decision based on heuristics
        if risk_level == "high":
            # High risk: 30% chance of approval
            return confidence > 0.9 and len(violations) == 0
        elif risk_level == "medium":
            # Medium risk: 70% chance of approval
            return (
                confidence > 0.7
                and len([v for v in violations if v.get("severity") == "high"]) == 0
            )
        else:
            # Low risk: 90% chance of approval
            return confidence > 0.5

    def get_approval_stats(self) -> Dict[str, Any]:
        """Get approval statistics."""
        if not self.approval_history:
            return {
                "total_approvals": 0,
                "approved_count": 0,
                "rejected_count": 0,
                "approval_rate": 0.0,
                "recent_decisions": [],
            }

        total = len(self.approval_history)
        approved = len([h for h in self.approval_history if h.get("approved", False)])
        rejected = total - approved
        approval_rate = approved / total if total > 0 else 0.0

        # Get recent decisions (last 10)
        recent = sorted(
            self.approval_history, key=lambda x: x.get("timestamp", ""), reverse=True
        )[:10]

        return {
            "total_approvals": total,
            "approved_count": approved,
            "rejected_count": rejected,
            "approval_rate": approval_rate,
            "recent_decisions": recent,
        }

    def _log_decision(self, proposal: Dict[str, Any], decision: Dict[str, Any]):
        """Log approval decision."""
        try:
            log_entry = {
                "proposal": proposal,
                "decision": decision,
                "timestamp": datetime.now().isoformat(),
            }

            self.approval_history.append(log_entry)

            # Keep only last 1000 entries
            if len(self.approval_history) > 1000:
                self.approval_history = self.approval_history[-1000:]

            # Save to file
            with open(self.approval_log_path, "w") as f:
                json.dump(self.approval_history, f, indent=2)

        except Exception as e:
            self.logger.error(f"Error logging decision: {e}")

    def _load_approval_history(self) -> list:
        """Load approval history from file."""
        try:
            if self.approval_log_path.exists():
                with open(self.approval_log_path, "r") as f:
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading approval history: {e}")

        return []

    def reset_approval_history(self):
        """Reset approval history."""
        self.approval_history = []
        if self.approval_log_path.exists():
            self.approval_log_path.unlink()
        self.logger.info("Approval history reset")

    def get_approval_summary(self) -> Dict[str, Any]:
        """Get a summary of approval decisions."""
        stats = self.get_approval_stats()

        # Analyze recent trends
        recent_decisions = stats.get("recent_decisions", [])
        if recent_decisions:
            recent_approval_rate = len(
                [
                    d
                    for d in recent_decisions
                    if d.get("decision", {}).get("approved", False)
                ]
            ) / len(recent_decisions)
        else:
            recent_approval_rate = 0.0

        return {
            "overall_stats": stats,
            "recent_approval_rate": recent_approval_rate,
            "total_history_entries": len(self.approval_history),
            "last_decision": recent_decisions[0] if recent_decisions else None,
        }
