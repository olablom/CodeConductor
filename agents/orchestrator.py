"""
AgentOrchestrator - Coordinates multi-agent discussions

This module implements the orchestrator that manages discussions between
different agents, collects their analyses and proposals, and reaches
consensus on the best approach.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


@dataclass
class DiscussionRound:
    """Represents a single round of agent discussion."""

    round_id: int
    task_context: Dict[str, Any]
    analyses: List[Dict[str, Any]]
    proposals: List[Dict[str, Any]]
    consensus: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class AgentOrchestrator:
    """
    Orchestrator for multi-agent discussions.

    This class coordinates discussions between different agents,
    collects their analyses and proposals, and reaches consensus
    on the best approach for a given task.
    """

    def __init__(self, agents: List[BaseAgent], config: Optional[Dict[str, Any]] = None):
        """
        Initialize the agent orchestrator.

        Args:
            agents: List of agents to coordinate
            config: Configuration for consensus and discussion logic
        """
        self.agents = agents
        self.discussion_history: List[DiscussionRound] = []

        # Default configuration
        default_config = {
            "consensus_strategy": "weighted_majority",  # "majority", "weighted_majority", "unanimous"
            "max_rounds": 3,  # Maximum discussion rounds
            "consensus_threshold": 0.7,  # Minimum agreement for consensus
            "agent_weights": {},  # Custom weights for agents
            "enable_voting": True,  # Enable voting mechanism
            "enable_feedback": True,  # Enable inter-agent feedback
            "timeout_seconds": 30,  # Timeout for discussion rounds
        }

        if config:
            default_config.update(config)

        self.config = default_config

        # Initialize agent weights if not provided
        if not self.config["agent_weights"]:
            self.config["agent_weights"] = {agent.name: 1.0 for agent in self.agents}

        # Add logger for backward compatibility
        self.logger = logger

        logger.info(f"Initialized AgentOrchestrator with {len(self.agents)} agents")
        logger.info(f"Agents: {[agent.name for agent in self.agents]}")

    def run_discussion(self, task_context: Dict[str, Any], max_rounds: Optional[int] = None) -> Dict[str, Any]:
        """
        Run a multi-agent discussion for the given task.

        Args:
            task_context: Context and requirements for the task
            max_rounds: Maximum number of discussion rounds (overrides config)

        Returns:
            Final consensus result with discussion metadata
        """
        if max_rounds is None:
            max_rounds = self.config["max_rounds"]

        logger.info(f"Starting discussion with {len(self.agents)} agents")
        logger.info(f"Task context: {task_context.get('task_type', 'unknown')}")

        current_round = 0
        consensus_reached = False
        final_consensus = None

        while current_round < max_rounds and not consensus_reached:
            current_round += 1
            logger.info(f"Discussion round {current_round}/{max_rounds}")

            # Run analysis phase
            analyses = self._run_analysis_phase(task_context, current_round)

            # Run proposal phase
            proposals = self._run_proposal_phase(analyses, current_round, task_context)

            # Try to reach consensus
            consensus = self._reach_consensus(proposals, current_round)

            # Create discussion round record
            discussion_round = DiscussionRound(
                round_id=current_round,
                task_context=task_context,
                analyses=analyses,
                proposals=proposals,
                consensus=consensus,
                metadata={
                    "consensus_reached": consensus is not None,
                    "agent_count": len(self.agents),
                    "timestamp": self._get_timestamp(),
                },
            )

            self.discussion_history.append(discussion_round)

            # Check if consensus was reached
            if consensus is not None:
                consensus_reached = True
                final_consensus = consensus
                logger.info(f"Consensus reached in round {current_round}")
            else:
                logger.info(f"No consensus in round {current_round}, continuing...")

                # Update task context with feedback for next round
                task_context = self._update_context_with_feedback(task_context, proposals)

        # Return final result
        result = {
            "consensus": final_consensus,
            "discussion_rounds": len(self.discussion_history),
            "consensus_reached": consensus_reached,
            "final_proposals": proposals if not consensus_reached else None,
            "discussion_summary": self._generate_discussion_summary(),
            "metadata": {
                "total_agents": len(self.agents),
                "agent_names": [agent.name for agent in self.agents],
                "config": self.config,
            },
        }

        logger.info(f"Discussion completed. Consensus reached: {consensus_reached}")
        return result

    def _run_analysis_phase(self, task_context: Dict[str, Any], round_id: int) -> List[Dict[str, Any]]:
        """
        Run the analysis phase where all agents analyze the task context.

        Args:
            task_context: Current task context
            round_id: Current discussion round

        Returns:
            List of analysis results from all agents
        """
        analyses = []

        for agent in self.agents:
            try:
                logger.debug(f"Agent {agent.name} analyzing task...")
                analysis = agent.analyze(task_context)
                analysis["agent_name"] = agent.name
                analysis["round_id"] = round_id
                analyses.append(analysis)
                logger.debug(f"Agent {agent.name} analysis completed")
            except Exception as e:
                logger.error(f"Agent {agent.name} failed to analyze: {e}")
                # Add error analysis
                analyses.append(
                    {
                        "agent_name": agent.name,
                        "round_id": round_id,
                        "error": str(e),
                        "status": "failed",
                    }
                )

        return analyses

    def _run_proposal_phase(
        self,
        analyses: List[Dict[str, Any]],
        round_id: int,
        task_context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Run the proposal phase where all agents propose solutions.

        Args:
            analyses: Analysis results from all agents
            round_id: Current discussion round

        Returns:
            List of proposal results from all agents
        """
        proposals = []

        for agent, analysis in zip(self.agents, analyses):
            try:
                if analysis.get("status") == "failed":
                    # Skip failed agents
                    proposals.append(
                        {
                            "agent_name": agent.name,
                            "round_id": round_id,
                            "error": analysis.get("error", "Unknown error"),
                            "status": "failed",
                        }
                    )
                    continue

                logger.debug(f"Agent {agent.name} proposing solution...")
                proposal = agent.propose(analysis, task_context)
                proposal["agent_name"] = agent.name
                proposal["round_id"] = round_id
                proposal["analysis_id"] = analysis.get("analysis_id", round_id)
                proposals.append(proposal)
                logger.debug(f"Agent {agent.name} proposal completed")
            except Exception as e:
                logger.error(f"Agent {agent.name} failed to propose: {e}")
                proposals.append(
                    {
                        "agent_name": agent.name,
                        "round_id": round_id,
                        "error": str(e),
                        "status": "failed",
                    }
                )

        return proposals

    def _reach_consensus(self, proposals: List[Dict[str, Any]], round_id: int) -> Optional[Dict[str, Any]]:
        """
        Attempt to reach consensus among agent proposals.

        Args:
            proposals: List of proposals from all agents
            round_id: Current discussion round

        Returns:
            Consensus result if reached, None otherwise
        """
        # Filter out failed proposals
        valid_proposals = [p for p in proposals if p.get("status") != "failed"]

        if not valid_proposals:
            logger.warning("No valid proposals to reach consensus on")
            return None

        if len(valid_proposals) == 1:
            # Only one valid proposal, use it as consensus
            logger.info("Single valid proposal, using as consensus")
            return valid_proposals[0]

        # Apply consensus strategy
        strategy = self.config["consensus_strategy"]

        if strategy == "majority":
            consensus = self._majority_consensus(valid_proposals)
        elif strategy == "weighted_majority":
            consensus = self._weighted_majority_consensus(valid_proposals)
        elif strategy == "unanimous":
            consensus = self._unanimous_consensus(valid_proposals)
        else:
            logger.warning(f"Unknown consensus strategy: {strategy}, using majority")
            consensus = self._majority_consensus(valid_proposals)

        # Check if consensus meets threshold
        if consensus and self._check_consensus_threshold(consensus, valid_proposals):
            return consensus

        return None

    def _majority_consensus(self, proposals: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Reach consensus using simple majority voting."""
        # For now, use the first proposal as consensus
        # TODO: Implement more sophisticated majority logic
        return proposals[0] if proposals else None

    def _weighted_majority_consensus(self, proposals: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Reach consensus using weighted majority voting."""
        # Calculate weighted scores for each proposal
        proposal_scores = {}

        for proposal in proposals:
            agent_name = proposal["agent_name"]
            weight = self.config["agent_weights"].get(agent_name, 1.0)
            confidence = proposal.get("confidence", 0.5)

            score = weight * confidence
            proposal_scores[agent_name] = score

        # Find the proposal with highest weighted score
        if proposal_scores:
            best_agent = max(proposal_scores.items(), key=lambda x: x[1])[0]
            best_proposal = next(p for p in proposals if p["agent_name"] == best_agent)

            # Add consensus metadata
            best_proposal["consensus_metadata"] = {
                "strategy": "weighted_majority",
                "scores": proposal_scores,
                "winning_agent": best_agent,
                "winning_score": proposal_scores[best_agent],
            }

            return best_proposal

        return None

    def _unanimous_consensus(self, proposals: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Reach consensus only if all agents agree."""
        # For now, return None (no unanimous consensus)
        # TODO: Implement unanimous consensus logic
        return None

    def _check_consensus_threshold(self, consensus: Dict[str, Any], proposals: List[Dict[str, Any]]) -> bool:
        """Check if consensus meets the required threshold."""
        threshold = self.config["consensus_threshold"]

        # Calculate agreement level
        agreement_count = 0
        total_weight = 0

        for proposal in proposals:
            agent_name = proposal["agent_name"]
            weight = self.config["agent_weights"].get(agent_name, 1.0)
            total_weight += weight

            # Simple agreement check (can be enhanced)
            if self._proposals_agree(consensus, proposal):
                agreement_count += weight

        agreement_level = agreement_count / total_weight if total_weight > 0 else 0
        return agreement_level >= threshold

    def _proposals_agree(self, proposal1: Dict[str, Any], proposal2: Dict[str, Any]) -> bool:
        """Check if two proposals agree on key aspects."""
        # Check approval status first
        approval1 = proposal1.get("approval", "")
        approval2 = proposal2.get("approval", "")

        # If approvals don't match, they don't agree
        if approval1 != approval2:
            return False

        # Check confidence levels if approvals match
        conf1 = proposal1.get("confidence", 0.5)
        conf2 = proposal2.get("confidence", 0.5)

        return abs(conf1 - conf2) < 0.2  # Within 20% confidence difference

    def _update_context_with_feedback(self, task_context: Dict[str, Any], proposals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Update task context with feedback from proposals for next round."""
        # Extract feedback from proposals
        feedback = []
        for proposal in proposals:
            if proposal.get("status") != "failed":
                feedback.append(
                    {
                        "agent": proposal["agent_name"],
                        "reasoning": proposal.get("reasoning", ""),
                        "confidence": proposal.get("confidence", 0.5),
                        "suggestions": proposal.get("suggestions", []),
                    }
                )

        # Update context with feedback
        updated_context = task_context.copy()
        updated_context["previous_feedback"] = feedback
        updated_context["discussion_round"] = len(self.discussion_history) + 1

        return updated_context

    def _generate_discussion_summary(self) -> Dict[str, Any]:
        """Generate a summary of the discussion."""
        if not self.discussion_history:
            return {"summary": "No discussion rounds completed"}

        total_rounds = len(self.discussion_history)
        consensus_rounds = sum(1 for round_data in self.discussion_history if round_data.consensus)

        agent_participation = {}
        for agent in self.agents:
            agent_name = agent.name
            participation = sum(
                1
                for round_data in self.discussion_history
                for analysis in round_data.analyses
                if analysis.get("agent_name") == agent_name and analysis.get("status") != "failed"
            )
            agent_participation[agent_name] = participation

        return {
            "total_rounds": total_rounds,
            "consensus_rounds": consensus_rounds,
            "success_rate": consensus_rounds / total_rounds if total_rounds > 0 else 0,
            "agent_participation": agent_participation,
            "final_consensus_reached": bool(self.discussion_history[-1].consensus),
        }

    def get_discussion_history(self) -> List[DiscussionRound]:
        """Get the complete discussion history."""
        return self.discussion_history.copy()

    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get statistics about agent performance."""
        stats = {
            "total_agents": len(self.agents),
            "agent_names": [agent.name for agent in self.agents],
            "total_discussions": len(self.discussion_history),
            "agent_weights": self.config["agent_weights"].copy(),
        }

        # Calculate agent-specific statistics
        agent_stats = {}
        for agent in self.agents:
            agent_name = agent.name
            successful_analyses = 0
            successful_proposals = 0

            for round_data in self.discussion_history:
                # Count successful analyses
                for analysis in round_data.analyses:
                    if analysis.get("agent_name") == agent_name and analysis.get("status") != "failed":
                        successful_analyses += 1

                # Count successful proposals
                for proposal in round_data.proposals:
                    if proposal.get("agent_name") == agent_name and proposal.get("status") != "failed":
                        successful_proposals += 1

            agent_stats[agent_name] = {
                "successful_analyses": successful_analyses,
                "successful_proposals": successful_proposals,
                "success_rate": successful_proposals / len(self.discussion_history) if self.discussion_history else 0,
            }

        stats["agent_performance"] = agent_stats
        return stats

    def reset_discussion(self):
        """Reset the discussion history."""
        self.discussion_history.clear()
        logger.info("Discussion history reset")

    def _get_timestamp(self) -> str:
        """Get current timestamp for logging."""
        from datetime import datetime

        return datetime.now().isoformat()

    # Additional methods for test compatibility
    def _calculate_consensus_score(self, proposals: List[Dict[str, Any]]) -> float:
        """Calculate consensus score for backward compatibility."""
        if not proposals:
            return 0.0

        # Calculate average confidence from proposals
        confidences = []
        for proposal in proposals:
            if isinstance(proposal, dict) and "confidence" in proposal:
                confidences.append(proposal["confidence"])
            elif isinstance(proposal, dict) and "score" in proposal:
                confidences.append(proposal["score"])

        return sum(confidences) / len(confidences) if confidences else 0.0

    def _run_analysis_phase(self, task_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run analysis phase for backward compatibility."""
        return self._run_analysis_phase(task_context, 1)

    def _run_proposal_phase(self, analyses: List[Dict[str, Any]], task_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run proposal phase for backward compatibility."""
        return self._run_proposal_phase(analyses, 1, task_context)

    def _reach_consensus(self, proposals: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Reach consensus for backward compatibility."""
        return self._reach_consensus(proposals, 1)

    def _majority_consensus(self, proposals: List[Dict[str, Any]]) -> tuple:
        """Majority consensus for backward compatibility."""
        if not proposals:
            return None, None, 0.0

        # Filter out failed proposals
        valid_proposals = [p for p in proposals if isinstance(p, dict) and "error" not in p]

        if not valid_proposals:
            return None, None, 0.0

        # Use first valid proposal as consensus
        consensus = valid_proposals[0]
        decision = "approve" if consensus.get("confidence", 0) > 0.5 else "reject"
        score = consensus.get("confidence", 0.5)

        return consensus, decision, score

    def _unanimous_consensus(self, proposals: List[Dict[str, Any]]) -> tuple:
        """Unanimous consensus for backward compatibility."""
        if not proposals:
            return None, None, 0.0

        # Filter out failed proposals
        valid_proposals = [p for p in proposals if isinstance(p, dict) and "error" not in p]

        if not valid_proposals:
            return None, None, 0.0

        # Check if all proposals have same confidence
        confidences = [p.get("confidence", 0) for p in valid_proposals]
        if len(set(confidences)) == 1:
            consensus = valid_proposals[0]
            decision = "approve" if confidences[0] > 0.5 else "reject"
            return consensus, decision, confidences[0]

        return None, None, 0.0

    def _weighted_majority_consensus(self, proposals: List[Dict[str, Any]]) -> tuple:
        """Weighted majority consensus for backward compatibility."""
        if not proposals:
            return None, None, 0.0

        # Filter out failed proposals
        valid_proposals = [p for p in proposals if isinstance(p, dict) and "error" not in p]

        if not valid_proposals:
            return None, None, 0.0

        # Calculate weighted average
        total_weight = 0
        weighted_sum = 0

        for proposal in valid_proposals:
            agent_name = proposal.get("agent_name", "unknown")
            weight = self.config["agent_weights"].get(agent_name, 1.0)
            confidence = proposal.get("confidence", 0.5)

            total_weight += weight
            weighted_sum += weight * confidence

        if total_weight == 0:
            return None, None, 0.0

        avg_confidence = weighted_sum / total_weight
        consensus = valid_proposals[0]  # Use first proposal as representative
        decision = "approve" if avg_confidence > 0.5 else "reject"

        return consensus, decision, avg_confidence
