"""
Orchestrator Service - Agent Orchestration Logic

This module contains the core orchestrator classes migrated from the main CodeConductor
codebase to the microservices architecture.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import asyncio
import httpx

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

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the agent orchestrator.

        Args:
            config: Configuration for consensus and discussion logic
        """
        self.discussion_history: List[DiscussionRound] = []
        self.agent_service_url = "http://agent-service:8001"

        # Default configuration
        default_config = {
            "consensus_strategy": "weighted_majority",  # "majority", "weighted_majority", "unanimous"
            "max_rounds": 3,  # Maximum discussion rounds
            "consensus_threshold": 0.7,  # Minimum agreement for consensus
            "agent_weights": {},  # Custom weights for agents
            "enable_voting": True,  # Enable voting mechanism
            "enable_feedback": True,  # Enable inter-agent feedback
            "timeout_seconds": 30,  # Timeout for discussion rounds
            "available_agents": [
                "codegen",
                "review",
                "architect",
            ],  # Available agent types
        }

        if config:
            default_config.update(config)

        self.config = default_config

        # Initialize agent weights if not provided
        if not self.config["agent_weights"]:
            self.config["agent_weights"] = {
                agent: 1.0 for agent in self.config["available_agents"]
            }

        logger.info(
            f"Initialized AgentOrchestrator with agents: {self.config['available_agents']}"
        )

    async def run_discussion(
        self,
        task_context: Dict[str, Any],
        agents: Optional[List[str]] = None,
        max_rounds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run a multi-agent discussion for the given task.

        Args:
            task_context: Context and requirements for the task
            agents: List of agent types to use (defaults to all available)
            max_rounds: Maximum number of discussion rounds (overrides config)

        Returns:
            Final consensus result with discussion metadata
        """
        if agents is None:
            agents = self.config["available_agents"]

        if max_rounds is None:
            max_rounds = self.config["max_rounds"]

        logger.info(f"Starting discussion with agents: {agents}")
        logger.info(f"Task context: {task_context.get('task', 'unknown')}")

        current_round = 0
        consensus_reached = False
        final_consensus = None

        while current_round < max_rounds and not consensus_reached:
            current_round += 1
            logger.info(f"Discussion round {current_round}/{max_rounds}")

            # Run analysis phase
            analyses = await self._run_analysis_phase(
                task_context, current_round, agents
            )

            # Run proposal phase
            proposals = await self._run_proposal_phase(
                analyses, current_round, task_context, agents
            )

            # Try to reach consensus
            consensus = await self._reach_consensus(proposals, current_round)

            # Store round data
            round_data = DiscussionRound(
                round_id=current_round,
                task_context=task_context.copy(),
                analyses=analyses,
                proposals=proposals,
                consensus=consensus,
                metadata={
                    "timestamp": self._get_timestamp(),
                    "agents_used": agents,
                    "round_duration": 0.1,  # Placeholder
                },
            )
            self.discussion_history.append(round_data)

            if consensus:
                consensus_reached = True
                final_consensus = consensus
                logger.info(f"Consensus reached in round {current_round}")
            else:
                logger.info(f"No consensus in round {current_round}, continuing...")
                # Update context with feedback for next round
                if self.config["enable_feedback"]:
                    task_context = self._update_context_with_feedback(
                        task_context, proposals
                    )

        # Generate final result
        result = {
            "consensus_reached": consensus_reached,
            "final_consensus": final_consensus,
            "discussion_rounds": current_round,
            "total_rounds": len(self.discussion_history),
            "agents_used": agents,
            "consensus_strategy": self.config["consensus_strategy"],
            "discussion_summary": self._generate_discussion_summary(),
            "metadata": {
                "timestamp": self._get_timestamp(),
                "execution_time": 0.1,  # Placeholder
                "config_used": self.config,
            },
        }

        logger.info(f"Discussion completed. Consensus reached: {consensus_reached}")
        return result

    async def _run_analysis_phase(
        self, task_context: Dict[str, Any], round_id: int, agents: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Run the analysis phase with all agents.

        Args:
            task_context: Task context for analysis
            round_id: Current round number
            agents: List of agent types to use

        Returns:
            List of analysis results from all agents
        """
        logger.info(f"Running analysis phase for round {round_id}")
        analyses = []

        async with httpx.AsyncClient() as client:
            for agent_type in agents:
                try:
                    # Call agent service for analysis
                    response = await client.post(
                        f"{self.agent_service_url}/agents/{agent_type}/analyze",
                        json={
                            "agent_type": agent_type,
                            "task_context": task_context,
                            "config": {},
                        },
                        timeout=self.config["timeout_seconds"],
                    )

                    if response.status_code == 200:
                        analysis_data = response.json()
                        analyses.append(
                            {
                                "agent_type": agent_type,
                                "analysis": analysis_data,
                                "round_id": round_id,
                                "timestamp": self._get_timestamp(),
                            }
                        )
                        logger.info(f"Analysis completed for {agent_type}")
                    else:
                        logger.error(
                            f"Analysis failed for {agent_type}: {response.status_code}"
                        )

                except Exception as e:
                    logger.error(f"Error in analysis phase for {agent_type}: {e}")
                    # Add fallback analysis
                    analyses.append(
                        {
                            "agent_type": agent_type,
                            "analysis": {
                                "agent_name": f"{agent_type}_agent",
                                "task": task_context.get("task", ""),
                                "language": task_context.get("language", "python"),
                                "requirements": task_context.get("requirements", []),
                                "patterns": [],
                                "challenges": [],
                                "complexity": "medium",
                                "recommended_approach": "Standard approach",
                                "quality_focus": ["Code quality"],
                                "performance_needs": [],
                                "security_needs": [],
                                "testing_strategy": "Unit tests",
                                "documentation_needs": {},
                            },
                            "round_id": round_id,
                            "timestamp": self._get_timestamp(),
                            "error": str(e),
                        }
                    )

        return analyses

    async def _run_proposal_phase(
        self,
        analyses: List[Dict[str, Any]],
        round_id: int,
        task_context: Dict[str, Any],
        agents: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Run the proposal phase with all agents.

        Args:
            analyses: Analysis results from previous phase
            round_id: Current round number
            task_context: Original task context
            agents: List of agent types to use

        Returns:
            List of proposal results from all agents
        """
        logger.info(f"Running proposal phase for round {round_id}")
        proposals = []

        async with httpx.AsyncClient() as client:
            for agent_type in agents:
                try:
                    # Find analysis for this agent
                    agent_analysis = next(
                        (a for a in analyses if a["agent_type"] == agent_type), None
                    )

                    if not agent_analysis:
                        logger.warning(f"No analysis found for {agent_type}")
                        continue

                    # Call agent service for proposal
                    response = await client.post(
                        f"{self.agent_service_url}/agents/{agent_type}/propose",
                        json={
                            "agent_type": agent_type,
                            "analysis": agent_analysis["analysis"],
                            "task_context": task_context,
                            "config": {},
                        },
                        timeout=self.config["timeout_seconds"],
                    )

                    if response.status_code == 200:
                        proposal_data = response.json()
                        proposals.append(
                            {
                                "agent_type": agent_type,
                                "proposal": proposal_data,
                                "round_id": round_id,
                                "timestamp": self._get_timestamp(),
                            }
                        )
                        logger.info(f"Proposal completed for {agent_type}")
                    else:
                        logger.error(
                            f"Proposal failed for {agent_type}: {response.status_code}"
                        )

                except Exception as e:
                    logger.error(f"Error in proposal phase for {agent_type}: {e}")
                    # Add fallback proposal
                    proposals.append(
                        {
                            "agent_type": agent_type,
                            "proposal": {
                                "agent_name": f"{agent_type}_agent",
                                "approach": "Standard approach",
                                "structure": {"main_file": "main.py"},
                                "implementation_plan": ["Implement core functionality"],
                                "code_template": "# Generated code template",
                                "quality_guidelines": ["Follow best practices"],
                                "documentation_plan": {"readme": "Documentation"},
                                "size_estimate": {"lines": 100, "files": 1},
                                "confidence": 0.5,
                                "reasoning": "Standard implementation approach",
                                "suggestions": ["Add tests"],
                                "analysis_summary": {"complexity": "medium"},
                            },
                            "round_id": round_id,
                            "timestamp": self._get_timestamp(),
                            "error": str(e),
                        }
                    )

        return proposals

    async def _reach_consensus(
        self, proposals: List[Dict[str, Any]], round_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Attempt to reach consensus among proposals.

        Args:
            proposals: List of proposals from all agents
            round_id: Current round number

        Returns:
            Consensus result if reached, None otherwise
        """
        if not proposals:
            return None

        logger.info(f"Attempting to reach consensus in round {round_id}")

        strategy = self.config["consensus_strategy"]

        if strategy == "majority":
            consensus = self._majority_consensus(proposals)
        elif strategy == "weighted_majority":
            consensus = self._weighted_majority_consensus(proposals)
        elif strategy == "unanimous":
            consensus = self._unanimous_consensus(proposals)
        else:
            logger.warning(f"Unknown consensus strategy: {strategy}")
            consensus = self._majority_consensus(proposals)

        if consensus:
            # Check if consensus meets threshold
            if self._check_consensus_threshold(consensus, proposals):
                return consensus
            else:
                logger.info("Consensus found but below threshold")
                return None

        return None

    def _majority_consensus(
        self, proposals: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Simple majority consensus."""
        if len(proposals) == 1:
            return proposals[0]["proposal"]

        # For now, return the first proposal as consensus
        # In a real implementation, you'd analyze similarity between proposals
        return proposals[0]["proposal"]

    def _weighted_majority_consensus(
        self, proposals: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Weighted majority consensus based on agent weights."""
        if len(proposals) == 1:
            return proposals[0]["proposal"]

        # Calculate weighted scores for each proposal
        proposal_scores = {}

        for proposal in proposals:
            agent_type = proposal["agent_type"]
            weight = self.config["agent_weights"].get(agent_type, 1.0)
            confidence = proposal["proposal"].get("confidence", 0.5)

            score = weight * confidence
            proposal_scores[agent_type] = score

        # Return the proposal with highest weighted score
        best_agent = max(proposal_scores, key=proposal_scores.get)
        return next(p["proposal"] for p in proposals if p["agent_type"] == best_agent)

    def _unanimous_consensus(
        self, proposals: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Unanimous consensus - all proposals must agree."""
        if len(proposals) <= 1:
            return proposals[0]["proposal"] if proposals else None

        # Check if all proposals are similar enough
        first_proposal = proposals[0]["proposal"]

        for proposal in proposals[1:]:
            if not self._proposals_agree(first_proposal, proposal["proposal"]):
                return None

        return first_proposal

    def _check_consensus_threshold(
        self, consensus: Dict[str, Any], proposals: List[Dict[str, Any]]
    ) -> bool:
        """Check if consensus meets the required threshold."""
        if not proposals:
            return False

        # Calculate consensus score
        consensus_score = self._calculate_consensus_score(proposals)
        threshold = self.config["consensus_threshold"]

        logger.info(f"Consensus score: {consensus_score:.2f}, threshold: {threshold}")
        return consensus_score >= threshold

    def _proposals_agree(
        self, proposal1: Dict[str, Any], proposal2: Dict[str, Any]
    ) -> bool:
        """Check if two proposals are similar enough to be considered in agreement."""
        # Simple similarity check - in practice, you'd use more sophisticated comparison
        approach1 = proposal1.get("approach", "").lower()
        approach2 = proposal2.get("approach", "").lower()

        # Check if approaches are similar
        similarity_threshold = 0.7
        common_words = len(set(approach1.split()) & set(approach2.split()))
        total_words = len(set(approach1.split()) | set(approach2.split()))

        if total_words == 0:
            return True

        similarity = common_words / total_words
        return similarity >= similarity_threshold

    def _update_context_with_feedback(
        self, task_context: Dict[str, Any], proposals: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Update task context with feedback from proposals."""
        updated_context = task_context.copy()

        # Collect suggestions from all proposals
        all_suggestions = []
        for proposal in proposals:
            suggestions = proposal["proposal"].get("suggestions", [])
            all_suggestions.extend(suggestions)

        # Add suggestions to context
        if all_suggestions:
            updated_context["feedback"] = {
                "suggestions": list(set(all_suggestions)),  # Remove duplicates
                "round": len(self.discussion_history),
            }

        return updated_context

    def _generate_discussion_summary(self) -> Dict[str, Any]:
        """Generate a summary of the discussion."""
        if not self.discussion_history:
            return {"message": "No discussion history"}

        total_rounds = len(self.discussion_history)
        agents_used = set()

        for round_data in self.discussion_history:
            for analysis in round_data.analyses:
                agents_used.add(analysis["agent_type"])

        return {
            "total_rounds": total_rounds,
            "agents_used": list(agents_used),
            "consensus_reached": any(
                round_data.consensus for round_data in self.discussion_history
            ),
            "final_round": self.discussion_history[-1]
            if self.discussion_history
            else None,
            "discussion_quality": "high" if total_rounds > 1 else "basic",
        }

    def get_discussion_history(self) -> List[DiscussionRound]:
        """Get the complete discussion history."""
        return self.discussion_history

    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get statistics about agent performance."""
        if not self.discussion_history:
            return {"message": "No discussion history"}

        agent_stats = {}

        for round_data in self.discussion_history:
            for analysis in round_data.analyses:
                agent_type = analysis["agent_type"]
                if agent_type not in agent_stats:
                    agent_stats[agent_type] = {
                        "analysis_count": 0,
                        "proposal_count": 0,
                        "errors": 0,
                        "avg_confidence": 0.0,
                    }

                agent_stats[agent_type]["analysis_count"] += 1

                if "error" in analysis:
                    agent_stats[agent_type]["errors"] += 1

            for proposal in round_data.proposals:
                agent_type = proposal["agent_type"]
                if agent_type not in agent_stats:
                    agent_stats[agent_type] = {
                        "analysis_count": 0,
                        "proposal_count": 0,
                        "errors": 0,
                        "avg_confidence": 0.0,
                    }

                agent_stats[agent_type]["proposal_count"] += 1
                confidence = proposal["proposal"].get("confidence", 0.5)
                agent_stats[agent_type]["avg_confidence"] += confidence

        # Calculate averages
        for agent_type, stats in agent_stats.items():
            if stats["proposal_count"] > 0:
                stats["avg_confidence"] /= stats["proposal_count"]

        return {
            "total_rounds": len(self.discussion_history),
            "agent_statistics": agent_stats,
            "consensus_strategy": self.config["consensus_strategy"],
            "config": self.config,
        }

    def reset_discussion(self):
        """Reset the discussion history."""
        self.discussion_history.clear()
        logger.info("Discussion history reset")

    def _get_timestamp(self) -> str:
        """Get current timestamp as string."""
        return datetime.now().isoformat()

    def _calculate_consensus_score(self, proposals: List[Dict[str, Any]]) -> float:
        """Calculate a consensus score based on proposal similarity."""
        if len(proposals) <= 1:
            return 1.0

        # Calculate average confidence
        total_confidence = sum(p["proposal"].get("confidence", 0.5) for p in proposals)
        avg_confidence = total_confidence / len(proposals)

        # Calculate approach similarity
        approaches = [p["proposal"].get("approach", "") for p in proposals]
        similarity_score = 0.0

        for i in range(len(approaches)):
            for j in range(i + 1, len(approaches)):
                if self._proposals_agree(
                    {"approach": approaches[i]}, {"approach": approaches[j]}
                ):
                    similarity_score += 1.0

        max_similarity = len(approaches) * (len(approaches) - 1) / 2
        if max_similarity > 0:
            similarity_score /= max_similarity

        # Combine confidence and similarity
        final_score = (avg_confidence + similarity_score) / 2
        return min(final_score, 1.0)


# Orchestrator factory for creating different orchestrator types
class OrchestratorFactory:
    """Factory for creating different types of orchestrators."""

    @staticmethod
    def create_orchestrator(
        orchestrator_type: str = "standard", config: Dict[str, Any] = None
    ) -> AgentOrchestrator:
        """
        Create an orchestrator of the specified type.

        Args:
            orchestrator_type: Type of orchestrator to create
            config: Orchestrator configuration

        Returns:
            Orchestrator instance
        """
        if orchestrator_type == "standard":
            return AgentOrchestrator(config)
        else:
            raise ValueError(f"Unknown orchestrator type: {orchestrator_type}")
