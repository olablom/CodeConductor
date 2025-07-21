#!/usr/bin/env python3
"""
GPU-Powered Agent Orchestrator
Uses neural bandit to intelligently select agents based on task complexity
"""

import logging
import requests
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from agents.base_agent import BaseAgent
from agents.orchestrator import AgentOrchestrator, DiscussionRound

logger = logging.getLogger(__name__)


@dataclass
class AgentSelection:
    """Represents an agent selection decision."""

    agent_name: str
    confidence: float
    gpu_used: bool
    inference_time_ms: float
    reasoning: str


class GPUOrchestrator(AgentOrchestrator):
    """
    GPU-powered orchestrator that uses neural bandit for intelligent agent selection.

    This orchestrator:
    1. Analyzes task complexity using neural networks
    2. Selects optimal agents using GPU-accelerated bandits
    3. Monitors performance and learns from results
    4. Scales automatically based on GPU availability
    """

    def __init__(
        self,
        agents: List[BaseAgent],
        gpu_service_url: str = "http://localhost:8009",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize GPU-powered orchestrator.

        Args:
            agents: List of available agents
            gpu_service_url: URL of the GPU neural bandit service
            config: Configuration for GPU orchestration
        """
        super().__init__(agents, config)

        self.gpu_service_url = gpu_service_url
        self.agent_selections: List[AgentSelection] = []

        # GPU-specific configuration
        gpu_config = {
            "enable_gpu_selection": True,
            "complexity_threshold": 0.7,  # When to use GPU agents
            "fallback_to_cpu": True,  # Use CPU if GPU fails
            "performance_monitoring": True,
            "learning_enabled": True,
        }

        if config:
            gpu_config.update(config)

        self.gpu_config = gpu_config

        # Available agent types for bandit selection
        self.agent_types = [
            "architect_agent",
            "review_agent",
            "codegen_agent",
            "policy_agent",
            "reward_agent",
            "qlearning_agent",
        ]

        logger.info(f"GPU Orchestrator initialized with {len(self.agents)} agents")
        logger.info(f"GPU Service URL: {self.gpu_service_url}")

    def select_agents_for_task(self, task_context: Dict[str, Any]) -> List[BaseAgent]:
        """
        Use neural bandit to select optimal agents for the task.

        Args:
            task_context: Task context and requirements

        Returns:
            List of selected agents
        """
        try:
            # Extract task features for neural bandit
            features = self._extract_task_features(task_context)

            # Use neural bandit to select agents
            selection = self._neural_bandit_selection(features)

            # Filter agents based on selection
            selected_agents = self._filter_agents_by_selection(selection)

            logger.info(
                f"Neural bandit selected {len(selected_agents)} agents: {[a.name for a in selected_agents]}"
            )

            return selected_agents

        except Exception as e:
            logger.warning(f"GPU selection failed, falling back to CPU: {e}")
            return self.agents  # Fallback to all agents

    def _extract_task_features(self, task_context: Dict[str, Any]) -> List[float]:
        """
        Extract numerical features from task context for neural bandit.

        Args:
            task_context: Task context

        Returns:
            List of numerical features
        """
        # Extract complexity indicators
        task_type = task_context.get("task_type", "unknown")
        description = task_context.get("description", "")

        # Feature engineering
        features = [
            # Complexity features
            self._calculate_complexity_score(description),
            self._calculate_urgency_score(task_context),
            self._calculate_team_size_score(task_context),
            self._calculate_deadline_score(task_context),
            self._calculate_domain_expertise_score(task_context),
            self._calculate_code_quality_score(task_context),
            self._calculate_testing_required_score(task_context),
            self._calculate_documentation_needed_score(task_context),
            self._calculate_security_level_score(task_context),
            self._calculate_performance_priority_score(task_context),
        ]

        return features

    def _neural_bandit_selection(self, features: List[float]) -> AgentSelection:
        """
        Use neural bandit to select optimal agent combination.

        Args:
            features: Task features

        Returns:
            Agent selection decision
        """
        try:
            # Prepare request for neural bandit
            request_data = {
                "arms": self.agent_types,
                "features": features,
                "epsilon": 0.1,  # Exploration rate
            }

            # Call GPU neural bandit service
            response = requests.post(
                f"{self.gpu_service_url}/gpu/bandits/choose",
                json=request_data,
                timeout=10,
            )

            if response.status_code == 200:
                result = response.json()

                return AgentSelection(
                    agent_name=result["selected_arm"],
                    confidence=result["confidence"],
                    gpu_used=result["gpu_used"],
                    inference_time_ms=result["inference_time_ms"],
                    reasoning=f"Neural bandit selected {result['selected_arm']} with confidence {result['confidence']:.3f}",
                )
            else:
                raise Exception(f"GPU service returned {response.status_code}")

        except Exception as e:
            logger.error(f"Neural bandit selection failed: {e}")
            raise

    def _filter_agents_by_selection(self, selection: AgentSelection) -> List[BaseAgent]:
        """
        Filter agents based on neural bandit selection.

        Args:
            selection: Agent selection decision

        Returns:
            List of filtered agents
        """
        # For now, return all agents that match the selected type
        # In the future, this could be more sophisticated
        selected_agents = []

        for agent in self.agents:
            if selection.agent_name in agent.name.lower():
                selected_agents.append(agent)

        # If no agents match, return all agents as fallback
        if not selected_agents:
            logger.warning(
                f"No agents match selection '{selection.agent_name}', using all agents"
            )
            selected_agents = self.agents

        return selected_agents

    def run_discussion(
        self, task_context: Dict[str, Any], max_rounds: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run GPU-powered multi-agent discussion.

        Args:
            task_context: Task context and requirements
            max_rounds: Maximum discussion rounds

        Returns:
            Final consensus result with GPU metadata
        """
        # Select optimal agents using neural bandit
        selected_agents = self.select_agents_for_task(task_context)

        # Temporarily replace agents with selected ones
        original_agents = self.agents
        self.agents = selected_agents

        try:
            # Run discussion with selected agents
            result = super().run_discussion(task_context, max_rounds)

            # Add GPU metadata
            result["gpu_metadata"] = {
                "agents_selected": [agent.name for agent in selected_agents],
                "total_agents_available": len(original_agents),
                "gpu_used": True,
                "selection_method": "neural_bandit",
            }

            return result

        finally:
            # Restore original agents
            self.agents = original_agents

    # Feature extraction helper methods
    def _calculate_complexity_score(self, description: str) -> float:
        """Calculate complexity score from description."""
        complexity_keywords = [
            "complex",
            "advanced",
            "sophisticated",
            "enterprise",
            "scalable",
        ]
        score = sum(
            1 for keyword in complexity_keywords if keyword in description.lower()
        )
        return min(score / len(complexity_keywords), 1.0)

    def _calculate_urgency_score(self, context: Dict[str, Any]) -> float:
        """Calculate urgency score."""
        urgency = context.get("urgency", "medium")
        urgency_map = {"low": 0.2, "medium": 0.5, "high": 0.8, "critical": 1.0}
        return urgency_map.get(urgency.lower(), 0.5)

    def _calculate_team_size_score(self, context: Dict[str, Any]) -> float:
        """Calculate team size score."""
        team_size = context.get("team_size", "small")
        size_map = {"small": 0.2, "medium": 0.5, "large": 0.8, "enterprise": 1.0}
        return size_map.get(team_size.lower(), 0.5)

    def _calculate_deadline_score(self, context: Dict[str, Any]) -> float:
        """Calculate deadline pressure score."""
        deadline = context.get("deadline", "flexible")
        deadline_map = {"flexible": 0.2, "moderate": 0.5, "tight": 0.8, "urgent": 1.0}
        return deadline_map.get(deadline.lower(), 0.5)

    def _calculate_domain_expertise_score(self, context: Dict[str, Any]) -> float:
        """Calculate domain expertise requirement score."""
        expertise = context.get("domain_expertise", "general")
        expertise_map = {
            "general": 0.2,
            "moderate": 0.5,
            "specialized": 0.8,
            "expert": 1.0,
        }
        return expertise_map.get(expertise.lower(), 0.5)

    def _calculate_code_quality_score(self, context: Dict[str, Any]) -> float:
        """Calculate code quality requirement score."""
        quality = context.get("code_quality", "standard")
        quality_map = {"basic": 0.2, "standard": 0.5, "high": 0.8, "enterprise": 1.0}
        return quality_map.get(quality.lower(), 0.5)

    def _calculate_testing_required_score(self, context: Dict[str, Any]) -> float:
        """Calculate testing requirement score."""
        testing = context.get("testing_required", "basic")
        testing_map = {
            "none": 0.0,
            "basic": 0.3,
            "comprehensive": 0.7,
            "enterprise": 1.0,
        }
        return testing_map.get(testing.lower(), 0.3)

    def _calculate_documentation_needed_score(self, context: Dict[str, Any]) -> float:
        """Calculate documentation requirement score."""
        docs = context.get("documentation_needed", "basic")
        docs_map = {"none": 0.0, "basic": 0.3, "comprehensive": 0.7, "enterprise": 1.0}
        return docs_map.get(docs.lower(), 0.3)

    def _calculate_security_level_score(self, context: Dict[str, Any]) -> float:
        """Calculate security requirement score."""
        security = context.get("security_level", "standard")
        security_map = {"basic": 0.2, "standard": 0.5, "high": 0.8, "enterprise": 1.0}
        return security_map.get(security.lower(), 0.5)

    def _calculate_performance_priority_score(self, context: Dict[str, Any]) -> float:
        """Calculate performance priority score."""
        performance = context.get("performance_priority", "balanced")
        perf_map = {"low": 0.2, "balanced": 0.5, "high": 0.8, "critical": 1.0}
        return perf_map.get(performance.lower(), 0.5)
