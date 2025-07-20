"""
BaseAgent - Abstract base class for all agents in CodeConductor v2.0

This module defines the core interface that all agents must implement.
Agents are responsible for analyzing context, proposing solutions, and reviewing code.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the CodeConductor system.

    All agents must implement the core methods: analyze(), propose(), and review().
    This ensures consistent interface across different agent types.
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base agent.

        Args:
            name: Unique identifier for this agent
            config: Configuration dictionary for agent behavior
        """
        self.name = name
        self.config = config or {}
        self.message_bus = None
        self.llm_client = None

        logger.info(f"Initialized agent: {self.name}")

    @abstractmethod
    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze context and return structured insights.

        This method should examine the input context and extract relevant
        information, patterns, and insights that will inform the proposal.

        Args:
            context: Dictionary containing context information
                     (e.g., requirements, existing code, constraints)

        Returns:
            Dictionary containing analysis results and insights
        """
        pass

    @abstractmethod
    def propose(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a proposal based on the analysis and context.

        This method should use the analysis results and original context to create a concrete
        proposal for how to proceed (e.g., code structure, architecture).

        Args:
            analysis: Dictionary containing analysis results from analyze()
            context: Dictionary containing original context information

        Returns:
            Dictionary containing the proposal and supporting information
        """
        pass

    @abstractmethod
    def review(self, proposal: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Review a proposal and provide feedback on quality and safety.

        This method should examine the proposal and provide feedback
        on various aspects like quality, security, performance, etc.

        Args:
            proposal: Dictionary containing the proposal to review
            context: Dictionary containing original context information

        Returns:
            Dictionary containing review results and recommendations
        """
        pass

    def set_message_bus(self, message_bus):
        """
        Set the message bus for inter-agent communication.

        Args:
            message_bus: Message bus instance for agent communication
        """
        self.message_bus = message_bus
        logger.debug(f"Agent {self.name} connected to message bus")

    def set_llm_client(self, llm_client):
        """
        Set the LLM client for this agent.

        Args:
            llm_client: LLM client instance for generating responses
        """
        self.llm_client = llm_client
        logger.debug(f"Agent {self.name} connected to LLM client")

    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the agent.

        Returns:
            Dictionary containing agent status information
        """
        return {
            "name": self.name,
            "config": self.config,
            "has_message_bus": self.message_bus is not None,
            "has_llm_client": self.llm_client is not None,
        }

    def __str__(self) -> str:
        """String representation of the agent."""
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __repr__(self) -> str:
        """Detailed string representation of the agent."""
        return f"{self.__class__.__name__}(name='{self.name}', config={self.config})"
