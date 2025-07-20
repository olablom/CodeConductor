"""
Orchestrator Service - API Schemas

This module defines the Pydantic models for request/response validation
in the Orchestrator Service API.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from enum import Enum


class ConsensusStrategy(str, Enum):
    """Available consensus strategies."""

    MAJORITY = "majority"
    WEIGHTED_MAJORITY = "weighted_majority"
    UNANIMOUS = "unanimous"


class TaskContext(BaseModel):
    """Task context for orchestration."""

    task: str = Field(..., description="The task to be performed")
    requirements: List[str] = Field(default=[], description="List of requirements")
    language: str = Field(default="python", description="Programming language")
    existing_code: Optional[str] = Field(None, description="Existing code if any")
    constraints: Optional[Dict[str, Any]] = Field(
        None, description="Additional constraints"
    )
    performance_requirements: Optional[bool] = Field(
        None, description="Performance requirements"
    )
    security_requirements: Optional[bool] = Field(
        None, description="Security requirements"
    )
    test_requirements: Optional[bool] = Field(None, description="Testing requirements")


class OrchestratorConfig(BaseModel):
    """Configuration for the orchestrator."""

    consensus_strategy: ConsensusStrategy = Field(
        default=ConsensusStrategy.WEIGHTED_MAJORITY,
        description="Strategy for reaching consensus",
    )
    max_rounds: int = Field(default=3, description="Maximum discussion rounds")
    consensus_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Minimum agreement for consensus"
    )
    agent_weights: Dict[str, float] = Field(
        default={}, description="Custom weights for agents"
    )
    enable_voting: bool = Field(default=True, description="Enable voting mechanism")
    enable_feedback: bool = Field(
        default=True, description="Enable inter-agent feedback"
    )
    timeout_seconds: int = Field(
        default=30, description="Timeout for discussion rounds"
    )


class DiscussionRequest(BaseModel):
    """Request to start a discussion."""

    task_context: TaskContext = Field(..., description="Task context")
    agents: Optional[List[str]] = Field(
        default=None,
        description="List of agent types to use (defaults to all available)",
    )
    config: Optional[OrchestratorConfig] = Field(
        default=None, description="Orchestrator configuration"
    )


class DiscussionRound(BaseModel):
    """Represents a single discussion round."""

    round_id: int = Field(..., description="Round number")
    task_context: Dict[str, Any] = Field(..., description="Task context for this round")
    analyses: List[Dict[str, Any]] = Field(
        ..., description="Analysis results from agents"
    )
    proposals: List[Dict[str, Any]] = Field(
        ..., description="Proposal results from agents"
    )
    consensus: Optional[Dict[str, Any]] = Field(
        None, description="Consensus result if reached"
    )
    metadata: Optional[Dict[str, Any]] = Field(None, description="Round metadata")


class DiscussionSummary(BaseModel):
    """Summary of the discussion."""

    total_rounds: int = Field(..., description="Total number of rounds")
    agents_used: List[str] = Field(..., description="List of agents used")
    consensus_reached: bool = Field(..., description="Whether consensus was reached")
    final_round: Optional[DiscussionRound] = Field(None, description="Final round data")
    discussion_quality: str = Field(..., description="Quality assessment of discussion")


class DiscussionResponse(BaseModel):
    """Response from a discussion."""

    consensus_reached: bool = Field(..., description="Whether consensus was reached")
    final_consensus: Optional[Dict[str, Any]] = Field(
        None, description="Final consensus result"
    )
    discussion_rounds: int = Field(..., description="Number of rounds completed")
    total_rounds: int = Field(..., description="Total rounds in history")
    agents_used: List[str] = Field(..., description="List of agents used")
    consensus_strategy: str = Field(..., description="Consensus strategy used")
    discussion_summary: DiscussionSummary = Field(..., description="Discussion summary")
    metadata: Dict[str, Any] = Field(..., description="Execution metadata")


class AgentStatistics(BaseModel):
    """Statistics for a single agent."""

    analysis_count: int = Field(..., description="Number of analyses performed")
    proposal_count: int = Field(..., description="Number of proposals generated")
    errors: int = Field(..., description="Number of errors encountered")
    avg_confidence: float = Field(..., description="Average confidence score")


class OrchestratorStatistics(BaseModel):
    """Overall orchestrator statistics."""

    total_rounds: int = Field(..., description="Total discussion rounds")
    agent_statistics: Dict[str, AgentStatistics] = Field(
        ..., description="Per-agent statistics"
    )
    consensus_strategy: str = Field(..., description="Consensus strategy used")
    config: Dict[str, Any] = Field(..., description="Configuration used")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    available_agents: List[str] = Field(..., description="Available agent types")
    consensus_strategies: List[str] = Field(
        ..., description="Available consensus strategies"
    )


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    discussion_id: Optional[str] = Field(
        None, description="Discussion ID if applicable"
    )


# Legacy models for backward compatibility
class Task(BaseModel):
    """Legacy task model."""

    name: str = Field(..., description="Task name")
    params: Dict[str, Any] = Field(..., description="Task parameters")


class OrchestrateRequest(BaseModel):
    """Legacy orchestrate request."""

    tasks: List[Task] = Field(..., description="List of tasks")


class OrchestrateResponse(BaseModel):
    """Legacy orchestrate response."""

    result: Any = Field(..., description="Orchestration result")
