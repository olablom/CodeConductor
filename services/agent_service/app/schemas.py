"""
Agent Service - API Schemas

This module defines the Pydantic models for request/response validation
in the Agent Service API.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from enum import Enum


class AgentType(str, Enum):
    """Supported agent types."""

    CODEGEN = "codegen"
    REVIEW = "review"
    ARCHITECT = "architect"
    REWARD = "reward"


class TaskContext(BaseModel):
    """Task context for agent analysis."""

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
    high_traffic: Optional[bool] = Field(None, description="High traffic requirements")
    large_data: Optional[bool] = Field(None, description="Large data processing")
    real_time: Optional[bool] = Field(None, description="Real-time processing")
    user_input: Optional[bool] = Field(None, description="User input handling")
    authentication: Optional[bool] = Field(None, description="Authentication required")
    sensitive_data: Optional[bool] = Field(None, description="Sensitive data handling")
    deployment_required: Optional[bool] = Field(
        None, description="Deployment instructions needed"
    )


class AgentAnalysisRequest(BaseModel):
    """Request for agent analysis."""

    agent_type: AgentType = Field(..., description="Type of agent to use")
    task_context: TaskContext = Field(..., description="Task context")
    config: Optional[Dict[str, Any]] = Field(None, description="Agent configuration")


class AgentAnalysisResponse(BaseModel):
    """Response from agent analysis."""

    agent_name: str = Field(..., description="Name of the agent")
    task: str = Field(..., description="Analyzed task")
    language: str = Field(..., description="Programming language")
    requirements: List[str] = Field(..., description="Extracted requirements")
    patterns: List[str] = Field(..., description="Identified patterns")
    challenges: List[str] = Field(..., description="Identified challenges")
    complexity: str = Field(..., description="Estimated complexity")
    recommended_approach: str = Field(..., description="Recommended approach")
    quality_focus: List[str] = Field(..., description="Quality focus areas")
    performance_needs: List[str] = Field(..., description="Performance needs")
    security_needs: List[str] = Field(..., description="Security needs")
    testing_strategy: str = Field(..., description="Testing strategy")
    documentation_needs: Dict[str, Any] = Field(..., description="Documentation needs")


class AgentProposalRequest(BaseModel):
    """Request for agent proposal."""

    agent_type: AgentType = Field(..., description="Type of agent to use")
    analysis: Dict[str, Any] = Field(..., description="Analysis results")
    task_context: TaskContext = Field(..., description="Original task context")
    config: Optional[Dict[str, Any]] = Field(None, description="Agent configuration")


class AgentProposalResponse(BaseModel):
    """Response from agent proposal."""

    agent_name: str = Field(..., description="Name of the agent")
    approach: str = Field(..., description="Proposed approach")
    structure: Dict[str, Any] = Field(..., description="Code structure")
    implementation_plan: List[str] = Field(..., description="Implementation plan")
    code_template: str = Field(..., description="Generated code template")
    quality_guidelines: List[str] = Field(..., description="Quality guidelines")
    documentation_plan: Dict[str, Any] = Field(..., description="Documentation plan")
    size_estimate: Dict[str, int] = Field(..., description="Code size estimate")
    confidence: float = Field(..., description="Confidence score")
    reasoning: str = Field(..., description="Reasoning for proposal")
    suggestions: List[str] = Field(..., description="Improvement suggestions")
    analysis_summary: Dict[str, Any] = Field(..., description="Analysis summary")


class AgentReviewRequest(BaseModel):
    """Request for agent review."""

    agent_type: AgentType = Field(..., description="Type of agent to use")
    proposal: Dict[str, Any] = Field(..., description="Proposal to review")
    task_context: TaskContext = Field(..., description="Original task context")
    config: Optional[Dict[str, Any]] = Field(None, description="Agent configuration")


class AgentReviewResponse(BaseModel):
    """Response from agent review."""

    agent_name: str = Field(..., description="Name of the agent")
    quality_score: float = Field(..., description="Code quality score")
    issues: List[str] = Field(..., description="Identified issues")
    improvements: List[str] = Field(..., description="Improvement suggestions")
    best_practices: Dict[str, bool] = Field(..., description="Best practices adherence")
    performance_notes: List[str] = Field(..., description="Performance notes")
    security_notes: List[str] = Field(..., description="Security notes")
    readability_score: float = Field(..., description="Readability score")
    maintainability_score: float = Field(..., description="Maintainability score")
    test_recommendations: List[str] = Field(..., description="Test recommendations")
    documentation_gaps: List[str] = Field(..., description="Documentation gaps")
    overall_assessment: str = Field(..., description="Overall assessment")
    proposal_quality: Dict[str, Any] = Field(
        ..., description="Proposal quality assessment"
    )
    review_summary: Dict[str, Any] = Field(..., description="Review summary")


class AgentStatusResponse(BaseModel):
    """Response for agent status."""

    agent_name: str = Field(..., description="Name of the agent")
    config: Dict[str, Any] = Field(..., description="Agent configuration")
    has_message_bus: bool = Field(..., description="Message bus connection status")
    has_llm_client: bool = Field(..., description="LLM client connection status")
    status: str = Field(..., description="Overall status")


class AgentConfig(BaseModel):
    """Agent configuration."""

    max_code_length: int = Field(default=1000, description="Maximum code length")
    quality_threshold: float = Field(default=0.7, description="Quality threshold")
    supported_languages: List[str] = Field(
        default=["python"], description="Supported languages"
    )
    code_style: str = Field(default="pep8", description="Code style guide")
    include_tests: bool = Field(default=True, description="Include tests")
    include_docs: bool = Field(default=True, description="Include documentation")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    agent_name: Optional[str] = Field(None, description="Agent name if applicable")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    agents_available: List[str] = Field(..., description="Available agent types")
