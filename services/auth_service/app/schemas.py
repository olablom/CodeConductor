"""
Auth Service - API Schemas

This module defines the Pydantic models for request/response validation
in the Auth Service API.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from enum import Enum


class RiskLevel(str, Enum):
    """Risk levels for approval requests."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ViolationType(str, Enum):
    """Types of policy violations."""

    DANGEROUS_PATTERN = "dangerous_pattern"
    POTENTIAL_SECRET = "potential_secret"
    NETWORK_OPERATION = "network_operation"
    SECURITY_ISSUE = "security_issue"


class ApprovalStrategy(str, Enum):
    """Approval strategies."""

    AUTO_APPROVE = "auto_approve"
    HUMAN_APPROVAL = "human_approval"
    CONDITIONAL_APPROVAL = "conditional_approval"
    REJECT = "reject"


class ApprovalRequest(BaseModel):
    """Request for approval."""

    context: Dict[str, Any] = Field(..., description="Context requiring approval")
    code: Optional[str] = Field(None, description="Code to be approved")
    task_type: Optional[str] = Field(None, description="Type of task")
    risk_level: RiskLevel = Field(default=RiskLevel.MEDIUM, description="Risk level")
    confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Confidence score"
    )
    agent_analyses: Optional[Dict[str, Any]] = Field(
        None, description="Agent analysis results"
    )


class ApprovalResponse(BaseModel):
    """Response from approval request."""

    approved: bool = Field(..., description="Whether the request was approved")
    strategy: ApprovalStrategy = Field(..., description="Approval strategy used")
    risk_level: RiskLevel = Field(..., description="Assessed risk level")
    violations: List[Dict[str, Any]] = Field(
        default=[], description="Policy violations found"
    )
    recommendations: List[str] = Field(default=[], description="Recommendations")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in decision")
    reasoning: str = Field(..., description="Reasoning for decision")
    timestamp: str = Field(..., description="Timestamp of decision")


class SafetyAnalysis(BaseModel):
    """Code safety analysis results."""

    safe: bool = Field(..., description="Whether code is safe")
    violations: List[Dict[str, Any]] = Field(
        default=[], description="Safety violations"
    )
    risk_level: RiskLevel = Field(..., description="Overall risk level")
    recommendations: List[str] = Field(default=[], description="Safety recommendations")
    total_violations: int = Field(..., description="Total number of violations")
    critical_violations: int = Field(..., description="Number of critical violations")
    high_violations: int = Field(..., description="Number of high risk violations")
    medium_violations: int = Field(..., description="Number of medium risk violations")
    low_violations: int = Field(..., description="Number of low risk violations")


class PolicyAnalysis(BaseModel):
    """Policy analysis results."""

    agent_name: str = Field(..., description="Name of the policy agent")
    task_type: str = Field(..., description="Type of task analyzed")
    safety_analysis: SafetyAnalysis = Field(..., description="Safety analysis results")
    policy_compliant: bool = Field(..., description="Whether compliant with policies")
    risk_assessment: RiskLevel = Field(..., description="Risk assessment")
    recommendations: List[str] = Field(default=[], description="Policy recommendations")
    timestamp: str = Field(..., description="Analysis timestamp")


class PolicyProposal(BaseModel):
    """Policy proposal based on analysis."""

    agent_name: str = Field(..., description="Name of the policy agent")
    strategy: ApprovalStrategy = Field(..., description="Proposed approval strategy")
    approved: bool = Field(..., description="Whether to approve")
    requires_human: bool = Field(..., description="Whether human approval required")
    risk_level: RiskLevel = Field(..., description="Risk level")
    violations: List[Dict[str, Any]] = Field(
        default=[], description="Policy violations"
    )
    recommendations: List[str] = Field(default=[], description="Recommendations")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in proposal")
    reasoning: str = Field(..., description="Reasoning for proposal")


class PolicyReview(BaseModel):
    """Final policy review results."""

    agent_name: str = Field(..., description="Name of the policy agent")
    final_approval: bool = Field(..., description="Final approval decision")
    strategy: ApprovalStrategy = Field(..., description="Final strategy")
    risk_level: RiskLevel = Field(..., description="Final risk level")
    violations: List[Dict[str, Any]] = Field(default=[], description="Final violations")
    recommendations: List[str] = Field(default=[], description="Final recommendations")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Final confidence")
    reasoning: str = Field(..., description="Final reasoning")
    timestamp: str = Field(..., description="Review timestamp")


class HumanApprovalStats(BaseModel):
    """Human approval statistics."""

    total_approvals: int = Field(..., description="Total number of approvals")
    approved_count: int = Field(..., description="Number of approved requests")
    rejected_count: int = Field(..., description="Number of rejected requests")
    approval_rate: float = Field(..., ge=0.0, le=1.0, description="Approval rate")
    recent_decisions: List[Dict[str, Any]] = Field(
        default=[], description="Recent decisions"
    )


class HumanApprovalSummary(BaseModel):
    """Human approval summary."""

    overall_stats: HumanApprovalStats = Field(..., description="Overall statistics")
    recent_approval_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Recent approval rate"
    )
    total_history_entries: int = Field(..., description="Total history entries")
    last_decision: Optional[Dict[str, Any]] = Field(
        None, description="Last decision made"
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    policy_agent_ready: bool = Field(..., description="Whether policy agent is ready")
    human_gate_ready: bool = Field(..., description="Whether human gate is ready")
    approval_stats: Optional[HumanApprovalStats] = Field(
        None, description="Approval statistics"
    )


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    request_id: Optional[str] = Field(None, description="Request ID if applicable")


# Violation detail models
class Violation(BaseModel):
    """Detailed violation information."""

    type: ViolationType = Field(..., description="Type of violation")
    severity: RiskLevel = Field(..., description="Severity of violation")
    pattern: str = Field(..., description="Pattern that triggered violation")
    line: int = Field(..., description="Line number where violation occurred")
    position: int = Field(..., description="Character position in code")
    description: str = Field(..., description="Human-readable description")


class CodeSafetyRequest(BaseModel):
    """Request for code safety analysis."""

    code: str = Field(..., description="Code to analyze")
    task_type: Optional[str] = Field(None, description="Type of task")


class CodeSafetyResponse(BaseModel):
    """Response from code safety analysis."""

    safety_analysis: SafetyAnalysis = Field(..., description="Safety analysis results")
    timestamp: str = Field(..., description="Analysis timestamp")
