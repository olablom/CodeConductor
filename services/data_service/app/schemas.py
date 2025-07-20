"""
Data Service - API Schemas

This module defines the Pydantic models for request/response validation
in the Data Service API.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union
from enum import Enum


class BanditType(str, Enum):
    """Types of bandit algorithms."""

    LINUCB = "linucb"
    UCB = "ucb"
    THOMPSON = "thompson"


class QLearningConfig(BaseModel):
    """Configuration for Q-learning agent."""

    learning_rate: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Learning rate (alpha)"
    )
    discount_factor: float = Field(
        default=0.9, ge=0.0, le=1.0, description="Discount factor (gamma)"
    )
    epsilon: float = Field(default=0.1, ge=0.0, le=1.0, description="Exploration rate")
    epsilon_decay: float = Field(
        default=0.995, ge=0.0, le=1.0, description="Epsilon decay rate"
    )
    epsilon_min: float = Field(
        default=0.01, ge=0.0, le=1.0, description="Minimum epsilon"
    )


class BanditConfig(BaseModel):
    """Configuration for bandit algorithms."""

    d: int = Field(..., description="Feature vector dimensions")
    alpha: float = Field(default=1.0, ge=0.0, description="Exploration parameter")
    bandit_type: BanditType = Field(
        default=BanditType.LINUCB, description="Bandit algorithm type"
    )


# Bandit schemas
class BanditChooseRequest(BaseModel):
    """Request to choose an arm from bandit."""

    arms: List[str] = Field(..., description="Available arms")
    features: List[float] = Field(..., description="Feature vector")
    bandit_name: str = Field(default="default", description="Bandit instance name")
    config: Optional[BanditConfig] = Field(None, description="Bandit configuration")


class BanditChooseResponse(BaseModel):
    """Response from bandit arm selection."""

    selected_arm: str = Field(..., description="Selected arm")
    ucb_values: Dict[str, float] = Field(..., description="UCB values for all arms")
    confidence_intervals: Dict[str, float] = Field(
        ..., description="Confidence intervals"
    )
    bandit_name: str = Field(..., description="Bandit instance name")
    exploration: bool = Field(..., description="Whether exploration was used")
    timestamp: str = Field(..., description="Selection timestamp")


class BanditUpdateRequest(BaseModel):
    """Request to update bandit with reward."""

    arm: str = Field(..., description="Arm that was pulled")
    features: List[float] = Field(..., description="Feature vector used")
    reward: float = Field(..., description="Observed reward")
    bandit_name: str = Field(default="default", description="Bandit instance name")


class BanditUpdateResponse(BaseModel):
    """Response from bandit update."""

    updated: bool = Field(..., description="Whether update was successful")
    arm: str = Field(..., description="Updated arm")
    reward: float = Field(..., description="Reward that was used")
    bandit_name: str = Field(..., description="Bandit instance name")
    timestamp: str = Field(..., description="Update timestamp")


class BanditStatsResponse(BaseModel):
    """Bandit statistics response."""

    bandit_name: str = Field(..., description="Bandit instance name")
    total_pulls: int = Field(..., description="Total number of pulls")
    arm_count: int = Field(..., description="Number of arms")
    arms: Dict[str, Dict[str, Any]] = Field(..., description="Per-arm statistics")
    config: Dict[str, Any] = Field(..., description="Bandit configuration")


# Q-learning schemas
class QLearningState(BaseModel):
    """Q-learning state representation."""

    task_type: str = Field(..., description="Task type")
    complexity: str = Field(..., description="Task complexity")
    language: str = Field(..., description="Programming language")
    agent_count: int = Field(..., description="Number of agents")
    context_hash: str = Field(..., description="Context hash")


class QLearningAction(BaseModel):
    """Q-learning action representation."""

    agent_combination: str = Field(..., description="Agent combination")
    prompt_strategy: str = Field(..., description="Prompt strategy")
    iteration_count: int = Field(..., description="Number of iterations")
    confidence_threshold: float = Field(..., description="Confidence threshold")


class QLearningRunRequest(BaseModel):
    """Request to run Q-learning episode."""

    context: Dict[str, Any] = Field(..., description="Context for Q-learning")
    config: Optional[QLearningConfig] = Field(
        None, description="Q-learning configuration"
    )
    agent_name: str = Field(default="qlearning_agent", description="Agent name")


class QLearningRunResponse(BaseModel):
    """Response from Q-learning episode."""

    agent_name: str = Field(..., description="Agent name")
    selected_action: QLearningAction = Field(..., description="Selected action")
    q_value: float = Field(..., description="Q-value for selected action")
    epsilon: float = Field(..., description="Current epsilon value")
    exploration: bool = Field(..., description="Whether exploration was used")
    confidence: float = Field(..., description="Confidence in selection")
    reasoning: str = Field(..., description="Reasoning for selection")
    timestamp: str = Field(..., description="Selection timestamp")


class QLearningUpdateRequest(BaseModel):
    """Request to update Q-learning with reward."""

    context: Dict[str, Any] = Field(..., description="Original context")
    action: QLearningAction = Field(..., description="Action that was taken")
    reward: float = Field(..., description="Observed reward")
    next_context: Optional[Dict[str, Any]] = Field(None, description="Next context")
    agent_name: str = Field(default="qlearning_agent", description="Agent name")


class QLearningUpdateResponse(BaseModel):
    """Response from Q-learning update."""

    updated: bool = Field(..., description="Whether update was successful")
    agent_name: str = Field(..., description="Agent name")
    old_q_value: float = Field(..., description="Previous Q-value")
    new_q_value: float = Field(..., description="Updated Q-value")
    reward: float = Field(..., description="Reward that was used")
    timestamp: str = Field(..., description="Update timestamp")


class QLearningStatsResponse(BaseModel):
    """Q-learning statistics response."""

    agent_name: str = Field(..., description="Agent name")
    total_episodes: int = Field(..., description="Total episodes")
    successful_episodes: int = Field(..., description="Successful episodes")
    success_rate: float = Field(..., description="Success rate")
    epsilon: float = Field(..., description="Current epsilon")
    q_table_size: int = Field(..., description="Q-table size")
    average_q_value: float = Field(..., description="Average Q-value")
    min_q_value: float = Field(..., description="Minimum Q-value")
    max_q_value: float = Field(..., description="Maximum Q-value")
    total_visits: int = Field(..., description="Total state-action visits")
    visit_distribution: Dict[str, float] = Field(..., description="Visit distribution")
    learning_rate: float = Field(..., description="Learning rate")
    discount_factor: float = Field(..., description="Discount factor")


# Prompt optimization schemas
class PromptAction(str, Enum):
    """Available prompt mutation actions."""

    ADD_TYPE_HINTS = "add_type_hints"
    ASK_FOR_OOP = "ask_for_oop"
    ADD_DOCSTRINGS = "add_docstrings"
    SIMPLIFY = "simplify"
    ADD_EXAMPLES = "add_examples"
    NO_CHANGE = "no_change"


class PromptOptimizeRequest(BaseModel):
    """Request to optimize prompt."""

    original_prompt: str = Field(..., description="Original prompt")
    task_id: str = Field(..., description="Task identifier")
    arm_prev: str = Field(..., description="Previous arm used")
    passed: bool = Field(..., description="Whether tests passed")
    blocked: bool = Field(..., description="Whether policy blocked")
    complexity: float = Field(..., ge=0.0, le=1.0, description="Task complexity")
    model_source: str = Field(..., description="Source model used")
    agent_name: str = Field(default="prompt_optimizer", description="Agent name")


class PromptOptimizeResponse(BaseModel):
    """Response from prompt optimization."""

    agent_name: str = Field(..., description="Agent name")
    selected_action: PromptAction = Field(..., description="Selected action")
    original_prompt: str = Field(..., description="Original prompt")
    mutated_prompt: str = Field(..., description="Mutated prompt")
    mutation: str = Field(..., description="Applied mutation")
    q_value: float = Field(..., description="Q-value for action")
    epsilon: float = Field(..., description="Current epsilon")
    exploration: bool = Field(..., description="Whether exploration was used")
    confidence: float = Field(..., description="Confidence in selection")
    reasoning: str = Field(..., description="Reasoning for selection")
    timestamp: str = Field(..., description="Optimization timestamp")


class PromptOptimizerStatsResponse(BaseModel):
    """Prompt optimizer statistics response."""

    agent_name: str = Field(..., description="Agent name")
    total_states: int = Field(..., description="Total states in Q-table")
    total_entries: int = Field(..., description="Total Q-table entries")
    average_q_value: float = Field(..., description="Average Q-value")
    min_q_value: float = Field(..., description="Minimum Q-value")
    max_q_value: float = Field(..., description="Maximum Q-value")
    epsilon: float = Field(..., description="Current epsilon")
    total_episodes: int = Field(..., description="Total episodes")
    successful_episodes: int = Field(..., description="Successful episodes")
    success_rate: float = Field(..., description="Success rate")
    action_stats: Dict[str, int] = Field(..., description="Action usage statistics")


# Health and status schemas
class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    bandits_ready: bool = Field(..., description="Whether bandits are ready")
    qlearning_ready: bool = Field(..., description="Whether Q-learning is ready")
    prompt_optimizer_ready: bool = Field(
        ..., description="Whether prompt optimizer is ready"
    )
    active_bandits: List[str] = Field(..., description="Active bandit instances")
    active_agents: List[str] = Field(..., description="Active agent instances")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    request_id: Optional[str] = Field(None, description="Request ID if applicable")


# Model state schemas
class ModelStateRequest(BaseModel):
    """Request to get/set model state."""

    model_name: str = Field(..., description="Model name")
    model_type: str = Field(
        ..., description="Model type (bandit, qlearning, prompt_optimizer)"
    )


class ModelStateResponse(BaseModel):
    """Response with model state."""

    model_name: str = Field(..., description="Model name")
    model_type: str = Field(..., description="Model type")
    state: Dict[str, Any] = Field(..., description="Model state")
    timestamp: str = Field(..., description="State timestamp")


class ModelResetRequest(BaseModel):
    """Request to reset model."""

    model_name: str = Field(..., description="Model name")
    model_type: str = Field(..., description="Model type")


class ModelResetResponse(BaseModel):
    """Response from model reset."""

    reset: bool = Field(..., description="Whether reset was successful")
    model_name: str = Field(..., description="Model name")
    model_type: str = Field(..., description="Model type")
    timestamp: str = Field(..., description="Reset timestamp")
