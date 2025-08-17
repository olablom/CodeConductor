"""
CodeConductor Debate System

Multi-agent debate system for code generation tasks.
"""

from .local_agent import LocalAIAgent
from .local_ai_agent import LocalAIAgent as LocalAIAgentV2, LocalDebateManager
from .single_model_agent import SingleModelAIAgent, SingleModelDebateManager
from .shared_model_agent import SharedModelAIAgent, SharedModelDebateManager
from .debate_manager import CodeConductorDebateManager
from .personas import get_persona_prompt, list_personas

__all__ = [
    "LocalAIAgent", "LocalAIAgentV2", "LocalDebateManager",
    "SingleModelAIAgent", "SingleModelDebateManager",
    "SharedModelAIAgent", "SharedModelDebateManager",
    "CodeConductorDebateManager",
    "get_persona_prompt", "list_personas",
]
