"""
CodeConductor Debate System

Multi-agent debate system for code generation tasks.
"""

from .local_agent import LocalAIAgent
from .debate_manager import CodeConductorDebateManager

__all__ = ["LocalAIAgent", "CodeConductorDebateManager"]
