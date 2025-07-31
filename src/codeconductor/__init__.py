"""
CodeConductor - Personal AI Development Platform

Advanced ensemble consensus algorithms, RTX 5090 optimization, and enterprise-grade AI patterns.
"""

__version__ = "0.1.0"
__author__ = "Ola Blom"
__email__ = "olablom@github.com"

# Core imports
from .app import CodeConductorApp
from .dashboard import ValidationDashboard
from .logger import ValidationLogger

# Ensemble imports
from .ensemble import (
    ModelManager,
    QueryDispatcher,
    ConsensusCalculator,
    HybridEnsemble,
    EnsembleEngine,
)

# Analysis imports
from .analysis import (
    PlannerAgent,
    ProjectAnalyzer,
    TreeSitterAnalyzer,
)

# Feedback imports
from .feedback import (
    LearningSystem,
    RLHFAgent,
    validate_cursor_output,
    CodeValidator,
    ValidationResult,
)

# Generator imports
from .generators import (
    PromptGenerator,
    SimplePromptGenerator,
)

# Integration imports
from .integrations import (
    ClipboardMonitor,
    CloudEscalator,
    CursorIntegration,
    CursorLocalAPI,
    HotkeyManager,
    get_hotkey_manager,
    start_global_hotkeys,
    stop_global_hotkeys,
    notify_success,
    notify_error,
)

# Context imports
from .context import RAGSystem

# Runner imports
from .runners import TestRunner

__all__ = [
    # Core
    "CodeConductorApp",
    "ValidationDashboard", 
    "ValidationLogger",
    
    # Ensemble
    "ModelManager",
    "QueryDispatcher",
    "ConsensusCalculator",
    "HybridEnsemble",
    "EnsembleEngine",
    
    # Analysis
    "PlannerAgent",
    "ProjectAnalyzer",
    "TreeSitterAnalyzer",
    
    # Feedback
    "LearningSystem",
    "RLHFAgent",
    "validate_cursor_output",
    "CodeValidator",
    "ValidationResult",
    
    # Generators
    "PromptGenerator",
    "SimplePromptGenerator",
    
    # Integrations
    "ClipboardMonitor",
    "CloudEscalator",
    "CursorIntegration",
    "CursorLocalAPI",
    "HotkeyManager",
    "get_hotkey_manager",
    "start_global_hotkeys",
    "stop_global_hotkeys",
    "notify_success",
    "notify_error",
    
    # Context
    "RAGSystem",
    
    # Runners
    "TestRunner",
] 