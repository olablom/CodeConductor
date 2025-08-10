"""
Integration Components

Handles external integrations like Cursor, clipboard management, and code extraction.
"""

from .cursor_integration import (
    ClipboardManager,
    CodeExtractor,
    CursorIntegration,
    ExtractedFile,
)
from .clipboard_monitor import ClipboardMonitor
from .cloud_escalator import CloudEscalator
from .hotkeys import (
    HotkeyManager,
    get_hotkey_manager,
    start_global_hotkeys,
    stop_global_hotkeys,
)
from .notifications import notify_success, notify_error

# Optional external adapters (behind ALLOW_NET)
try:
    from .net.github_code_search import GitHubCodeSearch  # type: ignore
except Exception:
    GitHubCodeSearch = None  # type: ignore

__all__ = [
    "ClipboardManager",
    "CodeExtractor",
    "CursorIntegration",
    "ExtractedFile",
    "ClipboardMonitor",
    "CloudEscalator",
    "HotkeyManager",
    "get_hotkey_manager",
    "start_global_hotkeys",
    "stop_global_hotkeys",
    "notify_success",
    "notify_error",
    "GitHubCodeSearch",
]
