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
from .cursor_local_api import CursorLocalAPI
from .hotkeys import HotkeyManager, get_hotkey_manager, start_global_hotkeys, stop_global_hotkeys
from .notifications import notify_success, notify_error

__all__ = [
    "ClipboardManager", 
    "CodeExtractor", 
    "CursorIntegration", 
    "ExtractedFile",
    "ClipboardMonitor",
    "CloudEscalator", 
    "CursorLocalAPI",
    "HotkeyManager",
    "get_hotkey_manager",
    "start_global_hotkeys", 
    "stop_global_hotkeys",
    "notify_success",
    "notify_error",
]
