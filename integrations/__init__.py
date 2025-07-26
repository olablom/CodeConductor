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

__all__ = ["ClipboardManager", "CodeExtractor", "CursorIntegration", "ExtractedFile"]
