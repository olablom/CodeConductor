#!/usr/bin/env python3
"""
Global Hotkeys for CodeConductor MVP

Provides keyboard shortcuts for common CodeConductor actions.
"""

import logging
from collections.abc import Callable

# Windows-specific imports
try:
    import keyboard

    WINDOWS_AVAILABLE = True
except ImportError:
    WINDOWS_AVAILABLE = False

logger = logging.getLogger(__name__)


class HotkeyManager:
    """Manages global hotkeys for CodeConductor workflow."""

    def __init__(self):
        self.enabled = WINDOWS_AVAILABLE
        self.is_running = False
        self.hotkey_thread = None
        self.callbacks: dict[str, Callable] = {}

        # Default hotkey mappings
        self.default_hotkeys = {
            "ctrl+shift+c": "copy_prompt",
            "ctrl+shift+v": "paste_from_cursor",
            "ctrl+shift+r": "rerun_last_task",
            "ctrl+shift+t": "run_tests",
            "ctrl+shift+s": "stop_pipeline",
        }

        if not self.enabled:
            logger.warning("Global hotkeys not available (keyboard library not installed)")

    def register_hotkey(self, hotkey: str, callback: Callable, description: str = ""):
        """
        Register a hotkey with callback function.

        Args:
            hotkey: Hotkey combination (e.g., "ctrl+shift+c")
            callback: Function to call when hotkey is pressed
            description: Description of what the hotkey does
        """
        if not self.enabled:
            logger.warning(f"Cannot register hotkey '{hotkey}' - hotkeys disabled")
            return

        try:
            keyboard.add_hotkey(hotkey, callback, suppress=True)
            self.callbacks[hotkey] = callback
            logger.info(f"Registered hotkey: {hotkey} - {description}")
        except Exception as e:
            logger.error(f"Failed to register hotkey '{hotkey}': {e}")

    def unregister_hotkey(self, hotkey: str):
        """Unregister a hotkey."""
        if not self.enabled:
            return

        try:
            keyboard.remove_hotkey(hotkey)
            if hotkey in self.callbacks:
                del self.callbacks[hotkey]
            logger.info(f"Unregistered hotkey: {hotkey}")
        except Exception as e:
            logger.error(f"Failed to unregister hotkey '{hotkey}': {e}")

    def register_default_hotkeys(self, callbacks: dict[str, Callable]):
        """
        Register default CodeConductor hotkeys.

        Args:
            callbacks: Dictionary mapping action names to callback functions
        """
        if not self.enabled:
            logger.warning("Cannot register default hotkeys - hotkeys disabled")
            return

        for hotkey, action in self.default_hotkeys.items():
            if action in callbacks:
                self.register_hotkey(hotkey, callbacks[action], f"Default: {action}")
            else:
                logger.warning(f"No callback provided for action: {action}")

    def start_listening(self):
        """Start listening for hotkeys."""
        if not self.enabled:
            logger.warning("Cannot start hotkey listening - hotkeys disabled")
            return

        if self.is_running:
            logger.warning("Hotkey listening already started")
            return

        self.is_running = True
        logger.info("Global hotkeys enabled")
        logger.info("Available hotkeys:")
        for hotkey, action in self.default_hotkeys.items():
            logger.info(f"  {hotkey}: {action}")

    def stop_listening(self):
        """Stop listening for hotkeys."""
        if not self.enabled:
            return

        self.is_running = False

        # Unregister all hotkeys
        for hotkey in list(self.callbacks.keys()):
            self.unregister_hotkey(hotkey)

        logger.info("Global hotkeys disabled")

    def get_hotkey_help(self) -> str:
        """Get help text for available hotkeys."""
        help_text = "🎹 CodeConductor Hotkeys:\n"
        help_text += "=" * 40 + "\n"

        for hotkey, action in self.default_hotkeys.items():
            help_text += f"{hotkey:15} - {action}\n"

        help_text += "\n💡 Tip: Press hotkeys from any application!"
        return help_text


# Global hotkey manager
_global_hotkeys: HotkeyManager | None = None


def get_hotkey_manager() -> HotkeyManager:
    """Get or create global hotkey manager."""
    global _global_hotkeys
    if _global_hotkeys is None:
        _global_hotkeys = HotkeyManager()
    return _global_hotkeys


def start_global_hotkeys(callbacks: dict[str, Callable]):
    """Start global hotkeys with provided callbacks."""
    manager = get_hotkey_manager()
    manager.register_default_hotkeys(callbacks)
    manager.start_listening()


def stop_global_hotkeys():
    """Stop global hotkeys."""
    manager = get_hotkey_manager()
    manager.stop_listening()


def show_hotkey_help():
    """Show available hotkeys."""
    manager = get_hotkey_manager()
    print(manager.get_hotkey_help())


# Example callback functions for integration
def example_copy_prompt():
    """Example callback for copying prompt to clipboard."""
    print("🎯 Copying prompt to clipboard...")
    # This would integrate with your clipboard manager
    # clipboard_manager.copy_to_clipboard(current_prompt)


def example_paste_from_cursor():
    """Example callback for pasting from Cursor."""
    print("🤖 Pasting from Cursor...")
    # This would integrate with your clipboard monitor
    # clipboard_monitor.wait_for_cursor_output()


def example_rerun_last_task():
    """Example callback for re-running last task."""
    print("🔄 Re-running last task...")
    # This would re-run the last pipeline


def example_run_tests():
    """Example callback for running tests."""
    print("🧪 Running tests...")
    # This would run tests on current directory


def example_stop_pipeline():
    """Example callback for stopping pipeline."""
    print("🛑 Stopping pipeline...")
    # This would stop any running pipeline


if __name__ == "__main__":
    # Test hotkeys
    import logging
    import time

    logging.basicConfig(level=logging.INFO)

    # Create example callbacks
    callbacks = {
        "copy_prompt": example_copy_prompt,
        "paste_from_cursor": example_paste_from_cursor,
        "rerun_last_task": example_rerun_last_task,
        "run_tests": example_run_tests,
        "stop_pipeline": example_stop_pipeline,
    }

    # Start hotkeys
    start_global_hotkeys(callbacks)

    print("🎹 Global hotkeys enabled!")
    print("Press Ctrl+Shift+C to copy prompt")
    print("Press Ctrl+Shift+V to paste from Cursor")
    print("Press Ctrl+Shift+R to re-run last task")
    print("Press Ctrl+Shift+T to run tests")
    print("Press Ctrl+Shift+S to stop pipeline")
    print("Press Ctrl+C to exit")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping hotkeys...")
        stop_global_hotkeys()
