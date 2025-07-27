#!/usr/bin/env python3
"""
Test Hotkeys for CodeConductor MVP

Simple script to test global hotkeys functionality.
"""

import time
import logging
from integrations.hotkeys import (
    start_global_hotkeys,
    stop_global_hotkeys,
    show_hotkey_help,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_copy_prompt():
    """Test callback for copy prompt hotkey."""
    print("🎯 Copy prompt hotkey pressed!")
    print("   This would copy the current prompt to clipboard")


def test_paste_from_cursor():
    """Test callback for paste from Cursor hotkey."""
    print("🤖 Paste from Cursor hotkey pressed!")
    print("   This would wait for Cursor output and paste it")


def test_rerun_last_task():
    """Test callback for re-run last task hotkey."""
    print("🔄 Re-run last task hotkey pressed!")
    print("   This would re-run the last pipeline task")


def test_run_tests():
    """Test callback for run tests hotkey."""
    print("🧪 Run tests hotkey pressed!")
    print("   This would run tests on the current directory")


def test_stop_pipeline():
    """Test callback for stop pipeline hotkey."""
    print("🛑 Stop pipeline hotkey pressed!")
    print("   This would stop any running pipeline")


def main():
    """Main test function."""
    print("🎹 CodeConductor Hotkey Test")
    print("=" * 40)

    # Show available hotkeys
    show_hotkey_help()
    print()

    # Define test callbacks
    callbacks = {
        "copy_prompt": test_copy_prompt,
        "paste_from_cursor": test_paste_from_cursor,
        "rerun_last_task": test_rerun_last_task,
        "run_tests": test_run_tests,
        "stop_pipeline": test_stop_pipeline,
    }

    # Start hotkeys
    print("🚀 Starting hotkey test...")
    start_global_hotkeys(callbacks)

    print("\n💡 Test Instructions:")
    print("   - Press Ctrl+Shift+C to test copy prompt")
    print("   - Press Ctrl+Shift+V to test paste from Cursor")
    print("   - Press Ctrl+Shift+R to test re-run last task")
    print("   - Press Ctrl+Shift+T to test run tests")
    print("   - Press Ctrl+Shift+S to test stop pipeline")
    print("   - Press Ctrl+C to exit")
    print("\n⏳ Waiting for hotkey presses...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping hotkey test...")
        stop_global_hotkeys()
        print("✅ Hotkey test completed!")


if __name__ == "__main__":
    main()
