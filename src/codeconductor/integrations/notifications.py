#!/usr/bin/env python3
"""
Windows Notifications for CodeConductor MVP

Provides user feedback with toast notifications and sounds.
"""

import logging
import time

# Windows-specific imports
try:
    import winsound

    from win10toast import ToastNotifier

    WINDOWS_AVAILABLE = True
except ImportError:
    WINDOWS_AVAILABLE = False

logger = logging.getLogger(__name__)


class NotificationManager:
    """Manages Windows notifications for CodeConductor workflow."""

    def __init__(self):
        self.toaster = None
        self.enabled = WINDOWS_AVAILABLE

        if self.enabled:
            try:
                self.toaster = ToastNotifier()
                logger.info("Windows notifications initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize notifications: {e}")
                self.enabled = False
        else:
            logger.warning("Windows notifications not available")

    def show_notification(
        self,
        title: str,
        message: str,
        duration: int = 5,
        icon_path: str | None = None,
        sound: bool = True,
    ):
        """
        Show a Windows toast notification.

        Args:
            title: Notification title
            message: Notification message
            duration: Display duration in seconds
            icon_path: Path to custom icon
            sound: Whether to play notification sound
        """
        if not self.enabled or not self.toaster:
            logger.info(f"Notification: {title} - {message}")
            return

        try:
            # Play notification sound
            if sound:
                self._play_notification_sound()

            # Show toast notification
            self.toaster.show_toast(
                title=title,
                msg=message,
                duration=duration,
                icon_path=icon_path,
                threaded=True,
            )

            logger.info(f"Notification shown: {title}")

        except Exception as e:
            logger.error(f"Failed to show notification: {e}")

    def _play_notification_sound(self, sound_type: str = "info"):
        """Play a notification sound."""
        if not WINDOWS_AVAILABLE:
            return

        try:
            # Use simple beep instead of MessageBeep for better compatibility
            if sound_type == "success":
                winsound.Beep(800, 200)  # Higher pitch, shorter duration
            elif sound_type == "error":
                winsound.Beep(400, 400)  # Lower pitch, longer duration
            elif sound_type == "warning":
                winsound.Beep(600, 300)  # Medium pitch, medium duration
            else:  # info
                winsound.Beep(700, 150)  # Standard info beep
        except Exception as e:
            logger.warning(f"Failed to play notification sound: {e}")

    def prompt_copied(self):
        """Notify that prompt has been copied to clipboard."""
        self.show_notification(
            title="🎯 CodeConductor",
            message="Prompt copied to clipboard! Switch to Cursor and paste.",
            duration=8,
            sound=True,
        )

    def code_detected(self, file_count: int = 0):
        """Notify that Cursor output has been detected."""
        message = "Code detected! Processing..."
        if file_count > 0:
            message = f"Code detected! Extracted {file_count} files."

        self.show_notification(
            title="🤖 CodeConductor", message=message, duration=5, sound=True
        )

    def tests_running(self):
        """Notify that tests are running."""
        self.show_notification(
            title="🧪 CodeConductor",
            message="Running tests... Please wait.",
            duration=3,
            sound=False,
        )

    def tests_passed(self, test_count: int = 0):
        """Notify that tests passed."""
        message = "Tests passed! Task complete ✅"
        if test_count > 0:
            message = f"Tests passed! ({test_count} tests) ✅"

        self.show_notification(
            title="✅ CodeConductor", message=message, duration=8, sound=True
        )

    def tests_failed(self, error_count: int = 0):
        """Notify that tests failed."""
        message = "Tests failed! Check errors ❌"
        if error_count > 0:
            message = f"Tests failed! ({error_count} errors) ❌"

        self.show_notification(
            title="❌ CodeConductor", message=message, duration=10, sound=True
        )

    def pipeline_complete(self, success: bool = True):
        """Notify that pipeline is complete."""
        if success:
            self.show_notification(
                title="🎉 CodeConductor",
                message="Pipeline completed successfully!",
                duration=8,
                sound=True,
            )
        else:
            self.show_notification(
                title="⚠️ CodeConductor",
                message="Pipeline completed with issues.",
                duration=8,
                sound=True,
            )

    def error_occurred(self, error_message: str):
        """Notify that an error occurred."""
        # Truncate long error messages
        if len(error_message) > 100:
            error_message = error_message[:97] + "..."

        self.show_notification(
            title="🚨 CodeConductor Error",
            message=f"Error: {error_message}",
            duration=10,
            sound=True,
        )


# Global notification manager
_global_notifications: NotificationManager | None = None


def get_notification_manager() -> NotificationManager:
    """Get or create global notification manager."""
    global _global_notifications
    if _global_notifications is None:
        _global_notifications = NotificationManager()
    return _global_notifications


def notify_prompt_copied():
    """Show notification that prompt was copied to clipboard."""
    get_notification_manager().show_notification(
        "🎯 CodeConductor",
        "📋 Prompt copied to clipboard! Ready for Cursor.",
        duration=3,
    )


def notify_waiting_for_cursor():
    """Show notification that we're waiting for Cursor output."""
    get_notification_manager().show_notification(
        "🤖 CodeConductor", "⏳ Waiting for Cursor to generate code...", duration=5
    )


def notify_code_detected(file_count: int = 0):
    """Global function to notify code detected."""
    get_notification_manager().code_detected(file_count)


def notify_tests_running():
    """Global function to notify tests running."""
    get_notification_manager().tests_running()


def notify_tests_passed(test_count: int = 0):
    """Global function to notify tests passed."""
    get_notification_manager().tests_passed(test_count)


def notify_tests_failed(error_count: int = 0):
    """Global function to notify tests failed."""
    get_notification_manager().tests_failed(error_count)


def notify_pipeline_complete(success: bool = True):
    """Global function to notify pipeline complete."""
    get_notification_manager().pipeline_complete(success)


def notify_error(error_message: str):
    """Global function to notify error."""
    get_notification_manager().error_occurred(error_message)


def notify_success(message: str):
    """Global function to notify success."""
    get_notification_manager().show_notification(
        "✅ CodeConductor", message, duration=5, sound=True
    )


if __name__ == "__main__":
    # Test notifications
    import logging

    logging.basicConfig(level=logging.INFO)

    notifications = NotificationManager()

    print("🔔 Testing notifications...")

    notifications.prompt_copied()
    time.sleep(2)

    notifications.code_detected(2)
    time.sleep(2)

    notifications.tests_running()
    time.sleep(2)

    notifications.tests_passed(5)
    time.sleep(2)

    notifications.pipeline_complete(True)

    print("✅ Notification tests complete!")
