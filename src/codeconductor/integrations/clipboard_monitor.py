#!/usr/bin/env python3
"""
Clipboard Monitor for CodeConductor MVP

Auto-detects Cursor output patterns and triggers pipeline steps.
"""

import logging
import re
import threading
import time
from collections.abc import Callable

# Windows-specific imports
try:
    import win32clipboard
    import win32con

    WINDOWS_AVAILABLE = True
except ImportError:
    WINDOWS_AVAILABLE = False

logger = logging.getLogger(__name__)


class ClipboardMonitor:
    """Monitors clipboard for Cursor output patterns."""

    def __init__(self, check_interval: float = 0.5):
        self.check_interval = check_interval
        self.last_content = ""
        self.is_monitoring = False
        self.monitor_thread = None
        self.callbacks: list[Callable] = []

        # Patterns that indicate Cursor has generated code
        self.code_patterns = [
            r"```python\s*\n",  # Python code block
            r"```javascript\s*\n",  # JavaScript code block
            r"```typescript\s*\n",  # TypeScript code block
            r"```java\s*\n",  # Java code block
            r"```cpp\s*\n",  # C++ code block
            r"```c\s*\n",  # C code block
            r"```go\s*\n",  # Go code block
            r"```rust\s*\n",  # Rust code block
            r"```ruby\s*\n",  # Ruby code block
            r"```php\s*\n",  # PHP code block
            r"```html\s*\n",  # HTML code block
            r"```css\s*\n",  # CSS code block
            r"```sql\s*\n",  # SQL code block
            r"```bash\s*\n",  # Bash code block
            r"```shell\s*\n",  # Shell code block
        ]

        # Combined pattern for efficiency
        self.combined_pattern = re.compile("|".join(self.code_patterns), re.IGNORECASE)

    def read_clipboard(self) -> str:
        """Read current clipboard content."""
        if not WINDOWS_AVAILABLE:
            logger.warning("Windows clipboard not available")
            return ""

        try:
            win32clipboard.OpenClipboard()
            try:
                data = win32clipboard.GetClipboardData(win32con.CF_UNICODETEXT)
                return data if data else ""
            except Exception as e:
                logger.warning(f"Failed to read clipboard: {e}")
                return ""
            finally:
                win32clipboard.CloseClipboard()
        except Exception as e:
            logger.error(f"Clipboard access error: {e}")
            return ""

    def is_cursor_output(self, content: str) -> bool:
        """Check if clipboard content looks like Cursor output."""
        if not content or len(content) < 50:
            return False

        # Check for code block patterns
        if self.combined_pattern.search(content):
            return True

        # Check for file paths in code blocks
        file_path_patterns = [
            r"#\s*[\w\-_]+\.py",  # Python file comments
            r"#\s*[\w\-_]+\.js",  # JavaScript file comments
            r"#\s*[\w\-_]+\.ts",  # TypeScript file comments
            r"//\s*[\w\-_]+\.\w+",  # File comments in other languages
        ]

        for pattern in file_path_patterns:
            if re.search(pattern, content):
                return True

        # Check for typical Cursor response structure
        cursor_indicators = [
            "Here's a",
            "I'll create",
            "Here's the implementation",
            "Here's a simple",
            "Here's the code",
            "Here's an implementation",
        ]

        content_lower = content.lower()
        for indicator in cursor_indicators:
            if indicator.lower() in content_lower:
                return True

        return False

    def add_callback(self, callback: Callable[[str], None]):
        """Add callback to be called when Cursor output is detected."""
        self.callbacks.append(callback)
        logger.info(f"Added clipboard callback (total: {len(self.callbacks)})")

    def remove_callback(self, callback: Callable[[str], None]):
        """Remove a callback."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            logger.info(f"Removed clipboard callback (total: {len(self.callbacks)})")

    def _monitor_loop(self):
        """Main monitoring loop."""
        logger.info("Starting clipboard monitoring...")

        while self.is_monitoring:
            try:
                current_content = self.read_clipboard()

                # Check if content changed and looks like Cursor output
                if current_content != self.last_content and self.is_cursor_output(current_content):
                    logger.info(f"Cursor output detected! ({len(current_content)} chars)")

                    # Notify all callbacks
                    for callback in self.callbacks:
                        try:
                            callback(current_content)
                        except Exception as e:
                            logger.error(f"Callback error: {e}")

                    self.last_content = current_content

                time.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(self.check_interval)

        logger.info("Clipboard monitoring stopped")

    def start_monitoring(self):
        """Start clipboard monitoring in background thread."""
        if self.is_monitoring:
            logger.warning("Already monitoring clipboard")
            return

        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Clipboard monitoring started")

    def stop_monitoring(self):
        """Stop clipboard monitoring."""
        if not self.is_monitoring:
            return

        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("Clipboard monitoring stopped")

    def wait_for_cursor_output(self, timeout: float = 60.0) -> str | None:
        """
        Wait for Cursor output with timeout.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            Clipboard content if Cursor output detected, None if timeout
        """
        logger.info(f"Waiting for Cursor output (timeout: {timeout}s)...")

        start_time = time.time()
        while time.time() - start_time < timeout:
            current_content = self.read_clipboard()

            if self.is_cursor_output(current_content):
                logger.info(f"Cursor output detected after {time.time() - start_time:.1f}s")
                return current_content

            time.sleep(self.check_interval)

        logger.warning(f"Timeout waiting for Cursor output ({timeout}s)")
        return None


# Global monitor instance
_global_monitor: ClipboardMonitor | None = None


def get_global_monitor() -> ClipboardMonitor:
    """Get or create global clipboard monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = ClipboardMonitor()
    return _global_monitor


def start_global_monitoring():
    """Start global clipboard monitoring."""
    monitor = get_global_monitor()
    monitor.start_monitoring()


def stop_global_monitoring():
    """Stop global clipboard monitoring."""
    global _global_monitor
    if _global_monitor:
        _global_monitor.stop_monitoring()


if __name__ == "__main__":
    # Test the clipboard monitor
    import logging

    logging.basicConfig(level=logging.INFO)

    monitor = ClipboardMonitor()

    def on_cursor_output(content: str):
        print("🎯 Cursor output detected!")
        print(f"   Length: {len(content)} chars")
        print(f"   Preview: {content[:100]}...")

    monitor.add_callback(on_cursor_output)
    monitor.start_monitoring()

    print("🔍 Monitoring clipboard for Cursor output...")
    print("   Copy some code to clipboard to test")
    print("   Press Ctrl+C to stop")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping monitor...")
        monitor.stop_monitoring()
