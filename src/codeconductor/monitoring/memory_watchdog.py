"""
Memory Watchdog for CodeConductor

Monitors VRAM usage and automatically triggers cleanup when memory usage gets too high.
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class MemoryThresholds:
    """Memory thresholds for different alert levels."""

    warning_percent: float = 75.0
    critical_percent: float = 85.0
    emergency_percent: float = 95.0


class MemoryWatchdog:
    """
    Memory watchdog that monitors VRAM usage and triggers cleanup.

    This watchdog runs in the background and monitors GPU memory usage.
    When thresholds are exceeded, it automatically triggers cleanup actions.
    """

    def __init__(
        self,
        model_manager,
        check_interval: float = 30.0,  # Check every 30 seconds
        thresholds: MemoryThresholds | None = None,
        cleanup_callback: Callable | None = None,
    ):
        self.model_manager = model_manager
        self.check_interval = check_interval
        self.thresholds = thresholds or MemoryThresholds()
        self.cleanup_callback = cleanup_callback

        # State tracking
        self.is_running = False
        self.last_check = None
        self.last_vram_percent = 0.0
        self.consecutive_high_usage = 0
        self.max_consecutive_high_usage = (
            3  # Trigger cleanup after 3 consecutive high readings
        )

        # Statistics
        self.total_checks = 0
        self.cleanup_triggers = 0
        self.emergency_triggers = 0

    async def start(self):
        """Start the memory watchdog."""
        if self.is_running:
            logger.warning("⚠️ Memory watchdog is already running")
            return

        logger.info("🔄 Starting memory watchdog...")
        self.is_running = True

        try:
            while self.is_running:
                await self._check_memory_usage()
                await asyncio.sleep(self.check_interval)
        except asyncio.CancelledError:
            logger.info("🔄 Memory watchdog cancelled")
        except Exception as e:
            logger.error(f"❌ Memory watchdog error: {e}")
        finally:
            self.is_running = False
            logger.info("🔄 Memory watchdog stopped")

    async def stop(self):
        """Stop the memory watchdog."""
        logger.info("🔄 Stopping memory watchdog...")
        self.is_running = False

    async def _check_memory_usage(self):
        """Check current memory usage and trigger cleanup if needed."""
        try:
            self.total_checks += 1

            # Get current VRAM usage
            gpu_info = await self.model_manager.get_gpu_memory_info()
            if not gpu_info:
                logger.warning("⚠️ Could not get GPU memory info")
                return

            current_vram_percent = gpu_info["usage_percent"]
            self.last_vram_percent = current_vram_percent
            self.last_check = datetime.now()

            logger.debug(f"🎮 VRAM usage: {current_vram_percent:.1f}%")

            # Check thresholds and trigger cleanup if needed
            if current_vram_percent > self.thresholds.emergency_percent:
                await self._handle_emergency_cleanup(current_vram_percent)
            elif current_vram_percent > self.thresholds.critical_percent:
                await self._handle_critical_cleanup(current_vram_percent)
            elif current_vram_percent > self.thresholds.warning_percent:
                await self._handle_warning_cleanup(current_vram_percent)
            else:
                # Reset consecutive counter if usage is normal
                if self.consecutive_high_usage > 0:
                    logger.info(
                        f"✅ VRAM usage normalized: {current_vram_percent:.1f}%"
                    )
                self.consecutive_high_usage = 0

        except Exception as e:
            logger.error(f"❌ Memory check failed: {e}")

    async def _handle_emergency_cleanup(self, vram_percent: float):
        """Handle emergency memory cleanup."""
        self.emergency_triggers += 1
        logger.warning(
            f"🚨 EMERGENCY: VRAM usage {vram_percent:.1f}% > {self.thresholds.emergency_percent}%"
        )

        try:
            # Emergency unload all models
            unloaded_count = await self.model_manager.emergency_unload_all()
            logger.info(
                f"🚨 Emergency cleanup completed: unloaded {unloaded_count} models"
            )

            # Call custom cleanup callback if provided
            if self.cleanup_callback:
                await self.cleanup_callback("emergency", vram_percent)

        except Exception as e:
            logger.error(f"❌ Emergency cleanup failed: {e}")

    async def _handle_critical_cleanup(self, vram_percent: float):
        """Handle critical memory cleanup."""
        self.consecutive_high_usage += 1
        logger.warning(
            f"⚠️ CRITICAL: VRAM usage {vram_percent:.1f}% > {self.thresholds.critical_percent}%"
        )

        # Trigger cleanup if we've had consecutive high usage
        if self.consecutive_high_usage >= self.max_consecutive_high_usage:
            self.cleanup_triggers += 1
            logger.warning(
                f"🧹 Triggering critical cleanup after {self.consecutive_high_usage} consecutive high readings"
            )

            try:
                # Smart cleanup to target 60% VRAM usage
                unloaded_count = await self.model_manager.smart_memory_cleanup(60.0)
                logger.info(
                    f"🧹 Critical cleanup completed: unloaded {unloaded_count} models"
                )

                # Call custom cleanup callback if provided
                if self.cleanup_callback:
                    await self.cleanup_callback("critical", vram_percent)

            except Exception as e:
                logger.error(f"❌ Critical cleanup failed: {e}")

    async def _handle_warning_cleanup(self, vram_percent: float):
        """Handle warning level memory monitoring."""
        self.consecutive_high_usage += 1
        logger.info(
            f"⚠️ WARNING: VRAM usage {vram_percent:.1f}% > {self.thresholds.warning_percent}%"
        )

        # Only trigger cleanup if we've had consecutive high usage
        if self.consecutive_high_usage >= self.max_consecutive_high_usage:
            self.cleanup_triggers += 1
            logger.info(
                f"🧹 Triggering warning cleanup after {self.consecutive_high_usage} consecutive high readings"
            )

            try:
                # Smart cleanup to target 50% VRAM usage
                unloaded_count = await self.model_manager.smart_memory_cleanup(50.0)
                logger.info(
                    f"🧹 Warning cleanup completed: unloaded {unloaded_count} models"
                )

                # Call custom cleanup callback if provided
                if self.cleanup_callback:
                    await self.cleanup_callback("warning", vram_percent)

            except Exception as e:
                logger.error(f"❌ Warning cleanup failed: {e}")

    def get_stats(self) -> dict:
        """Get watchdog statistics."""
        return {
            "is_running": self.is_running,
            "total_checks": self.total_checks,
            "cleanup_triggers": self.cleanup_triggers,
            "emergency_triggers": self.emergency_triggers,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "last_vram_percent": self.last_vram_percent,
            "consecutive_high_usage": self.consecutive_high_usage,
            "check_interval": self.check_interval,
            "thresholds": {
                "warning": self.thresholds.warning_percent,
                "critical": self.thresholds.critical_percent,
                "emergency": self.thresholds.emergency_percent,
            },
        }


# Global watchdog instance
_watchdog_instance: MemoryWatchdog | None = None


async def start_memory_watchdog(model_manager, check_interval: float = 30.0):
    """Start the global memory watchdog."""
    global _watchdog_instance

    if _watchdog_instance and _watchdog_instance.is_running:
        logger.warning("⚠️ Memory watchdog is already running")
        return _watchdog_instance

    _watchdog_instance = MemoryWatchdog(
        model_manager=model_manager,
        check_interval=check_interval,
    )

    # Start watchdog in background
    asyncio.create_task(_watchdog_instance.start())

    logger.info(f"🔄 Memory watchdog started (check interval: {check_interval}s)")
    return _watchdog_instance


async def stop_memory_watchdog():
    """Stop the global memory watchdog."""
    global _watchdog_instance

    if _watchdog_instance:
        await _watchdog_instance.stop()
        _watchdog_instance = None
        logger.info("🔄 Memory watchdog stopped")


def get_memory_watchdog() -> MemoryWatchdog | None:
    """Get the global memory watchdog instance."""
    return _watchdog_instance
