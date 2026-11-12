"""Application lifecycle.

Coordinates start-up sequencing, graceful shutdown, signal handling and crash
recovery. Components are started in dependency order and stopped in reverse
order.

Implemented in Phase 2 (Core Architecture).
"""

from __future__ import annotations

import signal
import threading
from typing import Callable, Optional

from trading_bot.logging_config import get_logger
from trading_bot.state import AppState

logger = get_logger(__name__)


class Lifecycle:
    """Manages application start-up, shutdown and recovery.

    The lifecycle owns the application :class:`AppState` and coordinates
    registered start/stop hooks. Signal handlers (SIGINT/SIGTERM) trigger a
    graceful stop.
    """

    def __init__(self, state: AppState) -> None:
        self.state = state
        self._start_hooks: list[Callable[[], None]] = []
        self._stop_hooks: list[Callable[[], None]] = []
        self._lock = threading.RLock()
        self._installed_signal_handlers = False
        self._original_handlers: dict[int, Optional[Callable]] = {}

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def on_start(self, hook: Callable[[], None]) -> None:
        with self._lock:
            self._start_hooks.append(hook)

    def on_stop(self, hook: Callable[[], None]) -> None:
        with self._lock:
            # Stop hooks run in reverse registration order later.
            self._stop_hooks.insert(0, hook)

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------
    def install_signal_handlers(self) -> None:
        """Install SIGINT/SIGTERM handlers that request a graceful stop."""
        if self._installed_signal_handlers:
            return

        def _handle(signum, frame):  # noqa: ARG001
            logger.warning(
                "Received signal %s; requesting graceful shutdown", signum
            )
            self.request_stop()

        try:
            self._original_handlers[signal.SIGINT] = signal.signal(
                signal.SIGINT, _handle
            )
            self._original_handlers[signal.SIGTERM] = signal.signal(
                signal.SIGTERM, _handle
            )
            self._installed_signal_handlers = True
            logger.debug("Signal handlers installed")
        except (ValueError, OSError) as exc:
            # Signal handlers can only be installed in the main thread.
            logger.debug("Could not install signal handlers: %s", exc)

    def restore_signal_handlers(self) -> None:
        for signum, handler in self._original_handlers.items():
            if handler is not None:
                try:
                    signal.signal(signum, handler)
                except (ValueError, OSError):
                    pass
        self._original_handlers.clear()
        self._installed_signal_handlers = False

    # ------------------------------------------------------------------
    # Start / stop
    # ------------------------------------------------------------------
    def request_stop(self) -> None:
        self.state.stop()

    def start(self, session_id: str) -> None:
        """Start the application: run start hooks, then mark state running."""
        with self._lock:
            if self.state.is_running():
                raise RuntimeError("application is already running")
            self.state.start(session_id)
            logger.info("Application starting (session=%s)", session_id)
            exc: Optional[BaseException] = None
            for hook in self._start_hooks:
                try:
                    hook()
                except Exception as e:  # noqa: BLE001
                    exc = e
                    logger.error("Start hook failed: %s", e, exc_info=True)
                    break
            if exc is not None:
                self.state.stop()
                raise RuntimeError("application start failed") from exc
            logger.info("Application started")

    def stop(self) -> None:
        """Stop the application: run stop hooks (reverse order), then mark stopped."""
        with self._lock:
            if not self.state.is_running() and self.state.stopped_at is not None:
                # Already stopped.
                return
            logger.info("Application stopping")
            errors = []
            for hook in self._stop_hooks:
                try:
                    hook()
                except Exception as e:  # noqa: BLE001
                    errors.append(str(e))
                    logger.error("Stop hook failed: %s", e, exc_info=True)
            self.state.stop()
            if errors:
                raise RuntimeError(
                    f"application stop completed with {len(errors)} error(s)"
                )
            logger.info("Application stopped")

    def is_running(self) -> bool:
        return self.state.is_running()
