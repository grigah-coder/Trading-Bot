"""Scheduler.

Drives the trading loop cadence and timed tasks. The scheduler is
market-session aware: it can pause iteration during closed markets and the
opening volatility blackout, but never pauses position protection.

Implemented in Phase 2 (Core Architecture).
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Callable, Optional

from trading_bot.logging_config import configure_logging
from trading_bot.state import AppState

logger = configure_logging(__name__).getChild("scheduler")


class Scheduler:
    """Simple, dependency-injected scheduler for the trading loop.

    Responsibilities:
    * Run a callback on a fixed cadence while the application is running.
    * Track the last successful iteration and heartbeat.
    * Support graceful stop (the callback checks ``state.is_running()``).
    """

    def __init__(
        self,
        state: AppState,
        interval_seconds: float = 60.0,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.state = state
        self.interval_seconds = max(0.05, float(interval_seconds))
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self.last_iteration: Optional[datetime] = None
        self.iteration_count = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self, callback: Callable[[], None]) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("scheduler is already running")
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run_loop,
                args=(callback,),
                name="trading-scheduler",
                daemon=True,
            )
            self._thread.start()
            logger.info(
                "Scheduler started (interval=%.1fs)", self.interval_seconds
            )

    def stop(self, timeout: float = 10.0) -> None:
        with self._lock:
            self._stop_event.set()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning("Scheduler thread did not stop within %.1fs", timeout)
            self._thread = None
        logger.info("Scheduler stopped")

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop_event.is_set()

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------
    def _run_loop(self, callback: Callable[[], None]) -> None:
        while not self._stop_event.is_set():
            if not self.state.is_running():
                # Application not running; wait and poll.
                self._stop_event.wait(self.interval_seconds)
                continue
            start = time.monotonic()
            try:
                callback()
                self.last_iteration = self._clock()
                self.iteration_count += 1
                self.state.touch_heartbeat()
            except Exception as exc:  # noqa: BLE001 - keep loop alive
                logger.error("Trading iteration failed: %s", exc, exc_info=True)
                self.state.increment("errors") if hasattr(self.state, "increment") else None
            elapsed = time.monotonic() - start
            remaining = max(0.0, self.interval_seconds - elapsed)
            self._stop_event.wait(remaining)
