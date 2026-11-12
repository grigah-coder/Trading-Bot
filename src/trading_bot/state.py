"""State management.

Holds mutable application state behind a controlled, thread-safe interface.
Avoids global mutable state by requiring components to receive an
:class:`AppState` instance through dependency injection.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class AppState:
    """Thread-safe application state container.

    Attributes are grouped by concern so the trading engine, risk engine,
    order manager and monitoring can each read/write only what they own.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # Lifecycle
        self.running = False
        self.started_at: Optional[datetime] = None
        self.stopped_at: Optional[datetime] = None
        self.session_id: Optional[str] = None
        # Component references (set during initialization)
        self.data_provider: Optional[Any] = None
        self.broker_adapter: Optional[Any] = None
        self.risk_engine: Optional[Any] = None
        self.strategy: Optional[Any] = None
        self.order_manager: Optional[Any] = None
        self.database: Optional[Any] = None
        self.monitoring: Optional[Any] = None
        # Risk/runtime snapshot
        self.circuit_breaker_tripped = False
        self.circuit_breaker_tripped_at: Optional[datetime] = None
        self.data_stale = False
        self.last_heartbeat: Optional[datetime] = None
        # Counters
        self.signals_generated = 0
        self.signals_approved = 0
        self.signals_rejected = 0
        self.orders_submitted = 0
        self.orders_filled = 0
        self.orders_rejected = 0
        self.trades_executed = 0
        self.errors: list[str] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self, session_id: str) -> None:
        with self._lock:
            if self.running:
                raise RuntimeError("application is already running")
            self.running = True
            self.started_at = datetime.now(timezone.utc)
            self.stopped_at = None
            self.session_id = session_id

    def stop(self) -> None:
        with self._lock:
            if not self.running:
                return
            self.running = False
            self.stopped_at = datetime.now(timezone.utc)

    def is_running(self) -> bool:
        with self._lock:
            return self.running

    # ------------------------------------------------------------------
    # Risk state
    # ------------------------------------------------------------------
    def trip_circuit_breaker(self, when: Optional[datetime] = None) -> None:
        with self._lock:
            self.circuit_breaker_tripped = True
            self.circuit_breaker_tripped_at = when or datetime.now(timezone.utc)

    def reset_circuit_breaker(self) -> None:
        with self._lock:
            self.circuit_breaker_tripped = False
            self.circuit_breaker_tripped_at = None

    def is_circuit_breaker_tripped(self) -> bool:
        with self._lock:
            return self.circuit_breaker_tripped

    def set_data_stale(self, stale: bool) -> None:
        with self._lock:
            self.data_stale = stale

    def is_data_stale(self) -> bool:
        with self._lock:
            return self.data_stale

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------
    def touch_heartbeat(self) -> None:
        with self._lock:
            self.last_heartbeat = datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Counters
    # ------------------------------------------------------------------
    def increment(self, counter: str, amount: int = 1) -> None:
        with self._lock:
            current = getattr(self, counter, None)
            if not isinstance(current, int):
                raise AttributeError(f"unknown counter: {counter}")
            setattr(self, counter, current + amount)

    def record_error(self, message: str) -> None:
        """Append ``message`` to the shared error log (bounded)."""
        text = str(message or "").strip()
        if not text:
            return
        with self._lock:
            self.errors.append(text)
            # Keep memory bounded on long-running sessions.
            if len(self.errors) > 1000:
                del self.errors[:-1000]

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "session_id": self.session_id,
                "running": self.running,
                "started_at": (
                    self.started_at.isoformat() if self.started_at else None
                ),
                "stopped_at": (
                    self.stopped_at.isoformat() if self.stopped_at else None
                ),
                "circuit_breaker_tripped": self.circuit_breaker_tripped,
                "data_stale": self.data_stale,
                "signals_generated": self.signals_generated,
                "signals_approved": self.signals_approved,
                "signals_rejected": self.signals_rejected,
                "orders_submitted": self.orders_submitted,
                "orders_filled": self.orders_filled,
                "orders_rejected": self.orders_rejected,
                "trades_executed": self.trades_executed,
                "error_count": len(self.errors),
            }
