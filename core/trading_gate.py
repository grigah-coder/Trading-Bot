"""Process-wide trading safety gate.

The gate is a small, thread-safe latch.  Recovery services block it when the
broker and persisted state cannot be reconciled; order executors and the engine
check it immediately before any order is submitted.

Blocking is reason-scoped: a reconciliation failure cannot be accidentally
cleared by an unrelated component, and clearing one reason leaves other active
blocks in place.
"""

from __future__ import annotations

import threading
from types import MappingProxyType
from typing import Mapping

from core.exceptions import TradingBlocked


class TradingGate:
    """Thread-safe collection of active trading blockers."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._reasons: dict[str, str] = {}

    @property
    def blocked(self) -> bool:
        with self._lock:
            return bool(self._reasons)

    @property
    def reasons(self) -> Mapping[str, str]:
        with self._lock:
            return MappingProxyType(dict(self._reasons))

    def block(self, reason: str, detail: str = "") -> None:
        key = _reason(reason)
        with self._lock:
            self._reasons[key] = str(detail or "")

    def unblock(self, reason: str) -> None:
        key = _reason(reason)
        with self._lock:
            self._reasons.pop(key, None)

    def clear(self) -> None:
        """Explicit operator override; never called during normal recovery."""
        with self._lock:
            self._reasons.clear()

    def require_open(self) -> None:
        with self._lock:
            if not self._reasons:
                return
            reasons = dict(self._reasons)
        first_reason, detail = next(iter(reasons.items()))
        raise TradingBlocked(
            "Trading is blocked",
            context={
                "reason": first_reason,
                "detail": detail,
                "reasons": tuple(reasons),
            },
        )


def _reason(value: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        raise ValueError("block reason cannot be empty")
    return text
