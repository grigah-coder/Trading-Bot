"""Signal quality, age, replay, and exchange-session validation."""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

from core.market_calendar import ExchangeMarketCalendar, MarketCalendar
from signals.models import Direction, Signal


class SignalValidationError(ValueError):
    """A signal cannot be safely consumed by downstream risk controls."""


class SignalValidator:
    def __init__(self, *, max_age: timedelta = timedelta(minutes=2),
                 calendar: MarketCalendar | None = None,
                 allowed_session: Callable[[datetime], bool] | None = None) -> None:
        if max_age.total_seconds() <= 0:
            raise ValueError("max_age must be positive")
        self.max_age = max_age
        self._calendar = calendar or (None if allowed_session else ExchangeMarketCalendar())
        self._allowed_session = allowed_session
        self._seen: set[tuple[str, str, str, datetime]] = set()

    def validate(self, signal: Signal, *, now: datetime | None = None) -> bool:
        decision_time = _utc(now or datetime.now(timezone.utc), "decision time")
        timestamp = _utc(signal.timestamp, "signal timestamp")
        if (not isinstance(signal.symbol, str) or not signal.symbol.strip()
                or not isinstance(signal.strategy_id, str) or not signal.strategy_id.strip()):
            raise SignalValidationError("Signal symbol and strategy identifier are required")
        if not isinstance(signal.reason, str) or not signal.reason.strip():
            raise SignalValidationError("Signal reason is required")
        if not isinstance(signal.direction, Direction):
            raise SignalValidationError("Signal direction is invalid")
        for name, price in (("entry", signal.entry_reference),
                            ("stop", signal.stop_loss_reference),
                            ("target", signal.take_profit_reference)):
            if price is not None and not _positive_finite(price):
                raise SignalValidationError(f"Signal {name} price must be positive and finite")
        if signal.direction is not Direction.EXIT:
            if signal.stop_loss_reference is None or signal.take_profit_reference is None:
                raise SignalValidationError("Entry signal requires stop and target references")
            if signal.direction is Direction.LONG and not (
                signal.stop_loss_reference < signal.entry_reference < signal.take_profit_reference
            ):
                raise SignalValidationError("Long signal stop and target are on invalid sides of entry")
            if signal.direction is Direction.SHORT and not (
                signal.take_profit_reference < signal.entry_reference < signal.stop_loss_reference
            ):
                raise SignalValidationError("Short signal stop and target are on invalid sides of entry")
        if signal.strength is not None and not _finite_between(signal.strength, 0, 1):
            raise SignalValidationError("Signal strength must be between zero and one")
        age = decision_time - timestamp
        if age < timedelta(0) or age > self.max_age:
            raise SignalValidationError("Signal is expired or from the future")
        if self._allowed_session is not None:
            allowed = self._allowed_session(timestamp)
        else:
            session = self._calendar.session_for(timestamp.date()) if self._calendar else None
            allowed = bool(session and session.opens_at <= timestamp <= session.closes_at)
        if not allowed:
            raise SignalValidationError("Signal is outside the allowed trading session")
        identity = signal.identity
        if identity in self._seen:
            raise SignalValidationError("Duplicate signal")
        self._seen.add(identity)
        return True

    def health_check(self) -> None:
        return None

    def shutdown(self) -> None:
        self._seen.clear()


def _utc(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise SignalValidationError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _positive_finite(value: object) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(value) and value > 0  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def _finite_between(value: object, lower: float, upper: float) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(value) and lower <= value <= upper  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
