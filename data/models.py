"""Provider-neutral market-data value objects."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass(frozen=True, slots=True)
class Timeframe:
    amount: int
    unit: str

    def __post_init__(self) -> None:
        normalized = self.unit.lower().rstrip("s")
        if self.amount <= 0 or normalized not in {"minute", "hour", "day", "week", "month"}:
            raise ValueError("Timeframe must use a positive minute, hour, day, week, or month")
        object.__setattr__(self, "unit", normalized)

    @property
    def duration(self) -> timedelta | None:
        if self.unit == "minute":
            return timedelta(minutes=self.amount)
        if self.unit == "hour":
            return timedelta(hours=self.amount)
        if self.unit == "day":
            return timedelta(days=self.amount)
        return None


@dataclass(frozen=True, slots=True)
class Candle:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float | None

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("Candle timestamp must be timezone-aware")
        if not self.symbol or not self.symbol.strip():
            raise ValueError("Candle symbol cannot be empty")
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        object.__setattr__(self, "timestamp", self.timestamp.astimezone(timezone.utc))


@dataclass(frozen=True, slots=True)
class LatestPrice:
    symbol: str
    price: float
    timestamp: datetime

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("Latest price timestamp must be timezone-aware")
        if not self.symbol or not self.symbol.strip():
            raise ValueError("Latest price symbol cannot be empty")
        if not math.isfinite(self.price) or self.price <= 0:
            raise ValueError("Latest price must be positive and finite")
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        object.__setattr__(self, "timestamp", self.timestamp.astimezone(timezone.utc))
