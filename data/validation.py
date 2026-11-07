"""Market-data quality and freshness checks."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Iterable

from core.exceptions import TradingBotError
from core.market_calendar import MarketCalendar
from data.models import Candle, LatestPrice, Timeframe


class DataQualityError(TradingBotError):
    code = "invalid_market_data"


class DataFreshnessError(TradingBotError):
    code = "stale_market_data"


def require_aware(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


class CandleValidator:
    def __init__(self, calendar: MarketCalendar | None = None) -> None:
        self.calendar = calendar

    def validate(self, candles: Iterable[Candle], *, timeframe: Timeframe | None = None) -> tuple[Candle, ...]:
        bars = tuple(candles)
        seen: set[tuple[str, datetime]] = set()
        previous: dict[str, datetime] = {}
        for bar in bars:
            values = (bar.open, bar.high, bar.low, bar.close)
            if any(not math.isfinite(value) or value <= 0 for value in values):
                raise DataQualityError("Candle prices must be positive and finite", context={"symbol": bar.symbol})
            if bar.high < max(bar.open, bar.close, bar.low) or bar.low > min(bar.open, bar.close, bar.high):
                raise DataQualityError("Invalid OHLC relationship", context={"symbol": bar.symbol})
            if bar.volume is None or not math.isfinite(bar.volume) or bar.volume < 0:
                raise DataQualityError("Candle volume must be present, finite, and non-negative", context={"symbol": bar.symbol})
            key = (bar.symbol, bar.timestamp)
            if key in seen:
                raise DataQualityError("Duplicate candle", context={"symbol": bar.symbol, "timestamp": bar.timestamp.isoformat()})
            seen.add(key)
            if bar.symbol in previous and bar.timestamp < previous[bar.symbol]:
                raise DataQualityError("Candles are out of order", context={"symbol": bar.symbol})
            previous[bar.symbol] = bar.timestamp
        if timeframe is not None and self.calendar is not None and timeframe.duration is not None:
            self._validate_gaps(bars, timeframe.duration)
        return bars

    def _validate_gaps(self, bars: tuple[Candle, ...], duration: timedelta) -> None:
        by_symbol: dict[str, list[datetime]] = {}
        for bar in bars:
            by_symbol.setdefault(bar.symbol, []).append(bar.timestamp)
        for symbol, timestamps in by_symbol.items():
            ordered = sorted(timestamps)
            for left, right in zip(ordered, ordered[1:]):
                cursor = left + duration
                while cursor < right:
                    session = self.calendar.session_for(cursor.date()) if self.calendar else None
                    if session and session.opens_at <= cursor < session.closes_at:
                        raise DataQualityError("Missing candle", context={"symbol": symbol, "timestamp": cursor.isoformat()})
                    cursor += duration

    def validate_latest(self, price: LatestPrice, *, now: datetime, max_age: timedelta) -> None:
        now_utc = require_aware(now, "now")
        timestamp = require_aware(price.timestamp, "price timestamp")
        if max_age.total_seconds() < 0:
            raise ValueError("max_age must be non-negative")
        age = now_utc - timestamp
        if age < timedelta(0) or age > max_age:
            raise DataFreshnessError("Latest price is stale or from the future", context={"symbol": price.symbol, "age_seconds": age.total_seconds()})
