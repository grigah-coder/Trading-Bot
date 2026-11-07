"""Cached historical retrieval with decision-time availability boundaries."""

from __future__ import annotations

import calendar as month_calendar
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from types import MappingProxyType

from core.market_calendar import ExchangeMarketCalendar, MarketCalendar
from data.cache import TTLCache
from data.models import Candle, Timeframe
from data.provider import MarketDataProvider
from data.validation import CandleValidator, require_aware


class HistoricalDataService:
    def __init__(self, provider: MarketDataProvider, validator: CandleValidator,
                 *, cache_ttl_seconds: float = 30, calendar: MarketCalendar | None = None) -> None:
        if cache_ttl_seconds <= 0:
            raise ValueError("cache_ttl_seconds must be positive")
        self.provider = provider
        self.calendar = calendar or validator.calendar or ExchangeMarketCalendar()
        self.validator = validator if validator.calendar is not None else CandleValidator(self.calendar)
        self.cache: TTLCache = TTLCache(cache_ttl_seconds)

    def invalidate(self) -> None:
        self.cache.clear()

    def get_history(self, symbols: Sequence[str], timeframe: Timeframe,
                    start: datetime, decision_at: datetime) -> Mapping[str, tuple[Candle, ...]]:
        start_utc = require_aware(start, "start")
        decision_utc = require_aware(decision_at, "decision_at")
        if start_utc >= decision_utc:
            raise ValueError("start must be before decision_at")
        normalized = tuple(sorted({symbol.strip().upper() for symbol in symbols if symbol.strip()}))
        if not normalized:
            raise ValueError("at least one symbol is required")
        key = (normalized, timeframe, start_utc, decision_utc)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        raw = self.provider.fetch_bars(normalized, timeframe, start_utc, decision_utc)
        output: dict[str, tuple[Candle, ...]] = {}
        for symbol in normalized:
            bars = self.validator.validate(raw.get(symbol, ()), timeframe=timeframe)
            eligible = tuple(bar for bar in bars if bar.symbol == symbol and self._available_at(bar, timeframe) <= decision_utc)
            output[symbol] = eligible
        result = MappingProxyType(dict(output))
        self.cache.set(key, result)
        return result

    def _available_at(self, candle: Candle, timeframe: Timeframe) -> datetime:
        timestamp = candle.timestamp
        duration = timeframe.duration
        if timeframe.unit == "day":
            session = self.calendar.session_for(timestamp.date())
            return session.closes_at if session else timestamp + (duration or timedelta(days=1))
        if duration is not None:
            return timestamp + duration
        if timeframe.unit == "week":
            return timestamp + timedelta(days=7 * timeframe.amount)
        if timeframe.unit == "month":
            year, month = timestamp.year, timestamp.month
            month_index = year * 12 + month - 1 + timeframe.amount
            end_year, end_month = divmod(month_index, 12)
            end_month += 1
            last_day = month_calendar.monthrange(end_year, end_month)[1]
            return timestamp.replace(year=end_year, month=end_month, day=last_day, hour=23, minute=59, second=59, microsecond=999999)
        session = self.calendar.session_for(timestamp.date())
        return session.closes_at if session else timestamp + timedelta(days=1)
