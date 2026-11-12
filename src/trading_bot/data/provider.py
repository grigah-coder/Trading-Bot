"""Alpaca market data provider.

Implements the :class:`DataProvider` protocol using alpaca-py's
``StockHistoricalDataClient`` against the IEX feed (default) — safe for
paper/free accounts that cannot query recent SIP data.

Responsibilities:
* historical bar retrieval (OHLCV) with validation,
* latest bar / trade / quote retrieval,
* data freshness checks,
* timezone normalization (UTC internally, market-local for sessions),
* market session detection,
* in-memory caching (TTL),
* retry / rate-limit handling via :func:`trading_bot.retry.call_with_retry`.

Look-ahead bias prevention:
* bars are never returned with timestamps after the provider clock,
* ``get_latest_bar`` returns the most recent *completed* bar only,
* historical requests clamp ``end`` to "now minus one minute" so a forming
  (incomplete) bar is never mistaken for a closed bar.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

from alpaca.data.enums import DataFeed  # pyright: ignore[reportMissingImports]
from alpaca.data.historical import StockHistoricalDataClient  # pyright: ignore[reportMissingImports]
from alpaca.data.requests import (  # pyright: ignore[reportMissingImports]
    StockBarsRequest,
    StockLatestQuoteRequest,
    StockLatestTradeRequest,
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit  # pyright: ignore[reportMissingImports]

from trading_bot.cache import TTLCache
from trading_bot.config.settings import Settings
from trading_bot.data.market_calendar import MarketCalendar
from trading_bot.data.validation import Bar, DataValidationError, validate_bar, validate_bars
from trading_bot.exceptions import DataStaleError, MarketDataError
from trading_bot.logging_config import get_logger
from trading_bot.retry import call_with_retry, is_retryable

logger = get_logger(__name__)

# Alpaca does not publish the most recent ~1 minute of data reliably; clamp
# historical requests to this many minutes before now so we never hand out a
# forming bar as closed.
RECENT_DATA_LAG_MINUTES = 1

# Timezone used for all market-local session logic.
MARKET_TZ = ZoneInfo("America/New_York")

_TIMEFRAME_MAP: dict[str, tuple[int, TimeFrameUnit]] = {
    "1 minute": (1, TimeFrameUnit.Minute),
    "5 minutes": (5, TimeFrameUnit.Minute),
    "15 minutes": (15, TimeFrameUnit.Minute),
    "30 minutes": (30, TimeFrameUnit.Minute),
    "1 hour": (1, TimeFrameUnit.Hour),
    "1 day": (1, TimeFrameUnit.Day),
}


def _parse_timeframe(value: str) -> TimeFrame:
    key = value.strip().lower()
    if key not in _TIMEFRAME_MAP:
        raise MarketDataError(f"unsupported timeframe {value!r}; use one of {list(_TIMEFRAME_MAP)}")
    amount, unit = _TIMEFRAME_MAP[key]
    return TimeFrame(amount, unit)


def _resolve_feed(value: str) -> DataFeed:
    try:
        return DataFeed(value.strip().lower())
    except ValueError:
        logger.warning("Unknown data feed %r; falling back to IEX", value)
        return DataFeed.IEX


class AlpacaDataProvider:
    """Alpaca-backed market data provider."""

    def __init__(self, settings: Settings, clock=None):
        self.settings = settings
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._calendar = MarketCalendar(
            market_timezone=settings.market_timezone,
            market_open_time=settings.market_open_time,
            market_close_time="16:00",
            opening_blackout_minutes=settings.opening_blackout_minutes,
        )
        self._feed = _resolve_feed(settings.alpaca_data_feed)
        self._client = StockHistoricalDataClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_api_secret,
        )
        # Caches (TTL from settings).
        self._bars_cache = TTLCache(ttl=settings.indicator_cache_ttl, maxsize=settings.cache_maxsize)
        self._latest_cache = TTLCache(ttl=max(1.0, settings.indicator_cache_ttl / 10.0), maxsize=settings.cache_maxsize)
        self._freshness_cache = TTLCache(ttl=30.0, maxsize=settings.cache_maxsize)

    # ------------------------------------------------------------------
    # DataProvider protocol
    # ------------------------------------------------------------------
    def get_historical_bars(
        self, symbol: str, limit: int, timeframe: str
    ) -> Optional[list[dict[str, Any]]]:
        """Return the last ``limit`` validated bars for ``symbol``.

        Bars are sorted oldest-first. The ``end`` is clamped to now minus
        ``RECENT_DATA_LAG_MINUTES`` so a forming bar is never returned as
        complete (look-ahead bias prevention).
        """
        symbol = str(symbol).upper()
        cache_key = f"bars:{symbol}:{limit}:{timeframe}"
        cached = self._bars_cache.get(cache_key)
        if cached is not None:
            return cached

        tf = _parse_timeframe(timeframe)
        end = self._clock() - timedelta(minutes=RECENT_DATA_LAG_MINUTES)
        # Request a wider window than strictly needed; Alpaca may return fewer
        # bars around weekends/holidays.
        start = end - timedelta(minutes=limit * _minutes_per_bar(tf) * 2 + 10_080)

        def _fetch() -> list[dict[str, Any]]:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=start,
                end=end,
                feed=self._feed,
                limit=limit,
            )
            try:
                barset = self._client.get_stock_bars(request)
            except Exception as exc:  # noqa: BLE001
                if is_retryable(exc):
                    raise
                raise MarketDataError(f"non-retryable Alpaca error for {symbol}: {exc}") from exc
            return self._normalize_barset(barset, symbol, timeframe)

        try:
            bars = call_with_retry(
                _fetch,
                attempts=self.settings.retry_attempts,
                base_delay=self.settings.retry_base_delay,
                max_delay=self.settings.retry_max_delay,
                backoff_factor=self.settings.retry_backoff_factor,
                jitter=self.settings.retry_jitter,
                description=f"alpaca_bars[{symbol}]",
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Historical bars failed for %s: %s", symbol, exc)
            return None

        self._bars_cache.set(cache_key, bars)
        return bars

    def get_latest_bar(self, symbol: str) -> Optional[dict[str, Any]]:
        """Return the most recent *completed* bar for ``symbol``.

        Uses the latest trade to synthesize a 1-minute bar anchored at the
        previous minute boundary, so the bar never includes the current
        forming minute (look-ahead bias prevention).
        """
        symbol = str(symbol).upper()
        cache_key = f"latest_bar:{symbol}"
        cached = self._latest_cache.get(cache_key)
        if cached is not None:
            return cached

        trade = self._get_latest_trade(symbol)
        if trade is None:
            return None
        price = trade.get("price")
        if price is None or not math.isfinite(price) or price <= 0:
            return None
        now = self._clock()
        # Anchor to the previous completed minute so the forming minute is excluded.
        ts = now.replace(second=0, microsecond=0) - timedelta(minutes=1)
        bar = {
            "timestamp": ts.isoformat(),
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": trade.get("size", 0) or 0,
            "symbol": symbol,
            "timeframe": "1 minute",
        }
        try:
            validated = validate_bar(
                timestamp=bar["timestamp"],
                open_price=bar["open"],
                high=bar["high"],
                low=bar["low"],
                close=bar["close"],
                volume=bar["volume"],
                symbol=symbol,
                timeframe=bar["timeframe"],
                clock=self._clock,
            )
            result = validated.to_dict()
        except DataValidationError as exc:
            logger.warning("Latest bar failed validation for %s: %s", symbol, exc)
            return None
        self._latest_cache.set(cache_key, result)
        return result

    def get_latest_trade(self, symbol: str) -> Optional[dict[str, Any]]:
        return self._get_latest_trade(str(symbol).upper())

    def get_latest_quote(self, symbol: str) -> Optional[dict[str, Any]]:
        symbol = str(symbol).upper()
        try:
            quote = call_with_retry(
                lambda: self._client.get_stock_latest_quote(
                    StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=self._feed)
                ),
                attempts=self.settings.retry_attempts,
                base_delay=self.settings.retry_base_delay,
                max_delay=self.settings.retry_max_delay,
                backoff_factor=self.settings.retry_backoff_factor,
                jitter=self.settings.retry_jitter,
                description=f"alpaca_quote[{symbol}]",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Latest quote failed for %s: %s", symbol, exc)
            return None
        return self._normalize_quote(quote, symbol)

    def is_market_open(self) -> bool:
        return self._calendar.is_market_open()

    def is_in_opening_blackout(self) -> bool:
        return self._calendar.is_in_opening_blackout()

    def is_data_fresh(self, symbol: str, max_age_minutes: float = 5.0) -> bool:
        """True when the latest known bar/trade for ``symbol`` is recent enough."""
        symbol = str(symbol).upper()
        cache_key = f"fresh:{symbol}:{max_age_minutes}"
        cached = self._freshness_cache.get(cache_key)
        if cached is not None:
            return cached

        fresh = False
        try:
            latest = self.get_latest_bar(symbol)
            if latest is not None:
                bar_ts = datetime.fromisoformat(latest["timestamp"])
                if bar_ts.tzinfo is None:
                    bar_ts = bar_ts.replace(tzinfo=timezone.utc)
                age = (self._clock() - bar_ts).total_seconds() / 60.0
                fresh = age <= max_age_minutes
        except Exception as exc:  # noqa: BLE001
            logger.debug("Freshness check failed for %s: %s", symbol, exc)
            fresh = False

        self._freshness_cache.set(cache_key, fresh)
        return fresh

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _get_latest_trade(self, symbol: str) -> Optional[dict[str, Any]]:
        try:
            trade = call_with_retry(
                lambda: self._client.get_stock_latest_trade(
                    StockLatestTradeRequest(symbol_or_symbols=symbol, feed=self._feed)
                ),
                attempts=self.settings.retry_attempts,
                base_delay=self.settings.retry_base_delay,
                max_delay=self.settings.retry_max_delay,
                backoff_factor=self.settings.retry_backoff_factor,
                jitter=self.settings.retry_jitter,
                description=f"alpaca_trade[{symbol}]",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Latest trade failed for %s: %s", symbol, exc)
            return None
        return self._normalize_trade(trade, symbol)

    def _normalize_barset(
        self, barset: Any, symbol: str, timeframe: str
    ) -> list[dict[str, Any]]:
        """Convert alpaca-py barset (DataFrame or dict) to validated bar dicts."""
        raw_bars: list[dict[str, Any]] = []
        df = getattr(barset, "df", None)
        if df is not None and hasattr(df, "iterrows"):
            # alpaca-py returns a DataFrame with a (symbol, timestamp) MultiIndex.
            for (sym, ts), row in df.iterrows():
                if str(sym).upper() != symbol:
                    continue
                raw_bars.append(
                    {
                        "timestamp": ts,
                        "open": float(row.get("open", row.get("o", 0))),
                        "high": float(row.get("high", row.get("h", 0))),
                        "low": float(row.get("low", row.get("l", 0))),
                        "close": float(row.get("close", row.get("c", 0))),
                        "volume": float(row.get("volume", row.get("v", 0))),
                    }
                )
        elif isinstance(barset, dict):
            for key, bars_list in barset.items():
                if str(key).upper() != symbol:
                    continue
                for raw in bars_list or []:
                    raw_bars.append(raw)
        elif isinstance(barset, list):
            raw_bars = list(barset)

        validated = validate_bars(raw_bars, symbol, timeframe, clock=self._clock)
        # Sort oldest-first, deduplicate by timestamp.
        validated.sort(key=lambda b: b.timestamp)
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for bar in validated:
            key = bar.timestamp.isoformat()
            if key in seen:
                continue
            seen.add(key)
            unique.append(bar.to_dict())
        return unique

    def _normalize_trade(self, trade: Any, symbol: str) -> dict[str, Any]:
        price = getattr(trade, "price", None) or getattr(trade, "p", None)
        size = getattr(trade, "size", None) or getattr(trade, "s", 0)
        ts = getattr(trade, "timestamp", None) or getattr(trade, "t", None)
        if ts is None:
            ts = self._clock()
        return {
            "price": float(price) if price is not None else None,
            "size": float(size) if size is not None else 0.0,
            "timestamp": _to_iso(ts),
            "symbol": symbol,
        }

    def _normalize_quote(self, quote: Any, symbol: str) -> dict[str, Any]:
        bid = getattr(quote, "bid_price", None) or getattr(quote, "bp", None)
        ask = getattr(quote, "ask_price", None) or getattr(quote, "ap", None)
        ts = getattr(quote, "timestamp", None) or getattr(quote, "t", None)
        if ts is None:
            ts = self._clock()
        return {
            "bid": float(bid) if bid is not None else None,
            "ask": float(ask) if ask is not None else None,
            "timestamp": _to_iso(ts),
            "symbol": symbol,
        }


def _minutes_per_bar(tf: TimeFrame) -> int:
    unit = getattr(tf, "unit", None)
    amount = max(1, int(getattr(tf, "amount", 1)))
    if unit == TimeFrameUnit.Minute:
        return amount
    if unit == TimeFrameUnit.Hour:
        return amount * 60
    if unit == TimeFrameUnit.Day:
        return amount * 1440
    return 1


def _to_iso(value: Any) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    return str(value)
