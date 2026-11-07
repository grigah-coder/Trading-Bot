"""Alpaca implementation of the provider-neutral market-data contract."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Any

import requests
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from config.settings import BrokerSettings, MarketDataSettings
from core.retry import call_with_retry
from data.models import Candle, LatestPrice, Timeframe
from data.validation import CandleValidator


class _TimeoutSession(requests.Session):
    def __init__(self, timeout_seconds: float) -> None:
        super().__init__()
        self._timeout_seconds = timeout_seconds

    def request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        kwargs.setdefault("timeout", self._timeout_seconds)
        return super().request(method, url, **kwargs)


class AlpacaMarketDataProvider:
    """Alpaca stock data adapter; credentials and feed are supplied by config."""

    def __init__(
        self,
        broker: BrokerSettings,
        settings: MarketDataSettings | None = None,
        *,
        historical_client: Any | None = None,
        stream_factory: Callable[..., Any] = StockDataStream,
        retry: Callable[..., Any] = call_with_retry,
    ) -> None:
        if not broker.api_key or not broker.api_secret:
            raise ValueError("Alpaca market-data credentials are required")
        self.settings = settings or MarketDataSettings()
        if historical_client is None:
            historical_client = StockHistoricalDataClient(broker.api_key, broker.api_secret)
            historical_client._session = _TimeoutSession(self.settings.request_timeout_seconds)
        self._client = historical_client
        self._api_key = broker.api_key
        self._api_secret = broker.api_secret
        self._stream_factory = stream_factory
        self._retry = retry
        self._stream: Any | None = None

    def fetch_bars(self, symbols: Sequence[str], timeframe: Timeframe,
                   start: datetime, end: datetime) -> Mapping[str, tuple[Candle, ...]]:
        start_utc = _utc(start, "start")
        end_utc = _utc(end, "end")
        if start_utc >= end_utc:
            raise ValueError("start must be before end")
        normalized = tuple(dict.fromkeys(_symbols(symbols)))
        request = StockBarsRequest(
            symbol_or_symbols=list(normalized),
            timeframe=_alpaca_timeframe(timeframe),
            start=start_utc,
            end=end_utc,
            feed=DataFeed(self.settings.feed.lower()),
        )
        response = self._retry(
            self._client.get_stock_bars,
            request,
            attempts=self.settings.retry_attempts,
            base_delay=self.settings.retry_base_delay_seconds,
            description="Alpaca historical bars",
        )
        bars = _bar_mapping(response)
        return {symbol: tuple(_candle(symbol, bar) for bar in bars.get(symbol, ())) for symbol in normalized}

    def latest_prices(self, symbols: Sequence[str]) -> Mapping[str, LatestPrice]:
        normalized = tuple(dict.fromkeys(_symbols(symbols)))
        request = StockLatestTradeRequest(
            symbol_or_symbols=list(normalized), feed=DataFeed(self.settings.feed.lower())
        )
        response = self._retry(
            self._client.get_stock_latest_trade,
            request,
            attempts=self.settings.retry_attempts,
            base_delay=self.settings.retry_base_delay_seconds,
            description="Alpaca latest trades",
        )
        trades = _bar_mapping(response)
        result: dict[str, LatestPrice] = {}
        for symbol in normalized:
            trade = trades.get(symbol)
            if trade is not None:
                result[symbol] = LatestPrice(symbol, float(trade.price), trade.timestamp)
        return result

    def stream_bars(self, symbols: Sequence[str], on_candle: Callable[[Candle], None]) -> None:
        normalized = tuple(dict.fromkeys(_symbols(symbols)))
        stream = self._stream_factory(
            self._api_key,
            self._api_secret,
            feed=DataFeed(self.settings.feed.lower()),
            data_timeout=self.settings.stream_timeout_seconds,
        )
        self._stream = stream

        async def handle(bar: Any) -> None:
            candle = _candle(bar.symbol, bar)
            CandleValidator().validate((candle,))
            on_candle(candle)

        stream.subscribe_bars(handle, *normalized)
        stream.run()

    def stop_stream(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream = None


def _symbols(symbols: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(symbol.strip().upper() for symbol in symbols)
    if not normalized or any(not symbol for symbol in normalized):
        raise ValueError("symbols must contain non-empty values")
    return normalized


def _utc(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    from datetime import timezone
    return value.astimezone(timezone.utc)


def _alpaca_timeframe(timeframe: Timeframe) -> TimeFrame:
    units = {
        "minute": TimeFrameUnit.Minute,
        "hour": TimeFrameUnit.Hour,
        "day": TimeFrameUnit.Day,
        "week": TimeFrameUnit.Week,
        "month": TimeFrameUnit.Month,
    }
    return TimeFrame(timeframe.amount, units[timeframe.unit])


def _bar_mapping(response: Any) -> Mapping[str, Any]:
    values = getattr(response, "data", response)
    if not isinstance(values, Mapping):
        raise TypeError("Alpaca returned an unexpected market-data response")
    return values


def _candle(symbol: str, bar: Any) -> Candle:
    candle = Candle(
        symbol=symbol,
        timestamp=bar.timestamp,
        open=float(bar.open),
        high=float(bar.high),
        low=float(bar.low),
        close=float(bar.close),
        volume=float(bar.volume) if getattr(bar, "volume", None) is not None else None,
    )
    CandleValidator().validate((candle,))
    return candle
