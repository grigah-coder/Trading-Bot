"""Provider-independent market data contract."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Protocol

from data.models import Candle, LatestPrice, Timeframe


class MarketDataProvider(Protocol):
    def fetch_bars(
        self, symbols: Sequence[str], timeframe: Timeframe, start: datetime, end: datetime
    ) -> Mapping[str, Sequence[Candle]]: ...

    def latest_prices(self, symbols: Sequence[str]) -> Mapping[str, LatestPrice]: ...

    def stream_bars(self, symbols: Sequence[str], on_candle: Callable[[Candle], None]) -> None:
        """Run the provider stream until stopped; callbacks receive normalized candles."""
        ...

    def stop_stream(self) -> None: ...
