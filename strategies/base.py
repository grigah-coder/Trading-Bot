"""Common strategy evaluation interface and validated decision-time snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from core.market_calendar import MarketSession
from data.models import Candle, Timeframe
from signals.models import Signal


@dataclass(frozen=True, slots=True)
class StrategySnapshot:
    symbol: str
    candles: tuple[Candle, ...]
    decision_at: datetime
    timeframe: Timeframe


class Strategy(Protocol):
    strategy_id: str

    def evaluate(self, snapshot: StrategySnapshot) -> Signal | None: ...

    def on_session_start(self, session: MarketSession) -> None: ...
