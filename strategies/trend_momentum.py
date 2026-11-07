"""Configurable long-only trend/momentum research strategy (not investment advice)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import timedelta, timezone
from typing import Mapping

from core.market_calendar import ExchangeMarketCalendar, MarketCalendar
from data.models import Candle
from data.validation import CandleValidator
from indicators.technical import atr, ema, momentum
from signals.models import Direction, Signal
from strategies.base import StrategySnapshot


@dataclass(frozen=True, slots=True)
class TrendMomentumConfig:
    fast_period: int = 10
    slow_period: int = 30
    momentum_period: int = 5
    atr_period: int = 14
    stop_atr_multiple: float = 2.0
    target_atr_multiple: float = 3.0
    minimum_atr_fraction: float = 0.001
    maximum_atr_fraction: float = 0.05
    maximum_data_age: timedelta = timedelta(minutes=3)

    def __post_init__(self) -> None:
        for name in ("fast_period", "slow_period", "momentum_period", "atr_period"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.fast_period >= self.slow_period:
            raise ValueError("fast_period must be shorter than slow_period")
        if not math.isfinite(self.stop_atr_multiple) or self.stop_atr_multiple <= 0:
            raise ValueError("stop_atr_multiple must be positive and finite")
        if not math.isfinite(self.target_atr_multiple) or self.target_atr_multiple <= 0:
            raise ValueError("target_atr_multiple must be positive and finite")
        if (not math.isfinite(self.minimum_atr_fraction) or not math.isfinite(self.maximum_atr_fraction)
                or self.minimum_atr_fraction < 0 or self.maximum_atr_fraction <= self.minimum_atr_fraction):
            raise ValueError("ATR regime bounds must be finite, non-negative, and ordered")
        if not isinstance(self.maximum_data_age, timedelta) or self.maximum_data_age.total_seconds() <= 0:
            raise ValueError("maximum_data_age must be positive")

    @classmethod
    def from_mapping(cls, values: Mapping[str, str]) -> TrendMomentumConfig:
        allowed = {"fast_period", "slow_period", "momentum_period", "atr_period",
                   "stop_atr_multiple", "target_atr_multiple", "minimum_atr_fraction",
                   "maximum_atr_fraction", "maximum_data_age_seconds"}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unknown trend strategy parameter(s): {', '.join(sorted(unknown))}")
        converted: dict[str, object] = dict(values)
        try:
            for name in ("fast_period", "slow_period", "momentum_period", "atr_period"):
                if name in converted:
                    converted[name] = int(converted[name])
            for name in ("stop_atr_multiple", "target_atr_multiple", "minimum_atr_fraction", "maximum_atr_fraction"):
                if name in converted:
                    converted[name] = float(converted[name])
            if "maximum_data_age_seconds" in converted:
                age = float(converted.pop("maximum_data_age_seconds"))
                if not math.isfinite(age):
                    raise ValueError("maximum_data_age_seconds must be finite")
                converted["maximum_data_age"] = timedelta(seconds=age)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Invalid trend strategy parameter value") from exc
        return cls(**converted)


class TrendMomentumStrategy:
    """Uses EMA trend, positive price momentum, and an ATR regime gate."""

    strategy_id = "trend_momentum"

    def __init__(self, config: TrendMomentumConfig = TrendMomentumConfig(),
                 *, calendar: MarketCalendar | None = None) -> None:
        self.config = config
        self.calendar = calendar or ExchangeMarketCalendar()
        self._validator = CandleValidator(self.calendar)
        self._in_position: set[str] = set()

    def on_session_start(self, session) -> None:
        """Retain position intent across sessions; this is not a day-reset strategy."""

    def health_check(self) -> None:
        return None

    def shutdown(self) -> None:
        self._in_position.clear()

    def evaluate(self, snapshot: StrategySnapshot) -> Signal | None:
        decision = _utc(snapshot.decision_at)
        needed = max(self.config.slow_period, self.config.atr_period, self.config.momentum_period + 1)
        normalized = snapshot.symbol.strip().upper()
        if not normalized:
            raise ValueError("Snapshot symbol is required")
        if not snapshot.candles:
            raise ValueError(f"Missing market data for {normalized}")
        bars = self._validator.validate(snapshot.candles, timeframe=snapshot.timeframe)
        if any(bar.symbol != normalized for bar in bars):
            raise ValueError("Snapshot candle symbol does not match snapshot symbol")
        final_bar = bars[-1]
        if snapshot.timeframe.unit == "day":
            session = self.calendar.session_for(final_bar.timestamp.date())
            if session is None:
                raise ValueError("Daily candle does not correspond to a market session")
            available_at = session.closes_at
        else:
            available_at = final_bar.timestamp + _duration(snapshot.timeframe)
        if available_at > decision:
            raise ValueError("Snapshot contains data unavailable at decision timestamp")
        if decision - available_at > self.config.maximum_data_age:
            raise ValueError("Snapshot market data is stale")
        if len(bars) < needed:
            raise ValueError(f"Insufficient bars for strategy; need at least {needed}")
        closes = [bar.close for bar in bars]
        last = closes[-1]
        fast = ema(closes, self.config.fast_period)
        slow = ema(closes, self.config.slow_period)
        impulse = momentum(closes, self.config.momentum_period)
        average_range = atr(bars, self.config.atr_period)
        if last <= 0:
            raise ValueError("Invalid closing price")
        atr_fraction = average_range / last
        regime_ok = self.config.minimum_atr_fraction <= atr_fraction <= self.config.maximum_atr_fraction
        long_setup = regime_ok and fast > slow and last > slow and impulse > 0
        bearish_exit = fast <= slow or impulse <= 0
        if normalized in self._in_position:
            if bearish_exit:
                self._in_position.remove(normalized)
                return Signal(normalized, Direction.EXIT, last, None, None, decision,
                              self.strategy_id, None,
                              f"Exit: trend or momentum confirmation failed; ATR={average_range:.6g}")
            return None
        if long_setup:
            stop = last - self.config.stop_atr_multiple * average_range
            target = last + self.config.target_atr_multiple * average_range
            if stop <= 0:
                return None
            self._in_position.add(normalized)
            return Signal(normalized, Direction.LONG, last, stop, target, decision,
                          self.strategy_id, None,
                          f"Entry: EMA trend and positive momentum confirmed; ATR={average_range:.6g}")
        return None


def _utc(value):
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("decision_at must be timezone-aware")
    return value.astimezone(timezone.utc)


def _duration(timeframe) -> timedelta:
    duration = timeframe.duration
    if duration is None:
        raise ValueError("Trend strategy requires minute, hour, or day timeframe")
    return duration
