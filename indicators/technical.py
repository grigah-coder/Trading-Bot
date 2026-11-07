"""Small deterministic technical indicators used by research strategies."""

from __future__ import annotations

from collections.abc import Sequence

from data.models import Candle


def _validate_period(values: Sequence[float], period: int) -> None:
    if isinstance(period, bool) or period <= 0:
        raise ValueError("period must be a positive integer")
    if len(values) < period:
        raise ValueError("not enough observations for indicator period")


def sma(values: Sequence[float], period: int) -> float:
    """Arithmetic mean of the most recent ``period`` values."""
    _validate_period(values, period)
    return sum(values[-period:]) / period


def ema(values: Sequence[float], period: int) -> float:
    """EMA seeded with the first-period SMA and updated through the series."""
    _validate_period(values, period)
    result = sum(values[:period]) / period
    alpha = 2.0 / (period + 1)
    for value in values[period:]:
        result = alpha * value + (1.0 - alpha) * result
    return result


def momentum(values: Sequence[float], period: int) -> float:
    """Absolute close change over ``period`` observations."""
    if isinstance(period, bool) or period <= 0 or len(values) <= period:
        raise ValueError("momentum requires more than a positive period of observations")
    return values[-1] - values[-period - 1]


def atr(candles: Sequence[Candle], period: int) -> float:
    """Simple average of true range over the most recent candles."""
    _validate_period(candles, period)
    ranges: list[float] = []
    previous_close: float | None = None
    for candle in candles:
        true_range = candle.high - candle.low
        if previous_close is not None:
            true_range = max(true_range, abs(candle.high - previous_close), abs(candle.low - previous_close))
        ranges.append(true_range)
        previous_close = candle.close
    return sum(ranges[-period:]) / period
