from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from data.models import Candle, Timeframe
from data.validation import DataQualityError
from indicators.technical import atr, ema, momentum, sma
from signals.models import Direction, Signal
from signals.validation import SignalValidationError, SignalValidator
from strategies.base import StrategySnapshot
from strategies.trend_momentum import TrendMomentumConfig, TrendMomentumStrategy

UTC = timezone.utc
NOW = datetime(2026, 11, 6, 15, 0, tzinfo=UTC)


def candles(closes: list[float], *, start: datetime | None = None) -> tuple[Candle, ...]:
    start = start or NOW - timedelta(minutes=len(closes))
    return tuple(Candle("AAPL", start + timedelta(minutes=i), close - 0.2,
                        close + 0.5, close - 0.5, close, 1000)
                 for i, close in enumerate(closes))


def test_indicators_return_expected_values() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert sma(values, 3) == 3.0
    assert ema(values, 3) == pytest.approx(3.0)
    assert momentum(values, 2) == pytest.approx(2.0)
    assert atr(candles([10, 11, 10, 12]), 2) > 0


def test_strategy_parameters_are_configurable_and_validated() -> None:
    config = TrendMomentumConfig.from_mapping({"fast_period": "3", "slow_period": "8",
                                               "maximum_atr_fraction": "0.04"})
    assert config.fast_period == 3 and config.slow_period == 8
    assert config.maximum_atr_fraction == 0.04
    with pytest.raises(ValueError, match="Unknown"):
        TrendMomentumConfig.from_mapping({"misspelled_period": "4"})
    with pytest.raises(ValueError):
        TrendMomentumConfig.from_mapping({"maximum_data_age_seconds": "inf"})


def test_strategy_emits_long_entry_for_confirmed_uptrend() -> None:
    config = TrendMomentumConfig(fast_period=3, slow_period=5, momentum_period=2,
                                 atr_period=3, minimum_atr_fraction=0,
                                 maximum_atr_fraction=0.2)
    strategy = TrendMomentumStrategy(config)
    values = [100, 100, 100, 100, 100, 101, 102, 103, 104, 105]
    snapshot = StrategySnapshot("AAPL", candles(values), NOW, Timeframe(1, "minute"))
    signal = strategy.evaluate(snapshot)
    assert signal is not None
    assert signal.direction is Direction.LONG
    assert signal.entry_reference == 105
    assert signal.stop_loss_reference < signal.entry_reference
    assert signal.take_profit_reference > signal.entry_reference
    assert signal.strategy_id == "trend_momentum"
    assert signal.reason


def test_strategy_emits_exit_when_trend_and_momentum_turn_down() -> None:
    config = TrendMomentumConfig(fast_period=3, slow_period=5, momentum_period=2,
                                 atr_period=3, minimum_atr_fraction=0,
                                 maximum_atr_fraction=0.2)
    strategy = TrendMomentumStrategy(config)
    rising = [100, 100, 100, 100, 100, 101, 102, 103, 104, 105]
    falling = [110, 112, 114, 116, 118, 120, 119, 117, 114, 110]
    strategy.evaluate(StrategySnapshot("AAPL", candles(rising), NOW, Timeframe(1, "minute")))
    signal = strategy.evaluate(StrategySnapshot("AAPL", candles(falling), NOW, Timeframe(1, "minute")))
    assert signal is not None
    assert signal.direction is Direction.EXIT


def test_strategy_skips_flat_or_excessively_volatile_regime() -> None:
    config = TrendMomentumConfig(fast_period=3, slow_period=5, momentum_period=2,
                                 atr_period=3, maximum_atr_fraction=0.005)
    strategy = TrendMomentumStrategy(config)
    values = [100, 100, 100, 100, 100, 101, 102, 103, 104, 105]
    assert strategy.evaluate(StrategySnapshot("AAPL", candles(values), NOW, Timeframe(1, "minute"))) is None
    flat_config = TrendMomentumConfig(fast_period=3, slow_period=5, momentum_period=2,
                                      atr_period=3, minimum_atr_fraction=0.02,
                                      maximum_atr_fraction=0.2)
    assert TrendMomentumStrategy(flat_config).evaluate(
        StrategySnapshot("AAPL", candles([100] * 10), NOW, Timeframe(1, "minute"))
    ) is None


def test_strategy_rejects_incomplete_and_lookahead_data() -> None:
    config = TrendMomentumConfig(fast_period=2, slow_period=3, momentum_period=1, atr_period=2)
    strategy = TrendMomentumStrategy(config)
    with pytest.raises(ValueError, match="bars"):
        strategy.evaluate(StrategySnapshot("AAPL", candles([1, 2]), NOW, Timeframe(1, "minute")))
    future = candles([100, 101, 102, 103], start=NOW)
    with pytest.raises(ValueError, match="decision"):
        strategy.evaluate(StrategySnapshot("AAPL", future, NOW, Timeframe(1, "minute")))


def test_strategy_rejects_missing_in_session_candle() -> None:
    config = TrendMomentumConfig(fast_period=3, slow_period=5, momentum_period=2,
                                 atr_period=3, minimum_atr_fraction=0,
                                 maximum_atr_fraction=0.2)
    bars = list(candles([100, 100, 100, 100, 100, 101, 102, 103, 104, 105]))
    bars.pop(4)
    with pytest.raises(DataQualityError, match="Missing candle"):
        TrendMomentumStrategy(config).evaluate(StrategySnapshot("AAPL", tuple(bars), NOW, Timeframe(1, "minute")))


def test_daily_bar_is_not_available_before_exchange_close() -> None:
    config = TrendMomentumConfig(fast_period=2, slow_period=3, momentum_period=1, atr_period=2)
    dates = [datetime(2026, 11, day, 14, 30, tzinfo=UTC) for day in (3, 4, 5, 6)]
    bars = tuple(Candle("AAPL", timestamp, 99 + index, 101 + index, 98 + index,
                        100 + index, 1000)
                 for index, timestamp in enumerate(dates))
    before_close = datetime(2026, 11, 6, 20, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="decision timestamp"):
        TrendMomentumStrategy(config).evaluate(StrategySnapshot("AAPL", bars, before_close, Timeframe(1, "day")))


def test_strategy_does_not_reenter_until_an_exit_signal() -> None:
    config = TrendMomentumConfig(fast_period=3, slow_period=5, momentum_period=2,
                                 atr_period=3, minimum_atr_fraction=0,
                                 maximum_atr_fraction=0.2)
    strategy = TrendMomentumStrategy(config)
    rising = StrategySnapshot("AAPL", candles([100, 100, 100, 100, 100, 101, 102, 103, 104, 105]), NOW, Timeframe(1, "minute"))
    assert strategy.evaluate(rising).direction is Direction.LONG
    assert strategy.evaluate(rising) is None


def test_signal_validator_rejects_invalid_duplicate_expired_and_out_of_session() -> None:
    signal = Signal("AAPL", Direction.LONG, 100, 95, 110, NOW, "test", 0.5, "trend")
    validator = SignalValidator(max_age=timedelta(minutes=2), allowed_session=lambda at: True)
    assert validator.validate(signal, now=NOW)
    with pytest.raises(SignalValidationError, match="Duplicate"):
        validator.validate(signal, now=NOW)
    expired = Signal("MSFT", Direction.LONG, 100, 95, 110,
                     NOW - timedelta(minutes=3), "test", None, "trend")
    with pytest.raises(SignalValidationError, match="expired"):
        SignalValidator(max_age=timedelta(minutes=2), allowed_session=lambda at: True).validate(expired, now=NOW)
    invalid = Signal("NVDA", Direction.LONG, 100, 101, 110, NOW, "test", None, "bad stop")
    with pytest.raises(SignalValidationError, match="stop"):
        SignalValidator(max_age=timedelta(minutes=2), allowed_session=lambda at: True).validate(invalid, now=NOW)
    malformed = Signal("NVDA", Direction.LONG, "not-a-price", 95, 110, NOW, "test", None, "bad price")
    with pytest.raises(SignalValidationError, match="entry price"):
        SignalValidator(max_age=timedelta(minutes=2), allowed_session=lambda at: True).validate(malformed, now=NOW)
    outside = SignalValidator(max_age=timedelta(minutes=2), allowed_session=lambda at: False)
    with pytest.raises(SignalValidationError, match="session"):
        outside.validate(signal, now=NOW)
