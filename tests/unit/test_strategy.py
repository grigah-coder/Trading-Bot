"""Unit tests for the Phase 6 strategy & signal engine."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pytest

from trading_bot.exceptions import StrategyError
from trading_bot.strategy.base import Signal, Strategy
from trading_bot.strategy.indicators import crossover, crossunder, ema, rsi, sma
from trading_bot.strategy.signals import MovingAverageCrossStrategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bar(close: float, offset_minutes: int = 0) -> Dict[str, Any]:
    ts = (datetime.now(timezone.utc) + timedelta(minutes=offset_minutes)).isoformat()
    return {
        "timestamp": ts,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": 100.0,
        "symbol": "AAPL",
        "timeframe": "5 minutes",
    }


def _bars(closes: List[float]) -> List[Dict[str, Any]]:
    return [bar for bar in (_bar(c, i) for i, c in enumerate(closes))]


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------

class TestSignal:
    def test_valid_signal(self):
        s = Signal(
            symbol="aapl",
            side="BUY",
            price=100.0,
            quantity=10.0,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            signal_id="sig-1",
            timestamp=datetime.now(timezone.utc),
        )
        assert s.symbol == "AAPL"
        assert s.side == "buy"
        assert s.to_dict()["symbol"] == "AAPL"
        assert s.to_dict()["side"] == "buy"
        assert s.to_dict()["signal_id"] == "sig-1"

    def test_invalid_side_raises(self):
        with pytest.raises(StrategyError):
            Signal(
                symbol="AAPL", side="hold", price=100.0, quantity=10.0,
                stop_loss_pct=0.02, take_profit_pct=0.04,
                signal_id="sig-1", timestamp=datetime.now(timezone.utc),
            )

    def test_nonpositive_price_raises(self):
        with pytest.raises(StrategyError):
            Signal(
                symbol="AAPL", side="buy", price=0.0, quantity=10.0,
                stop_loss_pct=0.02, take_profit_pct=0.04,
                signal_id="sig-1", timestamp=datetime.now(timezone.utc),
            )

    def test_strength_out_of_range_raises(self):
        with pytest.raises(StrategyError):
            Signal(
                symbol="AAPL", side="buy", price=100.0, quantity=10.0,
                stop_loss_pct=0.02, take_profit_pct=0.04,
                signal_id="sig-1", timestamp=datetime.now(timezone.utc),
                strength=1.5,
            )

    def test_empty_signal_id_auto_generated(self):
        s = Signal(
            symbol="AAPL", side="buy", price=100.0, quantity=10.0,
            stop_loss_pct=0.02, take_profit_pct=0.04,
            signal_id="", timestamp=datetime.now(timezone.utc),
        )
        assert s.signal_id.startswith("sig-")


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

class TestSMA:
    def test_basic(self):
        assert sma([1, 2, 3, 4, 5], 3) == [None, None, 2.0, 3.0, 4.0]

    def test_period_one_equals_input(self):
        assert sma([1, 2, 3], 1) == [1.0, 2.0, 3.0]

    def test_none_inputs_skipped(self):
        assert sma([1, None, 3, None, 5], 2) == [None, None, 2.0, None, 4.0]

    def test_invalid_period_raises(self):
        with pytest.raises(ValueError):
            sma([1, 2, 3], 0)


class TestEMA:
    def test_shape_matches_input(self):
        out = ema([1, 2, 3, 4, 5], 3)
        assert len(out) == 5
        # First period-1 entries are None until seed window full.
        assert out[0] is None and out[1] is None
        assert all(v is not None and v > 0 for v in out[2:])

    def test_invalid_period_raises(self):
        with pytest.raises(ValueError):
            ema([1, 2, 3], 0)


class TestRSI:
    def test_bounds(self):
        out = rsi([10, 11, 12, 11, 12, 13, 12, 11, 10, 11, 12], 3)
        for v in out:
            if v is not None:
                assert 0.0 <= v <= 100.0

    def test_all_up_is_100(self):
        out = rsi([1, 2, 3, 4, 5, 6, 7], 3)
        assert out[-1] == 100.0

    def test_all_down_is_0(self):
        out = rsi([7, 6, 5, 4, 3, 2, 1], 3)
        assert out[-1] == 0.0


class TestCrossoverCrossunder:
    def test_crossover_detected(self):
        fast = [None, None, None, 5.0, 6.0]
        slow = [None, None, None, 6.0, 5.0]
        assert crossover(fast, slow) == [False, False, False, False, True]

    def test_no_crossover_when_both_none(self):
        assert crossover([None, None], [None, None]) == [False, False]

    def test_crossunder_detected(self):
        fast = [None, None, None, 6.0, 5.0]
        slow = [None, None, None, 5.0, 6.0]
        assert crossunder(fast, slow) == [False, False, False, False, True]


# ---------------------------------------------------------------------------
# MovingAverageCrossStrategy
# ---------------------------------------------------------------------------

class TestMovingAverageCrossStrategy:
    def test_insufficient_bars_no_signal(self):
        strat = MovingAverageCrossStrategy(fast_period=3, slow_period=5)
        signals = strat.generate_signals({"AAPL": _bars([10, 11, 12])})
        assert signals == []

    def test_uptrend_emits_buy(self):
        # Crossover occurs exactly on the last bar (fast crosses above slow).
        closes = [10, 10, 10, 10, 10, 10, 10, 10, 10, 15]
        strat = MovingAverageCrossStrategy(fast_period=3, slow_period=5)
        signals = strat.generate_signals({"AAPL": _bars(closes)})
        assert len(signals) == 1
        buys = [s for s in signals if s["side"] == "buy"]
        assert len(buys) == 1
        assert buys[0]["symbol"] == "AAPL"
        assert buys[0]["price"] == pytest.approx(sma(closes, 3)[-1])
        assert buys[0]["signal_id"].startswith("ma_cross:AAPL:buy:")

    def test_downtrend_emits_sell(self):
        # Crossunder occurs exactly on the last bar (fast crosses below slow).
        closes = [15, 15, 15, 15, 15, 15, 15, 15, 15, 10]
        strat = MovingAverageCrossStrategy(fast_period=3, slow_period=5)
        signals = strat.generate_signals({"AAPL": _bars(closes)})
        sells = [s for s in signals if s["side"] == "sell"]
        assert len(sells) == 1

    def test_dedup_same_bars(self):
        closes = [10, 10, 10, 10, 10, 10, 10, 10, 10, 15]
        strat = MovingAverageCrossStrategy(fast_period=3, slow_period=5)
        bars = _bars(closes)
        first = strat.generate_signals({"AAPL": bars})
        second = strat.generate_signals({"AAPL": bars})
        # Same strategy instance: second call dedupes.
        assert len(first) == 1
        assert len(second) == 0

    def test_new_instance_rededups(self):
        closes = [10, 10, 10, 10, 10, 10, 10, 10, 10, 15]
        bars = _bars(closes)
        s1 = MovingAverageCrossStrategy(fast_period=3, slow_period=5)
        s2 = MovingAverageCrossStrategy(fast_period=3, slow_period=5)
        assert len(s1.generate_signals({"AAPL": bars})) == 1
        assert len(s2.generate_signals({"AAPL": bars})) == 1

    def test_signal_shape(self):
        closes = [10, 10, 10, 10, 10, 10, 10, 10, 10, 15]
        strat = MovingAverageCrossStrategy(fast_period=3, slow_period=5)
        signals = strat.generate_signals({"AAPL": _bars(closes)})
        s = signals[0]
        for key in ("symbol", "side", "price", "quantity", "stop_loss_pct",
                    "take_profit_pct", "signal_id", "timestamp",
                    "strategy_name", "strength", "metadata"):
            assert key in s, f"missing key {key}"
        assert s["strategy_name"] == "ma_cross"
        assert s["metadata"]["trigger_bar"] is not None

    def test_slow_not_greater_than_fast_raises(self):
        with pytest.raises(StrategyError):
            MovingAverageCrossStrategy(fast_period=30, slow_period=10)

    def test_multiple_symbols_independent(self):
        strat = MovingAverageCrossStrategy(fast_period=3, slow_period=5)
        bars = {
            "AAPL": _bars([10, 10, 10, 10, 10, 10, 10, 10, 10, 15]),
            "MSFT": _bars([15, 15, 15, 15, 15, 15, 15, 15, 15, 10]),
        }
        signals = strat.generate_signals(bars)
        symbols = {s["symbol"] for s in signals}
        assert "AAPL" in symbols and "MSFT" in symbols


class TestStrategyBase:
    def test_base_raises_not_implemented(self):
        strat = Strategy()
        with pytest.raises(NotImplementedError):
            strat.generate_signals({"AAPL": []})
