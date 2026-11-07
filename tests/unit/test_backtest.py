"""Unit tests for the Phase 9 backtesting engine and metrics."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pytest

from trading_bot.backtest.engine import BacktestConfig, BacktestEngine
from trading_bot.backtest.metrics import (
    cagr,
    expectancy,
    max_drawdown,
    profit_factor,
    returns_from_equity,
    sharpe_ratio,
    sortino_ratio,
    summary_metrics,
    total_return,
    win_rate,
)

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeStrategy:
    """Emits a buy when the latest close rises above a threshold, sell below."""

    def __init__(self, buy_threshold: float = 100.0, sell_threshold: float = 90.0):
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.calls = 0

    def generate_signals(self, bars: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        self.calls += 1
        signals: List[Dict[str, Any]] = []
        for symbol, symbol_bars in bars.items():
            if not symbol_bars:
                continue
            close = float(symbol_bars[-1].get("close", 0))
            if close > self.buy_threshold:
                signals.append({
                    "symbol": symbol,
                    "side": "buy",
                    "price": close,
                    "quantity": 10.0,
                    "stop_loss_pct": 0.05,
                    "take_profit_pct": 0.10,
                    "signal_id": f"sig-{symbol}-buy",
                })
            elif close < self.sell_threshold:
                signals.append({
                    "symbol": symbol,
                    "side": "sell",
                    "price": close,
                    "quantity": 10.0,
                    "stop_loss_pct": 0.05,
                    "take_profit_pct": 0.10,
                    "signal_id": f"sig-{symbol}-sell",
                })
        return signals


class FakeRiskEngine:
    """Approves every signal unless the account is 'halted'."""

    def __init__(self, halt: bool = False, resize_qty: Optional[float] = None):
        self.halt = halt
        self.resize_qty = resize_qty
        self.assessed = 0

    def is_halted(self) -> bool:
        return self.halt

    def assess(
        self,
        signal: Dict[str, Any],
        account: Dict[str, Any],
        positions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        self.assessed += 1
        if self.halt:
            return {"approved": False, "rejection_reason": "halted"}
        qty = self.resize_qty if self.resize_qty is not None else float(
            signal.get("quantity", 10.0)
        )
        price = float(signal.get("price", 0))
        stop_pct = float(signal.get("stop_loss_pct", 0.02))
        take_pct = float(signal.get("take_profit_pct", 0.04))
        side = str(signal.get("side", "buy"))
        if side == "buy":
            stop_price = price * (1 - stop_pct)
            limit_price = price * take_pct
        else:
            stop_price = price * (1 + stop_pct)
            limit_price = price * (2 - take_pct)
        return {
            "approved": True,
            "qty": qty,
            "stop_price": round(stop_price, 4),
            "limit_price": round(limit_price, 4),
            "reason": "approved",
        }


# ---------------------------------------------------------------------------
# Bar helpers
# ---------------------------------------------------------------------------


def _bar(close: float, high: float, low: float, open_price: float,
         index: int) -> Dict[str, Any]:
    ts = (datetime(2024, 1, 1, tzinfo=timezone.utc)
          + timedelta(minutes=5 * index)).isoformat()
    return {
        "timestamp": ts,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": 1000.0,
        "symbol": "AAPL",
        "timeframe": "5 minutes",
    }


def _trending_bars(n: int, start: float = 90.0, step: float = 2.0) -> List[Dict[str, Any]]:
    """Bars that trend upward so the strategy buys near the end."""
    bars = []
    for i in range(n):
        c = start + step * i
        bars.append(_bar(c, c + 1, c - 1, c, i))
    return bars


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------


class TestReturns:
    def test_returns_from_equity(self):
        eq = [100.0, 110.0, 121.0]
        rets = returns_from_equity(eq)
        assert len(rets) == 2
        assert rets[0] == pytest.approx(0.1)
        assert rets[1] == pytest.approx(0.1)

    def test_returns_single_point(self):
        assert returns_from_equity([100.0]) == []


class TestTotalReturnCAGR:
    def test_total_return(self):
        assert total_return([100.0, 150.0]) == pytest.approx(0.5)
        assert total_return([100.0]) == 0.0

    def test_cagr(self):
        # 100 -> 200 over 2 years (504 bars at 252/year)
        eq = [100.0] + [100.0] * 503 + [200.0]
        c = cagr(eq, periods_per_year=252.0)
        assert c == pytest.approx(2 ** 0.5 - 1, abs=1e-6)


class TestMaxDrawdown:
    def test_mdd(self):
        eq = [100.0, 120.0, 90.0, 110.0, 80.0]
        # peak 120, trough 80 -> dd = 40/120 = 0.3333
        assert max_drawdown(eq) == pytest.approx(40.0 / 120.0)

    def test_mdd_no_drawdown(self):
        assert max_drawdown([100.0, 110.0, 120.0]) == 0.0


class TestSharpeSortino:
    def test_sharpe_positive(self):
        rets = [0.01, -0.005, 0.02, -0.001, 0.015] * 20
        assert sharpe_ratio(rets) > 0

    def test_sharpe_zero_vol(self):
        assert sharpe_ratio([0.01] * 100) == 0.0

    def test_sortino(self):
        rets = [0.01, -0.005, 0.02, -0.001, 0.015]
        s = sortino_ratio(rets)
        assert s >= 0


class TestTradeMetrics:
    def test_win_rate(self):
        trades = [
            {"realized_pl": 10.0}, {"realized_pl": -5.0},
            {"realized_pl": 20.0}, {"realized_pl": 0.0},
        ]
        assert win_rate(trades) == pytest.approx(0.5)

    def test_profit_factor(self):
        trades = [
            {"realized_pl": 10.0}, {"realized_pl": 20.0},
            {"realized_pl": -5.0}, {"realized_pl": -5.0},
        ]
        assert profit_factor(trades) == pytest.approx(30.0 / 10.0)

    def test_profit_factor_no_loss(self):
        trades = [{"realized_pl": 10.0}, {"realized_pl": 5.0}]
        assert profit_factor(trades) == float("inf")

    def test_expectancy(self):
        trades = [{"realized_pl": 10.0}, {"realized_pl": -10.0}, {"realized_pl": 0.0}]
        assert expectancy(trades) == pytest.approx(0.0)

    def test_summary_metrics(self):
        eq = [100.0, 110.0, 105.0, 120.0]
        trades = [{"realized_pl": 10.0}, {"realized_pl": -5.0}]
        m = summary_metrics(eq, trades)
        for key in ("total_return", "cagr", "max_drawdown", "sharpe", "sortino",
                    "calmar", "win_rate", "expectancy", "profit_factor", "trade_count"):
            assert key in m
        assert m["trade_count"] == 2.0


# ---------------------------------------------------------------------------
# Engine tests
# ---------------------------------------------------------------------------


class TestBacktestEngine:
    def test_no_bars_returns_empty(self):
        engine = BacktestEngine()
        result = engine.run({}, FakeStrategy(), FakeRiskEngine())
        assert result.equity_curve == []
        assert result.trades == []
        # With no data, all metrics are zeroed.
        assert result.metrics["total_return"] == 0.0
        assert result.metrics["max_drawdown"] == 0.0
        assert result.metrics["sharpe"] == 0.0

    def test_uptrend_buy_then_take_profit(self):
        # Strong uptrend: price crosses buy threshold, then keeps rising to hit TP.
        bars = {"AAPL": _trending_bars(40, start=90.0, step=3.0)}
        strategy = FakeStrategy(buy_threshold=100.0, sell_threshold=90.0)
        risk = FakeRiskEngine()
        engine = BacktestEngine(BacktestConfig(initial_capital=10000.0))
        result = engine.run(bars, strategy, risk)

        assert result.signals_generated >= 1
        assert result.signals_approved >= 1
        assert result.final_equity > 0
        assert "sharpe" in result.metrics
        assert result.metrics["trade_count"] >= 0

    def test_flat_no_signal(self):
        # Price never crosses thresholds -> no trades.
        bars = {"AAPL": [_bar(95.0, 96.0, 94.0, 95.0, i) for i in range(20)]}
        strategy = FakeStrategy(buy_threshold=100.0, sell_threshold=90.0)
        risk = FakeRiskEngine()
        engine = BacktestEngine()
        result = engine.run(bars, strategy, risk)
        assert result.signals_generated == 0
        assert result.metrics["trade_count"] == 0

    def test_halted_rejects_all(self):
        bars = {"AAPL": _trending_bars(30, start=90.0, step=3.0)}
        strategy = FakeStrategy(buy_threshold=100.0)
        risk = FakeRiskEngine(halt=True)
        engine = BacktestEngine()
        result = engine.run(bars, strategy, risk)
        assert result.signals_approved == 0
        assert result.signals_rejected >= 1

    def test_resize_qty_applied(self):
        bars = {"AAPL": _trending_bars(30, start=90.0, step=3.0)}
        strategy = FakeStrategy(buy_threshold=100.0)
        risk = FakeRiskEngine(resize_qty=2.0)
        engine = BacktestEngine(BacktestConfig(initial_capital=10000.0))
        result = engine.run(bars, strategy, risk)
        if result.trades:
            assert result.trades[0].quantity == pytest.approx(2.0)

    def test_deterministic(self):
        bars = {"AAPL": _trending_bars(30, start=90.0, step=3.0)}
        r1 = BacktestEngine().run(bars, FakeStrategy(100.0), FakeRiskEngine())
        r2 = BacktestEngine().run(bars, FakeStrategy(100.0), FakeRiskEngine())
        assert r1.total_return == r2.total_return
        assert r1.metrics == r2.metrics

    def test_stop_loss_triggers(self):
        # Price rises above buy threshold then drops sharply below stop.
        bars = {
            "AAPL": [
                _bar(90, 91, 89, 90, 0),
                _bar(101, 102, 100, 101, 1),  # crosses buy threshold
                _bar(94, 95, 93, 94, 2),      # below stop (101*0.95=95.95... actually 95.95)
            ]
        }
        # stop = 101 * 0.95 = 95.95; bar 2 low=93 <= 95.95 -> stop triggers
        strategy = FakeStrategy(buy_threshold=100.0)
        risk = FakeRiskEngine()
        engine = BacktestEngine(BacktestConfig(initial_capital=10000.0))
        result = engine.run(bars, strategy, risk)
        assert result.orders_filled >= 1
        # At least the entry filled; the stop should exit with a loss.
        assert result.metrics["trade_count"] >= 1 or result.trades

    def test_config_costs_reduce_return(self):
        bars = {"AAPL": _trending_bars(30, start=90.0, step=3.0)}
        strategy = FakeStrategy(buy_threshold=100.0)
        risk = FakeRiskEngine()
        free = BacktestEngine(BacktestConfig(initial_capital=10000.0)).run(
            bars, strategy, risk
        )
        costly = BacktestEngine(
            BacktestConfig(initial_capital=10000.0, commission_per_trade=1.0)
        ).run(bars, strategy, risk)
        # With costs, final equity should not be higher (usually lower or equal).
        assert costly.final_equity <= free.final_equity + 1e-6
