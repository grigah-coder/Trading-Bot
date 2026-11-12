"""Integration: real strategy signal -> real risk engine decision.

Verifies the percentage convention end to end: strategies emit fractional
stop/take distances and the risk engine converts them to absolute prices with
the same semantics.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from trading_bot.config.settings import Settings
from trading_bot.risk.engine import RiskEngine
from trading_bot.strategy.signals import MovingAverageCrossStrategy


def _bar(close: float, index: int) -> Dict[str, Any]:
    return {
        "timestamp": f"2024-01-02T09:{30 + index:02d}:00Z",
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": 1000.0,
        "symbol": "AAPL",
        "timeframe": "5 minutes",
    }


def _account(equity: float = 10000.0) -> Dict[str, Any]:
    return {
        "status": "active",
        "cash": equity,
        "equity": equity,
        "account_blocked": False,
        "trading_blocked": False,
    }


def test_ma_crossover_signal_is_approved_with_fractional_pct_units():
    """A real crossover signal must pass risk with fraction-based prices."""
    # Fast SMA (3) crosses above slow SMA (5) exactly on the last bar.
    closes: List[float] = [10.0] * 9 + [15.0]
    bars = {"AAPL": [_bar(close, i) for i, close in enumerate(closes)]}

    strategy = MovingAverageCrossStrategy(fast_period=3, slow_period=5)
    signals = strategy.generate_signals(bars)
    assert len(signals) == 1
    signal = signals[0]
    assert signal["side"] == "buy"
    assert signal["stop_loss_pct"] == pytest.approx(0.012)
    assert signal["take_profit_pct"] == pytest.approx(0.04)

    settings = Settings()
    assert settings.stop_loss_min_pct <= signal["stop_loss_pct"]
    assert signal["stop_loss_pct"] <= settings.stop_loss_max_pct

    decision = RiskEngine(settings).assess(signal, _account(), [])

    assert decision["approved"] is True
    assert decision["order"] is not None
    price = signal["price"]
    assert decision["stop_price"] == pytest.approx(price * (1.0 - 0.012), rel=1e-4)
    assert decision["limit_price"] == pytest.approx(price * (1.0 + 0.04), rel=1e-4)
    assert decision["order"]["stop_price"] == decision["stop_price"]
    assert decision["order"]["limit_price"] == decision["limit_price"]
    assert decision["order"]["qty"] == pytest.approx(decision["qty"])
    assert decision["qty"] > 0
