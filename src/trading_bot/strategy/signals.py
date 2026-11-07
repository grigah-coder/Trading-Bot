"""Concrete strategies and signal generation.

Provides :class:`MovingAverageCrossStrategy`, a dual-SMA trend-following
strategy, plus helpers for building idempotent signals.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from trading_bot.config.settings import Settings
from trading_bot.exceptions import StrategyError
from trading_bot.logging_config import configure_logging
from trading_bot.strategy.base import Signal, Strategy
from trading_bot.strategy.indicators import crossover, crossunder, ema, sma  # noqa: F401

logger = configure_logging(__name__).getChild("signals")


class MovingAverageCrossStrategy(Strategy):
    """Dual-SMA trend-following strategy.

    Emits a ``buy`` signal when the fast SMA crosses above the slow SMA, and a
    ``sell`` signal when it crosses below. Signals are keyed by
    ``(symbol, side, last_bar_timestamp)`` so repeated calls with unchanged
    bars do not re-emit (idempotency).

    Parameters
    ----------
    fast_period:
        Fast SMA lookback.
    slow_period:
        Slow SMA lookback (must exceed ``fast_period``).
    stop_loss_pct:
        Fractional stop distance from the entry price.
    take_profit_pct:
        Fractional take-profit distance from the entry price.
    quantity:
        Default order quantity (risk engine may resize).
    strength:
        Signal confidence in ``[0, 1]``.
    settings:
        Optional :class:`Settings` (constructor params take precedence).

    """

    name = "ma_cross"

    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        fast_period: int = 10,
        slow_period: int = 30,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
        quantity: float = 10.0,
        strength: float = 1.0,
        clock=None,
    ) -> None:
        super().__init__(settings=settings, clock=clock)
        if slow_period <= fast_period:
            raise StrategyError(
                f"slow_period ({slow_period}) must exceed fast_period ({fast_period})"
            )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.quantity = quantity
        self.strength = strength

    # ------------------------------------------------------------------
    # Strategy protocol
    # ------------------------------------------------------------------
    def generate_signals(
        self, bars: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        signals: List[Dict[str, Any]] = []
        for symbol, symbol_bars in bars.items():
            if not symbol_bars:
                continue
            try:
                signal = self._eval_symbol(symbol, symbol_bars)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Signal eval failed for %s: %s", symbol, exc)
                continue
            if signal is not None:
                signals.append(signal.to_dict())
        return signals

    # ------------------------------------------------------------------
    # Per-symbol evaluation
    # ------------------------------------------------------------------
    def _eval_symbol(
        self, symbol: str, symbol_bars: List[Dict[str, Any]]
    ) -> Optional[Signal]:
        closes: List[Optional[float]] = []
        for bar in symbol_bars:
            try:
                closes.append(float(bar.get("close")))
            except (TypeError, ValueError):
                closes.append(None)

        if len(closes) < self.slow_period:
            return None

        fast = sma(closes, self.fast_period)
        slow = sma(closes, self.slow_period)
        if fast[-1] is None or slow[-1] is None:
            return None

        last_bar = symbol_bars[-1]
        price = fast[-1] if fast[-1] is not None else slow[-1]
        if price is None or price <= 0:
            return None

        last_ts = str(last_bar.get("timestamp", ""))
        # Crossover -> buy; crossunder -> sell.
        if crossover(fast, slow)[-1]:
            side = "buy"
        elif crossunder(fast, slow)[-1]:
            side = "sell"
        else:
            return None

        signal_id = f"{self.name}:{symbol.upper()}:{side}:{last_ts}"
        if not self._dedup(signal_id):
            return None

        signal = self._make_signal(
            symbol=symbol,
            side=side,
            price=price,
            quantity=self.quantity,
            stop_loss_pct=self.stop_loss_pct,
            take_profit_pct=self.take_profit_pct,
            signal_id=signal_id,
            strength=self.strength,
            metadata={
                "fast": fast[-1],
                "slow": slow[-1],
                "trigger_bar": last_ts,
            },
        )
        logger.info(
            "MA cross signal %s %s @ %.4f (fast=%.4f slow=%.4f)",
            side, symbol, price, fast[-1], slow[-1],
        )
        return signal
