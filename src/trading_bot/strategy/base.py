"""Strategy base classes and the canonical ``Signal`` type.

Defines the :class:`Signal` dataclass (the contract between strategies, the
risk engine and the order manager) and the :class:`Strategy` base class that
concrete strategies subclass. Concrete strategies live in
:mod:`trading_bot.strategy.signals`.

Signal shape (dict form, consumed by the risk engine and order manager)::

    {
        "symbol": "AAPL",
        "side": "buy",                 # "buy" | "sell"
        "price": 150.25,               # reference price at signal time
        "quantity": 10.0,              # desired qty (risk engine may resize)
        "stop_loss_pct": 0.012,        # fractional stop distance from price
        "take_profit_pct": 0.04,       # fractional take-profit distance
        "signal_id": "sig-<uuid>",     # idempotency key
        "timestamp": "...iso...",      # tz-aware UTC
        "strategy_name": "ma_cross",   # originating strategy
        "strength": 1.0,               # 0..1 confidence
        "metadata": {},                # strategy-specific extras
    }
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from trading_bot.exceptions import StrategyError
from trading_bot.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Signal:
    """A trading signal produced by a :class:`Strategy`.

    Frozen so strategies cannot mutate a signal after emission (protects
    against downstream look-ahead bias and double-counting).
    """

    symbol: str
    side: str
    price: float
    quantity: float
    stop_loss_pct: float
    take_profit_pct: float
    signal_id: str
    timestamp: datetime
    strategy_name: str = "unknown"
    strength: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        side = self.side.strip().lower()
        if side not in {"buy", "sell"}:
            raise StrategyError(f"invalid side {self.side!r}; use 'buy' or 'sell'")
        object.__setattr__(self, "side", side)
        if self.price <= 0:
            raise StrategyError(f"price must be positive, got {self.price!r}")
        if not (0.0 <= self.strength <= 1.0):
            raise StrategyError(
                f"strength must be in [0, 1], got {self.strength!r}"
            )
        if not self.symbol:
            raise StrategyError("symbol must not be empty")
        object.__setattr__(self, "symbol", str(self.symbol).upper())
        if not self.signal_id:
            object.__setattr__(self, "signal_id", f"sig-{uuid.uuid4().hex}")

    def to_dict(self) -> Dict[str, Any]:
        """Return the engine/risk-compatible dict form of this signal."""
        return {
            "symbol": self.symbol,
            "side": self.side,
            "price": self.price,
            "quantity": self.quantity,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "signal_id": self.signal_id,
            "timestamp": self.timestamp.isoformat(),
            "strategy_name": self.strategy_name,
            "strength": self.strength,
            "metadata": dict(self.metadata),
        }


class Strategy:
    """Base class for all strategies.

    Subclasses implement :meth:`generate_signals`. A strategy is stateless
    except for an optional internal ``_seen_ids`` set used for duplicate
    suppression within a single run.
    """

    name: str = "base_strategy"

    def __init__(self, settings=None, clock=None) -> None:
        self.settings = settings
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._seen_ids: set[str] = set()

    # ------------------------------------------------------------------
    # Protocol surface
    # ------------------------------------------------------------------
    def generate_signals(
        self, bars: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Generate signals from the latest market data.

        ``bars`` maps symbol -> list of validated bar dicts (oldest-first).
        Returns a list of signal dicts. Subclasses override this.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _now(self) -> datetime:
        return self._clock()

    def _new_signal_id(self) -> str:
        return f"sig-{uuid.uuid4().hex}"

    def _dedup(self, signal_id: str) -> bool:
        """Return True if the signal id is new (not yet seen)."""
        if signal_id in self._seen_ids:
            return False
        self._seen_ids.add(signal_id)
        return True

    def _latest_close(self, symbol_bars: List[Dict[str, Any]]) -> Optional[float]:
        """Return the most recent close price, or None when no bars."""
        if not symbol_bars:
            return None
        try:
            return float(symbol_bars[-1].get("close"))
        except (TypeError, ValueError):
            return None

    def _make_signal(
        self,
        *,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
        stop_loss_pct: float,
        take_profit_pct: float,
        signal_id: Optional[str] = None,
        strength: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Signal:
        return Signal(
            symbol=str(symbol).upper(),
            side=side,
            price=float(price),
            quantity=float(quantity),
            stop_loss_pct=float(stop_loss_pct),
            take_profit_pct=float(take_profit_pct),
            signal_id=signal_id or self._new_signal_id(),
            timestamp=self._now(),
            strategy_name=self.name,
            strength=strength,
            metadata=dict(metadata or {}),
        )
