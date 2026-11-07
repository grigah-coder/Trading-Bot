"""Persistence record dataclasses.

Plain, frozen dataclasses that describe every row the persistence layer can
store. They carry no trading logic and no broker dependency, so they are easy
to construct in tests and to serialize to SQLite / JSON.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class OrderRecord:
    """A persisted order."""

    order_id: str
    client_order_id: str
    symbol: str
    side: str
    quantity: float
    order_type: str = "market"
    status: str = "pending_new"
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    stop_price: Optional[float] = None
    limit_price: Optional[float] = None
    time_in_force: str = "day"
    signal_id: Optional[str] = None
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "order_type": self.order_type,
            "status": self.status,
            "filled_qty": self.filled_qty,
            "avg_fill_price": self.avg_fill_price,
            "stop_price": self.stop_price,
            "limit_price": self.limit_price,
            "time_in_force": self.time_in_force,
            "signal_id": self.signal_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class FillRecord:
    """A persisted execution fill."""

    fill_id: str
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    filled_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fill_id": self.fill_id,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "price": self.price,
            "filled_at": self.filled_at,
        }


@dataclass
class PositionRecord:
    """A persisted position snapshot."""

    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pl: float
    unrealized_plpc: float
    cost_basis: float = 0.0
    side: str = "long"
    recorded_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "qty": self.qty,
            "avg_entry_price": self.avg_entry_price,
            "current_price": self.current_price,
            "market_value": self.market_value,
            "unrealized_pl": self.unrealized_pl,
            "unrealized_plpc": self.unrealized_plpc,
            "cost_basis": self.cost_basis,
            "side": self.side,
            "recorded_at": self.recorded_at,
        }


@dataclass
class SignalRecord:
    """A persisted strategy signal."""

    signal_id: str
    symbol: str
    side: str
    price: float
    quantity: float
    stop_loss_pct: float
    take_profit_pct: float
    strategy_name: str = "unknown"
    strength: float = 1.0
    created_at: str = field(default_factory=_now_iso)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "side": self.side,
            "price": self.price,
            "quantity": self.quantity,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "strategy_name": self.strategy_name,
            "strength": self.strength,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


@dataclass
class RiskEventRecord:
    """A persisted risk-engine event (rejection, halt, breach, ...)."""

    event_type: str  # "rejected", "halted", "breached", "reset"
    symbol: Optional[str] = None
    side: Optional[str] = None
    quantity: Optional[float] = None
    price: Optional[float] = None
    reason: str = ""
    created_at: str = field(default_factory=_now_iso)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "price": self.price,
            "reason": self.reason,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


@dataclass
class AccountSnapshotRecord:
    """A persisted account snapshot."""

    account_id: str
    status: str
    equity: float
    cash: float
    buying_power: float
    portfolio_value: float
    recorded_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "account_id": self.account_id,
            "status": self.status,
            "equity": self.equity,
            "cash": self.cash,
            "buying_power": self.buying_power,
            "portfolio_value": self.portfolio_value,
            "recorded_at": self.recorded_at,
        }


@dataclass
class SystemEventRecord:
    """A persisted system-level event (iteration crash, startup, shutdown)."""

    event_type: str
    detail: str
    timestamp: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.event_type,
            "detail": self.detail,
            "timestamp": self.timestamp,
        }
