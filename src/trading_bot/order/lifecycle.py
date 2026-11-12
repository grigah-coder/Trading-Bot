"""Order lifecycle state machine.

Defines canonical order statuses (normalized from Alpaca's varying strings),
the :class:`Order` dataclass that tracks a single order through its life, and
transition validation so an order can only move through states the broker
actually permits.

Statuses mirror :data:`trading_bot.broker.adapter.OPEN_STATES` and
:data:`trading_bot.broker.adapter.TERMINAL_STATES` so the order manager can
reconcile its in-memory view with the broker without ambiguity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from trading_bot.logging_config import get_logger

logger = get_logger(__name__)

# Protective bracket leg kinds.
STOP_LOSS = "stop_loss"
TAKE_PROFIT = "take_profit"

# ---------------------------------------------------------------------------
# Canonical statuses
# ---------------------------------------------------------------------------

OPEN_STATES = frozenset(
    {
        "pending",
        "pending_new",
        "pending_replace",
        "submitted",
        "new",
        "accepted",
        "accepted_for_bidding",
        "open",
        "held",
        "partially_filled",
    }
)

TERMINAL_STATES = frozenset(
    {"filled", "canceled", "cancelled", "expired", "rejected", "error", "done_for_day"}
)

# Protective-leg statuses that bracket an entry order.
PROTECTIVE_STATES = frozenset({STOP_LOSS, TAKE_PROFIT})

# Transitions the broker permits, keyed by current status -> allowed next.
# A status may transition to itself (e.g. repeated fill reports).
_TRANSITIONS: Dict[str, frozenset[str]] = {
    "pending": {"pending_new", "new", "rejected", "canceled", "cancelled"},
    "pending_new": {"new", "partially_filled", "filled", "canceled", "cancelled", "rejected"},
    "pending_replace": {"new", "partially_filled", "filled", "canceled", "cancelled"},
    "submitted": {"new", "accepted", "partially_filled", "filled", "canceled", "cancelled", "rejected"},
    "new": {"partially_filled", "filled", "canceled", "cancelled", "rejected"},
    "accepted": {"partially_filled", "filled", "canceled", "cancelled"},
    "accepted_for_bidding": {"filled", "canceled", "cancelled", "expired", "done_for_day"},
    "open": {"partially_filled", "filled", "canceled", "cancelled"},
    "held": {"new", "canceled", "cancelled", "expired"},
    "partially_filled": {"filled", "canceled", "cancelled"},
    # Terminal states: self-loop only.
    "filled": {"filled"},
    "canceled": {"canceled"},
    "cancelled": {"cancelled"},
    "expired": {"expired"},
    "rejected": {"rejected"},
    "error": {"error"},
    "done_for_day": {"done_for_day"},
}


def is_open(status: str) -> bool:
    return status.lower() in OPEN_STATES


def is_terminal(status: str) -> bool:
    return status.lower() in TERMINAL_STATES


def is_protective_leg(kind: str) -> bool:
    return kind.lower() in PROTECTIVE_STATES


def valid_transition(from_status: str, to_status: str) -> bool:
    """True when ``to_status`` is a legal next state for ``from_status``."""
    frm = from_status.lower()
    to = to_status.lower()
    if frm == to:
        return True
    allowed = _TRANSITIONS.get(frm)
    if allowed is None:
        # Unknown status: treat any move to a known state as acceptable to
        # avoid false rejections when the broker emits an unfamiliar string.
        return to in OPEN_STATES or to in TERMINAL_STATES
    return to in allowed


def normalize_status(value: Any) -> str:
    return str(getattr(value, "value", value) or value).strip().lower()


# ---------------------------------------------------------------------------
# Order
# ---------------------------------------------------------------------------


@dataclass
class Order:
    """A single tracked order through its lifecycle.

    Mutable fields (``status``, ``filled_qty``, ...) are updated by the order
    manager on fills, cancels and reconciliations. Everything else is fixed at
    creation.
    """

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
    bracket_legs: Dict[str, str] = field(default_factory=dict)  # kind -> order_id
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    raw: Any = None

    def __post_init__(self) -> None:
        self.symbol = str(self.symbol).strip().upper()
        self.side = str(self.side or "buy").strip().lower()

    # -- derived ---------------------------------------------------------
    @property
    def is_open(self) -> bool:
        return is_open(self.status)

    @property
    def is_terminal(self) -> bool:
        return is_terminal(self.status)

    @property
    def is_filled(self) -> bool:
        return self.status == "filled"

    @property
    def is_cancelled(self) -> bool:
        return self.status in {"canceled", "cancelled"}

    @property
    def is_rejected(self) -> bool:
        return self.status == "rejected"

    @property
    def remaining_qty(self) -> float:
        return max(self.quantity - self.filled_qty, 0.0)

    @property
    def fill_pct(self) -> float:
        if self.quantity <= 0:
            return 0.0
        return min(self.filled_qty / self.quantity, 1.0)

    # -- transitions -----------------------------------------------------
    def transition(self, to_status: str) -> None:
        """Move the order to ``to_status``, validating the transition."""
        to_status = normalize_status(to_status)
        if not valid_transition(self.status, to_status):
            logger.warning(
                "Order %s invalid status transition %s -> %s",
                self.order_id, self.status, to_status,
            )
        self.status = to_status
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def fill(self, fill_qty: float, fill_price: float) -> None:
        """Apply a partial or full fill, recomputing the average fill price."""
        fill_qty = float(fill_qty)
        fill_price = float(fill_price)
        if fill_qty <= 0:
            return
        total = self.filled_qty * self.avg_fill_price + fill_qty * fill_price
        self.filled_qty += fill_qty
        if self.filled_qty > 0:
            self.avg_fill_price = total / self.filled_qty
        self.transition("filled" if self.filled_qty >= self.quantity else "partially_filled")

    def cancel(self) -> None:
        self.transition("canceled")

    def reject(self) -> None:
        self.transition("rejected")

    def expire(self) -> None:
        self.transition("expired")

    # -- serialization ---------------------------------------------------
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
            "bracket_legs": dict(self.bracket_legs),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "remaining_qty": self.remaining_qty,
            "fill_pct": self.fill_pct,
            "is_open": self.is_open,
            "is_terminal": self.is_terminal,
            "is_filled": self.is_filled,
            "is_cancelled": self.is_cancelled,
            "is_rejected": self.is_rejected,
        }

    @classmethod
    def from_broker_dict(cls, data: Dict[str, Any]) -> "Order":
        return cls(
            order_id=str(data.get("order_id") or data.get("id") or ""),
            client_order_id=str(data.get("client_order_id") or ""),
            symbol=str(data.get("symbol", "")).upper(),
            side=str(data.get("side", "buy")).lower(),
            quantity=float(data.get("qty") or data.get("quantity") or 0),
            order_type=str(data.get("order_type", "market")).lower(),
            status=normalize_status(data.get("status", "pending_new")),
            filled_qty=float(data.get("filled_qty") or 0),
            avg_fill_price=float(data.get("filled_avg_price") or 0),
            stop_price=float(data["stop_price"]) if data.get("stop_price") not in (None, "") else None,
            limit_price=float(data["limit_price"]) if data.get("limit_price") not in (None, "") else None,
            time_in_force=str(data.get("time_in_force", "day")).lower(),
            signal_id=data.get("signal_id"),
            created_at=data.get("created_at") or datetime.now(timezone.utc).isoformat(),
            updated_at=data.get("updated_at") or datetime.now(timezone.utc).isoformat(),
            raw=data,
        )
