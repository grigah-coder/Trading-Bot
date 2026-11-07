"""SQLite-backed persistence models for the trading bot.

Tables
------
* ``orders``            - every order the bot creates or the broker reports.
* ``executions``        - one row per partial/full fill.
* ``positions``         - reconciled broker positions.
* ``trade_history``     - realised P/L per closed trade (FIFO matched lots).
* ``signals``           - strategy signals emitted by the bot.
* ``risk_events``       - risk approvals, rejections and circuit-breaker trips.
* ``account_snapshots`` - periodic snapshots of the broker account.
* ``strategy_versions`` - immutable snapshots of the active strategy config.
* ``application_events``- lifecycle and recovery events.

All timestamps are stored as ISO-8601 strings in UTC.  The schema is owned by
``storage.schema`` so migrations can evolve independently of the repositories.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping


# ---------------------------------------------------------------------------
# Row dataclasses (plain, hashable where useful, no ORM coupling)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class OrderRow:
    order_id: str
    client_order_id: str
    symbol: str
    side: str
    order_type: str
    time_in_force: str
    status: str
    quantity: Decimal
    filled_quantity: Decimal
    limit_price: Decimal | None
    filled_average_price: Decimal | None
    submitted_at: str
    updated_at: str | None
    source: str
    raw_status: str | None = None


@dataclass(frozen=True, slots=True)
class ExecutionRow:
    execution_id: str
    order_id: str
    client_order_id: str
    symbol: str
    side: str
    quantity: Decimal
    price: Decimal
    filled_at: str
    is_partial: bool


@dataclass(frozen=True, slots=True)
class PositionRow:
    position_id: str
    symbol: str
    quantity: Decimal
    side: str
    average_entry_price: Decimal
    market_value: Decimal
    cost_basis: Decimal
    unrealized_pl: Decimal
    unrealized_pl_fraction: Decimal
    observed_at: str
    source: str


@dataclass(frozen=True, slots=True)
class TradeHistoryRow:
    trade_id: str
    symbol: str
    side: str
    quantity: Decimal
    entry_price: Decimal
    exit_price: Decimal
    realized_pl: Decimal
    opened_at: str
    closed_at: str
    lot_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SignalRow:
    signal_id: str
    symbol: str
    direction: str
    entry_reference: Decimal
    stop_loss_reference: Decimal | None
    take_profit_reference: Decimal | None
    timestamp: str
    strategy_id: str
    strength: Decimal | None
    reason: str


@dataclass(frozen=True, slots=True)
class RiskEventRow:
    event_id: str
    timestamp: str
    kind: str
    code: str
    message: str
    symbol: str | None
    client_order_id: str | None
    context: str | None = None


@dataclass(frozen=True, slots=True)
class AccountSnapshotRow:
    snapshot_id: str
    observed_at: str
    account_id: str
    status: str
    currency: str
    cash: Decimal
    buying_power: Decimal
    equity: Decimal
    portfolio_value: Decimal
    day_trade_count: int
    trading_blocked: bool


@dataclass(frozen=True, slots=True)
class StrategyVersionRow:
    version_id: str
    activated_at: str
    name: str
    parameters: str
    source: str


@dataclass(frozen=True, slots=True)
class ApplicationEventRow:
    event_id: str
    timestamp: str
    level: str
    phase: str
    message: str
    context: str | None = None


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _json_dumps(value: Any) -> str:
    import json
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def context_json(context: Mapping[str, Any] | None) -> str | None:
    if context is None:
        return None
    return _json_dumps(context)


def parse_context(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    import json
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def decimal_or_none(value: Any) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None


def decimal_or_zero(value: Any) -> Decimal:
    parsed = decimal_or_none(value)
    return parsed if parsed is not None else Decimal("0")


# Late import to keep the module importable without the decimal exception
from decimal import InvalidOperation  # noqa: E402  (intentional late import)
