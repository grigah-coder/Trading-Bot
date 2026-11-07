"""Repository pattern for SQLite persistence.

Each repository owns one table, validates every field before writing, stores
monetary values as decimal strings, and returns immutable row dataclasses.
Repositories never delete historical trading records; duplicate identifiers are
handled explicitly and reported instead of silently replacing data.

Transactions are provided by :class:`storage.database.TradingDatabase`.
Repositories use nested transactions (SQLite savepoints) so callers can compose
several repository writes into one atomic unit of work.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, TYPE_CHECKING

from storage.models import (
    AccountSnapshotRow,
    ApplicationEventRow,
    ExecutionRow,
    OrderRow,
    PositionRow,
    RiskEventRow,
    SignalRow,
    StrategyVersionRow,
    TradeHistoryRow,
    context_json,
)
from storage.validation import (
    DataValidationError,
    DuplicateRecordError,
    StorageError,
    normalize_side,
    normalize_status,
    optional_utc_timestamp,
    utc_now_timestamp,
    utc_timestamp,
    validate_boolean,
    validate_decimal,
    validate_integer,
    validate_symbol,
    validate_text,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from storage.database import TradingDatabase


# Orders in any of these states are still working at the broker.
OPEN_ORDER_STATUSES = frozenset(
    {
        "new",
        "accepted",
        "accepted_for_bidding",
        "pending_new",
        "pending_cancel",
        "pending_replace",
        "pending_review",
        "partially_filled",
        "calculated",
        "held",
        "stopped",
        "suspended",
    }
)

TERMINAL_ORDER_STATUSES = frozenset(
    {
        "filled",
        "canceled",
        "cancelled",
        "expired",
        "rejected",
        "done_for_day",
        "replaced",
    }
)


def _row_decimal(value: Any, name: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except Exception as exc:  # pragma: no cover - corrupt database guard
        raise StorageError(f"Stored {name} is not numeric") from exc
    if not parsed.is_finite():
        raise StorageError(f"Stored {name} is not finite")
    return parsed


def _optional_row_decimal(value: Any, name: str) -> Decimal | None:
    return None if value is None else _row_decimal(value, name)


def _bool(value: Any) -> bool:
    return bool(int(value))


class BaseRepository:
    """Shared transaction and query helpers."""

    def __init__(self, database: "TradingDatabase") -> None:
        self._db = database

    def count(self) -> int:
        row = self._db.fetchone(f"SELECT COUNT(*) FROM {self.table}")
        return int(row[0]) if row is not None else 0


class OrderRepository(BaseRepository):
    table = "orders"

    _COLUMNS = (
        "order_id",
        "client_order_id",
        "symbol",
        "side",
        "order_type",
        "time_in_force",
        "status",
        "quantity",
        "filled_quantity",
        "limit_price",
        "filled_average_price",
        "submitted_at",
        "updated_at",
        "source",
        "raw_status",
    )

    @classmethod
    def _normalize(cls, order: OrderRow) -> OrderRow:
        return OrderRow(
            order_id=validate_text(order.order_id, "order_id", max_length=128),
            client_order_id=validate_text(
                order.client_order_id, "client_order_id", max_length=128
            ),
            symbol=validate_symbol(order.symbol),
            side=normalize_side(order.side),
            order_type=validate_text(order.order_type, "order_type", max_length=64).lower(),
            time_in_force=validate_text(
                order.time_in_force, "time_in_force", max_length=16
            ).lower(),
            status=normalize_status(order.status),
            quantity=validate_decimal(order.quantity, "quantity", positive=True),
            filled_quantity=validate_decimal(
                order.filled_quantity, "filled_quantity", non_negative=True
            ),
            limit_price=validate_decimal(
                order.limit_price, "limit_price", positive=True, allow_none=True
            ),
            filled_average_price=validate_decimal(
                order.filled_average_price,
                "filled_average_price",
                positive=True,
                allow_none=True,
            ),
            submitted_at=utc_timestamp(order.submitted_at, "submitted_at"),
            updated_at=optional_utc_timestamp(order.updated_at, "updated_at"),
            source=validate_text(order.source, "source", max_length=64),
            raw_status=(
                None
                if order.raw_status is None
                else validate_text(order.raw_status, "raw_status", max_length=128)
            ),
        )

    @classmethod
    def _from_row(cls, row: sqlite3.Row) -> OrderRow:
        return OrderRow(
            order_id=row["order_id"],
            client_order_id=row["client_order_id"],
            symbol=row["symbol"],
            side=row["side"],
            order_type=row["order_type"],
            time_in_force=row["time_in_force"],
            status=row["status"],
            quantity=_row_decimal(row["quantity"], "quantity"),
            filled_quantity=_row_decimal(row["filled_quantity"], "filled_quantity"),
            limit_price=_optional_row_decimal(row["limit_price"], "limit_price"),
            filled_average_price=_optional_row_decimal(
                row["filled_average_price"], "filled_average_price"
            ),
            submitted_at=row["submitted_at"],
            updated_at=row["updated_at"],
            source=row["source"],
            raw_status=row["raw_status"],
        )

    def upsert(self, order: OrderRow) -> OrderRow:
        """Insert an order or update it in place by ``order_id``."""
        normalized = self._normalize(order)
        placeholders = ", ".join("?" for _ in self._COLUMNS)
        updates = ", ".join(
            f"{column}=excluded.{column}"
            for column in self._COLUMNS
            if column != "order_id"
        )
        sql = (
            f"INSERT INTO orders ({', '.join(self._COLUMNS)}) VALUES ({placeholders}) "
            f"ON CONFLICT(order_id) DO UPDATE SET {updates}"
        )
        try:
            with self._db.transaction():
                self._db.execute(sql, tuple(getattr(normalized, column) for column in self._COLUMNS))
        except sqlite3.IntegrityError as exc:
            raise DuplicateRecordError(
                "order conflicts with an existing unique identifier",
                context={
                    "order_id": normalized.order_id,
                    "client_order_id": normalized.client_order_id,
                },
            ) from exc
        return normalized

    def insert(self, order: OrderRow) -> OrderRow:
        """Alias for :meth:`upsert` kept for symmetry with other repositories."""
        return self.upsert(order)

    def get(self, order_id: str) -> OrderRow | None:
        row = self._db.fetchone(
            "SELECT * FROM orders WHERE order_id = ?", (validate_text(order_id, "order_id"),)
        )
        return None if row is None else self._from_row(row)

    def get_by_client_order_id(self, client_order_id: str) -> OrderRow | None:
        row = self._db.fetchone(
            "SELECT * FROM orders WHERE client_order_id = ?",
            (validate_text(client_order_id, "client_order_id"),),
        )
        return None if row is None else self._from_row(row)

    def update_status(
        self,
        order_id: str,
        status: str,
        *,
        filled_quantity: Decimal | None = None,
        filled_average_price: Decimal | None = None,
        raw_status: str | None = None,
        updated_at: str | None = None,
    ) -> bool:
        """Update mutable broker state without touching immutable order data."""
        identifier = validate_text(order_id, "order_id")
        normalized_status = normalize_status(status)
        values: dict[str, Any] = {
            "status": normalized_status,
            "updated_at": utc_timestamp(updated_at or utc_now_timestamp(), "updated_at"),
        }
        if filled_quantity is not None:
            values["filled_quantity"] = validate_decimal(
                filled_quantity, "filled_quantity", non_negative=True
            )
        if filled_average_price is not None:
            values["filled_average_price"] = validate_decimal(
                filled_average_price, "filled_average_price", positive=True
            )
        if raw_status is not None:
            values["raw_status"] = validate_text(raw_status, "raw_status", max_length=128)
        assignments = ", ".join(f"{column} = ?" for column in values)
        with self._db.transaction():
            cursor = self._db.execute(
                f"UPDATE orders SET {assignments} WHERE order_id = ?",
                tuple(values.values()) + (identifier,),
            )
        return cursor.rowcount > 0

    def list_open(self) -> tuple[OrderRow, ...]:
        placeholders = ", ".join("?" for _ in OPEN_ORDER_STATUSES)
        rows = self._db.fetchall(
            f"SELECT * FROM orders WHERE status IN ({placeholders}) "
            "ORDER BY submitted_at ASC, order_id ASC",
            tuple(sorted(OPEN_ORDER_STATUSES)),
        )
        return tuple(self._from_row(row) for row in rows)

    def list_for_symbol(self, symbol: str, *, limit: int | None = None) -> tuple[OrderRow, ...]:
        parameters: list[Any] = [validate_symbol(symbol)]
        sql = "SELECT * FROM orders WHERE symbol = ? ORDER BY submitted_at DESC"
        if limit is not None:
            sql += " LIMIT ?"
            parameters.append(validate_integer(limit, "limit", minimum=1))
        return tuple(self._from_row(row) for row in self._db.fetchall(sql, tuple(parameters)))

    def list_submitted_between(self, start: str, end: str) -> tuple[OrderRow, ...]:
        rows = self._db.fetchall(
            "SELECT * FROM orders WHERE submitted_at >= ? AND submitted_at < ? "
            "ORDER BY submitted_at ASC",
            (utc_timestamp(start, "start"), utc_timestamp(end, "end")),
        )
        return tuple(self._from_row(row) for row in rows)

    def latest(self) -> OrderRow | None:
        row = self._db.fetchone(
            "SELECT * FROM orders ORDER BY submitted_at DESC LIMIT 1"
        )
        return None if row is None else self._from_row(row)


class ExecutionRepository(BaseRepository):
    table = "executions"

    _COLUMNS = (
        "execution_id",
        "order_id",
        "client_order_id",
        "symbol",
        "side",
        "quantity",
        "price",
        "filled_at",
        "is_partial",
    )

    @classmethod
    def _normalize(cls, execution: ExecutionRow) -> ExecutionRow:
        return ExecutionRow(
            execution_id=validate_text(execution.execution_id, "execution_id", max_length=128),
            order_id=validate_text(execution.order_id, "order_id", max_length=128),
            client_order_id=validate_text(
                execution.client_order_id, "client_order_id", max_length=128
            ),
            symbol=validate_symbol(execution.symbol),
            side=normalize_side(execution.side),
            quantity=validate_decimal(execution.quantity, "quantity", positive=True),
            price=validate_decimal(execution.price, "price", positive=True),
            filled_at=utc_timestamp(execution.filled_at, "filled_at"),
            is_partial=validate_boolean(execution.is_partial, "is_partial"),
        )

    @classmethod
    def _from_row(cls, row: sqlite3.Row) -> ExecutionRow:
        return ExecutionRow(
            execution_id=row["execution_id"],
            order_id=row["order_id"],
            client_order_id=row["client_order_id"],
            symbol=row["symbol"],
            side=row["side"],
            quantity=_row_decimal(row["quantity"], "quantity"),
            price=_row_decimal(row["price"], "price"),
            filled_at=row["filled_at"],
            is_partial=_bool(row["is_partial"]),
        )

    def insert(self, execution: ExecutionRow) -> bool:
        """Insert a fill once.

        Returns ``True`` when a new row was written, ``False`` for an exact
        duplicate (broker replays are normal).  A reused execution id carrying
        different data is corrupting and raises :class:`DuplicateRecordError`.
        """
        normalized = self._normalize(execution)
        existing = self.get(normalized.execution_id)
        if existing is not None:
            if existing == normalized:
                return False
            raise DuplicateRecordError(
                "execution_id already exists with different data",
                context={"execution_id": normalized.execution_id},
            )
        placeholders = ", ".join("?" for _ in self._COLUMNS)
        try:
            with self._db.transaction():
                self._db.execute(
                    f"INSERT INTO executions ({', '.join(self._COLUMNS)}) "
                    f"VALUES ({placeholders})",
                    tuple(getattr(normalized, column) for column in self._COLUMNS),
                )
        except sqlite3.IntegrityError as exc:
            current = self.get(normalized.execution_id)
            if current == normalized:
                return False
            if self._db.fetchone(
                "SELECT 1 FROM orders WHERE order_id = ?", (normalized.order_id,)
            ) is None:
                raise DataValidationError(
                    "execution references an unknown order",
                    context={"order_id": normalized.order_id},
                ) from exc
            raise DuplicateRecordError(
                "execution conflicts with an existing record",
                context={"execution_id": normalized.execution_id},
            ) from exc
        return True

    record = insert

    def get(self, execution_id: str) -> ExecutionRow | None:
        row = self._db.fetchone(
            "SELECT * FROM executions WHERE execution_id = ?",
            (validate_text(execution_id, "execution_id"),),
        )
        return None if row is None else self._from_row(row)

    def list_for_order(self, order_id: str) -> tuple[ExecutionRow, ...]:
        rows = self._db.fetchall(
            "SELECT * FROM executions WHERE order_id = ? ORDER BY filled_at ASC",
            (validate_text(order_id, "order_id"),),
        )
        return tuple(self._from_row(row) for row in rows)

    def list_between(self, start: str, end: str) -> tuple[ExecutionRow, ...]:
        rows = self._db.fetchall(
            "SELECT * FROM executions WHERE filled_at >= ? AND filled_at < ? "
            "ORDER BY filled_at ASC",
            (utc_timestamp(start, "start"), utc_timestamp(end, "end")),
        )
        return tuple(self._from_row(row) for row in rows)


class PositionRepository(BaseRepository):
    table = "positions"

    _COLUMNS = (
        "position_id",
        "symbol",
        "quantity",
        "side",
        "average_entry_price",
        "market_value",
        "cost_basis",
        "unrealized_pl",
        "unrealized_pl_fraction",
        "observed_at",
        "source",
    )

    @classmethod
    def _normalize(cls, position: PositionRow) -> PositionRow:
        return PositionRow(
            position_id=validate_text(position.position_id, "position_id", max_length=128),
            symbol=validate_symbol(position.symbol),
            quantity=validate_decimal(position.quantity, "quantity"),
            side=validate_text(position.side, "side", max_length=16).lower(),
            average_entry_price=validate_decimal(
                position.average_entry_price, "average_entry_price", non_negative=True
            ),
            market_value=validate_decimal(position.market_value, "market_value"),
            cost_basis=validate_decimal(position.cost_basis, "cost_basis"),
            unrealized_pl=validate_decimal(position.unrealized_pl, "unrealized_pl"),
            unrealized_pl_fraction=validate_decimal(
                position.unrealized_pl_fraction, "unrealized_pl_fraction"
            ),
            observed_at=utc_timestamp(position.observed_at, "observed_at"),
            source=validate_text(position.source, "source", max_length=64),
        )

    @classmethod
    def _from_row(cls, row: sqlite3.Row) -> PositionRow:
        return PositionRow(
            position_id=row["position_id"],
            symbol=row["symbol"],
            quantity=_row_decimal(row["quantity"], "quantity"),
            side=row["side"],
            average_entry_price=_row_decimal(
                row["average_entry_price"], "average_entry_price"
            ),
            market_value=_row_decimal(row["market_value"], "market_value"),
            cost_basis=_row_decimal(row["cost_basis"], "cost_basis"),
            unrealized_pl=_row_decimal(row["unrealized_pl"], "unrealized_pl"),
            unrealized_pl_fraction=_row_decimal(
                row["unrealized_pl_fraction"], "unrealized_pl_fraction"
            ),
            observed_at=row["observed_at"],
            source=row["source"],
        )

    def record(self, position: PositionRow) -> bool:
        """Store an immutable position observation."""
        normalized = self._normalize(position)
        existing = self.get(normalized.position_id)
        if existing is not None:
            if existing == normalized:
                return False
            raise DuplicateRecordError(
                "position_id already exists with different data",
                context={"position_id": normalized.position_id},
            )
        placeholders = ", ".join("?" for _ in self._COLUMNS)
        with self._db.transaction():
            self._db.execute(
                f"INSERT INTO positions ({', '.join(self._COLUMNS)}) "
                f"VALUES ({placeholders})",
                tuple(getattr(normalized, column) for column in self._COLUMNS),
            )
        return True

    insert = record

    def get(self, position_id: str) -> PositionRow | None:
        row = self._db.fetchone(
            "SELECT * FROM positions WHERE position_id = ?",
            (validate_text(position_id, "position_id"),),
        )
        return None if row is None else self._from_row(row)

    def latest_by_symbol(self) -> tuple[PositionRow, ...]:
        rows = self._db.fetchall(
            """
            SELECT * FROM (
                SELECT p.*, ROW_NUMBER() OVER (
                    PARTITION BY symbol ORDER BY observed_at DESC, position_id DESC
                ) AS rank
                FROM positions p
            ) WHERE rank = 1 ORDER BY symbol ASC
            """
        )
        return tuple(self._from_row(row) for row in rows)

    def latest_for_symbol(self, symbol: str) -> PositionRow | None:
        row = self._db.fetchone(
            "SELECT * FROM positions WHERE symbol = ? "
            "ORDER BY observed_at DESC, position_id DESC LIMIT 1",
            (validate_symbol(symbol),),
        )
        return None if row is None else self._from_row(row)

    def history(self, symbol: str, *, limit: int | None = None) -> tuple[PositionRow, ...]:
        parameters: list[Any] = [validate_symbol(symbol)]
        sql = "SELECT * FROM positions WHERE symbol = ? ORDER BY observed_at DESC"
        if limit is not None:
            sql += " LIMIT ?"
            parameters.append(validate_integer(limit, "limit", minimum=1))
        return tuple(self._from_row(row) for row in self._db.fetchall(sql, tuple(parameters)))


class TradeHistoryRepository(BaseRepository):
    table = "trade_history"

    _COLUMNS = (
        "trade_id",
        "symbol",
        "side",
        "quantity",
        "entry_price",
        "exit_price",
        "realized_pl",
        "opened_at",
        "closed_at",
        "lot_ids",
    )

    @classmethod
    def _normalize(cls, trade: TradeHistoryRow) -> TradeHistoryRow:
        return TradeHistoryRow(
            trade_id=validate_text(trade.trade_id, "trade_id", max_length=128),
            symbol=validate_symbol(trade.symbol),
            side=normalize_side(trade.side),
            quantity=validate_decimal(trade.quantity, "quantity", positive=True),
            entry_price=validate_decimal(trade.entry_price, "entry_price", positive=True),
            exit_price=validate_decimal(trade.exit_price, "exit_price", positive=True),
            realized_pl=validate_decimal(trade.realized_pl, "realized_pl"),
            opened_at=utc_timestamp(trade.opened_at, "opened_at"),
            closed_at=utc_timestamp(trade.closed_at, "closed_at"),
            lot_ids=tuple(
                validate_text(item, "lot_id", max_length=128) for item in trade.lot_ids
            ),
        )

    @classmethod
    def _from_row(cls, row: sqlite3.Row) -> TradeHistoryRow:
        try:
            lot_ids = tuple(json.loads(row["lot_ids"])) if row["lot_ids"] else ()
        except (TypeError, ValueError):
            lot_ids = ()
        return TradeHistoryRow(
            trade_id=row["trade_id"],
            symbol=row["symbol"],
            side=row["side"],
            quantity=_row_decimal(row["quantity"], "quantity"),
            entry_price=_row_decimal(row["entry_price"], "entry_price"),
            exit_price=_row_decimal(row["exit_price"], "exit_price"),
            realized_pl=_row_decimal(row["realized_pl"], "realized_pl"),
            opened_at=row["opened_at"],
            closed_at=row["closed_at"],
            lot_ids=tuple(str(item) for item in lot_ids),
        )

    def insert(self, trade: TradeHistoryRow) -> bool:
        normalized = self._normalize(trade)
        existing = self.get(normalized.trade_id)
        if existing is not None:
            if existing == normalized:
                return False
            raise DuplicateRecordError(
                "trade_id already exists with different data",
                context={"trade_id": normalized.trade_id},
            )
        parameters = list(getattr(normalized, column) for column in self._COLUMNS)
        parameters[-1] = json.dumps(list(normalized.lot_ids), separators=(",", ":"))
        placeholders = ", ".join("?" for _ in self._COLUMNS)
        with self._db.transaction():
            self._db.execute(
                f"INSERT INTO trade_history ({', '.join(self._COLUMNS)}) "
                f"VALUES ({placeholders})",
                tuple(parameters),
            )
        return True

    record = insert

    def get(self, trade_id: str) -> TradeHistoryRow | None:
        row = self._db.fetchone(
            "SELECT * FROM trade_history WHERE trade_id = ?",
            (validate_text(trade_id, "trade_id"),),
        )
        return None if row is None else self._from_row(row)

    def list_between(self, start: str, end: str) -> tuple[TradeHistoryRow, ...]:
        rows = self._db.fetchall(
            "SELECT * FROM trade_history WHERE closed_at >= ? AND closed_at < ? "
            "ORDER BY closed_at ASC",
            (utc_timestamp(start, "start"), utc_timestamp(end, "end")),
        )
        return tuple(self._from_row(row) for row in rows)

    def list_for_symbol(self, symbol: str) -> tuple[TradeHistoryRow, ...]:
        rows = self._db.fetchall(
            "SELECT * FROM trade_history WHERE symbol = ? ORDER BY closed_at ASC",
            (validate_symbol(symbol),),
        )
        return tuple(self._from_row(row) for row in rows)


class SignalRepository(BaseRepository):
    table = "signals"

    _COLUMNS = (
        "signal_id",
        "symbol",
        "direction",
        "entry_reference",
        "stop_loss_reference",
        "take_profit_reference",
        "timestamp",
        "strategy_id",
        "strength",
        "reason",
    )

    @classmethod
    def _normalize(cls, signal: SignalRow) -> SignalRow:
        return SignalRow(
            signal_id=validate_text(signal.signal_id, "signal_id", max_length=128),
            symbol=validate_symbol(signal.symbol),
            direction=validate_text(signal.direction, "direction", max_length=16).lower(),
            entry_reference=validate_decimal(
                signal.entry_reference, "entry_reference", non_negative=True
            ),
            stop_loss_reference=validate_decimal(
                signal.stop_loss_reference,
                "stop_loss_reference",
                non_negative=True,
                allow_none=True,
            ),
            take_profit_reference=validate_decimal(
                signal.take_profit_reference,
                "take_profit_reference",
                non_negative=True,
                allow_none=True,
            ),
            timestamp=utc_timestamp(signal.timestamp, "timestamp"),
            strategy_id=validate_text(signal.strategy_id, "strategy_id", max_length=128),
            strength=validate_decimal(signal.strength, "strength", allow_none=True),
            reason=validate_text(signal.reason, "reason"),
        )

    @classmethod
    def _from_row(cls, row: sqlite3.Row) -> SignalRow:
        return SignalRow(
            signal_id=row["signal_id"],
            symbol=row["symbol"],
            direction=row["direction"],
            entry_reference=_row_decimal(row["entry_reference"], "entry_reference"),
            stop_loss_reference=_optional_row_decimal(
                row["stop_loss_reference"], "stop_loss_reference"
            ),
            take_profit_reference=_optional_row_decimal(
                row["take_profit_reference"], "take_profit_reference"
            ),
            timestamp=row["timestamp"],
            strategy_id=row["strategy_id"],
            strength=_optional_row_decimal(row["strength"], "strength"),
            reason=row["reason"],
        )

    def insert(self, signal: SignalRow) -> bool:
        normalized = self._normalize(signal)
        existing = self.get(normalized.signal_id)
        if existing is not None:
            if existing == normalized:
                return False
            raise DuplicateRecordError(
                "signal_id already exists with different data",
                context={"signal_id": normalized.signal_id},
            )
        placeholders = ", ".join("?" for _ in self._COLUMNS)
        with self._db.transaction():
            self._db.execute(
                f"INSERT INTO signals ({', '.join(self._COLUMNS)}) "
                f"VALUES ({placeholders})",
                tuple(getattr(normalized, column) for column in self._COLUMNS),
            )
        return True

    record = insert

    def get(self, signal_id: str) -> SignalRow | None:
        row = self._db.fetchone(
            "SELECT * FROM signals WHERE signal_id = ?",
            (validate_text(signal_id, "signal_id"),),
        )
        return None if row is None else self._from_row(row)

    def list_since(self, since: str, *, limit: int | None = None) -> tuple[SignalRow, ...]:
        parameters: list[Any] = [utc_timestamp(since, "since")]
        sql = "SELECT * FROM signals WHERE timestamp >= ? ORDER BY timestamp ASC"
        if limit is not None:
            sql += " LIMIT ?"
            parameters.append(validate_integer(limit, "limit", minimum=1))
        return tuple(self._from_row(row) for row in self._db.fetchall(sql, tuple(parameters)))

    def latest(self) -> SignalRow | None:
        row = self._db.fetchone(
            "SELECT * FROM signals ORDER BY timestamp DESC LIMIT 1"
        )
        return None if row is None else self._from_row(row)


class RiskEventRepository(BaseRepository):
    table = "risk_events"

    _COLUMNS = (
        "event_id",
        "timestamp",
        "kind",
        "code",
        "message",
        "symbol",
        "client_order_id",
        "context",
    )

    @classmethod
    def _normalize(cls, event: RiskEventRow) -> RiskEventRow:
        return RiskEventRow(
            event_id=validate_text(event.event_id, "event_id", max_length=128),
            timestamp=utc_timestamp(event.timestamp, "timestamp"),
            kind=validate_text(event.kind, "kind", max_length=64).lower(),
            code=validate_text(event.code, "code", max_length=64).lower(),
            message=validate_text(event.message, "message"),
            symbol=None if event.symbol is None else validate_symbol(event.symbol),
            client_order_id=(
                None
                if event.client_order_id is None
                else validate_text(event.client_order_id, "client_order_id", max_length=128)
            ),
            context=event.context,
        )

    @classmethod
    def _from_row(cls, row: sqlite3.Row) -> RiskEventRow:
        return RiskEventRow(
            event_id=row["event_id"],
            timestamp=row["timestamp"],
            kind=row["kind"],
            code=row["code"],
            message=row["message"],
            symbol=row["symbol"],
            client_order_id=row["client_order_id"],
            context=row["context"],
        )

    def insert(self, event: RiskEventRow) -> bool:
        normalized = self._normalize(event)
        existing = self.get(normalized.event_id)
        if existing is not None:
            if existing == normalized:
                return False
            raise DuplicateRecordError(
                "event_id already exists with different data",
                context={"event_id": normalized.event_id},
            )
        placeholders = ", ".join("?" for _ in self._COLUMNS)
        with self._db.transaction():
            self._db.execute(
                f"INSERT INTO risk_events ({', '.join(self._COLUMNS)}) "
                f"VALUES ({placeholders})",
                tuple(getattr(normalized, column) for column in self._COLUMNS),
            )
        return True

    record = insert

    def get(self, event_id: str) -> RiskEventRow | None:
        row = self._db.fetchone(
            "SELECT * FROM risk_events WHERE event_id = ?",
            (validate_text(event_id, "event_id"),),
        )
        return None if row is None else self._from_row(row)

    def list_since(self, since: str) -> tuple[RiskEventRow, ...]:
        rows = self._db.fetchall(
            "SELECT * FROM risk_events WHERE timestamp >= ? ORDER BY timestamp ASC",
            (utc_timestamp(since, "since"),),
        )
        return tuple(self._from_row(row) for row in rows)

    def list_by_kind(self, kind: str) -> tuple[RiskEventRow, ...]:
        rows = self._db.fetchall(
            "SELECT * FROM risk_events WHERE kind = ? ORDER BY timestamp ASC",
            (validate_text(kind, "kind", max_length=64).lower(),),
        )
        return tuple(self._from_row(row) for row in rows)


class AccountSnapshotRepository(BaseRepository):
    table = "account_snapshots"

    _COLUMNS = (
        "snapshot_id",
        "observed_at",
        "account_id",
        "status",
        "currency",
        "cash",
        "buying_power",
        "equity",
        "portfolio_value",
        "day_trade_count",
        "trading_blocked",
    )

    @classmethod
    def _normalize(cls, snapshot: AccountSnapshotRow) -> AccountSnapshotRow:
        return AccountSnapshotRow(
            snapshot_id=validate_text(snapshot.snapshot_id, "snapshot_id", max_length=128),
            observed_at=utc_timestamp(snapshot.observed_at, "observed_at"),
            account_id=validate_text(snapshot.account_id, "account_id", max_length=128),
            status=validate_text(snapshot.status, "status", max_length=64).lower(),
            currency=validate_text(snapshot.currency, "currency", max_length=8).upper(),
            cash=validate_decimal(snapshot.cash, "cash"),
            buying_power=validate_decimal(snapshot.buying_power, "buying_power"),
            equity=validate_decimal(snapshot.equity, "equity"),
            portfolio_value=validate_decimal(snapshot.portfolio_value, "portfolio_value"),
            day_trade_count=validate_integer(
                snapshot.day_trade_count, "day_trade_count", minimum=0
            ),
            trading_blocked=validate_boolean(snapshot.trading_blocked, "trading_blocked"),
        )

    @classmethod
    def _from_row(cls, row: sqlite3.Row) -> AccountSnapshotRow:
        return AccountSnapshotRow(
            snapshot_id=row["snapshot_id"],
            observed_at=row["observed_at"],
            account_id=row["account_id"],
            status=row["status"],
            currency=row["currency"],
            cash=_row_decimal(row["cash"], "cash"),
            buying_power=_row_decimal(row["buying_power"], "buying_power"),
            equity=_row_decimal(row["equity"], "equity"),
            portfolio_value=_row_decimal(row["portfolio_value"], "portfolio_value"),
            day_trade_count=int(row["day_trade_count"]),
            trading_blocked=_bool(row["trading_blocked"]),
        )

    def insert(self, snapshot: AccountSnapshotRow) -> bool:
        normalized = self._normalize(snapshot)
        existing = self.get(normalized.snapshot_id)
        if existing is not None:
            if existing == normalized:
                return False
            raise DuplicateRecordError(
                "snapshot_id already exists with different data",
                context={"snapshot_id": normalized.snapshot_id},
            )
        values = list(getattr(normalized, column) for column in self._COLUMNS)
        values[-1] = 1 if normalized.trading_blocked else 0
        placeholders = ", ".join("?" for _ in self._COLUMNS)
        with self._db.transaction():
            self._db.execute(
                f"INSERT INTO account_snapshots ({', '.join(self._COLUMNS)}) "
                f"VALUES ({placeholders})",
                tuple(values),
            )
        return True

    record = insert

    def get(self, snapshot_id: str) -> AccountSnapshotRow | None:
        row = self._db.fetchone(
            "SELECT * FROM account_snapshots WHERE snapshot_id = ?",
            (validate_text(snapshot_id, "snapshot_id"),),
        )
        return None if row is None else self._from_row(row)

    def latest(self) -> AccountSnapshotRow | None:
        row = self._db.fetchone(
            "SELECT * FROM account_snapshots ORDER BY observed_at DESC, snapshot_id DESC LIMIT 1"
        )
        return None if row is None else self._from_row(row)

    def list_between(self, start: str, end: str) -> tuple[AccountSnapshotRow, ...]:
        rows = self._db.fetchall(
            "SELECT * FROM account_snapshots WHERE observed_at >= ? AND observed_at < ? "
            "ORDER BY observed_at ASC",
            (utc_timestamp(start, "start"), utc_timestamp(end, "end")),
        )
        return tuple(self._from_row(row) for row in rows)


class StrategyVersionRepository(BaseRepository):
    table = "strategy_versions"

    _COLUMNS = ("version_id", "activated_at", "name", "parameters", "source")

    @classmethod
    def _normalize(cls, version: StrategyVersionRow) -> StrategyVersionRow:
        return StrategyVersionRow(
            version_id=validate_text(version.version_id, "version_id", max_length=128),
            activated_at=utc_timestamp(version.activated_at, "activated_at"),
            name=validate_text(version.name, "name", max_length=128),
            parameters=validate_text(version.parameters, "parameters"),
            source=validate_text(version.source, "source", max_length=128),
        )

    @classmethod
    def _from_row(cls, row: sqlite3.Row) -> StrategyVersionRow:
        return StrategyVersionRow(
            version_id=row["version_id"],
            activated_at=row["activated_at"],
            name=row["name"],
            parameters=row["parameters"],
            source=row["source"],
        )

    def insert(self, version: StrategyVersionRow) -> bool:
        normalized = self._normalize(version)
        existing = self.get(normalized.version_id)
        if existing is not None:
            if existing == normalized:
                return False
            raise DuplicateRecordError(
                "version_id already exists with different data",
                context={"version_id": normalized.version_id},
            )
        placeholders = ", ".join("?" for _ in self._COLUMNS)
        with self._db.transaction():
            self._db.execute(
                f"INSERT INTO strategy_versions ({', '.join(self._COLUMNS)}) "
                f"VALUES ({placeholders})",
                tuple(getattr(normalized, column) for column in self._COLUMNS),
            )
        return True

    def record(
        self,
        name: str,
        parameters: Any,
        source: str,
        *,
        activated_at: str | None = None,
        version_id: str | None = None,
    ) -> StrategyVersionRow:
        """Persist a strategy configuration, skipping an identical latest version."""
        if isinstance(parameters, str):
            serialized = parameters
        else:
            serialized = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
        candidate = StrategyVersionRow(
            version_id=version_id or f"strategy_{utc_now_timestamp()}_{name}",
            activated_at=activated_at or utc_now_timestamp(),
            name=name,
            parameters=serialized,
            source=source,
        )
        normalized = self._normalize(candidate)
        with self._db.transaction():
            latest = self.latest()
            if (
                latest is not None
                and latest.name == normalized.name
                and latest.parameters == normalized.parameters
                and latest.source == normalized.source
            ):
                return latest
            self.insert(normalized)
        return normalized

    def get(self, version_id: str) -> StrategyVersionRow | None:
        row = self._db.fetchone(
            "SELECT * FROM strategy_versions WHERE version_id = ?",
            (validate_text(version_id, "version_id"),),
        )
        return None if row is None else self._from_row(row)

    def latest(self) -> StrategyVersionRow | None:
        row = self._db.fetchone(
            "SELECT * FROM strategy_versions ORDER BY activated_at DESC, version_id DESC LIMIT 1"
        )
        return None if row is None else self._from_row(row)

    def list_versions(self) -> tuple[StrategyVersionRow, ...]:
        rows = self._db.fetchall(
            "SELECT * FROM strategy_versions ORDER BY activated_at ASC, version_id ASC"
        )
        return tuple(self._from_row(row) for row in rows)


class ApplicationEventRepository(BaseRepository):
    table = "application_events"

    _COLUMNS = ("event_id", "timestamp", "level", "phase", "message", "context")

    @classmethod
    def _normalize(cls, event: ApplicationEventRow) -> ApplicationEventRow:
        return ApplicationEventRow(
            event_id=validate_text(event.event_id, "event_id", max_length=128),
            timestamp=utc_timestamp(event.timestamp, "timestamp"),
            level=validate_text(event.level, "level", max_length=16).upper(),
            phase=validate_text(event.phase, "phase", max_length=64),
            message=validate_text(event.message, "message"),
            context=event.context,
        )

    @classmethod
    def _from_row(cls, row: sqlite3.Row) -> ApplicationEventRow:
        return ApplicationEventRow(
            event_id=row["event_id"],
            timestamp=row["timestamp"],
            level=row["level"],
            phase=row["phase"],
            message=row["message"],
            context=row["context"],
        )

    def insert(self, event: ApplicationEventRow) -> bool:
        normalized = self._normalize(event)
        existing = self.get(normalized.event_id)
        if existing is not None:
            if existing == normalized:
                return False
            raise DuplicateRecordError(
                "event_id already exists with different data",
                context={"event_id": normalized.event_id},
            )
        placeholders = ", ".join("?" for _ in self._COLUMNS)
        with self._db.transaction():
            self._db.execute(
                f"INSERT INTO application_events ({', '.join(self._COLUMNS)}) "
                f"VALUES ({placeholders})",
                tuple(getattr(normalized, column) for column in self._COLUMNS),
            )
        return True

    def record(
        self,
        level: str,
        phase: str,
        message: str,
        *,
        context: dict[str, Any] | None = None,
        timestamp: str | None = None,
        event_id: str | None = None,
    ) -> ApplicationEventRow:
        event = ApplicationEventRow(
            event_id=event_id or f"app_{utc_now_timestamp()}_{phase}",
            timestamp=timestamp or utc_now_timestamp(),
            level=level,
            phase=phase,
            message=message,
            context=context_json(context),
        )
        normalized = self._normalize(event)
        self.insert(normalized)
        return normalized

    def get(self, event_id: str) -> ApplicationEventRow | None:
        row = self._db.fetchone(
            "SELECT * FROM application_events WHERE event_id = ?",
            (validate_text(event_id, "event_id"),),
        )
        return None if row is None else self._from_row(row)

    def list_since(self, since: str) -> tuple[ApplicationEventRow, ...]:
        rows = self._db.fetchall(
            "SELECT * FROM application_events WHERE timestamp >= ? ORDER BY timestamp ASC",
            (utc_timestamp(since, "since"),),
        )
        return tuple(self._from_row(row) for row in rows)

    def list_for_phase(self, phase: str) -> tuple[ApplicationEventRow, ...]:
        rows = self._db.fetchall(
            "SELECT * FROM application_events WHERE phase = ? ORDER BY timestamp ASC",
            (validate_text(phase, "phase", max_length=64),),
        )
        return tuple(self._from_row(row) for row in rows)

    def latest(self) -> ApplicationEventRow | None:
        row = self._db.fetchone(
            "SELECT * FROM application_events ORDER BY timestamp DESC, event_id DESC LIMIT 1"
        )
        return None if row is None else self._from_row(row)


@dataclass(frozen=True, slots=True)
class RepositoryRegistry:
    """Bundle of repositories sharing one database connection."""

    orders: OrderRepository
    executions: ExecutionRepository
    positions: PositionRepository
    trade_history: TradeHistoryRepository
    signals: SignalRepository
    risk_events: RiskEventRepository
    account_snapshots: AccountSnapshotRepository
    strategy_versions: StrategyVersionRepository
    application_events: ApplicationEventRepository

    @classmethod
    def create(cls, database: "TradingDatabase") -> "RepositoryRegistry":
        return cls(
            orders=OrderRepository(database),
            executions=ExecutionRepository(database),
            positions=PositionRepository(database),
            trade_history=TradeHistoryRepository(database),
            signals=SignalRepository(database),
            risk_events=RiskEventRepository(database),
            account_snapshots=AccountSnapshotRepository(database),
            strategy_versions=StrategyVersionRepository(database),
            application_events=ApplicationEventRepository(database),
        )
