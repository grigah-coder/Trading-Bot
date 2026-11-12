"""SQLite-backed persistence repository.

Implements the :class:`trading_bot.engine.Database` protocol on top of SQLite
(see :mod:`trading_bot.db.migrations` for the schema). Every ``save_*`` method
is idempotent (INSERT OR REPLACE keyed by the record's natural key) and
never raises: persistence failures are logged and swallowed so the trading
loop never crashes because the database is temporarily unavailable.

A :class:`trading_bot.trade_ledger.TradeLedger` is also wired in so that
every persisted fill is simultaneously recorded in the FIFO trade ledger
(realised P/L tracking) when a ``trade_ledger_file`` is configured.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from trading_bot.config.settings import Settings
from trading_bot.db.migrations import migrate
from trading_bot.db.models import (
    AccountSnapshotRecord,
    FillRecord,
    OrderRecord,
    PositionRecord,
    RiskEventRecord,
    SignalRecord,
    SystemEventRecord,
)
from trading_bot.logging_config import get_logger
from trading_bot.trade_ledger import TradeLedger

logger = get_logger(__name__)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _as_str(value: Any, default: str = "") -> str:
    return "" if value is None else str(value)


def _iso(value: Any) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    return str(value or datetime.now(timezone.utc).isoformat())


class Repository:
    """SQLite persistence for the trading bot.

    Parameters
    ----------
    settings:
        Validated application configuration. ``db_path`` locates the SQLite
        file and ``trade_ledger_file``/``risk_state_file`` locate side-car
        stores.

    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        from trading_bot.config.settings import load_settings

        self.settings = settings or load_settings()
        self._lock = threading.RLock()
        migrate(self.settings.db_path)
        self._ledger = TradeLedger(
            path=self.settings.trade_ledger_file,
            keep_days=90,
        )

    # ------------------------------------------------------------------
    # Connection handling
    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.settings.db_path, timeout=30.0)
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _execute(self, sql: str, params: tuple = ()) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(sql, params)
                conn.commit()
            except Exception as exc:  # noqa: BLE001 - persistence is best-effort
                logger.warning("Database write failed: %s", exc)
                try:
                    conn.rollback()
                except Exception:  # noqa: BLE001
                    pass
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Database protocol
    # ------------------------------------------------------------------
    def save_order(self, order: Dict[str, Any]) -> None:
        rec = OrderRecord(
            order_id=_as_str(order.get("order_id") or order.get("id")),
            client_order_id=_as_str(
                order.get("client_order_id") or order.get("client_order_id")
            ),
            symbol=_as_str(order.get("symbol", "")).upper(),
            side=_as_str(order.get("side", "buy")),
            quantity=_as_float(order.get("quantity") or order.get("qty")),
            order_type=_as_str(order.get("order_type", "market")),
            status=_as_str(order.get("status", "pending_new")),
            filled_qty=_as_float(order.get("filled_qty")),
            avg_fill_price=_as_float(
                order.get("avg_fill_price") or order.get("filled_avg_price")
            ),
            stop_price=_as_optional_float(order.get("stop_price")),
            limit_price=_as_optional_float(order.get("limit_price")),
            time_in_force=_as_str(order.get("time_in_force", "day")),
            signal_id=_as_optional_str(order.get("signal_id")),
            created_at=_iso(order.get("created_at")),
            updated_at=_iso(order.get("updated_at")),
        )
        self._execute(
            """
            INSERT OR REPLACE INTO orders
            (order_id, client_order_id, symbol, side, quantity, order_type,
             status, filled_qty, avg_fill_price, stop_price, limit_price,
             time_in_force, signal_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rec.order_id, rec.client_order_id, rec.symbol, rec.side,
                rec.quantity, rec.order_type, rec.status, rec.filled_qty,
                rec.avg_fill_price, rec.stop_price, rec.limit_price,
                rec.time_in_force, rec.signal_id, rec.created_at, rec.updated_at,
            ),
        )

    def save_fill(self, fill: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Persist a fill and return the trade-ledger result (``None`` if dup)."""
        rec = FillRecord(
            fill_id=_as_str(fill.get("fill_id") or fill.get("id")),
            order_id=_as_str(fill.get("order_id")),
            symbol=_as_str(fill.get("symbol", "")).upper(),
            side=_as_str(fill.get("side", "buy")),
            quantity=_as_float(fill.get("quantity") or fill.get("qty")),
            price=_as_float(fill.get("price") or fill.get("fill_price")),
            filled_at=_iso(fill.get("filled_at")),
        )
        self._execute(
            """
            INSERT OR REPLACE INTO fills
            (fill_id, order_id, symbol, side, quantity, price, filled_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rec.fill_id, rec.order_id, rec.symbol, rec.side,
                rec.quantity, rec.price, rec.filled_at,
            ),
        )
        # Also record in the FIFO trade ledger for realised-P/L tracking.
        try:
            return self._ledger.record_fill(
                symbol=rec.symbol,
                side=rec.side,
                quantity=rec.quantity,
                price=rec.price,
                fill_id=rec.fill_id or rec.order_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Trade ledger write failed: %s", exc)
            return None

    def save_position(self, position: Dict[str, Any]) -> None:
        rec = PositionRecord(
            symbol=_as_str(position.get("symbol", "")).upper(),
            qty=_as_float(position.get("qty")),
            avg_entry_price=_as_float(position.get("avg_entry_price")),
            current_price=_as_float(position.get("current_price")),
            market_value=_as_float(position.get("market_value")),
            unrealized_pl=_as_float(position.get("unrealized_pl")),
            unrealized_plpc=_as_float(position.get("unrealized_plpc")),
            cost_basis=_as_float(position.get("cost_basis")),
            side=_as_str(position.get("side", "long")),
            recorded_at=_iso(position.get("recorded_at")),
        )
        self._execute(
            """
            INSERT OR REPLACE INTO positions
            (symbol, qty, avg_entry_price, current_price, market_value,
             unrealized_pl, unrealized_plpc, cost_basis, side, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rec.symbol, rec.qty, rec.avg_entry_price, rec.current_price,
                rec.market_value, rec.unrealized_pl, rec.unrealized_plpc,
                rec.cost_basis, rec.side, rec.recorded_at,
            ),
        )

    def save_signal(self, signal: Dict[str, Any]) -> None:
        rec = SignalRecord(
            signal_id=_as_str(signal.get("signal_id")),
            symbol=_as_str(signal.get("symbol", "")).upper(),
            side=_as_str(signal.get("side", "buy")),
            price=_as_float(signal.get("price")),
            quantity=_as_float(signal.get("quantity")),
            stop_loss_pct=_as_float(signal.get("stop_loss_pct")),
            take_profit_pct=_as_float(signal.get("take_profit_pct")),
            strategy_name=_as_str(signal.get("strategy_name", "unknown")),
            strength=_as_float(signal.get("strength"), 1.0),
            created_at=_iso(signal.get("created_at")),
            metadata=dict(signal.get("metadata") or {}),
        )
        self._execute(
            """
            INSERT OR REPLACE INTO signals
            (signal_id, symbol, side, price, quantity, stop_loss_pct,
             take_profit_pct, strategy_name, strength, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rec.signal_id, rec.symbol, rec.side, rec.price, rec.quantity,
                rec.stop_loss_pct, rec.take_profit_pct, rec.strategy_name,
                rec.strength, rec.created_at,
                self._json(rec.metadata),
            ),
        )

    def save_risk_event(self, event: Dict[str, Any]) -> None:
        rec = RiskEventRecord(
            event_type=_as_str(event.get("event_type") or event.get("type")),
            symbol=_as_optional_str(event.get("symbol")),
            side=_as_optional_str(event.get("side")),
            quantity=_as_optional_float(event.get("quantity")),
            price=_as_optional_float(event.get("price")),
            reason=_as_str(event.get("reason")),
            created_at=_iso(event.get("created_at") or event.get("timestamp")),
            metadata=dict(event.get("metadata") or {}),
        )
        self._execute(
            """
            INSERT INTO risk_events
            (event_type, symbol, side, quantity, price, reason, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rec.event_type, rec.symbol, rec.side, rec.quantity, rec.price,
                rec.reason, rec.created_at, self._json(rec.metadata),
            ),
        )

    def save_account_snapshot(self, snapshot: Dict[str, Any]) -> None:
        rec = AccountSnapshotRecord(
            account_id=_as_str(snapshot.get("account_id")),
            status=_as_str(snapshot.get("status")),
            equity=_as_float(snapshot.get("equity")),
            cash=_as_float(snapshot.get("cash")),
            buying_power=_as_float(snapshot.get("buying_power")),
            portfolio_value=_as_float(snapshot.get("portfolio_value")),
            recorded_at=_iso(snapshot.get("recorded_at")),
        )
        self._execute(
            """
            INSERT INTO account_snapshots
            (account_id, status, equity, cash, buying_power, portfolio_value, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rec.account_id, rec.status, rec.equity, rec.cash,
                rec.buying_power, rec.portfolio_value, rec.recorded_at,
            ),
        )

    def save_system_event(self, event: Dict[str, Any]) -> None:
        rec = SystemEventRecord(
            event_type=_as_str(event.get("type") or event.get("event_type")),
            detail=_as_str(event.get("detail")),
            timestamp=_iso(event.get("timestamp")),
        )
        self._execute(
            """
            INSERT INTO system_events (event_type, detail, timestamp)
            VALUES (?, ?, ?)
            """,
            (rec.event_type, rec.detail, rec.timestamp),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _json(value: Any) -> str:
        import json

        try:
            return json.dumps(value, default=str)
        except Exception:  # noqa: BLE001
            return "{}"

    # ------------------------------------------------------------------
    # Convenience queries (beyond the protocol)
    # ------------------------------------------------------------------
    def get_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                if symbol:
                    rows = conn.execute(
                        "SELECT * FROM orders WHERE symbol=? ORDER BY created_at",
                        (symbol.upper(),),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM orders ORDER BY created_at"
                    ).fetchall()
                return [dict(r) for r in rows]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Database read failed: %s", exc)
                return []
            finally:
                conn.close()

    def get_positions(self) -> List[Dict[str, Any]]:
        """Return persisted position snapshots ordered by symbol."""
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM positions ORDER BY symbol"
                ).fetchall()
                return [dict(r) for r in rows]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Database read failed: %s", exc)
                return []
            finally:
                conn.close()

    def prune_positions(self, open_symbols: Any) -> int:
        """Delete persisted positions no longer held at the broker.

        Returns the number of rows removed. An empty symbol set clears the
        table, which is correct when the broker reports no open positions.
        """
        symbols = {
            str(symbol).strip().upper()
            for symbol in (open_symbols or ())
            if str(symbol).strip()
        }
        with self._lock:
            conn = self._connect()
            try:
                if symbols:
                    placeholders = ",".join("?" for _ in symbols)
                    cursor = conn.execute(
                        f"DELETE FROM positions WHERE symbol NOT IN ({placeholders})",
                        tuple(sorted(symbols)),
                    )
                else:
                    cursor = conn.execute("DELETE FROM positions")
                conn.commit()
                return cursor.rowcount or 0
            except Exception as exc:  # noqa: BLE001
                logger.warning("Position prune failed: %s", exc)
                try:
                    conn.rollback()
                except Exception:  # noqa: BLE001
                    pass
                return 0
            finally:
                conn.close()

    def ledger_stats(self, day: Optional[str] = None) -> Dict[str, Any]:
        """Delegate to the trade ledger's daily stats."""
        try:
            return self._ledger.stats(day)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Trade ledger stats failed: %s", exc)
            return {}


def _as_optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_optional_str(value: Any) -> Optional[str]:
    if value is None or value == "":
        return None
    return str(value)


__all__ = ["Repository"]
