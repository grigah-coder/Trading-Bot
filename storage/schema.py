"""Versioned SQLite schema and migration runner.

The schema is intentionally plain SQL so it stays readable, testable and
independent of any ORM.  Each migration is applied in its own transaction and
recorded in ``schema_migrations``; ``PRAGMA user_version`` mirrors the latest
applied version for tooling that does not know about the table.

Migrations are additive.  No migration in this module drops or rewrites
historical trading records.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Iterable

from storage.validation import StorageError, utc_now_timestamp

SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    description: str
    statements: tuple[str, ...]


_V1_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS schema_migrations (
        version   INTEGER PRIMARY KEY,
        applied_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS orders (
        order_id             TEXT PRIMARY KEY,
        client_order_id      TEXT NOT NULL,
        symbol               TEXT NOT NULL,
        side                 TEXT NOT NULL,
        order_type           TEXT NOT NULL,
        time_in_force        TEXT NOT NULL,
        status               TEXT NOT NULL,
        quantity             TEXT NOT NULL,
        filled_quantity      TEXT NOT NULL,
        limit_price          TEXT,
        filled_average_price TEXT,
        submitted_at         TEXT NOT NULL,
        updated_at           TEXT,
        source               TEXT NOT NULL,
        raw_status           TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS executions (
        execution_id     TEXT PRIMARY KEY,
        order_id         TEXT NOT NULL,
        client_order_id  TEXT NOT NULL,
        symbol           TEXT NOT NULL,
        side             TEXT NOT NULL,
        quantity         TEXT NOT NULL,
        price            TEXT NOT NULL,
        filled_at        TEXT NOT NULL,
        is_partial       INTEGER NOT NULL,
        FOREIGN KEY (order_id) REFERENCES orders(order_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS positions (
        position_id            TEXT PRIMARY KEY,
        symbol                 TEXT NOT NULL,
        quantity               TEXT NOT NULL,
        side                   TEXT NOT NULL,
        average_entry_price    TEXT NOT NULL,
        market_value           TEXT NOT NULL,
        cost_basis             TEXT NOT NULL,
        unrealized_pl          TEXT NOT NULL,
        unrealized_pl_fraction TEXT NOT NULL,
        observed_at            TEXT NOT NULL,
        source                 TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS trade_history (
        trade_id    TEXT PRIMARY KEY,
        symbol      TEXT NOT NULL,
        side        TEXT NOT NULL,
        quantity    TEXT NOT NULL,
        entry_price TEXT NOT NULL,
        exit_price  TEXT NOT NULL,
        realized_pl TEXT NOT NULL,
        opened_at   TEXT NOT NULL,
        closed_at   TEXT NOT NULL,
        lot_ids     TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS signals (
        signal_id               TEXT PRIMARY KEY,
        symbol                  TEXT NOT NULL,
        direction               TEXT NOT NULL,
        entry_reference         TEXT NOT NULL,
        stop_loss_reference     TEXT,
        take_profit_reference   TEXT,
        timestamp               TEXT NOT NULL,
        strategy_id             TEXT NOT NULL,
        strength                TEXT,
        reason                  TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS risk_events (
        event_id        TEXT PRIMARY KEY,
        timestamp       TEXT NOT NULL,
        kind            TEXT NOT NULL,
        code            TEXT NOT NULL,
        message         TEXT NOT NULL,
        symbol          TEXT,
        client_order_id TEXT,
        context         TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS account_snapshots (
        snapshot_id     TEXT PRIMARY KEY,
        observed_at     TEXT NOT NULL,
        account_id      TEXT NOT NULL,
        status          TEXT NOT NULL,
        currency        TEXT NOT NULL,
        cash            TEXT NOT NULL,
        buying_power    TEXT NOT NULL,
        equity          TEXT NOT NULL,
        portfolio_value TEXT NOT NULL,
        day_trade_count INTEGER NOT NULL,
        trading_blocked INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS strategy_versions (
        version_id   TEXT PRIMARY KEY,
        activated_at TEXT NOT NULL,
        name         TEXT NOT NULL,
        parameters   TEXT NOT NULL,
        source       TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS application_events (
        event_id   TEXT PRIMARY KEY,
        timestamp  TEXT NOT NULL,
        level      TEXT NOT NULL,
        phase      TEXT NOT NULL,
        message    TEXT NOT NULL,
        context    TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_orders_client_order_id ON orders(client_order_id)",
    "CREATE INDEX IF NOT EXISTS idx_orders_symbol_status ON orders(symbol, status)",
    "CREATE INDEX IF NOT EXISTS idx_executions_order_id ON executions(order_id)",
    "CREATE INDEX IF NOT EXISTS idx_executions_filled_at ON executions(filled_at)",
    "CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol)",
    "CREATE INDEX IF NOT EXISTS idx_trade_history_symbol ON trade_history(symbol)",
    "CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_risk_events_timestamp ON risk_events(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_account_snapshots_observed ON account_snapshots(observed_at)",
    "CREATE INDEX IF NOT EXISTS idx_application_events_timestamp ON application_events(timestamp)",
)

_V2_STATEMENTS: tuple[str, ...] = (
    # A broker client order id identifies exactly one intent.  The unique index
    # makes duplicate writes fail loudly instead of silently creating twins.
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_orders_client_order_id_unique "
    "ON orders(client_order_id)",
    "CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)",
    "CREATE INDEX IF NOT EXISTS idx_orders_submitted_at ON orders(submitted_at)",
    "CREATE INDEX IF NOT EXISTS idx_executions_symbol ON executions(symbol)",
    "CREATE INDEX IF NOT EXISTS idx_executions_symbol_filled_at "
    "ON executions(symbol, filled_at)",
    "CREATE INDEX IF NOT EXISTS idx_positions_symbol_observed "
    "ON positions(symbol, observed_at)",
    "CREATE INDEX IF NOT EXISTS idx_positions_observed_at ON positions(observed_at)",
    "CREATE INDEX IF NOT EXISTS idx_trade_history_closed_at ON trade_history(closed_at)",
    "CREATE INDEX IF NOT EXISTS idx_signals_symbol_timestamp "
    "ON signals(symbol, timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_signals_strategy_id ON signals(strategy_id)",
    "CREATE INDEX IF NOT EXISTS idx_risk_events_kind ON risk_events(kind)",
    "CREATE INDEX IF NOT EXISTS idx_account_snapshots_account_observed "
    "ON account_snapshots(account_id, observed_at)",
    "CREATE INDEX IF NOT EXISTS idx_strategy_versions_activated_at "
    "ON strategy_versions(activated_at)",
    "CREATE INDEX IF NOT EXISTS idx_application_events_phase ON application_events(phase)",
    "CREATE INDEX IF NOT EXISTS idx_application_events_level_timestamp "
    "ON application_events(level, timestamp)",
)

_MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        1,
        "Initial trading schema: orders, executions, positions, trade history, "
        "signals, risk events, account snapshots, strategy versions, application events.",
        _V1_STATEMENTS,
    ),
    Migration(
        2,
        "Uniqueness and query indexes for identifiers, status, symbol and time-window lookups.",
        _V2_STATEMENTS,
    ),
)


def _ensure_migration_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version   INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL
        )
        """
    )


def _user_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def _begin(conn: sqlite3.Connection) -> bool:
    """Begin a transaction unless the connection already has one open."""
    if conn.in_transaction:
        return False
    conn.execute("BEGIN IMMEDIATE")
    return True


def _set_user_version(conn: sqlite3.Connection, version: int) -> None:
    # PRAGMA does not support bound parameters; version is an int by contract.
    conn.execute(f"PRAGMA user_version = {int(version)}")


def initialize_schema(conn: sqlite3.Connection) -> int:
    """Apply every pending migration and return the resulting schema version."""
    return migrate(conn)


def current_version(conn: sqlite3.Connection) -> int:
    _ensure_migration_table(conn)
    row = conn.execute(
        "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def migrate(conn: sqlite3.Connection, *, target: int | None = None) -> int:
    """Apply pending migrations and return the new schema version.

    Raises:
        ValueError: when a downgrade is requested.
        StorageError: when a migration fails; the migration is rolled back and
            the schema is left at the previous version.
    """
    desired = SCHEMA_VERSION if target is None else int(target)
    if desired < 0:
        raise ValueError("target schema version cannot be negative")
    _ensure_migration_table(conn)
    current = current_version(conn)
    if desired < current:
        raise ValueError("Cannot downgrade schema via migrate()")

    for migration in _MIGRATIONS:
        if migration.version <= current:
            continue
        if migration.version > desired:
            break
        began = _begin(conn)
        try:
            for statement in migration.statements:
                conn.execute(statement)
            conn.execute(
                "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                (migration.version, utc_now_timestamp()),
            )
            _set_user_version(conn, migration.version)
        except Exception as exc:
            if began:
                conn.execute("ROLLBACK")
            raise StorageError(
                f"Schema migration {migration.version} failed",
                context={
                    "version": migration.version,
                    "description": migration.description,
                },
            ) from exc
        else:
            if began:
                conn.execute("COMMIT")
            current = migration.version
    return current


def applied_migrations(conn: sqlite3.Connection) -> tuple[tuple[int, str], ...]:
    """Return ``(version, applied_at)`` pairs in ascending order."""
    _ensure_migration_table(conn)
    rows = conn.execute(
        "SELECT version, applied_at FROM schema_migrations ORDER BY version"
    ).fetchall()
    return tuple((int(row[0]), str(row[1])) for row in rows)


def all_tables() -> Iterable[str]:
    return (
        "orders",
        "executions",
        "positions",
        "trade_history",
        "signals",
        "risk_events",
        "account_snapshots",
        "strategy_versions",
        "application_events",
        "schema_migrations",
    )
