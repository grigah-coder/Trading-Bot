"""SQLite schema definition and migration runner.

The schema is intentionally plain SQL so it stays readable, testable and
independent of any ORM.  ``SCHEMA_VERSION`` is bumped whenever a migration is
added; ``migrate()`` applies pending migrations inside a single transaction
per migration and records the applied version in ``schema_migrations``.
"""

from __future__ import annotations

import sqlite3
from typing import Iterable

SCHEMA_VERSION = 1

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version   INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    order_id           TEXT PRIMARY KEY,
    client_order_id    TEXT NOT NULL,
    symbol             TEXT NOT NULL,
    side               TEXT NOT NULL,
    order_type         TEXT NOT NULL,
    time_in_force      TEXT NOT NULL,
    status             TEXT NOT NULL,
    quantity           TEXT NOT NULL,
    filled_quantity    TEXT NOT NULL,
    limit_price        TEXT,
    filled_average_price TEXT,
    submitted_at       TEXT NOT NULL,
    updated_at         TEXT,
    source             TEXT NOT NULL,
    raw_status         TEXT
);

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
);

CREATE TABLE IF NOT EXISTS positions (
    position_id            TEXT PRIMARY KEY,
    symbol                 TEXT NOT NULL,
    quantity               TEXT NOT NULL,
    side                   TEXT NOT NULL,
    average_entry_price    TEXT NOT NULL,
    market_value           TEXT NOT NULL,
    cost_basis             TEXT NOT NULL,
    unrealized_pl           TEXT NOT NULL,
    unrealized_pl_fraction TEXT NOT NULL,
    observed_at            TEXT NOT NULL,
    source                 TEXT NOT NULL
);

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
);

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
);

CREATE TABLE IF NOT EXISTS risk_events (
    event_id        TEXT PRIMARY KEY,
    timestamp       TEXT NOT NULL,
    kind            TEXT NOT NULL,
    code            TEXT NOT NULL,
    message         TEXT NOT NULL,
    symbol         TEXT,
    client_order_id TEXT,
    context        TEXT
);

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
);

CREATE TABLE IF NOT EXISTS strategy_versions (
    version_id   TEXT PRIMARY KEY,
    activated_at TEXT NOT NULL,
    name         TEXT NOT NULL,
    parameters   TEXT NOT NULL,
    source       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS application_events (
    event_id   TEXT PRIMARY KEY,
    timestamp  TEXT NOT NULL,
    level      TEXT NOT NULL,
    phase      TEXT NOT NULL,
    message    TEXT NOT NULL,
    context    TEXT
);

CREATE INDEX IF NOT EXISTS idx_orders_client_order_id ON orders(client_order_id);
CREATE INDEX IF NOT EXISTS idx_orders_symbol_status ON orders(symbol, status);
CREATE INDEX IF NOT EXISTS idx_executions_order_id ON executions(order_id);
CREATE INDEX IF NOT EXISTS idx_executions_filled_at ON executions(filled_at);
CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
CREATE INDEX IF NOT EXISTS idx_trade_history_symbol ON trade_history(symbol);
CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp);
CREATE INDEX IF NOT EXISTS idx_risk_events_timestamp ON risk_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_account_snapshots_observed ON account_snapshots(observed_at);
CREATE INDEX IF NOT EXISTS idx_application_events_timestamp ON application_events(timestamp);
"""

_MIGRATIONS: tuple[tuple[int, str], ...] = (
    (1, "Initial schema with orders, executions, positions, trade history, signals, risk events, account snapshots, strategy versions and application events."),
)


def _user_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def _set_user_version(conn: sqlite3.Connection, version: int) -> None:
    conn.execute("PRAGMA user_version = ?", (version,))


def initialize_schema(conn: sqlite3.Connection) -> None:
    """Create every table and index if they do not exist yet."""
    conn.executescript(_CREATE_SQL)
    conn.commit()


def current_version(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def migrate(conn: sqlite3.Connection, *, target: int | None = None) -> int:
    """Apply pending migrations and return the new schema version."""
    initialize_schema(conn)
    current = current_version(conn)
    desired = SCHEMA_VERSION if target is None else target
    if desired < current:
        raise ValueError("Cannot downgrade schema via migrate()")
    for version, _description in _MIGRATIONS:
        if version <= current or version > desired:
            continue
        conn.execute(
            "INSERT INTO schema_migrations (version, applied_at) VALUES (?, datetime('now'))",
            (version,),
        )
        _set_user_version(conn, version)
        current = version
    conn.commit()
    return current


def all_tables() -> Iterable[str]:
    return (
        "orders", "executions", "positions", "trade_history", "signals",
        "risk_events", "account_snapshots", "strategy_versions",
        "application_events", "schema_migrations",
    )
