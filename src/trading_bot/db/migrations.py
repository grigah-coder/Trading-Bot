"""SQLite schema migrations.

Defines the on-disk schema for every persisted record and a :func:`migrate`
runner that brings an existing database file up to the latest version.
Migrations are additive only: tables use ``CREATE TABLE IF NOT EXISTS`` so a
re-run is safe, and a ``schema_version`` table tracks the applied version.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

# Bump this when the schema changes in a non-additive way.
SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS orders (
        order_id        TEXT PRIMARY KEY,
        client_order_id TEXT NOT NULL,
        symbol          TEXT NOT NULL,
        side            TEXT NOT NULL,
        quantity        REAL NOT NULL,
        order_type      TEXT NOT NULL DEFAULT 'market',
        status          TEXT NOT NULL DEFAULT 'pending_new',
        filled_qty      REAL NOT NULL DEFAULT 0,
        avg_fill_price  REAL NOT NULL DEFAULT 0,
        stop_price      REAL,
        limit_price     REAL,
        time_in_force   TEXT NOT NULL DEFAULT 'day',
        signal_id       TEXT,
        created_at      TEXT NOT NULL,
        updated_at      TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_orders_symbol
        ON orders(symbol);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_orders_signal
        ON orders(signal_id);
    """,
    """
    CREATE TABLE IF NOT EXISTS fills (
        fill_id   TEXT PRIMARY KEY,
        order_id  TEXT NOT NULL,
        symbol    TEXT NOT NULL,
        side      TEXT NOT NULL,
        quantity  REAL NOT NULL,
        price     REAL NOT NULL,
        filled_at TEXT NOT NULL,
        FOREIGN KEY(order_id) REFERENCES orders(order_id)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_fills_order
        ON fills(order_id);
    """,
    """
    CREATE TABLE IF NOT EXISTS positions (
        symbol            TEXT PRIMARY KEY,
        qty               REAL NOT NULL,
        avg_entry_price   REAL NOT NULL,
        current_price     REAL NOT NULL,
        market_value      REAL NOT NULL,
        unrealized_pl     REAL NOT NULL,
        unrealized_plpc   REAL NOT NULL,
        cost_basis        REAL NOT NULL DEFAULT 0,
        side              TEXT NOT NULL DEFAULT 'long',
        recorded_at       TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS signals (
        signal_id     TEXT PRIMARY KEY,
        symbol        TEXT NOT NULL,
        side          TEXT NOT NULL,
        price         REAL NOT NULL,
        quantity      REAL NOT NULL,
        stop_loss_pct REAL NOT NULL,
        take_profit_pct REAL NOT NULL,
        strategy_name TEXT NOT NULL DEFAULT 'unknown',
        strength      REAL NOT NULL DEFAULT 1,
        created_at    TEXT NOT NULL,
        metadata      TEXT NOT NULL DEFAULT '{}'
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_signals_symbol
        ON signals(symbol);
    """,
    """
    CREATE TABLE IF NOT EXISTS risk_events (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        event_type    TEXT NOT NULL,
        symbol        TEXT,
        side          TEXT,
        quantity      REAL,
        price         REAL,
        reason        TEXT NOT NULL DEFAULT '',
        created_at    TEXT NOT NULL,
        metadata      TEXT NOT NULL DEFAULT '{}'
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_risk_events_type
        ON risk_events(event_type);
    """,
    """
    CREATE TABLE IF NOT EXISTS account_snapshots (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        account_id      TEXT NOT NULL,
        status          TEXT NOT NULL,
        equity          REAL NOT NULL,
        cash            REAL NOT NULL,
        buying_power    REAL NOT NULL,
        portfolio_value REAL NOT NULL,
        recorded_at     TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_account_recorded_at
        ON account_snapshots(recorded_at);
    """,
    """
    CREATE TABLE IF NOT EXISTS system_events (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        event_type  TEXT NOT NULL,
        detail      TEXT NOT NULL DEFAULT '',
        timestamp   TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_system_events_type
        ON system_events(event_type);
    """,
)


def _connect(db_path: str) -> sqlite3.Connection:
    """Open a SQLite connection with sane defaults."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def migrate(db_path: str) -> int:
    """Apply pending migrations to ``db_path``.

    Creates the database file and all tables when absent; otherwise verifies
    the schema version and upgrades if needed. Returns the final schema
    version.
    """
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN")
        # schema_version may not exist yet on a brand-new database.
        try:
            row = conn.execute(
                "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
            ).fetchone()
        except sqlite3.OperationalError:
            row = None
        current = int(row["version"]) if row else 0
        if current < SCHEMA_VERSION:
            for statement in SCHEMA_STATEMENTS:
                conn.execute(statement)
            conn.execute(
                "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
                (SCHEMA_VERSION,),
            )
        conn.commit()
        return SCHEMA_VERSION
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


__all__ = ["SCHEMA_VERSION", "migrate", "_connect"]
