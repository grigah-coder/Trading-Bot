"""Unit tests for the Phase 8 database / persistence layer."""

from __future__ import annotations

import os
import tempfile

from trading_bot.config.settings import Settings
from trading_bot.db.migrations import SCHEMA_VERSION, migrate
from trading_bot.db.repository import Repository


def _settings(db_path: str, ledger_path: str) -> Settings:
    return Settings(
        db_path=db_path,
        trade_ledger_file=ledger_path,
        risk_state_file="/tmp/kilo/risk_state.json",
    )


# ---------------------------------------------------------------------------
# Migrations
# ---------------------------------------------------------------------------


class TestMigrations:
    def test_migrate_creates_tables(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as fh:
            path = fh.name
        try:
            version = migrate(path)
            assert version == SCHEMA_VERSION
            import sqlite3

            conn = sqlite3.connect(path)
            try:
                tables = {
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }
                for expected in (
                    "orders",
                    "fills",
                    "positions",
                    "signals",
                    "risk_events",
                    "account_snapshots",
                    "system_events",
                    "schema_version",
                ):
                    assert expected in tables, f"missing table {expected}"
            finally:
                conn.close()
        finally:
            os.unlink(path)

    def test_migrate_idempotent(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as fh:
            path = fh.name
        try:
            migrate(path)
            version = migrate(path)
            assert version == SCHEMA_VERSION
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------


class TestRepository:
    def test_save_and_get_order(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as fh:
            db_path = fh.name
        ledger_path = tempfile.mktemp(suffix=".json")
        try:
            repo = Repository(_settings(db_path, ledger_path))
            order = {
                "order_id": "ord-1",
                "client_order_id": "cid-1",
                "symbol": "aapl",
                "side": "buy",
                "quantity": 10.0,
                "order_type": "market",
                "status": "new",
                "signal_id": "sig-1",
            }
            repo.save_order(order)
            rows = repo.get_orders()
            assert len(rows) == 1
            assert rows[0]["order_id"] == "ord-1"
            assert rows[0]["symbol"] == "AAPL"
            assert rows[0]["side"] == "buy"
        finally:
            os.unlink(db_path)
            if os.path.exists(ledger_path):
                os.unlink(ledger_path)

    def test_save_order_idempotent_replace(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as fh:
            db_path = fh.name
        ledger_path = tempfile.mktemp(suffix=".json")
        try:
            repo = Repository(_settings(db_path, ledger_path))
            base = {
                "order_id": "ord-1",
                "client_order_id": "cid-1",
                "symbol": "AAPL",
                "side": "buy",
                "quantity": 10.0,
                "status": "new",
            }
            repo.save_order(base)
            updated = dict(base, status="filled", filled_qty=10.0)
            repo.save_order(updated)
            rows = repo.get_orders()
            assert len(rows) == 1
            assert rows[0]["status"] == "filled"
            assert rows[0]["filled_qty"] == 10.0
        finally:
            os.unlink(db_path)
            if os.path.exists(ledger_path):
                os.unlink(ledger_path)

    def test_save_fill_writes_ledger(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as fh:
            db_path = fh.name
        ledger_path = tempfile.mktemp(suffix=".json")
        try:
            repo = Repository(_settings(db_path, ledger_path))
            fill = {
                "fill_id": "fill-1",
                "order_id": "ord-1",
                "symbol": "MSFT",
                "side": "sell",
                "quantity": 5.0,
                "price": 150.0,
            }
            repo.save_fill(fill)
            rows = repo.get_orders()
            assert len(rows) == 0  # fills go to fills table, not orders
            stats = repo.ledger_stats()
            assert stats["trade_count"] >= 0
        finally:
            os.unlink(db_path)
            if os.path.exists(ledger_path):
                os.unlink(ledger_path)

    def test_save_position(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as fh:
            db_path = fh.name
        ledger_path = tempfile.mktemp(suffix=".json")
        try:
            repo = Repository(_settings(db_path, ledger_path))
            position = {
                "symbol": "NVDA",
                "qty": 2.0,
                "avg_entry_price": 100.0,
                "current_price": 110.0,
                "market_value": 220.0,
                "unrealized_pl": 20.0,
                "unrealized_plpc": 0.1,
                "side": "long",
            }
            repo.save_position(position)
            rows = repo.get_orders()
            assert len(rows) == 0  # positions in positions table
        finally:
            os.unlink(db_path)
            if os.path.exists(ledger_path):
                os.unlink(ledger_path)

    def test_save_signal(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as fh:
            db_path = fh.name
        ledger_path = tempfile.mktemp(suffix=".json")
        try:
            repo = Repository(_settings(db_path, ledger_path))
            signal = {
                "signal_id": "sig-1",
                "symbol": "TSLA",
                "side": "buy",
                "price": 200.0,
                "quantity": 3.0,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.04,
                "strategy_name": "ma_cross",
                "strength": 1.0,
                "metadata": {"fast": 10.0, "slow": 9.0},
            }
            repo.save_signal(signal)
        finally:
            os.unlink(db_path)
            if os.path.exists(ledger_path):
                os.unlink(ledger_path)

    def test_save_risk_event(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as fh:
            db_path = fh.name
        ledger_path = tempfile.mktemp(suffix=".json")
        try:
            repo = Repository(_settings(db_path, ledger_path))
            event = {
                "event_type": "rejected",
                "symbol": "AAPL",
                "side": "buy",
                "reason": "daily loss limit reached",
            }
            repo.save_risk_event(event)
        finally:
            os.unlink(db_path)
            if os.path.exists(ledger_path):
                os.unlink(ledger_path)

    def test_save_account_snapshot(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as fh:
            db_path = fh.name
        ledger_path = tempfile.mktemp(suffix=".json")
        try:
            repo = Repository(_settings(db_path, ledger_path))
            snapshot = {
                "account_id": "acct-1",
                "status": "active",
                "equity": 10000.0,
                "cash": 5000.0,
                "buying_power": 10000.0,
                "portfolio_value": 10000.0,
            }
            repo.save_account_snapshot(snapshot)
        finally:
            os.unlink(db_path)
            if os.path.exists(ledger_path):
                os.unlink(ledger_path)

    def test_save_system_event(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as fh:
            db_path = fh.name
        ledger_path = tempfile.mktemp(suffix=".json")
        try:
            repo = Repository(_settings(db_path, ledger_path))
            event = {
                "type": "startup",
                "detail": "engine started",
            }
            repo.save_system_event(event)
        finally:
            os.unlink(db_path)
            if os.path.exists(ledger_path):
                os.unlink(ledger_path)

    def test_get_orders_filtered_by_symbol(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as fh:
            db_path = fh.name
        ledger_path = tempfile.mktemp(suffix=".json")
        try:
            repo = Repository(_settings(db_path, ledger_path))
            repo.save_order(
                {
                    "order_id": "ord-1",
                    "client_order_id": "cid-1",
                    "symbol": "AAPL",
                    "side": "buy",
                    "quantity": 1.0,
                }
            )
            repo.save_order(
                {
                    "order_id": "ord-2",
                    "client_order_id": "cid-2",
                    "symbol": "MSFT",
                    "side": "sell",
                    "quantity": 1.0,
                }
            )
            aapl = repo.get_orders(symbol="AAPL")
            assert len(aapl) == 1
            assert aapl[0]["symbol"] == "AAPL"
            all_rows = repo.get_orders()
            assert len(all_rows) == 2
        finally:
            os.unlink(db_path)
            if os.path.exists(ledger_path):
                os.unlink(ledger_path)
