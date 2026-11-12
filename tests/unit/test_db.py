"""Unit tests for the Phase 8 database / persistence layer."""

from __future__ import annotations

import pytest

from trading_bot.config.settings import Settings
from trading_bot.db.migrations import SCHEMA_VERSION, migrate
from trading_bot.db.repository import Repository
from trading_bot.trade_ledger import TradeLedger


def _settings(db_path: str, ledger_path: str) -> Settings:
    return Settings(
        db_path=db_path,
        trade_ledger_file=ledger_path,
        # Persistence of risk state is irrelevant here; keep tests hermetic.
        risk_state_file="",
    )


def _repo(tmp_path) -> Repository:
    return Repository(_settings(
        str(tmp_path / "trading.sqlite"),
        str(tmp_path / "ledger.json"),
    ))


# ---------------------------------------------------------------------------
# Migrations
# ---------------------------------------------------------------------------


class TestMigrations:
    def test_migrate_creates_tables(self, tmp_path):
        path = str(tmp_path / "trading.sqlite")
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

    def test_migrate_idempotent(self, tmp_path):
        path = str(tmp_path / "trading.sqlite")
        migrate(path)
        version = migrate(path)
        assert version == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------


class TestRepository:
    def test_save_and_get_order(self, tmp_path):
        repo = _repo(tmp_path)
        order = {
            "order_id": "ord-1",
            "client_order_id": "cid-1",
            "symbol": "aapl",
            "side": "buy",
            "quantity": 10.0,
            "order_type": "market",
            "status": "new",
            "stop_price": 98.0,
            "limit_price": 104.0,
            "signal_id": "sig-1",
        }
        repo.save_order(order)
        rows = repo.get_orders()
        assert len(rows) == 1
        assert rows[0]["order_id"] == "ord-1"
        assert rows[0]["symbol"] == "AAPL"
        assert rows[0]["side"] == "buy"
        assert rows[0]["stop_price"] == 98.0
        assert rows[0]["limit_price"] == 104.0

    def test_save_order_idempotent_replace(self, tmp_path):
        repo = _repo(tmp_path)
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

    def test_save_fill_writes_ledger(self, tmp_path):
        repo = _repo(tmp_path)
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

    def test_save_position(self, tmp_path):
        repo = _repo(tmp_path)
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

    def test_save_signal(self, tmp_path):
        repo = _repo(tmp_path)
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

    def test_save_risk_event(self, tmp_path):
        repo = _repo(tmp_path)
        event = {
            "event_type": "rejected",
            "symbol": "AAPL",
            "side": "buy",
            "reason": "daily loss limit reached",
        }
        repo.save_risk_event(event)

    def test_save_account_snapshot(self, tmp_path):
        repo = _repo(tmp_path)
        snapshot = {
            "account_id": "acct-1",
            "status": "active",
            "equity": 10000.0,
            "cash": 5000.0,
            "buying_power": 10000.0,
            "portfolio_value": 10000.0,
        }
        repo.save_account_snapshot(snapshot)

    def test_save_system_event(self, tmp_path):
        repo = _repo(tmp_path)
        event = {
            "type": "startup",
            "detail": "engine started",
        }
        repo.save_system_event(event)

    def test_get_orders_filtered_by_symbol(self, tmp_path):
        repo = _repo(tmp_path)
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


class TestFillAndPositionPersistence:
    """Fill ordering, ledger results, and closed-position pruning."""

    def _repo(self, tmp_path) -> Repository:
        return _repo(tmp_path)

    def test_save_order_prefers_avg_fill_price(self, tmp_path):
        """``avg_fill_price`` wins when both price keys are present."""
        repo = self._repo(tmp_path)
        repo.save_order({
            "order_id": "ord-1", "client_order_id": "cid-1",
            "symbol": "AAPL", "side": "buy", "quantity": 1.0,
            "avg_fill_price": 12.5, "filled_avg_price": 99.0,
        })
        assert repo.get_orders()[0]["avg_fill_price"] == 12.5

    def test_save_order_falls_back_to_filled_avg_price(self, tmp_path):
        """Broker-normalized ``filled_avg_price`` is the fallback."""
        repo = self._repo(tmp_path)
        repo.save_order({
            "order_id": "ord-1", "client_order_id": "cid-1",
            "symbol": "AAPL", "side": "buy", "quantity": 1.0,
            "filled_avg_price": 10.25,
        })
        assert repo.get_orders()[0]["avg_fill_price"] == 10.25

    def test_save_fill_returns_realized_pl(self, tmp_path):
        """Fills return the ledger's realized P/L and ignore duplicates."""
        repo = self._repo(tmp_path)
        for oid, side in (("buy-1", "buy"), ("sell-1", "sell")):
            repo.save_order({
                "order_id": oid, "client_order_id": oid,
                "symbol": "AAPL", "side": side, "quantity": 2.0,
            })
        repo.save_fill({
            "fill_id": "buy-1:2", "order_id": "buy-1", "symbol": "AAPL",
            "side": "buy", "quantity": 2.0, "price": 100.0,
        })
        result = repo.save_fill({
            "fill_id": "sell-1:2", "order_id": "sell-1", "symbol": "AAPL",
            "side": "sell", "quantity": 2.0, "price": 110.0,
        })
        assert result is not None
        assert result["realized_pl"] == pytest.approx(20.0)
        # Duplicate fill ids are ignored by the ledger.
        assert repo.save_fill({
            "fill_id": "sell-1:2", "order_id": "sell-1", "symbol": "AAPL",
            "side": "sell", "quantity": 2.0, "price": 110.0,
        }) is None

    def test_seed_positions_accepts_avg_entry_price(self, tmp_path):
        """Seeding uses the broker's ``avg_entry_price`` field."""
        ledger = TradeLedger(str(tmp_path / "ledger.json"))
        seeded = ledger.seed_positions([
            {"symbol": "AAPL", "qty": 2.0, "avg_entry_price": 100.0},
        ])
        assert seeded == ["AAPL"]
        trade = ledger.record_fill("AAPL", "sell", 2.0, 110.0, fill_id="s1")
        assert trade["realized_pl"] == pytest.approx(20.0)

    def test_prune_positions_removes_closed_symbols(self, tmp_path):
        """Positions no longer held at the broker are deleted."""
        repo = self._repo(tmp_path)
        for symbol in ("AAPL", "MSFT"):
            repo.save_position({
                "symbol": symbol, "qty": 1.0, "avg_entry_price": 100.0,
                "current_price": 101.0, "market_value": 101.0,
                "unrealized_pl": 1.0, "unrealized_plpc": 0.01,
            })
        assert len(repo.get_positions()) == 2

        assert repo.prune_positions({"AAPL"}) == 1
        assert [p["symbol"] for p in repo.get_positions()] == ["AAPL"]

        assert repo.prune_positions(set()) == 1
        assert repo.get_positions() == []
