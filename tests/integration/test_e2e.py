"""End-to-end integration tests for Phase 11 final wiring.

These tests run a single :meth:`TradingEngine.run_iteration` with fully mocked
components to verify the data -> strategy -> risk -> order -> database ->
monitoring pipeline is correctly wired. No real credentials or network are used.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from trading_bot.config.settings import Settings
from trading_bot.engine import TradingEngine
from trading_bot.exceptions import ConfigurationError
from trading_bot.state import AppState

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_settings() -> Settings:
    return Settings(
        alpaca_api_key="key",
        alpaca_api_secret="secret",
        alpaca_paper=True,
        trading_mode="paper",
        live_trading_enabled=False,
        watchlist=["AAPL"],
        indicator_lookback=60,
        indicator_timeframe="5 minutes",
        position_size_pct=0.03,
        max_position_notional=2500.0,
        max_portfolio_risk_pct=0.01,
        max_shares_per_trade=100,
        max_concurrent_positions=2,
        stop_loss_min_pct=0.01,
        stop_loss_max_pct=0.05,
        stop_loss_pct=0.985,
        take_profit_pct=1.04,
        min_reward_risk_ratio=1.0,
        daily_loss_limit_enabled=False,
    )


class _FakeDataProvider:
    def __init__(self, bars: Dict[str, List[Dict[str, Any]]], fresh: bool = True):
        self.bars = bars
        self.fresh = fresh
        self.calls: List[str] = []

    def get_historical_bars(self, symbol, limit, timeframe):
        self.calls.append(symbol)
        return self.bars.get(symbol)

    def is_data_fresh(self, symbol):
        return self.fresh

    def is_market_open(self):
        return True


class _FakeBroker:
    def __init__(self):
        self.account = {
            "account_id": "acct-1",
            "status": "active",
            "cash": 10000.0,
            "equity": 10000.0,
            "buying_power": 10000.0,
            "trading_blocked": False,
            "account_blocked": False,
        }
        self.positions: List[Dict[str, Any]] = []
        self.orders: List[Dict[str, Any]] = []
        self.submitted: List[Dict[str, Any]] = []

    def get_account(self):
        return dict(self.account)

    def get_positions(self):
        return list(self.positions)

    def get_orders(self):
        return list(self.orders)

    def submit_market_order(self, **kwargs):
        self.submitted.append(dict(kwargs))
        return {
            "order_id": "ord-1",
            "client_order_id": kwargs.get("client_order_id"),
            "symbol": kwargs["symbol"],
            "status": "new",
        }

    def is_connected(self):
        return True


class _FakeRiskEngine:
    def __init__(self, approve: bool = True):
        self.approve = approve
        self.assessed: List[Dict[str, Any]] = []
        self.halted = False

    def is_halted(self):
        return self.halted

    def assess(self, signal, account, positions):
        self.assessed.append(signal)
        if self.halted:
            return {"approved": False, "rejection_reason": "halted"}
        return {
            "approved": self.approve,
            "qty": 3.0,
            "stop_price": 98.0,
            "limit_price": 104.0,
            "reason": "approved",
        }


class _FakeStrategy:
    def __init__(self, signals: List[Dict[str, Any]]):
        self.signals = signals
        self.calls = 0

    def generate_signals(self, bars):
        self.calls += 1
        return list(self.signals)


class _FakeOrderManager:
    def __init__(self):
        self.submitted: List[Dict[str, Any]] = []
        self.fills: List[Dict[str, Any]] = []

    def submit(self, order):
        self.submitted.append(dict(order))
        return {"order_id": "om-1", "status": "new"}

    def on_fill(self, fill):
        self.fills.append(dict(fill))

    def reconcile(self, positions, orders):
        pass


class _FakeDatabase:
    def __init__(self):
        self.orders: List[Dict[str, Any]] = []
        self.fills: List[Dict[str, Any]] = []
        self.positions: List[Dict[str, Any]] = []
        self.signals: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []

    def save_order(self, order):
        self.orders.append(dict(order))

    def save_fill(self, fill):
        self.fills.append(dict(fill))

    def save_position(self, position):
        self.positions.append(dict(position))

    def save_signal(self, signal):
        self.signals.append(dict(signal))

    def save_risk_event(self, event):
        pass

    def save_account_snapshot(self, snapshot):
        pass

    def save_system_event(self, event):
        self.events.append(dict(event))


class _FakeMonitoring:
    def __init__(self):
        self.metrics: List[tuple] = []
        self.alerts: List[tuple] = []

    def health_check(self):
        return {"ok": True, "status": "ok"}

    def alert(self, message, level="error"):
        self.alerts.append((message, level))

    def record_metric(self, name, value):
        self.metrics.append((name, value))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunIteration:
    def test_full_loop_approves_and_submits(self):
        signal = {
            "symbol": "AAPL",
            "side": "buy",
            "price": 100.0,
            "quantity": 10.0,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
            "signal_id": "sig-1",
        }
        bars = {"AAPL": [{"close": 100.0, "timestamp": "2024-01-01T09:30:00Z"}]}
        settings = _make_settings()
        state = AppState()
        state.start("session-test")

        data = _FakeDataProvider(bars)
        broker = _FakeBroker()
        risk = _FakeRiskEngine(approve=True)
        strategy = _FakeStrategy([signal])
        order_mgr = _FakeOrderManager()
        database = _FakeDatabase()
        monitoring = _FakeMonitoring()

        engine = TradingEngine(
            settings=settings,
            state=state,
            data_provider=data,
            broker_adapter=broker,
            risk_engine=risk,
            strategy=strategy,
            order_manager=order_mgr,
            database=database,
            monitoring=monitoring,
        )

        summary = engine.run_iteration()

        # Strategy ran.
        assert strategy.calls == 1
        # Risk assessed the signal.
        assert len(risk.assessed) == 1
        # Approved -> submitted.
        assert summary["approved"] == 1
        assert summary["orders"] == 1
        assert len(order_mgr.submitted) == 1
        assert order_mgr.submitted[0]["symbol"] == "AAPL"
        # Database persisted order + signal.
        assert len(database.orders) == 1
        assert len(database.signals) == 1
        # Monitoring recorded metrics.
        assert ("signals_generated", 1) in monitoring.metrics
        assert ("signals_approved", 1) in monitoring.metrics
        assert ("orders_submitted", 1) in monitoring.metrics
        assert summary["errors"] == []

    def test_halted_risk_blocks_all(self):
        bars = {"AAPL": [{"close": 100.0, "timestamp": "2024-01-01T09:30:00Z"}]}
        settings = _make_settings()
        state = AppState()
        state.start("session-test")

        data = _FakeDataProvider(bars)
        broker = _FakeBroker()
        risk = _FakeRiskEngine(approve=True)
        risk.halted = True
        strategy = _FakeStrategy([
            {"symbol": "AAPL", "side": "buy", "price": 100.0, "quantity": 10.0,
             "stop_loss_pct": 0.02, "take_profit_pct": 0.04, "signal_id": "s1"}
        ])
        order_mgr = _FakeOrderManager()
        database = _FakeDatabase()
        monitoring = _FakeMonitoring()

        engine = TradingEngine(
            settings=settings, state=state, data_provider=data,
            broker_adapter=broker, risk_engine=risk, strategy=strategy,
            order_manager=order_mgr, database=database,
            monitoring=monitoring,
        )

        summary = engine.run_iteration()
        assert summary["approved"] == 0
        assert summary["rejected"] == 0
        assert summary["orders"] == 0
        assert any("circuit breaker halted" in e for e in summary["errors"])

    def test_rejected_signal_not_submitted(self):
        bars = {"AAPL": [{"close": 100.0, "timestamp": "2024-01-01T09:30:00Z"}]}
        settings = _make_settings()
        state = AppState()
        state.start("session-test")

        data = _FakeDataProvider(bars)
        broker = _FakeBroker()
        risk = _FakeRiskEngine(approve=False)
        strategy = _FakeStrategy([
            {"symbol": "AAPL", "side": "buy", "price": 100.0, "quantity": 10.0,
             "stop_loss_pct": 0.02, "take_profit_pct": 0.04, "signal_id": "s1"}
        ])
        order_mgr = _FakeOrderManager()
        database = _FakeDatabase()
        monitoring = _FakeMonitoring()

        engine = TradingEngine(
            settings=settings, state=state, data_provider=data,
            broker_adapter=broker, risk_engine=risk, strategy=strategy,
            order_manager=order_mgr, database=database,
            monitoring=monitoring,
        )

        summary = engine.run_iteration()
        assert summary["approved"] == 0
        assert summary["rejected"] == 1
        assert summary["orders"] == 0
        assert len(order_mgr.submitted) == 0

    def test_not_running_returns_safe(self):
        settings = _make_settings()
        state = AppState()  # not started
        engine = TradingEngine(settings=settings, state=state)
        summary = engine.run_iteration()
        assert summary["orders"] == 0
        assert any("not running" in e for e in summary["errors"])

    def test_validate_requires_risk_engine(self):
        settings = _make_settings()
        state = AppState()
        engine = TradingEngine(settings=settings, state=state)  # no risk engine
        with pytest.raises(ConfigurationError):
            engine.validate()

    def test_validate_rejects_live_mode(self):
        with pytest.raises(ValueError):
            Settings(
                alpaca_api_key="key", alpaca_api_secret="secret",
                alpaca_paper=True,
                trading_mode="paper",
                live_trading_enabled=True,  # live -> rejected
            )


class TestBuildComponentsSmoke:
    def test_build_components_creates_all(self):
        from trading_bot.__main__ import _build_components

        settings = _make_settings()
        state = AppState()
        comps = _build_components(settings, state)
        expected = {
            "data_provider", "broker_adapter", "risk_engine", "strategy",
            "order_manager", "database", "monitoring",
        }
        assert set(comps) == expected
        assert comps["risk_engine"].settings is settings
        # Monitoring wraps broker + data + risk health checks.
        assert set(comps["monitoring"].components) == {
            "broker", "data", "risk",
        }
        assert comps["monitoring"].components["broker"] == comps[
            "broker_adapter"
        ].is_connected
        assert comps["monitoring"].components["data"] == comps[
            "data_provider"
        ].is_market_open
