"""Unit tests for trading engine (Phase 2).

Uses mock components to verify orchestration, risk gating, fail-closed
behaviour, and error handling without touching the network or broker.
"""


import pytest

from trading_bot.config.settings import Settings
from trading_bot.engine import TradingEngine
from trading_bot.exceptions import ConfigurationError, RiskDataUnavailableError
from trading_bot.state import AppState


def _make_settings(**overrides):
    data = dict(
        alpaca_api_key="key",
        alpaca_api_secret="secret",
        alpaca_paper=True,
        trading_mode="paper",
        live_trading_enabled=False,
        watchlist=["AAPL"],
        indicator_lookback=50,
        indicator_timeframe="5 minutes",
    )
    data.update(overrides)
    return Settings(**data)


class _MockRisk:
    def __init__(self, halted=False, approve=True, raise_on=None):
        self.halted = halted
        self.approve = approve
        self.raise_on = raise_on
        self.calls = []

    def is_halted(self):
        return self.halted

    def assess(self, signal, account, positions):
        self.calls.append((signal, account, positions))
        if self.raise_on == "unavailable":
            raise RiskDataUnavailableError("account data unavailable")
        return {"approved": self.approve}


class _MockStrategy:
    def __init__(self, signals=None):
        self.signals = signals or []
        self.calls = 0

    def generate_signals(self, bars):
        self.calls += 1
        return list(self.signals)


class _MockDataProvider:
    def __init__(self, fresh=True):
        self.fresh = fresh

    def get_historical_bars(self, symbol, limit, timeframe):
        return [{"close": 100.0}] * limit

    def is_data_fresh(self, symbol):
        return self.fresh


class _MockBroker:
    def __init__(self, account=None, positions=None):
        self.account = account or {"equity": 100000.0}
        self.positions = positions or []

    def get_account(self):
        return self.account

    def get_positions(self):
        return self.positions


class _MockOrderManager:
    def __init__(self, submitted=None):
        self.submitted = submitted or {"status": "accepted"}
        self.submissions = []

    def submit(self, order):
        self.submissions.append(order)
        return self.submitted

    def on_fill(self, fill):
        pass

    def reconcile(self, positions, orders):
        pass


class _MockDatabase:
    def __init__(self):
        self.orders = []
        self.fills = []
        self.positions = []
        self.signals = []
        self.events = []

    def save_order(self, order):
        self.orders.append(order)

    def save_fill(self, fill):
        self.fills.append(fill)

    def save_position(self, position):
        self.positions.append(position)

    def save_signal(self, signal):
        self.signals.append(signal)

    def save_risk_event(self, event):
        self.events.append(event)

    def save_account_snapshot(self, snapshot):
        pass

    def save_system_event(self, event):
        self.events.append(event)


class _MockMonitoring:
    def __init__(self):
        self.metrics = {}
        self.alerts = []

    def health_check(self):
        return {"ok": True}

    def alert(self, message, level="error"):
        self.alerts.append((message, level))

    def record_metric(self, name, value):
        self.metrics[name] = value


def _engine(settings=None, risk=None, strategy=None, data=None, broker=None,
            order_mgr=None, db=None, mon=None):
    if settings is None:
        settings = _make_settings()
    state = AppState()
    state.start("session-test")
    return TradingEngine(
        settings=settings,
        state=state,
        data_provider=data,
        broker_adapter=broker,
        risk_engine=risk,
        strategy=strategy,
        order_manager=order_mgr,
        database=db,
        monitoring=mon,
    ), state


def test_engine_validation_paper_only():
    # Bypass pydantic construction validation to simulate a live-mode
    # settings object; engine.validate must reject it.
    settings = Settings.model_construct(
        alpaca_api_key="key",
        alpaca_api_secret="secret",
        alpaca_paper=False,
        trading_mode="live",
        live_trading_enabled=True,
        watchlist=["AAPL"],
        indicator_lookback=50,
        indicator_timeframe="5 minutes",
    )
    state = AppState()
    engine = TradingEngine(settings=settings, state=state)
    with pytest.raises(ConfigurationError, match="paper trading"):
        engine.validate()


def test_engine_validation_missing_credentials():
    settings = _make_settings(alpaca_api_key="", alpaca_api_secret="")
    state = AppState()
    engine = TradingEngine(settings=settings, state=state)
    with pytest.raises(ConfigurationError, match="ALPACA_API_KEY"):
        engine.validate()


def test_engine_validation_requires_risk_engine():
    settings = _make_settings()
    state = AppState()
    engine = TradingEngine(settings=settings, state=state, risk_engine=None)
    with pytest.raises(ConfigurationError, match="risk engine is required"):
        engine.validate()


def test_run_iteration_halted_risk_blocks_all():
    settings = _make_settings()
    risk = _MockRisk(halted=True)
    strategy = _MockStrategy(signals=[{"symbol": "AAPL", "side": "buy", "quantity": 10}])
    engine, state = _engine(settings=settings, risk=risk, strategy=strategy,
                            data=_MockDataProvider())
    summary = engine.run_iteration()
    # Halted -> fail closed before strategy; no signals generated, no orders.
    assert summary["signals"] == 0
    assert summary["approved"] == 0
    assert summary["orders"] == 0
    assert any("halted" in e for e in summary["errors"])
    assert risk.calls == []  # no assessment when halted
    assert state.is_data_stale() is True


def test_run_iteration_approves_and_submits():
    settings = _make_settings()
    risk = _MockRisk(approve=True)
    strategy = _MockStrategy(signals=[{"symbol": "AAPL", "side": "buy", "quantity": 10}])
    order_mgr = _MockOrderManager()
    db = _MockDatabase()
    mon = _MockMonitoring()
    engine, state = _engine(settings=settings, risk=risk, strategy=strategy,
                            data=_MockDataProvider(), broker=_MockBroker(),
                            order_mgr=order_mgr, db=db, mon=mon)
    summary = engine.run_iteration()
    assert summary["signals"] == 1
    assert summary["approved"] == 1
    assert summary["orders"] == 1
    assert len(order_mgr.submissions) == 1
    assert len(db.signals) == 1
    assert len(db.orders) == 1
    assert mon.metrics["signals_approved"] == 1


def test_run_iteration_rejects_signal():
    settings = _make_settings()
    risk = _MockRisk(approve=False)
    strategy = _MockStrategy(signals=[{"symbol": "AAPL", "side": "buy", "quantity": 10}])
    engine, state = _engine(settings=settings, risk=risk, strategy=strategy,
                            data=_MockDataProvider())
    summary = engine.run_iteration()
    assert summary["approved"] == 0
    assert summary["rejected"] == 1
    assert summary["orders"] == 0


def test_run_iteration_risk_data_unavailable_is_fail_closed():
    settings = _make_settings()
    risk = _MockRisk(raise_on="unavailable")
    strategy = _MockStrategy(signals=[
        {"symbol": "AAPL", "side": "buy", "quantity": 10},
        {"symbol": "TSLA", "side": "buy", "quantity": 10},
    ])
    engine, state = _engine(settings=settings, risk=risk, strategy=strategy,
                            data=_MockDataProvider(), broker=_MockBroker())
    summary = engine.run_iteration()
    # First signal triggers RiskDataUnavailableError -> fail closed, stop processing
    assert summary["approved"] == 0
    assert summary["orders"] == 0
    assert any("unavailable" in e for e in summary["errors"])
    assert state.is_data_stale() is True


def test_run_iteration_stale_data_blocks():
    settings = _make_settings()
    risk = _MockRisk(approve=True)
    strategy = _MockStrategy(signals=[{"symbol": "AAPL", "side": "buy", "quantity": 10}])
    engine, state = _engine(settings=settings, risk=risk, strategy=strategy,
                            data=_MockDataProvider(fresh=False))
    summary = engine.run_iteration()
    assert summary["orders"] == 0
    assert any("stale" in e.lower() or "fresh" in e.lower() for e in summary["errors"])


def test_state_snapshot_recorded():
    settings = _make_settings()
    risk = _MockRisk(approve=True)
    strategy = _MockStrategy(signals=[{"symbol": "AAPL", "side": "buy", "quantity": 10}])
    engine, state = _engine(settings=settings, risk=risk, strategy=strategy,
                            data=_MockDataProvider(), broker=_MockBroker())
    engine.run_iteration()
    snap = state.snapshot()
    assert snap["signals_generated"] == 1
    assert snap["signals_approved"] == 1
