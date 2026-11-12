"""Unit tests for trading engine (Phase 2).

Uses mock components to verify orchestration, risk gating, fail-closed
behaviour, and error handling without touching the network or broker.
"""


import pytest

from trading_bot.config.settings import Settings
from trading_bot.engine import TradingEngine
from trading_bot.exceptions import ConfigurationError, RiskDataUnavailableError
from trading_bot.risk.engine import RiskEngine
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
        risk_state_file="",
    )
    data.update(overrides)
    return Settings(**data)


class _MockRisk:
    def __init__(self, halted=False, approve=True, raise_on=None,
                 qty=3.0, stop_price=98.0, limit_price=104.0):
        self.halted = halted
        self.approve = approve
        self.raise_on = raise_on
        self.qty = qty
        self.stop_price = stop_price
        self.limit_price = limit_price
        self.calls = []
        self.losses = []

    def is_halted(self):
        return self.halted

    def record_loss(self, usd):
        self.losses.append(usd)

    def assess(self, signal, account, positions):
        self.calls.append((signal, account, positions))
        if self.raise_on == "unavailable":
            raise RiskDataUnavailableError("account data unavailable")
        return {
            "approved": self.approve,
            "qty": self.qty,
            "stop_price": self.stop_price,
            "limit_price": self.limit_price,
        }


class _MockStrategy:
    def __init__(self, signals=None):
        self.signals = signals or []
        self.calls = 0

    def generate_signals(self, bars):
        self.calls += 1
        return list(self.signals)


class _MockDataProvider:
    def __init__(self, fresh=True, market_open=True, blackout=False):
        self.fresh = fresh
        self.market_open = market_open
        self.blackout = blackout

    def get_historical_bars(self, symbol, limit, timeframe):
        return [{"close": 100.0}] * limit

    def is_data_fresh(self, symbol):
        return self.fresh

    def is_market_open(self):
        return self.market_open

    def is_in_opening_blackout(self):
        return self.blackout


class _MockBroker:
    def __init__(self, account=None, positions=None, orders=None):
        self.account = account or {"equity": 100000.0}
        self.positions = positions or []
        self.orders = orders or []

    def get_account(self):
        return self.account

    def get_positions(self):
        return self.positions

    def get_orders(self):
        return self.orders


class _MockOrderManager:
    def __init__(self, submitted=None, fail_submit=False):
        self.submitted = submitted or {"status": "accepted"}
        self.fail_submit = fail_submit
        self.submissions = []
        self.cancelled_brackets = []
        self.fills = []
        self.reconciled = []

    def submit(self, order):
        self.submissions.append(order)
        if self.fail_submit:
            return None
        return self.submitted

    def cancel_bracket_legs(self, symbol):
        self.cancelled_brackets.append(symbol)
        return []

    def on_fill(self, fill):
        self.fills.append(fill)

    def reconcile(self, positions, orders):
        self.reconciled.append((list(positions), list(orders)))


class _MockDatabase:
    def __init__(self):
        self.orders = []
        self.fills = []
        self.positions = []
        self.signals = []
        self.events = []
        self.pruned = []
        self.realized_pl = None

    def save_order(self, order):
        self.orders.append(order)

    def save_fill(self, fill):
        self.fills.append(fill)
        if fill.get("side") == "sell" and self.realized_pl is not None:
            return {"realized_pl": self.realized_pl}
        return {"side": fill.get("side")}

    def prune_positions(self, symbols):
        self.pruned.append(set(symbols))

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
        self.breaker_syncs = []

    def health_check(self):
        return {"ok": True}

    def alert(self, message, level="error"):
        self.alerts.append((message, level))

    def record_metric(self, name, value):
        self.metrics[name] = value

    def set_circuit_breaker(self, tripped, reason=""):
        self.breaker_syncs.append((tripped, reason))


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
    order = order_mgr.submissions[0]
    # Risk resized 10 -> 3 and supplied protective prices.
    assert order["quantity"] == 3.0
    assert order["stop_price"] == 98.0
    assert order["limit_price"] == 104.0
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


# ---------------------------------------------------------------------------
# Portfolio gating and long-only exits
# ---------------------------------------------------------------------------


def _risk_signal(symbol):
    """Build a buy signal that passes every rule of the real risk engine."""
    return {
        "symbol": symbol,
        "side": "buy",
        "price": 100.0,
        "stop_loss_pct": 0.012,
        "take_profit_pct": 0.04,
    }


def _tradable_account():
    return {
        "status": "active",
        "equity": 100000.0,
        "cash": 100000.0,
        "account_blocked": False,
        "trading_blocked": False,
    }


def test_concurrent_position_cap_limits_a_single_batch():
    """Approved-but-unfilled entries count toward the concurrency cap."""
    settings = _make_settings(max_concurrent_positions=2)
    risk = RiskEngine(settings)
    strategy = _MockStrategy(signals=[
        _risk_signal("AAPL"),
        _risk_signal("MSFT"),
        _risk_signal("TSLA"),
    ])
    order_mgr = _MockOrderManager()
    engine, _ = _engine(
        settings=settings, risk=risk, strategy=strategy,
        data=_MockDataProvider(),
        broker=_MockBroker(account=_tradable_account()),
        order_mgr=order_mgr,
    )
    summary = engine.run_iteration()
    assert summary["signals"] == 3
    assert summary["approved"] == 2
    assert summary["orders"] == 2
    assert summary["rejected"] == 1
    assert [o["symbol"] for o in order_mgr.submissions] == ["AAPL", "MSFT"]


def test_sell_on_held_position_exits_without_risk_assessment():
    """Long-only exit: cancel protective legs, then market-sell the position."""
    settings = _make_settings()
    risk = _MockRisk(approve=False)
    strategy = _MockStrategy(signals=[{"symbol": "AAPL", "side": "sell"}])
    order_mgr = _MockOrderManager()
    db = _MockDatabase()
    broker = _MockBroker(positions=[
        {"symbol": "AAPL", "qty": 5.0, "side": "long"},
    ])
    engine, _ = _engine(
        settings=settings, risk=risk, strategy=strategy,
        data=_MockDataProvider(), broker=broker, order_mgr=order_mgr, db=db,
    )
    summary = engine.run_iteration()
    # Sells never go through risk assessment (which would reject here).
    assert risk.calls == []
    assert summary["exits"] == 1
    assert summary["ignored_sells"] == 0
    assert summary["rejected"] == 0
    assert order_mgr.cancelled_brackets == ["AAPL"]
    assert len(order_mgr.submissions) == 1
    exit_order = order_mgr.submissions[0]
    assert exit_order["symbol"] == "AAPL"
    assert exit_order["side"] == "sell"
    assert exit_order["quantity"] == 5.0
    assert exit_order["order_type"] == "market"
    assert "stop_price" not in exit_order
    assert "limit_price" not in exit_order
    assert len(db.orders) == 1


def test_sell_on_flat_symbol_is_ignored():
    """A sell with no held position must not open a short."""
    settings = _make_settings()
    risk = _MockRisk()
    strategy = _MockStrategy(signals=[{"symbol": "AAPL", "side": "sell"}])
    order_mgr = _MockOrderManager()
    engine, _ = _engine(
        settings=settings, risk=risk, strategy=strategy,
        data=_MockDataProvider(), broker=_MockBroker(), order_mgr=order_mgr,
    )
    summary = engine.run_iteration()
    assert risk.calls == []
    assert summary["ignored_sells"] == 1
    assert summary["exits"] == 0
    assert summary["orders"] == 0
    assert order_mgr.submissions == []
    assert order_mgr.cancelled_brackets == []


# ---------------------------------------------------------------------------
# Fills, persistence and daily loss
# ---------------------------------------------------------------------------


def _broker_order(**overrides):
    order = {
        "order_id": "ord-1",
        "client_order_id": "cid-1",
        "symbol": "AAPL",
        "side": "buy",
        "qty": 3.0,
        "filled_qty": 3.0,
        "filled_avg_price": 100.0,
        "order_type": "market",
        "status": "filled",
        "filled_at": "2024-01-02T15:00:00+00:00",
        "legs": [],
    }
    order.update(overrides)
    return order


def _reconcile_engine(orders, db=None, risk=None):
    settings = _make_settings()
    order_mgr = _MockOrderManager()
    db = db or _MockDatabase()
    risk = risk or _MockRisk()
    engine, _ = _engine(
        settings=settings, risk=risk, strategy=_MockStrategy(signals=[]),
        data=_MockDataProvider(),
        broker=_MockBroker(orders=list(orders)), order_mgr=order_mgr, db=db,
    )
    return engine, order_mgr, db, risk


def test_new_buy_fill_updates_manager_database_and_counter():
    """A broker fill updates the manager, DB, and filled counter."""
    engine, order_mgr, db, risk = _reconcile_engine([_broker_order()])
    engine.run_iteration()

    assert len(order_mgr.fills) == 1
    assert order_mgr.fills[0]["filled_qty"] == 3.0
    assert len(db.fills) == 1
    assert db.fills[0]["fill_id"] == "ord-1:3"
    assert db.fills[0]["quantity"] == 3.0
    assert any(o.get("order_id") == "ord-1" for o in db.orders)
    assert engine.state.snapshot()["orders_filled"] == 1
    assert risk.losses == []


def test_sell_fill_loss_is_recorded_with_risk_engine():
    """Realized sell losses feed the risk engine's daily counter."""
    db = _MockDatabase()
    db.realized_pl = -25.0
    engine, _, _, risk = _reconcile_engine(
        [_broker_order(side="sell", filled_avg_price=90.0)], db=db,
    )
    engine.run_iteration()

    assert risk.losses == [25.0]
    assert engine.state.snapshot()["orders_filled"] == 1


def test_same_fill_is_not_counted_twice():
    """Repeated reconciliation must not double-record a fill or loss."""
    db = _MockDatabase()
    db.realized_pl = -25.0
    engine, _, _, risk = _reconcile_engine(
        [_broker_order(side="sell", filled_avg_price=90.0)], db=db,
    )
    engine.run_iteration()
    engine.run_iteration()

    assert len(db.fills) == 1
    assert risk.losses == [25.0]
    assert engine.state.snapshot()["orders_filled"] == 1


def test_bracket_exit_leg_fill_is_recorded():
    """Nested bracket exit legs are persisted and counted as fills."""
    db = _MockDatabase()
    db.realized_pl = -12.0
    parent = _broker_order(legs=[
        _broker_order(
            order_id="tp-1", side="sell", filled_avg_price=104.0,
            order_type="limit", limit_price=104.0,
        ),
    ])
    engine, order_mgr, _, risk = _reconcile_engine([parent], db=db)
    engine.run_iteration()

    assert {f["order_id"] for f in order_mgr.fills} == {"ord-1", "tp-1"}
    assert {f["order_id"] for f in db.fills} == {"ord-1", "tp-1"}
    assert risk.losses == [12.0]
    assert engine.state.snapshot()["orders_filled"] == 2


def test_terminal_order_state_is_persisted():
    """Canceled/filled terminal states are written back to the database."""
    engine, _, db, _ = _reconcile_engine([
        _broker_order(filled_qty=0.0, filled_avg_price=0.0, status="canceled"),
    ])
    engine.run_iteration()

    assert any(
        o.get("order_id") == "ord-1" and o.get("status") == "canceled"
        for o in db.orders
    )


def test_positions_are_pruned_to_broker_holdings():
    """Closed symbols are removed from the persisted positions table."""
    settings = _make_settings()
    db = _MockDatabase()
    engine, _ = _engine(
        settings=settings, risk=_MockRisk(), strategy=_MockStrategy(signals=[]),
        data=_MockDataProvider(),
        broker=_MockBroker(
            positions=[{"symbol": "AAPL", "qty": 2.0}], orders=[],
        ),
        order_mgr=_MockOrderManager(), db=db,
    )
    engine.run_iteration()
    assert db.pruned[-1] == {"AAPL"}


def test_rejected_signal_writes_risk_event():
    """Risk rejections are persisted as risk events."""
    settings = _make_settings()
    db = _MockDatabase()
    engine, _ = _engine(
        settings=settings, risk=_MockRisk(approve=False),
        strategy=_MockStrategy(signals=[
            {"symbol": "AAPL", "side": "buy", "quantity": 1, "price": 100.0},
        ]),
        data=_MockDataProvider(), broker=_MockBroker(), db=db,
    )
    engine.run_iteration()

    rejected = [e for e in db.events if e.get("event_type") == "rejected"]
    assert len(rejected) == 1
    assert rejected[0]["symbol"] == "AAPL"
    assert rejected[0]["reason"]


def test_halted_engine_writes_risk_event_once():
    """A breaker halt is persisted once, not every iteration."""
    settings = _make_settings()
    db = _MockDatabase()
    engine, _ = _engine(
        settings=settings, risk=_MockRisk(halted=True),
        strategy=_MockStrategy(signals=[]), data=_MockDataProvider(),
        broker=_MockBroker(), db=db,
    )
    engine.run_iteration()
    engine.run_iteration()

    halted = [e for e in db.events if e.get("event_type") == "halted"]
    assert len(halted) == 1


def test_failed_order_submission_is_counted_as_failed():
    """A None submission counts as failed, not as rejected or submitted."""
    order_mgr = _MockOrderManager(fail_submit=True)
    engine, _ = _engine(
        settings=_make_settings(), risk=_MockRisk(),
        strategy=_MockStrategy(signals=[
            {"symbol": "AAPL", "side": "buy", "quantity": 1, "price": 100.0},
        ]),
        data=_MockDataProvider(), broker=_MockBroker(),
        order_mgr=order_mgr, db=_MockDatabase(),
    )
    summary = engine.run_iteration()

    assert len(order_mgr.submissions) == 1
    assert summary["approved"] == 1
    assert summary["orders"] == 0
    assert summary["failed"] == 1
    assert summary["rejected"] == 0
    assert any("submission failed" in e for e in summary["errors"])


# ---------------------------------------------------------------------------
# Runtime correctness: session gating, stale flags, alerts, startup reconcile
# ---------------------------------------------------------------------------


def _buy_signal():
    return {"symbol": "AAPL", "side": "buy", "quantity": 1, "price": 100.0}


def test_data_stale_is_cleared_after_a_healthy_pass():
    """A previous stale flag must not survive a fully healthy data pass."""
    engine, state = _engine(
        risk=_MockRisk(), strategy=_MockStrategy(signals=[]),
        data=_MockDataProvider(fresh=True), broker=_MockBroker(),
    )
    state.set_data_stale(True)
    engine.run_iteration()
    assert state.is_data_stale() is False


def test_one_stale_symbol_marks_data_stale():
    """Stale data on any symbol marks the whole snapshot stale."""
    class PartialProvider(_MockDataProvider):
        def is_data_fresh(self, symbol):
            return symbol != "AAPL"

    engine, state = _engine(
        risk=_MockRisk(), strategy=_MockStrategy(signals=[]),
        data=PartialProvider(), broker=_MockBroker(),
    )
    engine.run_iteration()
    assert state.is_data_stale() is True


def test_closed_market_blocks_new_entries():
    """New buys are suppressed when the market is closed."""
    order_mgr = _MockOrderManager()
    engine, _ = _engine(
        risk=_MockRisk(), strategy=_MockStrategy(signals=[_buy_signal()]),
        data=_MockDataProvider(market_open=False), broker=_MockBroker(),
        order_mgr=order_mgr,
    )
    summary = engine.run_iteration()

    assert summary["entries_blocked"] == 1
    assert summary["approved"] == 0
    assert summary["orders"] == 0
    assert order_mgr.submissions == []


def test_closed_market_still_allows_exits():
    """Protection/exits must run even when new entries are blocked."""
    broker = _MockBroker(positions=[
        {"symbol": "AAPL", "qty": 5.0, "side": "long"},
    ])
    order_mgr = _MockOrderManager()
    engine, _ = _engine(
        risk=_MockRisk(approve=False),
        strategy=_MockStrategy(signals=[{"symbol": "AAPL", "side": "sell"}]),
        data=_MockDataProvider(market_open=False), broker=broker,
        order_mgr=order_mgr,
    )
    summary = engine.run_iteration()

    assert summary["exits"] == 1
    assert order_mgr.submissions[0]["side"] == "sell"


def test_opening_blackout_blocks_entries_even_with_filter_disabled():
    """The opening blackout is a risk rule, independent of the time filter."""
    settings = _make_settings(time_filter_enabled=False)
    order_mgr = _MockOrderManager()
    engine, _ = _engine(
        settings=settings, risk=_MockRisk(),
        strategy=_MockStrategy(signals=[_buy_signal()]),
        data=_MockDataProvider(market_open=True, blackout=True),
        broker=_MockBroker(), order_mgr=order_mgr,
    )
    summary = engine.run_iteration()

    assert summary["entries_blocked"] == 1
    assert order_mgr.submissions == []


def test_disabled_time_filter_ignores_market_open():
    """With TIME_FILTER_ENABLED=false, entries are not market-hours gated."""
    settings = _make_settings(time_filter_enabled=False)
    order_mgr = _MockOrderManager()
    engine, _ = _engine(
        settings=settings, risk=_MockRisk(),
        strategy=_MockStrategy(signals=[_buy_signal()]),
        data=_MockDataProvider(market_open=False, blackout=False),
        broker=_MockBroker(), order_mgr=order_mgr,
    )
    summary = engine.run_iteration()

    assert summary["entries_blocked"] == 0
    assert summary["orders"] == 1
    assert len(order_mgr.submissions) == 1


def test_halt_alerts_and_syncs_monitoring_breaker():
    """A risk halt raises an alert and mirrors the breaker onto monitoring."""
    mon = _MockMonitoring()
    engine, _ = _engine(
        risk=_MockRisk(halted=True), strategy=_MockStrategy(signals=[]),
        data=_MockDataProvider(), broker=_MockBroker(),
        db=_MockDatabase(), mon=mon,
    )
    engine.run_iteration()

    assert any(
        "circuit breaker tripped" in message for message, _ in mon.alerts
    )
    assert mon.breaker_syncs[0][0] is True


def test_breaker_reset_is_synced_to_monitoring():
    """When the risk engine clears its halt, monitoring is reset too."""
    mon = _MockMonitoring()
    risk = _MockRisk(halted=True)
    engine, _ = _engine(
        risk=risk, strategy=_MockStrategy(signals=[]),
        data=_MockDataProvider(), broker=_MockBroker(),
        db=_MockDatabase(), mon=mon,
    )
    engine.run_iteration()
    risk.halted = False
    engine.run_iteration()

    assert mon.breaker_syncs[-1][0] is False


def test_iteration_crash_alerts_and_records_error():
    """An unexpected iteration failure raises a critical alert."""
    class BadStrategy:
        def generate_signals(self, bars):
            return object()  # len() raises -> outer crash handler

    mon = _MockMonitoring()
    engine, state = _engine(
        risk=_MockRisk(), strategy=BadStrategy(),
        data=_MockDataProvider(), broker=_MockBroker(),
        db=_MockDatabase(), mon=mon,
    )
    summary = engine.run_iteration()

    assert any("crashed" in message for message, _ in mon.alerts)
    assert any("unexpected" in error for error in summary["errors"])
    assert any("unexpected" in error for error in state.errors)


def test_reconcile_startup_runs_before_first_iteration():
    """Startup reconciliation processes broker orders and fills."""
    broker = _MockBroker(
        positions=[{"symbol": "AAPL", "qty": 2.0}],
        orders=[_broker_order()],
    )
    order_mgr = _MockOrderManager()
    db = _MockDatabase()
    engine, _ = _engine(
        risk=_MockRisk(), strategy=_MockStrategy(signals=[]),
        data=_MockDataProvider(), broker=broker,
        order_mgr=order_mgr, db=db,
    )

    engine.reconcile_startup()

    assert order_mgr.reconciled
    assert order_mgr.reconciled[0][0] == broker.positions
    assert len(db.fills) == 1
