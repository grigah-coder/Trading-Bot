from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest

from config.settings import RiskSettings
from execution.broker import BrokerOrder, OrderSide, OrderStatus
from risk.engine import RiskEngine
from risk.exceptions import CircuitBreakerActive, RiskRejected
from execution.risk_gate import RiskGatedOrderExecutor
from risk.models import (
    AccountRiskData,
    EstimatedTransactionCosts,
    InstrumentConstraints,
    MarketConditions,
    PortfolioRiskContext,
    PortfolioRiskPosition,
    RiskRecoveryStatus,
    TradeProposal,
)

UTC = timezone.utc
NOW = datetime(2026, 11, 6, 15, 0, tzinfo=UTC)


def context(*, equity: str = "100000", day_start: str = "100000", peak: str = "100000",
            buying_power: str = "100000", positions: tuple = (), broker_available: bool = True,
            reconciled: bool = True, execution_consistent: bool = True,
            observed_at: datetime = NOW, session_open: bool = True,
            bid: str = "100", ask: str = "100.10", adv: str = "10000000",
            volatility: str = "0.02", sector: str | None = "Technology",
            sector_reliable: bool = True) -> PortfolioRiskContext:
    account = None if not broker_available else AccountRiskData(
        Decimal(equity), Decimal(day_start), Decimal(peak), Decimal(buying_power)
    )
    return PortfolioRiskContext(
        account=account,
        positions=positions,
        positions_reconciled=reconciled,
        execution_state_consistent=execution_consistent,
        broker_available=broker_available,
        market=MarketConditions(Decimal(bid), Decimal(ask), Decimal(adv),
                                Decimal(volatility), observed_at, session_open),
        instrument=InstrumentConstraints(
            quantity_increment=Decimal("1"), minimum_quantity=Decimal("1"),
            maximum_quantity=Decimal("10000"), minimum_notional=Decimal("1"),
            sector=sector, sector_classification_reliable=sector_reliable,
        ),
        costs=EstimatedTransactionCosts(Decimal("0.01"), Decimal("1")),
        trading_session_id="2026-11-06",
    )


def proposal(*, side: str = "buy", stop: str = "95", quantity: str | None = None,
             symbol: str = "AAPL", order_type: str = "market", limit_price: str | None = None) -> TradeProposal:
    return TradeProposal(symbol, side, Decimal(stop), "intent-1",
                         requested_quantity=Decimal(quantity) if quantity else None,
                         order_type=order_type,
                         limit_price=Decimal(limit_price) if limit_price else None)


def permissive_settings(**overrides) -> RiskSettings:
    values = dict(
        max_risk_per_trade_fraction=0.005,
        max_position_fraction=1.0,
        daily_loss_limit_fraction=0.02,
        max_open_positions=10,
        max_drawdown_fraction=0.10,
        max_portfolio_exposure_fraction=1.0,
        max_order_notional_fraction=1.0,
        max_sector_exposure_fraction=1.0,
        minimum_average_daily_dollar_volume=1000000.0,
        maximum_spread_fraction=0.01,
        maximum_daily_volatility_fraction=0.10,
        market_data_max_age_seconds=60,
        regular_session_only=True,
    )
    values.update(overrides)
    return RiskSettings(**values)


def test_position_size_respects_risk_budget_and_transaction_costs() -> None:
    decision = RiskEngine(permissive_settings()).approve(proposal(), context(), now=NOW)
    assert decision.quantity == Decimal("97")
    assert decision.estimated_risk == Decimal("496.67")
    assert decision.entry_price == Decimal("100.10")


def test_sell_position_uses_ask_side_stop_distance() -> None:
    decision = RiskEngine(permissive_settings()).approve(proposal(side="sell", stop="105"), context(), now=NOW)
    assert decision.quantity == Decimal("99")


@pytest.mark.parametrize("side,stop", [("buy", "100.10"), ("buy", "101"), ("sell", "100"), ("sell", "99")])
def test_invalid_stop_loss_direction_is_rejected(side: str, stop: str) -> None:
    with pytest.raises(RiskRejected, match="stop"):
        RiskEngine(permissive_settings()).approve(proposal(side=side, stop=stop), context(), now=NOW)


def test_requested_order_over_max_position_exposure_is_rejected() -> None:
    settings = permissive_settings(max_position_fraction=0.01)
    with pytest.raises(RiskRejected, match="position exposure"):
        RiskEngine(settings).approve(proposal(quantity="6"), context(positions=(
            PortfolioRiskPosition("AAPL", Decimal("500")),
        )), now=NOW)


def test_daily_loss_threshold_activates_breaker() -> None:
    engine = RiskEngine(permissive_settings())
    with pytest.raises(CircuitBreakerActive, match="Daily loss"):
        engine.approve(proposal(), context(equity="97500"))
    assert engine.breaker_active
    assert "daily_loss_limit" in {reason.value for reason in engine.breaker_reasons}


def test_drawdown_threshold_activates_breaker() -> None:
    engine = RiskEngine(permissive_settings())
    with pytest.raises(CircuitBreakerActive, match="drawdown"):
        engine.approve(proposal(), context(equity="89000", day_start="89000", peak="100000"))
    assert engine.breaker_active


def test_missing_account_data_activates_broker_unavailable_breaker() -> None:
    engine = RiskEngine(permissive_settings())
    with pytest.raises(CircuitBreakerActive, match="account"):
        engine.approve(proposal(), context(broker_available=False))
    assert "broker_unavailable" in {reason.value for reason in engine.breaker_reasons}


def test_position_reconciliation_failure_activates_breaker() -> None:
    engine = RiskEngine(permissive_settings())
    with pytest.raises(CircuitBreakerActive, match="reconcil"):
        engine.approve(proposal(), context(reconciled=False))
    assert engine.breaker_active


def test_stale_market_data_activates_breaker_and_recovers() -> None:
    engine = RiskEngine(permissive_settings())
    stale = context(observed_at=NOW - timedelta(minutes=2))
    with pytest.raises(CircuitBreakerActive, match="stale"):
        engine.approve(proposal(), stale, now=NOW)
    recovered = engine.recover(RiskRecoveryStatus(True, True, True, True), session_id="2026-11-06")
    assert recovered
    assert not engine.breaker_active
    assert engine.approve(proposal(), context(), now=NOW).quantity > 0


def test_market_controls_reject_low_liquidity_wide_spread_volatility_and_closed_session() -> None:
    settings = permissive_settings()
    for invalid in (context(adv="500000"), context(bid="99", ask="102"),
                    context(volatility="0.15"), context(session_open=False)):
        with pytest.raises(RiskRejected):
            RiskEngine(settings).approve(proposal(), invalid, now=NOW)


def test_reliable_sector_exposure_is_enforced_but_unclassified_is_not_guessed() -> None:
    engine = RiskEngine(permissive_settings(max_sector_exposure_fraction=0.05))
    with pytest.raises(RiskRejected, match="sector exposure"):
        engine.approve(proposal(quantity="60"), context(positions=(
            PortfolioRiskPosition("MSFT", Decimal("4500"), "Technology", True),
        )), now=NOW)
    assert engine.approve(proposal(quantity="60"), context(sector=None,
        sector_reliable=False, positions=(PortfolioRiskPosition("MSFT", Decimal("4500")),)), now=NOW).quantity == 60


def test_circuit_breaker_transient_recovery_requires_all_health_checks() -> None:
    engine = RiskEngine(permissive_settings())
    engine.activate_breaker("market_data_stale")
    assert not engine.recover(RiskRecoveryStatus(True, True, False, True), session_id="day")
    assert engine.breaker_active
    assert engine.recover(RiskRecoveryStatus(True, True, True, True), session_id="day")
    assert not engine.breaker_active


def test_daily_loss_breaker_recovers_only_in_new_session() -> None:
    engine = RiskEngine(permissive_settings())
    engine.activate_breaker("daily_loss_limit", session_id="day-1")
    assert not engine.recover(RiskRecoveryStatus(True, True, True, True), session_id="day-1")
    assert engine.recover(RiskRecoveryStatus(True, True, True, True), session_id="day-2")


def test_execution_gate_rejects_before_broker_submission() -> None:
    class BrokerSpy:
        def __init__(self): self.submissions = []
        def submit_market_order(self, *args, **kwargs): self.submissions.append((args, kwargs)); return object()
        def submit_limit_order(self, *args, **kwargs): self.submissions.append((args, kwargs)); return object()
        def get_account(self): return object()
        def get_open_positions(self): return ()
        def get_open_orders(self): return ()
        def get_historical_orders(self, *args): return ()
        def cancel_order(self, *args): return object()
        def replace_order(self, *args, **kwargs): return object()
        def get_order(self, *args): return object()
        def get_order_by_client_order_id(self, *args): return object()
        def shutdown(self): pass
    class ContextSource:
        def load(self, trade): return context(equity="97500")
        def health_check(self): pass
        def shutdown(self): pass
    broker = BrokerSpy()
    executor = RiskGatedOrderExecutor(RiskEngine(permissive_settings()), broker, ContextSource(), clock=lambda: NOW)
    with pytest.raises(CircuitBreakerActive):
        executor.execute(proposal())
    assert broker.submissions == []


def test_execution_gate_uses_risk_sized_quantity_for_broker_order() -> None:
    class BrokerSpy:
        def __init__(self): self.submissions = []
        def submit_market_order(self, *args, **kwargs):
            self.submissions.append((args, kwargs))
            return SimpleNamespace(status=OrderStatus.NEW)
        def submit_limit_order(self, *args, **kwargs): raise AssertionError("unexpected limit")
        def get_account(self): return object()
        def get_open_positions(self): return ()
        def get_open_orders(self): return ()
        def get_historical_orders(self, *args): return ()
        def cancel_order(self, *args): return object()
        def replace_order(self, *args, **kwargs): return object()
        def get_order(self, *args): return object()
        def get_order_by_client_order_id(self, *args): return object()
        def shutdown(self): pass
    class ContextSource:
        def load(self, trade): return context()
        def health_check(self): pass
        def shutdown(self): pass
    broker = BrokerSpy()
    executor = RiskGatedOrderExecutor(RiskEngine(permissive_settings()), broker, ContextSource(), clock=lambda: NOW)
    executor.execute(proposal())
    assert broker.submissions[0][0][1] == Decimal("97")


@pytest.mark.parametrize(("settings", "positions", "buying_power", "quantity", "message"), [
    (dict(max_order_notional_fraction=0.005), (), "100000", "6", "order notional"),
    (dict(max_portfolio_exposure_fraction=0.01), (PortfolioRiskPosition("MSFT", Decimal("500")),), "100000", "6", "portfolio exposure"),
    (dict(max_position_fraction=0.01), (PortfolioRiskPosition("AAPL", Decimal("500")),), "100000", "6", "position exposure"),
    (dict(), (), "100", "1", "buying power"),
])
def test_exposure_and_buying_power_caps_reject_oversized_orders(settings, positions,
                                                                 buying_power, quantity, message):
    risk_settings = permissive_settings(**settings)
    with pytest.raises(RiskRejected, match=message):
        RiskEngine(risk_settings).approve(
            proposal(quantity=quantity),
            context(positions=positions, buying_power=buying_power), now=NOW,
        )


def test_maximum_open_positions_is_enforced_including_existing_symbols():
    engine = RiskEngine(permissive_settings(max_open_positions=1))
    with pytest.raises(RiskRejected, match="maximum open positions"):
        engine.approve(proposal(), context(positions=(
            PortfolioRiskPosition("MSFT", Decimal("100")),
        )), now=NOW)


def test_invalid_market_price_is_rejected_without_using_fabricated_price():
    with pytest.raises(RiskRejected, match="bid"):
        RiskEngine(permissive_settings()).approve(proposal(), context(bid="0"), now=NOW)


def test_internal_risk_calculation_failure_latches_breaker():
    engine = RiskEngine(permissive_settings())
    def fail(*args, **kwargs):
        raise ArithmeticError("unexpected arithmetic failure")
    engine._assess = fail
    with pytest.raises(CircuitBreakerActive, match="Risk calculation failed"):
        engine.approve(proposal(), context(), now=NOW)
    assert "risk_calculation_failure" in {reason.value for reason in engine.breaker_reasons}


def test_execution_state_mismatch_latches_breaker():
    engine = RiskEngine(permissive_settings())
    with pytest.raises(CircuitBreakerActive, match="Execution state"):
        engine.approve(proposal(), context(execution_consistent=False), now=NOW)
    assert "unexpected_execution_state" in {reason.value for reason in engine.breaker_reasons}


def test_execution_gate_halts_on_unknown_broker_status():
    class BrokerSpy:
        def submit_market_order(self, *args, **kwargs):
            return SimpleNamespace(status=OrderStatus.UNKNOWN, order_id="x", raw_status="new-v2")
    class ContextSource:
        def load(self, trade): return context()
        def health_check(self): pass
        def shutdown(self): pass
    engine = RiskEngine(permissive_settings())
    executor = RiskGatedOrderExecutor(engine, BrokerSpy(), ContextSource(), clock=lambda: NOW)
    with pytest.raises(CircuitBreakerActive, match="unknown order status"):
        executor.execute(proposal())
    assert "unexpected_execution_state" in {reason.value for reason in engine.breaker_reasons}


def test_trading_engine_cannot_bypass_independent_execution_risk_gate():
    from core.engine import EngineComponents, TradingEngine
    from core.exceptions import EngineCycleError
    from core.state import ApplicationState, StateMachine

    class Component:
        def health_check(self): pass
        def shutdown(self): pass
    class Market(Component):
        def fetch_snapshot(self): return object()
    class Strategy(Component):
        def evaluate(self, snapshot): return proposal()
        def on_session_start(self, session): pass
    class Validator(Component):
        def validate(self, signal): return True
    class Positions(Component):
        def manage(self): pass
    class Monitor(Component):
        def check(self): return False
    class BrokerSpy:
        def __init__(self): self.submissions = []
        def submit_market_order(self, *args, **kwargs): self.submissions.append(args)
    class ContextSource:
        def load(self, trade): return context(equity="97500")
        def health_check(self): pass
        def shutdown(self): pass
    broker = BrokerSpy()
    gate = RiskGatedOrderExecutor(RiskEngine(permissive_settings()), broker,
                                  ContextSource(), clock=lambda: NOW)
    parts = EngineComponents(Market(), Strategy(), Validator(), Validator(), gate,
                             Positions(), Monitor())
    state = StateMachine()
    state.transition(ApplicationState.STARTING)
    state.transition(ApplicationState.RUNNING)
    engine = TradingEngine(parts, state=state)
    with pytest.raises(EngineCycleError):
        engine.run_cycle()
    assert broker.submissions == []


def test_drawdown_breaker_requires_healthy_acknowledged_recovery():
    engine = RiskEngine(permissive_settings())
    engine.activate_breaker("maximum_drawdown", session_id="day")
    assert not engine.recover(
        RiskRecoveryStatus(True, True, True, True, Decimal("0.11")),
        session_id="day", acknowledge_drawdown=True,
    )
    assert engine.breaker_active
    assert engine.recover(
        RiskRecoveryStatus(True, True, True, True, Decimal("0.03")),
        session_id="day", acknowledge_drawdown=True,
    )
    assert not engine.breaker_active


def test_non_numeric_risk_input_is_rejected_without_approving_order():
    with pytest.raises(RiskRejected, match="finite"):
        RiskEngine(permissive_settings()).approve(proposal(), context(bid="NaN"), now=NOW)
