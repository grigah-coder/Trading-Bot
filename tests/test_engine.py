from datetime import date, datetime, timezone

import pytest

from core.engine import EngineComponents, TradingEngine
from core.exceptions import EngineStateError
from core.market_calendar import MarketSession
from core.state import ApplicationState, StateMachine


class Component:
    def __init__(self, name: str, events: list[str]) -> None:
        self.name = name
        self.events = events

    def health_check(self) -> None:
        self.events.append(f"health:{self.name}")

    def shutdown(self) -> None:
        self.events.append(f"close:{self.name}")


class MarketData(Component):
    def fetch_snapshot(self) -> dict[str, int]:
        self.events.append("market_data")
        return {"price": 10}


class Strategy(Component):
    def evaluate(self, snapshot: dict[str, int]) -> str:
        self.events.append(f"strategy:{snapshot['price']}")
        return "signal"

    def on_session_start(self, session: MarketSession) -> None:
        self.events.append(f"session:{session.session_date}")


class Validator(Component):
    def __init__(self, name: str, events: list[str], result: bool = True) -> None:
        super().__init__(name, events)
        self.result = result

    def validate(self, signal: str) -> bool:
        self.events.append(f"validate:{self.name}:{signal}")
        return self.result


class Executor(Component):
    risk_gate_enforced = True
    def execute(self, signal: str) -> None:
        self.events.append(f"execute:{signal}")


class PositionManager(Component):
    def manage(self) -> None:
        self.events.append("manage_positions")


class PortfolioMonitor(Component):
    def __init__(self, events: list[str], halt: bool = False) -> None:
        super().__init__("portfolio", events)
        self.halt = halt

    def check(self) -> bool:
        self.events.append("monitor_portfolio")
        return self.halt


def make_engine(*, risk_approved: bool = True, halt: bool = False):
    events: list[str] = []
    state = StateMachine()
    state.transition(ApplicationState.STARTING)
    state.transition(ApplicationState.RUNNING)
    components = EngineComponents(
        market_data=MarketData("market", events),
        strategy=Strategy("strategy", events),
        signal_validator=Validator("signal", events),
        risk_validator=Validator("risk", events, result=risk_approved),
        order_executor=Executor("executor", events),
        position_manager=PositionManager("positions", events),
        portfolio_monitor=PortfolioMonitor(events, halt=halt),
    )
    return TradingEngine(components, state=state), state, events


def test_engine_coordinates_validated_cycle_and_position_monitoring() -> None:
    engine, _, events = make_engine()

    result = engine.run_cycle()

    assert result.signal_found and result.signal_valid and result.risk_approved
    assert result.order_executed and not result.halted
    assert events == [
        "market_data",
        "strategy:10",
        "validate:signal:signal",
        "validate:risk:signal",
        "execute:signal",
        "manage_positions",
        "monitor_portfolio",
    ]


def test_engine_never_executes_when_risk_validation_rejects_signal() -> None:
    engine, _, events = make_engine(risk_approved=False)

    result = engine.run_cycle()

    assert result.signal_found and result.signal_valid
    assert not result.risk_approved and not result.order_executed
    assert "execute:signal" not in events
    assert events[-2:] == ["manage_positions", "monitor_portfolio"]


def test_portfolio_halt_changes_application_state() -> None:
    engine, state, _ = make_engine(halt=True)

    result = engine.run_cycle()

    assert result.halted
    assert state.current is ApplicationState.HALTED


def test_engine_refuses_to_run_before_startup() -> None:
    engine, _, _ = make_engine()
    stopped = StateMachine()
    engine = TradingEngine(engine.components, state=stopped)

    with pytest.raises(EngineStateError):
        engine.run_cycle()


def test_engine_rejects_unprotected_order_executor() -> None:
    from dataclasses import replace
    from core.exceptions import RiskGateRequired

    engine, state, events = make_engine()
    class UnprotectedExecutor(Component):
        def execute(self, signal: str) -> None:
            events.append("unprotected")
    components = replace(engine.components, order_executor=UnprotectedExecutor("raw", events))
    with pytest.raises(RiskGateRequired):
        TradingEngine(components, state=state)
