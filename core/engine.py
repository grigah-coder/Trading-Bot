"""Dependency-injected coordination of the bot's application components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from core.exceptions import (
    ComponentHealthError,
    EngineCycleError,
    EngineShutdownError,
    EngineStateError,
    RiskGateRequired,
)
from core.market_calendar import MarketSession
from core.state import ApplicationState, StateMachine
from core.trading_gate import TradingGate

SnapshotT = TypeVar("SnapshotT")
SignalT = TypeVar("SignalT")


class ManagedComponent(Protocol):
    def health_check(self) -> None: ...

    def shutdown(self) -> None: ...


class MarketDataSource(ManagedComponent, Protocol[SnapshotT]):
    def fetch_snapshot(self) -> SnapshotT: ...


class StrategyEvaluator(ManagedComponent, Protocol[SnapshotT, SignalT]):
    def evaluate(self, snapshot: SnapshotT) -> SignalT | None: ...

    def on_session_start(self, session: MarketSession) -> None: ...


class SignalValidator(ManagedComponent, Protocol[SignalT]):
    def validate(self, signal: SignalT) -> bool: ...


class RiskValidator(ManagedComponent, Protocol[SignalT]):
    def validate(self, signal: SignalT) -> bool: ...


class OrderExecutor(ManagedComponent, Protocol[SignalT]):
    risk_gate_enforced: bool

    def execute(self, signal: SignalT) -> None: ...


class PositionManager(ManagedComponent, Protocol):
    def manage(self) -> None: ...


class PortfolioMonitor(ManagedComponent, Protocol):
    def check(self) -> bool:
        """Return True when the application must halt trading."""
        ...


@dataclass(frozen=True, slots=True)
class EngineComponents(Generic[SnapshotT, SignalT]):
    market_data: MarketDataSource[SnapshotT]
    strategy: StrategyEvaluator[SnapshotT, SignalT]
    signal_validator: SignalValidator[SignalT]
    risk_validator: RiskValidator[SignalT]
    order_executor: OrderExecutor[SignalT]
    position_manager: PositionManager
    portfolio_monitor: PortfolioMonitor

    def lifecycle_order(self) -> tuple[tuple[str, ManagedComponent], ...]:
        return (
            ("market_data", self.market_data),
            ("strategy", self.strategy),
            ("signal_validator", self.signal_validator),
            ("risk_validator", self.risk_validator),
            ("order_executor", self.order_executor),
            ("position_manager", self.position_manager),
            ("portfolio_monitor", self.portfolio_monitor),
        )


@dataclass(frozen=True, slots=True)
class CycleResult:
    signal_found: bool
    signal_valid: bool
    risk_approved: bool
    order_executed: bool
    halted: bool
    trading_blocked: bool = False


class TradingEngine(Generic[SnapshotT, SignalT]):
    """Coordinates one cycle without embedding strategy or broker behavior."""

    def __init__(
        self,
        components: EngineComponents[SnapshotT, SignalT],
        *,
        state: StateMachine,
        trading_gate: TradingGate | None = None,
    ) -> None:
        if getattr(components.order_executor, "risk_gate_enforced", False) is not True:
            raise RiskGateRequired(
                "TradingEngine requires an order executor that enforces independent risk approval"
            )
        self.components = components
        self.state = state
        self.trading_gate = trading_gate

    def health_check(self) -> None:
        for name, component in self.components.lifecycle_order():
            try:
                component.health_check()
            except Exception as exc:
                raise ComponentHealthError(
                    f"Health check failed for {name}", context={"component": name}
                ) from exc

    def start_session(self, session: MarketSession) -> None:
        self.components.strategy.on_session_start(session)

    def run_cycle(self) -> CycleResult:
        if self.state.current is not ApplicationState.RUNNING:
            raise EngineStateError(
                "Engine cycle requires RUNNING state",
                context={"state": self.state.current.value},
            )

        try:
            snapshot = self.components.market_data.fetch_snapshot()
            signal = self.components.strategy.evaluate(snapshot)
            found = signal is not None
            valid = False
            approved = False
            executed = False
            blocked = bool(self.trading_gate is not None and self.trading_gate.blocked)
            if signal is not None:
                valid = self.components.signal_validator.validate(signal)
                if valid:
                    approved = self.components.risk_validator.validate(signal)
                if approved and not blocked:
                    self.components.order_executor.execute(signal)
                    executed = True
            self.components.position_manager.manage()
            halted = self.components.portfolio_monitor.check()
            if halted:
                self.state.transition(ApplicationState.HALTED)
            return CycleResult(found, valid, approved, executed, halted, blocked)
        except EngineStateError:
            raise
        except Exception as exc:
            raise EngineCycleError("Engine cycle failed") from exc

    def shutdown(self) -> None:
        failures: list[str] = []
        for name, component in reversed(self.components.lifecycle_order()):
            try:
                component.shutdown()
            except Exception:
                failures.append(name)
        if failures:
            raise EngineShutdownError(
                "One or more components failed to shut down",
                context={"components": tuple(failures)},
            )
