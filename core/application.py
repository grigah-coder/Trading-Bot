"""Application startup, recovery, run-loop, and graceful shutdown."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Protocol

from config.settings import Settings
from core.engine import EngineComponents, TradingEngine
from core.exceptions import (
    ApplicationShutdownError,
    ApplicationStartupError,
    ConfigurationError,
    RecoveryError,
)
from core.market_calendar import ExchangeMarketCalendar, MarketSession
from core.scheduler import MarketScheduler
from core.state import ApplicationState, StateMachine


class Scheduler(Protocol):
    def run_pending(
        self,
        now: datetime,
        *,
        on_session_start: Callable[[MarketSession], None],
        evaluate: Callable[[], object],
    ) -> bool: ...


@dataclass(frozen=True, slots=True)
class RecoveryPolicy:
    max_attempts: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_attempts, int)
            or isinstance(self.max_attempts, bool)
            or self.max_attempts < 0
        ):
            raise ValueError("max_attempts cannot be negative")
        delays = (self.initial_delay_seconds, self.max_delay_seconds)
        if any(not math.isfinite(value) or value < 0 for value in delays):
            raise ValueError("recovery delays cannot be negative")

    def delay_for(self, attempt: int) -> float:
        multiplier = 2 ** max(attempt - 1, 0)
        return min(self.initial_delay_seconds * multiplier, self.max_delay_seconds)


SettingsLoader = Callable[[], Settings]
ComponentFactory = Callable[[Settings], EngineComponents]
SchedulerFactory = Callable[[Settings], Scheduler]


def build_market_scheduler(settings: Settings) -> MarketScheduler:
    """Build the production scheduler using the validated market settings."""
    calendar = ExchangeMarketCalendar(settings.trading.calendar)
    if calendar.timezone_name != settings.trading.timezone:
        raise ConfigurationError(
            "TRADING_TIMEZONE must match the selected exchange calendar timezone",
            context={
                "configured": settings.trading.timezone,
                "calendar_timezone": calendar.timezone_name,
            },
        )
    return MarketScheduler(
        calendar,
        timezone_name=settings.trading.timezone,
        interval=timedelta(seconds=settings.trading.evaluation_interval_seconds),
    )


class TradingApplication:
    """Owns component initialization and the process-level lifecycle."""

    def __init__(
        self,
        *,
        settings_loader: SettingsLoader = Settings.from_env,
        component_factory: ComponentFactory,
        scheduler_factory: SchedulerFactory = build_market_scheduler,
        state: StateMachine | None = None,
        clock: Callable[[], datetime] | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        recovery_policy: RecoveryPolicy | None = None,
    ) -> None:
        self.settings_loader = settings_loader
        self.component_factory = component_factory
        self.scheduler_factory = scheduler_factory
        self.scheduler: Scheduler | None = None
        self.state_machine = state or StateMachine()
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.sleeper = sleeper
        self.recovery_policy = recovery_policy or RecoveryPolicy()
        self.settings: Settings | None = None
        self.engine: TradingEngine | None = None
        self._shutdown_requested = False

    @property
    def state(self) -> ApplicationState:
        return self.state_machine.current

    def start(self) -> None:
        self.state_machine.transition(ApplicationState.STARTING)
        self._shutdown_requested = False
        try:
            self.settings = self.settings_loader()
            self.scheduler = self.scheduler_factory(self.settings)
            components = self.component_factory(self.settings)
            self.engine = TradingEngine(components, state=self.state_machine)
            self.engine.health_check()
            self.state_machine.transition(ApplicationState.RUNNING)
        except Exception as exc:
            if self.engine is not None:
                try:
                    self.engine.shutdown()
                except Exception:
                    pass
                self.engine = None
            if self.state is ApplicationState.STARTING:
                self.state_machine.transition(ApplicationState.ERROR)
            raise ApplicationStartupError(
                "Application startup failed", context={"phase": "initialization"}
            ) from exc

    def request_shutdown(self) -> None:
        self._shutdown_requested = True

    def run_once(self, now: datetime | None = None) -> bool:
        if self.state is not ApplicationState.RUNNING:
            return False
        if self.engine is None or self.scheduler is None:
            raise ApplicationStartupError("Application has no initialized engine")
        return self.scheduler.run_pending(
            now or self.clock(),
            on_session_start=self.engine.start_session,
            evaluate=self.engine.run_cycle,
        )

    def run(
        self,
        *,
        max_iterations: int | None = None,
        now_factory: Callable[[], datetime] | None = None,
    ) -> None:
        if max_iterations is not None and max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.state in {ApplicationState.STOPPED, ApplicationState.ERROR}:
            self.start()
        if self.state not in {ApplicationState.RUNNING, ApplicationState.HALTED}:
            raise ApplicationStartupError(
                "Application cannot enter run loop from current state",
                context={"state": self.state.value},
            )

        get_now = now_factory or self.clock
        iterations = 0
        consecutive_failures = 0
        poll_seconds = (
            self.settings.trading.scheduler_poll_seconds if self.settings else 1.0
        )
        while not self._shutdown_requested and (
            max_iterations is None or iterations < max_iterations
        ):
            if self.state is ApplicationState.RUNNING:
                try:
                    self.run_once(get_now())
                except Exception as exc:
                    consecutive_failures += 1
                    self._recover(exc, consecutive_failures)
                else:
                    consecutive_failures = 0
            iterations += 1
            if not self._shutdown_requested and (
                max_iterations is None or iterations < max_iterations
            ):
                self.sleeper(poll_seconds)

        self.stop()

    def _recover(self, cause: Exception, attempt: int) -> None:
        self.state_machine.transition(ApplicationState.PAUSED)
        if attempt > self.recovery_policy.max_attempts:
            self.state_machine.transition(ApplicationState.ERROR)
            self._shutdown_preserving_error()
            raise RecoveryError(
                "Application recovery attempts were exhausted",
                context={"attempts": attempt},
            ) from cause

        try:
            if self.engine is None:
                raise RuntimeError("engine is unavailable")
            self.engine.health_check()
        except Exception as exc:
            self.state_machine.transition(ApplicationState.ERROR)
            self._shutdown_preserving_error()
            raise RecoveryError(
                "Application health check failed during recovery",
                context={"attempt": attempt},
            ) from exc

        self.sleeper(self.recovery_policy.delay_for(attempt))
        self.state_machine.transition(ApplicationState.RUNNING)

    def stop(self) -> None:
        if self.state is ApplicationState.STOPPED:
            return
        if self.state is ApplicationState.STOPPING:
            return
        self.state_machine.transition(ApplicationState.STOPPING)
        try:
            if self.engine is not None:
                self.engine.shutdown()
        except Exception as exc:
            self.engine = None
            self.state_machine.transition(ApplicationState.ERROR)
            raise ApplicationShutdownError(
                "Application shutdown failed", context={"phase": "component_shutdown"}
            ) from exc
        self.engine = None
        self.state_machine.transition(ApplicationState.STOPPED)

    def _shutdown_preserving_error(self) -> None:
        if self.engine is not None:
            try:
                self.engine.shutdown()
            except Exception:
                # The recovery error remains the primary failure; cleanup is best effort.
                pass
            finally:
                self.engine = None
