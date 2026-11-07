from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from config.settings import Settings
from core.application import RecoveryPolicy, TradingApplication, build_market_scheduler
from core.engine import EngineComponents
from core.exceptions import ApplicationStartupError, ConfigurationError
from core.market_calendar import MarketSession
from core.scheduler import MarketScheduler
from core.state import ApplicationState


class AlwaysOpenCalendar:
    def __init__(self, trading_date: date) -> None:
        zone = ZoneInfo("America/New_York")
        self.session = MarketSession(
            trading_date,
            datetime.combine(trading_date, datetime.min.time(), tzinfo=zone).replace(
                hour=9, minute=30
            ),
            datetime.combine(trading_date, datetime.min.time(), tzinfo=zone).replace(
                hour=16
            ),
        )

    def session_for(self, trading_date: date) -> MarketSession | None:
        return self.session if trading_date == self.session.session_date else None


class Component:
    def __init__(self, name: str, events: list[str]) -> None:
        self.name = name
        self.events = events

    def health_check(self) -> None:
        self.events.append(f"health:{self.name}")

    def shutdown(self) -> None:
        self.events.append(f"close:{self.name}")


class MarketData(Component):
    def fetch_snapshot(self) -> object:
        self.events.append("market")
        return object()


class Strategy(Component):
    def __init__(self, events: list[str], failures: int = 0) -> None:
        super().__init__("strategy", events)
        self.failures = failures
        self.evaluations = 0

    def on_session_start(self, session: MarketSession) -> None:
        self.events.append("session_start")

    def evaluate(self, snapshot: object) -> None:
        self.evaluations += 1
        self.events.append("strategy")
        if self.failures:
            self.failures -= 1
            raise RuntimeError("temporary strategy failure")
        return None


class Validator(Component):
    def validate(self, signal: object) -> bool:
        return True


class Executor(Component):
    risk_gate_enforced = True
    def execute(self, signal: object) -> None:
        self.events.append("execute")


class PositionManager(Component):
    def manage(self) -> None:
        self.events.append("positions")


class PortfolioMonitor(Component):
    def check(self) -> bool:
        self.events.append("portfolio")
        return False


def components(events: list[str], failures: int = 0) -> tuple[EngineComponents, Strategy]:
    strategy = Strategy(events, failures=failures)
    return (
        EngineComponents(
            market_data=MarketData("market", events),
            strategy=strategy,
            signal_validator=Validator("signal", events),
            risk_validator=Validator("risk", events),
            order_executor=Executor("executor", events),
            position_manager=PositionManager("positions", events),
            portfolio_monitor=PortfolioMonitor("monitor", events),
        ),
        strategy,
    )


def make_application(
    *, failures: int = 0, factory_error: Exception | None = None, health_error: bool = False
):
    trading_date = date(2026, 7, 2)
    now = datetime(2026, 7, 2, 14, 0, tzinfo=timezone.utc)
    events: list[str] = []
    created: dict[str, Strategy] = {}

    def component_factory(settings: Settings) -> EngineComponents:
        events.append("initialize")
        if factory_error:
            raise factory_error
        built, strategy = components(events, failures)
        if health_error:
            def fail_health() -> None:
                raise RuntimeError("market data unavailable")

            built.market_data.health_check = fail_health
        created["strategy"] = strategy
        return built

    application = TradingApplication(
        settings_loader=lambda: Settings(),
        component_factory=component_factory,
        scheduler_factory=lambda settings: MarketScheduler(
            AlwaysOpenCalendar(trading_date),
            timezone_name=settings.trading.timezone,
            interval=timedelta(seconds=settings.trading.evaluation_interval_seconds),
        ),
        clock=lambda: now,
        sleeper=lambda _: None,
        recovery_policy=RecoveryPolicy(max_attempts=2, initial_delay_seconds=0),
    )
    return application, events, created, now


def test_startup_initializes_and_health_checks_injected_components() -> None:
    application, events, _, _ = make_application()

    application.start()

    assert application.state is ApplicationState.RUNNING
    assert events[0] == "initialize"
    assert sum(event.startswith("health:") for event in events) == 7


def test_graceful_shutdown_closes_components_in_reverse_order() -> None:
    application, events, _, _ = make_application()
    application.start()

    application.stop()

    assert application.state is ApplicationState.STOPPED
    assert [event for event in events if event.startswith("close:")] == [
        "close:monitor",
        "close:positions",
        "close:executor",
        "close:risk",
        "close:signal",
        "close:strategy",
        "close:market",
    ]


def test_startup_failure_sets_error_and_reports_structured_exception() -> None:
    application, _, _, _ = make_application(factory_error=RuntimeError("init failed"))

    with pytest.raises(ApplicationStartupError) as raised:
        application.start()

    assert raised.value.code == "application_startup_failed"
    assert application.state is ApplicationState.ERROR


def test_invalid_configuration_fails_during_startup() -> None:
    application, _, _, _ = make_application()
    application.settings_loader = lambda: Settings.from_env({"ALPACA_PAPER": "false"})

    with pytest.raises(ApplicationStartupError):
        application.start()

    assert application.state is ApplicationState.ERROR


def test_health_failure_during_startup_closes_initialized_components() -> None:
    application, events, _, _ = make_application(health_error=True)

    with pytest.raises(ApplicationStartupError):
        application.start()

    assert application.state is ApplicationState.ERROR
    assert len([event for event in events if event.startswith("close:")]) == 7


def test_production_scheduler_uses_matching_exchange_timezone() -> None:
    scheduler = build_market_scheduler(Settings())

    assert scheduler.timezone.key == "America/New_York"

    incompatible = Settings.from_env({"TRADING_TIMEZONE": "America/Los_Angeles"})
    with pytest.raises(ConfigurationError, match="must match"):
        build_market_scheduler(incompatible)


def test_recovery_policy_rejects_non_finite_delays() -> None:
    with pytest.raises(ValueError):
        RecoveryPolicy(initial_delay_seconds=float("nan"))


def test_unexpected_cycle_failure_recovers_and_continues() -> None:
    application, events, created, now = make_application(failures=1)
    application.start()

    application.run(max_iterations=2, now_factory=lambda: now)

    assert application.state is ApplicationState.STOPPED
    assert created["strategy"].evaluations == 2
    assert events.count("session_start") == 1
    assert "close:monitor" in events


def test_invalid_risk_configuration_prevents_application_startup() -> None:
    application, _, _, _ = make_application()
    application.settings_loader = lambda: Settings.from_env({"RISK_MAX_DRAWDOWN_FRACTION": "1.2"})

    with pytest.raises(ApplicationStartupError):
        application.start()

    assert application.state is ApplicationState.ERROR
