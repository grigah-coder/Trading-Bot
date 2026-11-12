"""Unit tests for runtime wiring in ``trading_bot.__main__``."""

from __future__ import annotations

from typing import Any, Dict, List

from trading_bot.config.settings import Settings


class _FakeMonitoring:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _settings() -> Settings:
    return Settings(
        alpaca_api_key="key",
        alpaca_api_secret="secret",
        risk_state_file="",
        sleep_time="1M",
    )


def test_parse_sleep_time_supports_units_and_fallback():
    """SLEEPTIME values like ``1M`` are parsed into seconds."""
    from trading_bot.__main__ import _parse_sleep_time

    assert _parse_sleep_time("1M") == 60.0
    assert _parse_sleep_time("30s") == 30.0
    assert _parse_sleep_time("2H") == 7200.0
    assert _parse_sleep_time("90") == 90.0
    assert _parse_sleep_time("bogus") == 60.0
    assert _parse_sleep_time(0) == 0.05


def test_main_reconciles_before_scheduling_with_sleep_interval(monkeypatch):
    """``main`` reconciles at startup and schedules at SLEEPTIME cadence."""
    from trading_bot import __main__ as entry

    calls: Dict[str, Any] = {"events": []}
    monitoring = _FakeMonitoring()

    class FakeEngine:
        def __init__(self, **kwargs: Any) -> None:
            calls["engine_kwargs"] = kwargs

        def validate(self) -> None:
            calls["events"].append("validate")

        def reconcile_startup(self) -> None:
            calls["events"].append("reconcile")

        def run_iteration(self) -> Dict[str, Any]:
            calls["events"].append("iteration")
            return {}

    class FakeScheduler:
        def __init__(self, state: Any, interval_seconds: float = 60.0) -> None:
            calls["interval"] = interval_seconds
            self.state = state

        def start(self, callback: Any) -> None:
            calls["events"].append("scheduler-start")
            callback()

        def stop(self) -> None:
            calls["events"].append("scheduler-stop")

        def is_running(self) -> bool:
            return False

    monkeypatch.setattr(entry, "configure_logging", lambda **kwargs: None)
    monkeypatch.setattr(entry, "load_settings", _settings)
    monkeypatch.setattr(
        entry, "_build_components",
        lambda settings, state: {
            "data_provider": None,
            "broker_adapter": None,
            "risk_engine": None,
            "strategy": None,
            "order_manager": None,
            "database": None,
            "monitoring": monitoring,
        },
    )
    monkeypatch.setattr(entry, "TradingEngine", FakeEngine)
    monkeypatch.setattr(entry, "Scheduler", FakeScheduler)
    monkeypatch.setattr(
        entry.Lifecycle, "install_signal_handlers", lambda self: None,
    )
    monkeypatch.setattr(
        entry.Lifecycle, "restore_signal_handlers", lambda self: None,
    )

    exit_code = entry.main(backtest=False)

    assert exit_code == 0
    assert calls["interval"] == 60.0
    events: List[str] = calls["events"]
    assert events.index("validate") < events.index("reconcile")
    assert events.index("reconcile") < events.index("scheduler-start")
    assert "iteration" in events
    assert monitoring.closed is True
