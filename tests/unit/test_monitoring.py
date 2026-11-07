"""Unit tests for the Phase 10 monitoring package."""

from __future__ import annotations

from typing import Any, Dict, List

from trading_bot.monitoring.health import HealthMonitor
from trading_bot.monitoring.notifier import (
    ConsoleNotifier,
    Notifier,
    TelegramNotifier,
    build_notifier,
)

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class RecordingNotifier(Notifier):
    """Captures notifications instead of sending them."""

    def __init__(self) -> None:
        self.sent: List[Dict[str, Any]] = []

    def notify(self, message: str, level: str = "error") -> None:
        self.sent.append({"message": message, "level": level})


class FailingNotifier(Notifier):
    """Always raises to confirm HealthMonitor swallows notifier failures."""

    def notify(self, message: str, level: str = "error") -> None:
        raise RuntimeError("notifier down")


# ---------------------------------------------------------------------------
# HealthMonitor
# ---------------------------------------------------------------------------


class TestHealthMonitorProtocol:
    def test_implements_monitoring_protocol(self):
        from trading_bot.engine import Monitoring as Proto

        monitor = HealthMonitor()
        assert isinstance(monitor, Proto)
        assert callable(monitor.health_check)
        assert callable(monitor.alert)
        assert callable(monitor.record_metric)

    def test_default_health_ok(self):
        monitor = HealthMonitor()
        h = monitor.health_check()
        assert h["ok"] is True
        assert h["status"] == "ok"
        assert h["circuit_breaker_tripped"] is False
        assert h["data_stale"] is False
        assert h["uptime_seconds"] >= 0
        assert "components" in h
        assert "metrics" in h


class TestAlert:
    def test_alert_records_and_logs(self):
        monitor = HealthMonitor()
        monitor.alert("something broke", level="error")
        alerts = monitor.recent_alerts()
        assert len(alerts) == 1
        assert alerts[0]["message"] == "something broke"
        assert alerts[0]["level"] == "error"
        assert "timestamp" in alerts[0]

    def test_alert_forwards_to_notifier(self):
        notifier = RecordingNotifier()
        monitor = HealthMonitor(notifier=notifier)
        monitor.alert("critical issue", level="critical")
        assert len(notifier.sent) == 1
        assert notifier.sent[0]["message"] == "critical issue"
        assert notifier.sent[0]["level"] == "critical"

    def test_alert_swallows_notifier_failure(self):
        monitor = HealthMonitor(notifier=FailingNotifier())
        # Should not raise even though the notifier always fails.
        monitor.alert("boom", level="error")
        alerts = monitor.recent_alerts()
        assert len(alerts) == 1

    def test_alert_levels(self):
        monitor = HealthMonitor()
        for level in ("critical", "error", "warning", "info"):
            monitor.alert(f"msg {level}", level=level)
        levels = [a["level"] for a in monitor.recent_alerts()]
        assert levels == ["critical", "error", "warning", "info"]


class TestRecordMetric:
    def test_record_metric(self):
        monitor = HealthMonitor()
        monitor.record_metric("signals_generated", 5.0)
        monitor.record_metric("signals_generated", 7.0)
        h = monitor.health_check()
        m = h["metrics"]["signals_generated"]
        assert m["count"] == 2
        assert m["latest"] == 7.0

    def test_record_metric_ignores_nonfinite(self):
        monitor = HealthMonitor()
        monitor.record_metric("bad", float("nan"))
        monitor.record_metric("bad", float("inf"))
        monitor.record_metric("bad", "not a number")
        assert "bad" not in monitor.health_check()["metrics"]


class TestCircuitBreaker:
    def test_trip_and_reset(self):
        monitor = HealthMonitor()
        monitor.trip_circuit_breaker("daily loss limit")
        h = monitor.health_check()
        assert h["circuit_breaker_tripped"] is True
        assert h["circuit_breaker_reason"] == "daily loss limit"
        assert h["status"] == "halted"
        assert h["ok"] is False

        monitor.reset_circuit_breaker()
        h = monitor.health_check()
        assert h["circuit_breaker_tripped"] is False
        assert h["status"] == "ok"
        assert h["ok"] is True

    def test_trip_sends_critical_alert(self):
        notifier = RecordingNotifier()
        monitor = HealthMonitor(notifier=notifier)
        monitor.trip_circuit_breaker("halted")
        assert any("circuit breaker tripped" in a["message"]
                   and a["level"] == "critical" for a in notifier.sent)


class TestDataStale:
    def test_data_stale_affects_health(self):
        monitor = HealthMonitor()
        monitor.set_data_stale(True)
        h = monitor.health_check()
        assert h["data_stale"] is True
        assert h["ok"] is False
        assert h["status"] == "degraded"

        monitor.set_data_stale(False)
        h = monitor.health_check()
        assert h["data_stale"] is False


class TestComponents:
    def test_component_health_aggregated(self):
        def broker_check() -> Dict[str, Any]:
            return {"ok": True, "connected": True}

        def data_check() -> Dict[str, Any]:
            return {"ok": False, "error": "stale"}

        monitor = HealthMonitor(components={"broker": broker_check, "data": data_check})
        h = monitor.health_check()
        assert h["components"]["broker"]["ok"] is True
        assert h["components"]["data"]["ok"] is False
        assert h["ok"] is False

    def test_component_exception_swallowed(self):
        def bad_check():
            raise RuntimeError("boom")

        monitor = HealthMonitor(components={"bad": bad_check})
        h = monitor.health_check()
        assert h["components"]["bad"]["ok"] is False
        assert h["ok"] is False

    def test_add_remove_component(self):
        monitor = HealthMonitor()
        monitor.add_component("x", lambda: {"ok": True})
        assert "x" in monitor.health_check()["components"]
        monitor.remove_component("x")
        assert "x" not in monitor.health_check()["components"]


class TestHeartbeat:
    def test_touch_heartbeat(self):
        monitor = HealthMonitor()
        monitor.touch_heartbeat()
        h = monitor.health_check()
        assert h["last_heartbeat"] is not None


# ---------------------------------------------------------------------------
# Notifier
# ---------------------------------------------------------------------------


class TestConsoleNotifier:
    def test_notify_does_not_raise(self):
        notifier = ConsoleNotifier()
        notifier.notify("hello", level="error")
        notifier.notify("warn", level="warning")


class TestBuildNotifier:
    def test_console_when_telegram_disabled(self):
        class S:
            telegram_enabled = False
            telegram_token = ""
            telegram_chat_id = ""
            telegram_timeout = 5.0
            telegram_retry_attempts = 1

        n = build_notifier(S())
        assert isinstance(n, ConsoleNotifier)

    def test_console_when_missing_credentials(self):
        class S:
            telegram_enabled = True
            telegram_token = ""
            telegram_chat_id = "123"
            telegram_timeout = 5.0
            telegram_retry_attempts = 1

        n = build_notifier(S())
        assert isinstance(n, ConsoleNotifier)

    def test_telegram_when_configured(self):
        class S:
            telegram_enabled = True
            telegram_token = "123:ABC"
            telegram_chat_id = "-100123"
            telegram_timeout = 5.0
            telegram_retry_attempts = 1

        n = build_notifier(S())
        assert isinstance(n, TelegramNotifier)
        assert n.token == "123:ABC"
        assert n.chat_id == "-100123"
