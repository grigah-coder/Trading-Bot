"""Unit tests for state management (Phase 2)."""

import threading

import pytest

from trading_bot.state import AppState


def test_start_and_stop():
    state = AppState()
    assert state.is_running() is False
    state.start("session-1")
    assert state.is_running() is True
    assert state.session_id == "session-1"
    assert state.started_at is not None
    state.stop()
    assert state.is_running() is False
    assert state.stopped_at is not None


def test_start_while_running_raises():
    state = AppState()
    state.start("session-1")
    with pytest.raises(RuntimeError, match="already running"):
        state.start("session-2")
    state.stop()


def test_stop_when_not_running_is_noop():
    state = AppState()
    state.stop()  # should not raise
    assert state.is_running() is False


def test_circuit_breaker():
    state = AppState()
    assert state.is_circuit_breaker_tripped() is False
    state.trip_circuit_breaker()
    assert state.is_circuit_breaker_tripped() is True
    state.reset_circuit_breaker()
    assert state.is_circuit_breaker_tripped() is False


def test_data_stale_flag():
    state = AppState()
    assert state.is_data_stale() is False
    state.set_data_stale(True)
    assert state.is_data_stale() is True
    state.set_data_stale(False)
    assert state.is_data_stale() is False


def test_counters():
    state = AppState()
    state.increment("signals_generated", 3)
    state.increment("orders_submitted")
    assert state.signals_generated == 3
    assert state.orders_submitted == 1
    with pytest.raises(AttributeError):
        state.increment("nonexistent_counter")


def test_snapshot():
    state = AppState()
    state.start("session-1")
    snap = state.snapshot()
    assert snap["session_id"] == "session-1"
    assert snap["running"] is True
    assert isinstance(snap["started_at"], str)
    assert snap["error_count"] == 0


def test_thread_safety():
    state = AppState()
    state.start("session-1")

    def worker():
        for _ in range(100):
            state.increment("signals_generated")
            state.touch_heartbeat()

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert state.signals_generated == 1000
    state.stop()


def test_record_error_appends_and_bounds():
    """Errors are appended thread-safely and the list stays bounded."""
    state = AppState()
    state.record_error("boom")
    assert state.errors == ["boom"]

    for index in range(1100):
        state.record_error(f"error-{index}")
    assert len(state.errors) == 1000
    assert state.snapshot()["error_count"] == 1000
