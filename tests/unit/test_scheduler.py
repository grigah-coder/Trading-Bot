"""Unit tests for scheduler (Phase 2)."""

import threading

import pytest

from trading_bot.scheduler import Scheduler
from trading_bot.state import AppState


def test_scheduler_runs_callback():
    state = AppState()
    state.start("session-1")
    calls = []
    second_call = threading.Event()

    def callback():
        calls.append(1)
        if len(calls) >= 2:
            second_call.set()

    sched = Scheduler(state, interval_seconds=0.05)
    try:
        sched.start(callback)
        assert second_call.wait(2.0)
    finally:
        sched.stop()

    assert len(calls) >= 2
    assert sched.iteration_count >= 2
    assert sched.last_iteration is not None


def test_scheduler_respects_stopped_state():
    state = AppState()  # not started -> not running
    ran = threading.Event()

    def callback():
        ran.set()

    sched = Scheduler(state, interval_seconds=0.05)
    try:
        sched.start(callback)
        # Event-based wait: the callback must never fire while stopped.
        assert not ran.wait(0.3)
    finally:
        sched.stop()

    assert not ran.is_set()


def test_scheduler_stop_idempotent():
    state = AppState()
    sched = Scheduler(state, interval_seconds=0.05)
    sched.stop()  # not started, should not raise
    sched.stop()


def test_scheduler_start_twice_raises():
    state = AppState()
    state.start("session-1")
    sched = Scheduler(state, interval_seconds=0.05)
    try:
        sched.start(lambda: None)
        with pytest.raises(RuntimeError, match="already running"):
            sched.start(lambda: None)
    finally:
        sched.stop()
    state.stop()


def test_scheduler_interval_clamped_to_positive():
    state = AppState()
    sched = Scheduler(state, interval_seconds=-5)
    assert sched.interval_seconds == 0.05


def test_scheduler_records_callback_errors():
    """A failing callback is recorded on AppState, never crashes the loop."""
    state = AppState()
    state.start("session-1")
    recorded = threading.Event()
    original_record_error = state.record_error

    def callback():
        raise RuntimeError("iteration boom")

    def record_error(message):
        original_record_error(message)
        recorded.set()

    state.record_error = record_error
    sched = Scheduler(state, interval_seconds=0.05)
    try:
        sched.start(callback)
        assert recorded.wait(2.0)
    finally:
        sched.stop()

    assert any("iteration boom" in error for error in state.errors)
