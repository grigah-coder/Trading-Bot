"""Unit tests for scheduler (Phase 2)."""

import time

import pytest

from trading_bot.scheduler import Scheduler
from trading_bot.state import AppState


def test_scheduler_runs_callback():
    state = AppState()
    state.start("session-1")
    calls = []

    def callback():
        calls.append(1)

    sched = Scheduler(state, interval_seconds=0.05)
    try:
        sched.start(callback)
        time.sleep(0.2)
    finally:
        sched.stop()

    assert len(calls) >= 2
    assert sched.iteration_count >= 2
    assert sched.last_iteration is not None


def test_scheduler_respects_stopped_state():
    state = AppState()  # not started -> not running
    calls = []

    def callback():
        calls.append(1)

    sched = Scheduler(state, interval_seconds=0.05)
    try:
        sched.start(callback)
        time.sleep(0.2)
    finally:
        sched.stop()

    # Callback should not run because application is not running.
    assert calls == []


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
