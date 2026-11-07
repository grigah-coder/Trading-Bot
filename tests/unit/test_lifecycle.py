"""Unit tests for lifecycle (Phase 2)."""

import pytest

from trading_bot.lifecycle import Lifecycle
from trading_bot.state import AppState


def test_start_and_stop_runs_hooks():
    state = AppState()
    lifecycle = Lifecycle(state)

    start_calls = []
    stop_calls = []

    lifecycle.on_start(lambda: start_calls.append(1))
    lifecycle.on_start(lambda: start_calls.append(2))
    lifecycle.on_stop(lambda: stop_calls.append("first"))  # runs last
    lifecycle.on_stop(lambda: stop_calls.append("second"))  # runs first

    lifecycle.start("session-1")
    assert start_calls == [1, 2]
    assert state.is_running() is True

    lifecycle.stop()
    # Stop hooks inserted at front -> ["second", "first"] registered,
    # but executed in the order stored (reverse of registration).
    assert stop_calls == ["second", "first"]
    assert state.is_running() is False


def test_start_twice_raises():
    state = AppState()
    lifecycle = Lifecycle(state)
    lifecycle.start("session-1")
    with pytest.raises(RuntimeError, match="already running"):
        lifecycle.start("session-2")
    lifecycle.stop()


def test_stop_when_not_running_is_noop():
    state = AppState()
    lifecycle = Lifecycle(state)
    lifecycle.stop()  # should not raise


def test_request_stop():
    state = AppState()
    lifecycle = Lifecycle(state)
    lifecycle.start("session-1")
    assert state.is_running() is True
    lifecycle.request_stop()
    assert state.is_running() is False


def test_is_running():
    state = AppState()
    lifecycle = Lifecycle(state)
    assert lifecycle.is_running() is False
    lifecycle.start("session-1")
    assert lifecycle.is_running() is True
    lifecycle.stop()
    assert lifecycle.is_running() is False
