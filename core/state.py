"""Application lifecycle state and its validated transition graph."""

from __future__ import annotations

from enum import Enum
from types import MappingProxyType

from core.exceptions import StateTransitionError


class ApplicationState(str, Enum):
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"
    HALTED = "HALTED"


_TRANSITIONS = MappingProxyType({
    ApplicationState.STOPPED: frozenset({ApplicationState.STARTING}),
    ApplicationState.STARTING: frozenset(
        {ApplicationState.RUNNING, ApplicationState.ERROR, ApplicationState.STOPPING}
    ),
    ApplicationState.RUNNING: frozenset(
        {
            ApplicationState.PAUSED,
            ApplicationState.HALTED,
            ApplicationState.STOPPING,
            ApplicationState.ERROR,
        }
    ),
    ApplicationState.PAUSED: frozenset(
        {
            ApplicationState.RUNNING,
            ApplicationState.HALTED,
            ApplicationState.STOPPING,
            ApplicationState.ERROR,
        }
    ),
    ApplicationState.HALTED: frozenset(
        {ApplicationState.STOPPING, ApplicationState.ERROR}
    ),
    ApplicationState.ERROR: frozenset(
        {ApplicationState.STARTING, ApplicationState.STOPPING}
    ),
    ApplicationState.STOPPING: frozenset(
        {ApplicationState.STOPPED, ApplicationState.ERROR}
    ),
})


class StateMachine:
    """Small synchronous state machine; callers own any cross-thread access."""

    def __init__(self, initial: ApplicationState = ApplicationState.STOPPED) -> None:
        self._current = initial

    @property
    def current(self) -> ApplicationState:
        return self._current

    def transition(self, target: ApplicationState) -> None:
        if target not in _TRANSITIONS[self._current]:
            raise StateTransitionError(
                f"Cannot transition from {self._current.value} to {target.value}",
                context={"current": self._current.value, "requested": target.value},
            )
        self._current = target
