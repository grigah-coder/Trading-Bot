import pytest

from core.exceptions import StateTransitionError
from core.state import ApplicationState, StateMachine


def test_valid_lifecycle_and_recovery_transitions() -> None:
    state = StateMachine()

    state.transition(ApplicationState.STARTING)
    state.transition(ApplicationState.RUNNING)
    state.transition(ApplicationState.PAUSED)
    state.transition(ApplicationState.RUNNING)
    state.transition(ApplicationState.ERROR)
    state.transition(ApplicationState.STARTING)
    state.transition(ApplicationState.RUNNING)
    state.transition(ApplicationState.HALTED)
    state.transition(ApplicationState.STOPPING)
    state.transition(ApplicationState.STOPPED)

    assert state.current is ApplicationState.STOPPED


def test_invalid_state_transition_is_rejected_without_mutating_state() -> None:
    state = StateMachine()

    with pytest.raises(StateTransitionError):
        state.transition(ApplicationState.RUNNING)

    assert state.current is ApplicationState.STOPPED
