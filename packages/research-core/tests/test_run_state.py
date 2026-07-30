from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta

import pytest
from openalpha_research.errors import RunStateTransitionError
from openalpha_research.run_state import RunState, RunStateEvent, RunStateJournal

NOW = datetime(2026, 7, 29, 6, 0, tzinfo=UTC)


def test_state_transitions_append_immutable_events() -> None:
    draft = RunStateJournal.start(
        run_id="run_01",
        experiment_id="exp_" + "a" * 64,
        attempt_id="attempt_01",
        occurred_at=NOW,
    )
    validated = draft.transition(RunState.VALIDATED, occurred_at=NOW + timedelta(seconds=1))

    assert draft.current_state is RunState.DRAFT
    assert len(draft.events) == 1
    assert validated.current_state is RunState.VALIDATED
    assert [event.sequence for event in validated.events] == [1, 2]
    assert validated.events[0] is draft.events[0]
    with pytest.raises(FrozenInstanceError):
        validated.events[0].state = RunState.FAILED  # type: ignore[misc]


def test_invalid_transition_and_non_monotonic_time_are_rejected() -> None:
    draft = RunStateJournal.start(
        run_id="run_01",
        experiment_id="exp_" + "a" * 64,
        attempt_id="attempt_01",
        occurred_at=NOW,
    )

    with pytest.raises(RunStateTransitionError, match="draft.*running"):
        draft.transition(RunState.RUNNING, occurred_at=NOW + timedelta(seconds=1))
    with pytest.raises(RunStateTransitionError, match="strictly after"):
        draft.transition(RunState.VALIDATED, occurred_at=NOW)


def test_terminal_completed_attempt_cannot_transition() -> None:
    journal = _running_journal().transition(
        RunState.COMPLETED, occurred_at=NOW + timedelta(seconds=4)
    )

    with pytest.raises(RunStateTransitionError, match="terminal"):
        journal.transition(RunState.QUEUED, occurred_at=NOW + timedelta(seconds=5))


def test_failed_run_retries_as_a_new_attempt_without_rewriting_history() -> None:
    failed = _running_journal().transition(
        RunState.FAILED,
        occurred_at=NOW + timedelta(seconds=4),
        reason="provider timeout",
    )

    with pytest.raises(RunStateTransitionError, match="new attempt"):
        failed.retry("attempt_01", occurred_at=NOW + timedelta(seconds=5))
    retried = failed.retry("attempt_02", occurred_at=NOW + timedelta(seconds=5))

    assert failed.current_state is RunState.FAILED
    assert retried.current_state is RunState.QUEUED
    assert retried.current_attempt_id == "attempt_02"
    assert len(retried.events) == len(failed.events) + 1
    assert retried.events[:-1] == failed.events


def test_direct_constructor_rejects_forged_or_rewritten_history() -> None:
    forged = RunStateEvent(
        sequence=7,
        run_id="run_01",
        experiment_id="exp_" + "a" * 64,
        attempt_id="attempt_01",
        state=RunState.COMPLETED,
        occurred_at=NOW,
    )

    with pytest.raises(ValueError, match="begin with sequence 1 in draft"):
        RunStateJournal(events=(forged,))


def test_direct_constructor_revalidates_every_transition() -> None:
    draft = RunStateJournal.start(
        run_id="run_01",
        experiment_id="exp_" + "a" * 64,
        attempt_id="attempt_01",
        occurred_at=NOW,
    )
    forged_running = RunStateEvent(
        sequence=2,
        run_id="run_01",
        experiment_id="exp_" + "a" * 64,
        attempt_id="attempt_01",
        state=RunState.RUNNING,
        occurred_at=NOW + timedelta(seconds=1),
    )

    with pytest.raises(ValueError, match="invalid recorded transition"):
        RunStateJournal(events=(draft.events[0], forged_running))


def _running_journal() -> RunStateJournal:
    journal = RunStateJournal.start(
        run_id="run_01",
        experiment_id="exp_" + "a" * 64,
        attempt_id="attempt_01",
        occurred_at=NOW,
    )
    for offset, state in enumerate(
        (RunState.VALIDATED, RunState.QUEUED, RunState.RUNNING), start=1
    ):
        journal = journal.transition(state, occurred_at=NOW + timedelta(seconds=offset))
    return journal
