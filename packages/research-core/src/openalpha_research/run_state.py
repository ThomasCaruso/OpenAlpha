from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from .errors import RunStateTransitionError


class RunState(StrEnum):
    DRAFT = "draft"
    VALIDATED = "validated"
    QUEUED = "queued"
    RUNNING = "running"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INVALID = "invalid"
    COMPLETED = "completed"


_ALLOWED_TRANSITIONS: dict[RunState, frozenset[RunState]] = {
    RunState.DRAFT: frozenset({RunState.VALIDATED}),
    RunState.VALIDATED: frozenset({RunState.QUEUED}),
    RunState.QUEUED: frozenset({RunState.RUNNING}),
    RunState.RUNNING: frozenset(
        {RunState.FAILED, RunState.CANCELLED, RunState.INVALID, RunState.COMPLETED}
    ),
}
_TERMINAL_STATES = frozenset({RunState.CANCELLED, RunState.INVALID, RunState.COMPLETED})
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_EXPERIMENT_ID = re.compile(r"^exp_[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class RunStateEvent:
    sequence: int
    run_id: str
    experiment_id: str
    attempt_id: str
    state: RunState
    occurred_at: datetime
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class RunStateJournal:
    events: tuple[RunStateEvent, ...]

    def __post_init__(self) -> None:
        if not self.events:
            raise ValueError("run-state journal requires at least one event")
        first = self.events[0]
        if first.sequence != 1 or first.state is not RunState.DRAFT:
            raise ValueError("run-state history must begin with sequence 1 in draft")
        _validate_event(first)
        for previous, current in zip(self.events, self.events[1:], strict=False):
            _validate_event(current)
            if current.sequence != previous.sequence + 1:
                raise ValueError("run-state event sequences must be contiguous")
            if current.run_id != first.run_id or current.experiment_id != first.experiment_id:
                raise ValueError("run-state history cannot change run or experiment identity")
            if current.occurred_at <= previous.occurred_at:
                raise ValueError("run-state event times must be strictly increasing")
            if previous.state is RunState.FAILED:
                if (
                    current.state is not RunState.QUEUED
                    or current.attempt_id == previous.attempt_id
                ):
                    raise ValueError("invalid recorded retry transition")
                continue
            if previous.state in _TERMINAL_STATES:
                raise ValueError("terminal run-state history cannot contain later events")
            if current.state not in _ALLOWED_TRANSITIONS.get(previous.state, frozenset()):
                raise ValueError(
                    f"invalid recorded transition from {previous.state.value} "
                    f"to {current.state.value}"
                )
            if current.attempt_id != previous.attempt_id:
                raise ValueError("attempt identity can change only on an explicit retry")

    @classmethod
    def start(
        cls,
        *,
        run_id: str,
        experiment_id: str,
        attempt_id: str,
        occurred_at: datetime,
    ) -> RunStateJournal:
        _validate_identifier("run_id", run_id)
        _validate_identifier("attempt_id", attempt_id)
        if not _EXPERIMENT_ID.fullmatch(experiment_id):
            raise ValueError("experiment_id must be exp_ followed by a SHA-256 digest")
        _validate_timestamp(occurred_at)
        return cls(
            events=(
                RunStateEvent(
                    sequence=1,
                    run_id=run_id,
                    experiment_id=experiment_id,
                    attempt_id=attempt_id,
                    state=RunState.DRAFT,
                    occurred_at=occurred_at,
                ),
            )
        )

    @property
    def current_state(self) -> RunState:
        return self.events[-1].state

    @property
    def current_attempt_id(self) -> str:
        return self.events[-1].attempt_id

    def transition(
        self,
        state: RunState,
        *,
        occurred_at: datetime,
        reason: str | None = None,
    ) -> RunStateJournal:
        previous = self.events[-1]
        self._validate_later_timestamp(occurred_at)
        if previous.state in _TERMINAL_STATES:
            raise RunStateTransitionError(
                f"terminal state {previous.state.value} cannot transition to {state.value}"
            )
        if previous.state is RunState.FAILED:
            raise RunStateTransitionError("failed run requires retry with a new attempt")
        if state not in _ALLOWED_TRANSITIONS.get(previous.state, frozenset()):
            raise RunStateTransitionError(
                f"invalid transition from {previous.state.value} to {state.value}"
            )
        return self._append(state, occurred_at, previous.attempt_id, reason)

    def retry(self, new_attempt_id: str, *, occurred_at: datetime) -> RunStateJournal:
        previous = self.events[-1]
        self._validate_later_timestamp(occurred_at)
        if previous.state is not RunState.FAILED:
            raise RunStateTransitionError("only a failed run can be retried")
        _validate_identifier("new_attempt_id", new_attempt_id)
        if new_attempt_id == previous.attempt_id:
            raise RunStateTransitionError("retry must create a new attempt")
        return self._append(RunState.QUEUED, occurred_at, new_attempt_id, "explicit retry")

    def _append(
        self,
        state: RunState,
        occurred_at: datetime,
        attempt_id: str,
        reason: str | None,
    ) -> RunStateJournal:
        previous = self.events[-1]
        event = RunStateEvent(
            sequence=previous.sequence + 1,
            run_id=previous.run_id,
            experiment_id=previous.experiment_id,
            attempt_id=attempt_id,
            state=state,
            occurred_at=occurred_at,
            reason=reason,
        )
        return RunStateJournal(events=(*self.events, event))

    def _validate_later_timestamp(self, occurred_at: datetime) -> None:
        _validate_timestamp(occurred_at)
        if occurred_at <= self.events[-1].occurred_at:
            raise RunStateTransitionError(
                "new run-state event time must be strictly after the previous event"
            )


def _validate_identifier(field: str, value: str) -> None:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{field} contains unsupported characters")


def _validate_timestamp(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("run-state timestamps must be timezone-aware")


def _validate_event(event: RunStateEvent) -> None:
    _validate_identifier("run_id", event.run_id)
    _validate_identifier("attempt_id", event.attempt_id)
    if not _EXPERIMENT_ID.fullmatch(event.experiment_id):
        raise ValueError("experiment_id must be exp_ followed by a SHA-256 digest")
    _validate_timestamp(event.occurred_at)
