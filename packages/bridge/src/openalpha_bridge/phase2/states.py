"""Versioned terminal state machine for the Phase 2 execution pipeline.

Stages cannot be skipped, the reconstruction-test partition cannot open before
the checkpoint is frozen, and a final evaluation cannot be repeated without an
explicit immutable reference to the prior result.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from itertools import pairwise
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..errors import BridgeFailure, BridgeTransformError, FailureCategory

__all__ = [
    "ALLOWED_TRANSITIONS",
    "STATE_MACHINE_VERSION",
    "EvidenceClass",
    "Phase2State",
    "StateJournal",
    "StateTransition",
    "assert_transition_allowed",
    "is_terminal",
]

STATE_MACHINE_VERSION = "openalpha.bridge.phase2.states.v1"


class Phase2State(StrEnum):
    CREATED = "CREATED"
    PREFLIGHT_PASSED = "PREFLIGHT_PASSED"
    DATA_RETRIEVED = "DATA_RETRIEVED"
    DATA_VALIDATED = "DATA_VALIDATED"
    WINDOWS_BUILT = "WINDOWS_BUILT"
    COVERAGE_PASSED = "COVERAGE_PASSED"
    ASSETS_RESOLVED = "ASSETS_RESOLVED"
    STAGE_A_PASSED = "STAGE_A_PASSED"
    STAGE_B_PASSED = "STAGE_B_PASSED"
    STAGE_C_TRAINED = "STAGE_C_TRAINED"
    CHECKPOINT_FROZEN = "CHECKPOINT_FROZEN"
    TEST_OPENED = "TEST_OPENED"
    TEST_EVALUATED = "TEST_EVALUATED"
    EXTERNAL_EVALUATED = "EXTERNAL_EVALUATED"
    FINALIZED = "FINALIZED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"


class EvidenceClass(StrEnum):
    """Separates real Phase 2 evidence from synthetic pipeline validation."""

    REAL_PHASE2 = "real_phase2"
    SYNTHETIC_PIPELINE_VALIDATION = "synthetic_pipeline_validation"


#: The single linear success path. Every state may also fail or block.
_SUCCESS_PATH: tuple[Phase2State, ...] = (
    Phase2State.CREATED,
    Phase2State.PREFLIGHT_PASSED,
    Phase2State.DATA_RETRIEVED,
    Phase2State.DATA_VALIDATED,
    Phase2State.WINDOWS_BUILT,
    Phase2State.COVERAGE_PASSED,
    Phase2State.ASSETS_RESOLVED,
    Phase2State.STAGE_A_PASSED,
    Phase2State.STAGE_B_PASSED,
    Phase2State.STAGE_C_TRAINED,
    Phase2State.CHECKPOINT_FROZEN,
    Phase2State.TEST_OPENED,
    Phase2State.TEST_EVALUATED,
    Phase2State.EXTERNAL_EVALUATED,
    Phase2State.FINALIZED,
)

_TERMINAL_STATES = frozenset({Phase2State.FINALIZED, Phase2State.BLOCKED})

#: A failed run is recoverable by resuming from its last verified artifact; a
#: blocked or finalized run is terminal.
_RECOVERABLE = frozenset({Phase2State.FAILED})


def _build_transitions() -> dict[Phase2State, frozenset[Phase2State]]:
    transitions: dict[Phase2State, set[Phase2State]] = {}
    for current, following in pairwise(_SUCCESS_PATH):
        transitions.setdefault(current, set()).add(following)
    # Any non-terminal state may fail or block.
    for state in _SUCCESS_PATH:
        if state not in _TERMINAL_STATES:
            transitions.setdefault(state, set()).update({Phase2State.FAILED, Phase2State.BLOCKED})
    # A failed run resumes only by re-entering the state it failed from, which
    # the resume path validates against the last verified artifact.
    transitions[Phase2State.FAILED] = set(_SUCCESS_PATH) - {Phase2State.CREATED}
    return {state: frozenset(targets) for state, targets in transitions.items()}


ALLOWED_TRANSITIONS: dict[Phase2State, frozenset[Phase2State]] = _build_transitions()


def is_terminal(state: Phase2State) -> bool:
    return state in _TERMINAL_STATES


def _fail(code: str, message: str) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            message=message,
        )
    )


def assert_transition_allowed(current: Phase2State, target: Phase2State) -> None:
    """Fail closed on skipped stages and illegal transitions."""
    if current in _TERMINAL_STATES:
        raise _fail(
            "TERMINAL_STATE_TRANSITION",
            f"{current.value} is terminal and cannot transition to {target.value}",
        )
    allowed = ALLOWED_TRANSITIONS.get(current, frozenset())
    if target not in allowed:
        raise _fail(
            "ILLEGAL_STATE_TRANSITION",
            f"transition {current.value} -> {target.value} is not permitted",
        )
    if target is Phase2State.TEST_OPENED and current not in (
        Phase2State.CHECKPOINT_FROZEN,
        Phase2State.FAILED,
    ):
        raise _fail(
            "TEST_OPENED_BEFORE_CHECKPOINT_FROZEN",
            "the reconstruction-test partition may open only after CHECKPOINT_FROZEN",
        )


class StateTransition(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    schema_version: Literal["openalpha.bridge.phase2.transition.v1"] = (
        "openalpha.bridge.phase2.transition.v1"
    )
    sequence: int = Field(ge=1)
    state: Phase2State
    occurred_at: datetime
    reason: str | None = None
    artifact_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class StateJournal(BaseModel):
    """Append-only ordered record of a run's state transitions."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    schema_version: Literal["openalpha.bridge.phase2.journal.v1"] = (
        "openalpha.bridge.phase2.journal.v1"
    )
    state_machine_version: Literal["openalpha.bridge.phase2.states.v1"] = STATE_MACHINE_VERSION
    run_id: str = Field(min_length=1)
    evidence_class: EvidenceClass
    transitions: tuple[StateTransition, ...]

    @classmethod
    def start(
        cls,
        *,
        run_id: str,
        evidence_class: EvidenceClass,
        occurred_at: datetime,
    ) -> StateJournal:
        return cls(
            run_id=run_id,
            evidence_class=evidence_class,
            transitions=(
                StateTransition(sequence=1, state=Phase2State.CREATED, occurred_at=occurred_at),
            ),
        )

    @property
    def current_state(self) -> Phase2State:
        return self.transitions[-1].state

    def advance(
        self,
        target: Phase2State,
        *,
        occurred_at: datetime,
        reason: str | None = None,
        artifact_sha256: str | None = None,
    ) -> StateJournal:
        previous = self.transitions[-1]
        assert_transition_allowed(previous.state, target)
        if occurred_at <= previous.occurred_at:
            raise _fail(
                "NON_MONOTONIC_TRANSITION_TIME",
                "state transition timestamps must strictly increase",
            )
        transition = StateTransition(
            sequence=previous.sequence + 1,
            state=target,
            occurred_at=occurred_at,
            reason=reason,
            artifact_sha256=artifact_sha256,
        )
        return StateJournal(
            run_id=self.run_id,
            evidence_class=self.evidence_class,
            transitions=(*self.transitions, transition),
        )

    def has_reached(self, state: Phase2State) -> bool:
        return any(transition.state is state for transition in self.transitions)
