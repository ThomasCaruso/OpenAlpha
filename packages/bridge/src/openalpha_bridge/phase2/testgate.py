"""One-time reconstruction-test partition guard.

The test partition may be loaded exactly once, only after every precondition
below holds, and only through an explicit ``open_test_partition`` transition.
The resulting record is immutable: it cannot be deleted or recreated under the
same run ID.

Synthetic runs use a separate namespace and can never set the real
``test_partition_opened`` flag.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..errors import BridgeFailure, BridgeTransformError, FailureCategory
from .identity import RunIdentity, canonical_json, canonical_sha256
from .states import EvidenceClass, Phase2State, StateJournal

__all__ = [
    "TestOpeningPreconditions",
    "TestOpeningRecord",
    "is_test_partition_opened",
    "open_test_partition",
]

_REAL_LEDGER = "test_opening_record.json"
_SYNTHETIC_LEDGER = "synthetic_test_opening_record.json"


def _fail(code: str, message: str, *, field: str | None = None) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            field=field,
            message=message,
        )
    )


class TestOpeningPreconditions(BaseModel):
    """Every condition that must hold before the test partition may open."""

    __test__ = False  # not a pytest collection target

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    stage_a_passed: bool
    stage_b_passed: bool
    stage_c_completed: bool
    selected_checkpoint_fixed: bool
    checkpoint_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    frozen_kronos_weights_verified: bool
    validation_selection_report_sealed: bool
    experiment_hash_verified: bool
    preprocessing_state_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    feature_manifest_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    def unmet(self) -> tuple[str, ...]:
        missing: list[str] = []
        for field in (
            "stage_a_passed",
            "stage_b_passed",
            "stage_c_completed",
            "selected_checkpoint_fixed",
            "frozen_kronos_weights_verified",
            "validation_selection_report_sealed",
            "experiment_hash_verified",
        ):
            if not getattr(self, field):
                missing.append(field)
        for field in (
            "checkpoint_sha256",
            "preprocessing_state_sha256",
            "feature_manifest_sha256",
        ):
            if getattr(self, field) is None:
                missing.append(field)
        return tuple(missing)


class TestOpeningRecord(BaseModel):
    """Immutable evidence that the test partition was opened once."""

    __test__ = False  # not a pytest collection target

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    schema_version: Literal["openalpha.bridge.phase2.test_opening.v1"] = (
        "openalpha.bridge.phase2.test_opening.v1"
    )
    opened_at: datetime
    run_id: str = Field(min_length=1)
    evidence_class: EvidenceClass
    experiment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_commit: str | None
    preprocessing_state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    operator_command: str = Field(min_length=1)
    prior_ledger_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @property
    def ledger_sha256(self) -> str:
        return canonical_sha256(self.model_dump(mode="json"))


def _ledger_path(run_directory: Path, evidence_class: EvidenceClass) -> Path:
    name = (
        _REAL_LEDGER
        if evidence_class is EvidenceClass.REAL_PHASE2
        else _SYNTHETIC_LEDGER
    )
    return run_directory / name


def is_test_partition_opened(run_directory: Path, evidence_class: EvidenceClass) -> bool:
    """Whether the test partition has been opened for this run and namespace."""
    return _ledger_path(run_directory, evidence_class).is_file()


def open_test_partition(
    *,
    run_directory: Path,
    identity: RunIdentity,
    journal: StateJournal,
    preconditions: TestOpeningPreconditions,
    operator_command: str,
    opened_at: datetime,
    prior_ledger_sha256: str | None = None,
) -> tuple[TestOpeningRecord, StateJournal]:
    """Open the test partition exactly once, or fail closed."""
    if journal.evidence_class is not identity.evidence_class:
        raise _fail(
            "EVIDENCE_CLASS_MISMATCH",
            "journal and run identity disagree on evidence class",
        )
    if journal.current_state is not Phase2State.CHECKPOINT_FROZEN:
        raise _fail(
            "TEST_OPENED_BEFORE_CHECKPOINT_FROZEN",
            (
                "the test partition may open only from CHECKPOINT_FROZEN, "
                f"current state is {journal.current_state.value}"
            ),
        )

    unmet = preconditions.unmet()
    if unmet:
        raise _fail(
            "TEST_OPENING_PRECONDITION_UNMET",
            f"unmet test-opening preconditions: {', '.join(unmet)}",
        )

    ledger = _ledger_path(run_directory, identity.evidence_class)
    if ledger.exists():
        raise _fail(
            "TEST_PARTITION_ALREADY_OPENED",
            (
                f"run {identity.run_id} already opened the test partition; "
                "the record is immutable and cannot be recreated"
            ),
        )

    # Narrowed by preconditions.unmet(), which rejects None for each of these.
    assert preconditions.checkpoint_sha256 is not None
    assert preconditions.preprocessing_state_sha256 is not None
    assert preconditions.feature_manifest_sha256 is not None

    record = TestOpeningRecord(
        opened_at=opened_at,
        run_id=identity.run_id,
        evidence_class=identity.evidence_class,
        experiment_sha256=identity.experiment_sha256,
        checkpoint_sha256=preconditions.checkpoint_sha256,
        source_commit=identity.source_commit,
        preprocessing_state_sha256=preconditions.preprocessing_state_sha256,
        feature_manifest_sha256=preconditions.feature_manifest_sha256,
        operator_command=operator_command,
        prior_ledger_sha256=prior_ledger_sha256,
    )

    run_directory.mkdir(parents=True, exist_ok=True)
    # Exclusive create: a concurrent opener loses rather than overwrites.
    try:
        with ledger.open("xb") as handle:
            handle.write(canonical_json(record.model_dump(mode="json")))
    except FileExistsError as error:
        raise _fail(
            "TEST_PARTITION_ALREADY_OPENED",
            f"run {identity.run_id} already opened the test partition",
        ) from error

    advanced = journal.advance(
        Phase2State.TEST_OPENED,
        occurred_at=opened_at,
        reason="explicit open_test_partition",
        artifact_sha256=record.ledger_sha256,
    )
    return record, advanced
