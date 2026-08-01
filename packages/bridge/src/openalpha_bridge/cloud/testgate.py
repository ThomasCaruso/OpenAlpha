"""One-time test-partition gate backed by immutable cloud storage.

The filesystem guard becomes an object-store conditional create. Exactly one
``evaluation/test_opening_record.json`` may exist per run; a second attempt,
including one from a resumed worker, fails.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..errors import BridgeFailure, BridgeTransformError, FailureCategory
from ..phase2.states import EvidenceClass, Phase2State
from ..phase2.testgate import TestOpeningPreconditions
from .identity import CloudRunIdentity
from .journal import CloudJournal
from .objectstore import ObjectStore, get_model, put_json, run_prefix

__all__ = [
    "CloudTestOpeningRecord",
    "cloud_test_opening_key",
    "is_cloud_test_partition_opened",
    "open_cloud_test_partition",
]


def _fail(code: str, message: str) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            message=message,
        )
    )


def cloud_test_opening_key(run_id: str, evidence_class: EvidenceClass) -> str:
    return f"{run_prefix(run_id, evidence_class)}/evaluation/test_opening_record.json"


class CloudTestOpeningRecord(BaseModel):
    """Immutable proof that the test partition was opened once, in the cloud."""

    __test__ = False  # not a pytest collection target

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    schema_version: Literal["openalpha.bridge.phase2.cloud_test_opening.v1"] = (
        "openalpha.bridge.phase2.cloud_test_opening.v1"
    )
    run_id: str = Field(min_length=1)
    experiment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    frozen_weight_verified: bool
    preprocessing_state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_commit: str
    cloud_image_digest: str
    operator: str
    opened_at: datetime
    prior_ledger_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    evidence_class: EvidenceClass


def is_cloud_test_partition_opened(
    store: ObjectStore, *, run_id: str, evidence_class: EvidenceClass
) -> bool:
    return store.exists(cloud_test_opening_key(run_id, evidence_class))


def open_cloud_test_partition(
    store: ObjectStore,
    *,
    identity: CloudRunIdentity,
    journal: CloudJournal,
    preconditions: TestOpeningPreconditions,
    opened_at: datetime,
) -> CloudTestOpeningRecord:
    """Open the reconstruction-test partition exactly once, or fail closed."""
    if journal.current_state() is not Phase2State.CHECKPOINT_FROZEN:
        raise _fail(
            "TEST_OPENED_BEFORE_CHECKPOINT_FROZEN",
            (
                "the test partition may open only from CHECKPOINT_FROZEN, current state is "
                f"{journal.current_state().value}"
            ),
        )

    unmet = preconditions.unmet()
    if unmet:
        raise _fail(
            "TEST_OPENING_PRECONDITION_UNMET",
            f"unmet test-opening preconditions: {', '.join(unmet)}",
        )

    # Narrowed by unmet(), which rejects None for each of these.
    assert preconditions.checkpoint_sha256 is not None
    assert preconditions.preprocessing_state_sha256 is not None
    assert preconditions.feature_manifest_sha256 is not None

    latest = journal.latest()
    record = CloudTestOpeningRecord(
        run_id=identity.run_id,
        experiment_hash=identity.active_experiment_sha256,
        checkpoint_sha256=preconditions.checkpoint_sha256,
        frozen_weight_verified=preconditions.frozen_kronos_weights_verified,
        preprocessing_state_sha256=preconditions.preprocessing_state_sha256,
        feature_manifest_sha256=preconditions.feature_manifest_sha256,
        source_commit=identity.source_commit,
        cloud_image_digest=identity.cloud_image_digest,
        operator=identity.operator,
        opened_at=opened_at,
        prior_ledger_sha256=latest.entry_sha256 if latest else None,
        evidence_class=identity.evidence_class,
    )

    key = cloud_test_opening_key(identity.run_id, identity.evidence_class)
    try:
        put_json(
            store,
            key,
            record.model_dump(mode="json"),
            schema_version=record.schema_version,
            run_id=identity.run_id,
            experiment_hash=identity.active_experiment_sha256,
            evidence_class=identity.evidence_class,
            immutable=True,
            prior_journal_sha256=record.prior_ledger_sha256,
            created_at=opened_at,
        )
    except BridgeTransformError as error:
        raise _fail(
            "TEST_PARTITION_ALREADY_OPENED",
            (
                f"run {identity.run_id} already opened the reconstruction-test partition; "
                "the record is immutable and no resume may recreate it"
            ),
        ) from error

    journal.append(
        Phase2State.TEST_OPENED,
        occurred_at=opened_at,
        stage="open_test_partition",
        reason="explicit one-time cloud test opening",
    )
    return record


def read_cloud_test_opening_record(
    store: ObjectStore, *, run_id: str, evidence_class: EvidenceClass
) -> CloudTestOpeningRecord | None:
    key = cloud_test_opening_key(run_id, evidence_class)
    if not store.exists(key):
        return None
    return get_model(store, key, CloudTestOpeningRecord)
