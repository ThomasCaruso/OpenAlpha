"""One immutable terminal artifact per zero-shot benchmark invocation.

Its own root, object name, success schema and failure schema. A benchmark key
cannot be produced from a mini or base run identifier, and neither of theirs can
be produced from a ``zsb_`` one, so no study can overwrite, shadow or be
mistaken for another.

An operational failure records the exception class and a fixed message. Exception
text can carry a provider URL, a credential or a slice of a response, and an
artifact is durable and readable by anyone who can read the bucket.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any, Final, Literal

from openalpha_research.failures import FailureCategory, ResearchFailure, ResearchFailureError
from openalpha_research.objectstore import ObjectStore, get_json, put_json
from pydantic import BaseModel, ConfigDict

from openalpha_kronos.studies.provenance import EvidenceClass
from openalpha_kronos.studies.structural_validity.mini.safe_logging import StageTracker

from .invocation import ZeroShotInvocation
from .runner import ZeroShotBenchmarkArtifact
from .spec import (
    ZERO_SHOT_CLAIM_BOUNDARY,
    ZERO_SHOT_EXPERIMENT_ID,
    ZERO_SHOT_FAILURE_SCHEMA_VERSION,
    ZERO_SHOT_SPECIFICATION_NAME,
    ZERO_SHOT_SPECIFICATION_SHA256,
    ZERO_SHOT_SUCCESS_SCHEMA_VERSION,
    zero_shot_artifact_key,
)

__all__ = [
    "ZERO_SHOT_FAILURE_CODE",
    "ZERO_SHOT_OPERATIONAL_FAILURE_CODE",
    "ZERO_SHOT_OPERATIONAL_FAILURE_MESSAGE",
    "ZERO_SHOT_SUCCESS_CODE",
    "ZeroShotFailure",
    "ZeroShotTerminalArtifact",
    "find_existing_zero_shot_artifact",
    "publish_zero_shot_artifact",
    "verify_existing_zero_shot_artifact",
]

ZERO_SHOT_SUCCESS_CODE: Final[str] = "KRONOS_ZERO_SHOT_BENCHMARK_COMPLETED"
ZERO_SHOT_FAILURE_CODE: Final[str] = "KRONOS_ZERO_SHOT_BENCHMARK_FAILED"
ZERO_SHOT_OPERATIONAL_FAILURE_CODE: Final[str] = (
    "KRONOS_ZERO_SHOT_BENCHMARK_OPERATIONAL_FAILURE"
)

ZERO_SHOT_OPERATIONAL_FAILURE_MESSAGE: Final[str] = (
    "the zero-shot benchmark did not reach a decision; see the worker logs for "
    "detail, which are not reproduced here because exception text can carry "
    "credentials, provider responses, or local paths"
)

_SCHEMA_FOR_OUTCOME: Final[dict[str, str]] = {
    ZERO_SHOT_SUCCESS_CODE: ZERO_SHOT_SUCCESS_SCHEMA_VERSION,
    ZERO_SHOT_FAILURE_CODE: ZERO_SHOT_FAILURE_SCHEMA_VERSION,
    ZERO_SHOT_OPERATIONAL_FAILURE_CODE: ZERO_SHOT_FAILURE_SCHEMA_VERSION,
}


class ZeroShotFailure(BaseModel):
    """A benchmark run that did not reach a decision."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    schema_version: Literal["openalpha.bridge.zero_shot.kronos_zero_shot_failure.v1"] = (
        ZERO_SHOT_FAILURE_SCHEMA_VERSION
    )
    claim_boundary: Literal["DEVELOPMENT BENCHMARK - NOT HOLDOUT OR TRADING EVIDENCE"] = (
        ZERO_SHOT_CLAIM_BOUNDARY
    )
    outcome: Literal[
        "KRONOS_ZERO_SHOT_BENCHMARK_FAILED",
        "KRONOS_ZERO_SHOT_BENCHMARK_OPERATIONAL_FAILURE",
    ]
    decision_outcome: str | None = None
    evidence_class: EvidenceClass = EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY
    experiment_id: Literal["openalpha-kronos-zero-shot-benchmark-v1"] = ZERO_SHOT_EXPERIMENT_ID
    model_family: Literal["kronos-base"] = "kronos-base"
    study_type: Literal["zero_shot_forecast_benchmark"] = "zero_shot_forecast_benchmark"
    run_id: str
    source_commit: str
    deployed_commit: str
    specification_name: str
    specification_sha256: str
    failure_stage: str
    failure_code: str
    exception_class: str | None = None
    message: str
    completed_at: datetime
    training_performed: Literal[False] = False
    optimizer_constructed: Literal[False] = False
    held_out_partition_opened: Literal[False] = False
    authorizes_training: Literal[False] = False
    authorizes_fine_tuning: Literal[False] = False
    authorizes_representation_probe: Literal[False] = False
    authorizes_stage_b: Literal[False] = False
    authorizes_stage_c: Literal[False] = False
    authorizes_test_opening: Literal[False] = False
    authorizes_production_inference: Literal[False] = False
    authorizes_trading_claims: Literal[False] = False
    scientific_result_available: Literal[False] = False


class ZeroShotTerminalArtifact(BaseModel):
    """What was written, and whether this invocation wrote it."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    key: str
    content_sha256: str
    outcome: str
    decision_outcome: str | None
    evidence_class: Literal[EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY] = (
        EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY
    )
    already_existed: bool
    payload: dict[str, Any]


def _corrupt(key: str, detail: str) -> ResearchFailureError:
    return ResearchFailureError(
        ResearchFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code="ZERO_SHOT_TERMINAL_ARTIFACT_CORRUPT",
            message=f"the existing zero-shot artifact at {key} {detail}",
        )
    )


def _mismatch(code: str, key: str, detail: str) -> ResearchFailureError:
    return ResearchFailureError(
        ResearchFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            message=f"the existing zero-shot artifact at {key} {detail}",
        )
    )


def find_existing_zero_shot_artifact(
    store: ObjectStore, *, invocation: ZeroShotInvocation
) -> ZeroShotTerminalArtifact | None:
    """The verified artifact for this run, if one exists. Checked first."""
    if not store.exists(zero_shot_artifact_key(invocation.run_id)):
        return None
    return verify_existing_zero_shot_artifact(store, invocation=invocation)


def verify_existing_zero_shot_artifact(
    store: ObjectStore, *, invocation: ZeroShotInvocation
) -> ZeroShotTerminalArtifact:
    """Read the stored artifact and check it end to end before returning it."""
    key = zero_shot_artifact_key(invocation.run_id)
    stored = store.get(key)

    observed = hashlib.sha256(stored.body).hexdigest()
    if observed != stored.metadata.content_sha256:
        raise _corrupt(
            key,
            f"hashes to {observed} but its metadata records {stored.metadata.content_sha256}",
        )

    existing = get_json(store, key)
    if not isinstance(existing, dict):
        raise _corrupt(key, "is not a JSON object")

    outcome = existing.get("outcome")
    schema_version = existing.get("schema_version")
    if not isinstance(outcome, str) or outcome not in _SCHEMA_FOR_OUTCOME:
        raise _corrupt(key, f"records an unrecognised outcome {outcome!r}")
    if _SCHEMA_FOR_OUTCOME[outcome] != schema_version:
        raise _corrupt(
            key, f"pairs outcome {outcome!r} with schema {schema_version!r}, which disagree"
        )

    if existing.get("experiment_id") != ZERO_SHOT_EXPERIMENT_ID:
        raise _mismatch(
            "ZERO_SHOT_TERMINAL_ARTIFACT_EXPERIMENT_MISMATCH",
            key,
            f"was produced under experiment {existing.get('experiment_id')!r}",
        )
    if existing.get("study_type") != "zero_shot_forecast_benchmark":
        raise _mismatch(
            "ZERO_SHOT_TERMINAL_ARTIFACT_STUDY_TYPE_MISMATCH",
            key,
            f"records study type {existing.get('study_type')!r}",
        )
    if existing.get("specification_name") != ZERO_SHOT_SPECIFICATION_NAME:
        raise _mismatch(
            "ZERO_SHOT_TERMINAL_ARTIFACT_SPECIFICATION_MISMATCH",
            key,
            f"records specification {existing.get('specification_name')!r}",
        )
    if existing.get("specification_sha256") != ZERO_SHOT_SPECIFICATION_SHA256:
        raise _mismatch(
            "ZERO_SHOT_TERMINAL_ARTIFACT_SPECIFICATION_MISMATCH",
            key,
            "records a different specification digest",
        )
    if existing.get("evidence_class") != EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY.value:
        raise _mismatch(
            "ZERO_SHOT_TERMINAL_ARTIFACT_EVIDENCE_CLASS_MISMATCH",
            key,
            "is not development benchmark evidence",
        )
    if stored.metadata.evidence_class != EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY.value:
        raise _mismatch(
            "ZERO_SHOT_TERMINAL_ARTIFACT_EVIDENCE_CLASS_MISMATCH",
            key,
            "carries object metadata for a different evidence class",
        )
    if existing.get("run_id") != invocation.run_id or stored.metadata.run_id != invocation.run_id:
        raise _mismatch(
            "ZERO_SHOT_TERMINAL_ARTIFACT_RUN_ID_MISMATCH", key, "belongs to a different run"
        )
    if existing.get("source_commit") != invocation.source_commit:
        raise _mismatch(
            "ZERO_SHOT_TERMINAL_ARTIFACT_COMMIT_MISMATCH",
            key,
            "records a different requested source_commit",
        )
    if existing.get("deployed_commit") != invocation.deployed_commit:
        raise _mismatch(
            "ZERO_SHOT_TERMINAL_ARTIFACT_COMMIT_MISMATCH",
            key,
            "was produced by an image built from a different commit",
        )
    if existing.get("claim_boundary") != ZERO_SHOT_CLAIM_BOUNDARY:
        raise _mismatch(
            "ZERO_SHOT_TERMINAL_ARTIFACT_CLAIM_BOUNDARY_MISMATCH",
            key,
            "does not carry the benchmark claim boundary",
        )

    return ZeroShotTerminalArtifact(
        key=key,
        content_sha256=observed,
        outcome=outcome,
        decision_outcome=_decision_outcome_of(existing),
        already_existed=True,
        payload=existing,
    )


def _decision_outcome_of(payload: dict[str, Any]) -> str | None:
    decision = payload.get("decision")
    if isinstance(decision, dict):
        value = decision.get("outcome")
        if isinstance(value, str):
            return value
    value = payload.get("decision_outcome")
    return value if isinstance(value, str) else None


def _write(
    store: ObjectStore,
    *,
    invocation: ZeroShotInvocation,
    payload: dict[str, Any],
    schema_version: str,
    outcome: str,
    decision_outcome: str | None,
    stage: StageTracker | None = None,
) -> ZeroShotTerminalArtifact:
    tracker = stage or StageTracker()
    key = zero_shot_artifact_key(invocation.run_id)
    tracker.enter("write_artifact")
    try:
        metadata = put_json(
            store,
            key,
            payload,
            schema_version=schema_version,
            run_id=invocation.run_id,
            experiment_hash=ZERO_SHOT_SPECIFICATION_SHA256,
            evidence_class=EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY,
            immutable=True,
        )
    except ResearchFailureError as error:
        if error.failures[0].code != "OBJECT_ALREADY_EXISTS":
            raise
        tracker.enter("verify_existing_artifact")
        return verify_existing_zero_shot_artifact(store, invocation=invocation)
    return ZeroShotTerminalArtifact(
        key=key,
        content_sha256=metadata.content_sha256,
        outcome=outcome,
        decision_outcome=decision_outcome,
        already_existed=False,
        payload=payload,
    )


def publish_zero_shot_artifact(
    store: ObjectStore,
    *,
    invocation: ZeroShotInvocation,
    execute: Any,
    now: datetime | None = None,
    stage: StageTracker | None = None,
) -> ZeroShotTerminalArtifact:
    """Return the existing artifact, or run the benchmark and write one.

    The existence check comes first, so a duplicate invocation issues no
    provider request and loads no weight.
    """
    tracker = stage or StageTracker()

    tracker.enter("verify_existing_artifact")
    already = find_existing_zero_shot_artifact(store, invocation=invocation)
    if already is not None:
        return already

    stamped = now or datetime.now(UTC)
    try:
        result: ZeroShotBenchmarkArtifact = execute()
    except ResearchFailureError as error:
        tracker.enter("serialize_artifact")
        first = error.failures[0]
        failure = ZeroShotFailure(
            outcome=ZERO_SHOT_FAILURE_CODE,
            run_id=invocation.run_id,
            source_commit=invocation.source_commit,
            deployed_commit=invocation.deployed_commit,
            specification_name=ZERO_SHOT_SPECIFICATION_NAME,
            specification_sha256=ZERO_SHOT_SPECIFICATION_SHA256,
            failure_stage="kronos_zero_shot_benchmark",
            failure_code=first.code,
            exception_class=type(error).__name__,
            message=first.message[:2000],
            completed_at=stamped,
        )
    except Exception as error:  # noqa: BLE001 - nothing may escape unrecorded
        tracker.enter("serialize_artifact")
        failure = ZeroShotFailure(
            outcome=ZERO_SHOT_OPERATIONAL_FAILURE_CODE,
            run_id=invocation.run_id,
            source_commit=invocation.source_commit,
            deployed_commit=invocation.deployed_commit,
            specification_name=ZERO_SHOT_SPECIFICATION_NAME,
            specification_sha256=ZERO_SHOT_SPECIFICATION_SHA256,
            failure_stage="kronos_zero_shot_benchmark",
            failure_code="ZERO_SHOT_OPERATIONAL_FAILURE",
            # The class, and nothing else.
            exception_class=type(error).__name__,
            message=ZERO_SHOT_OPERATIONAL_FAILURE_MESSAGE,
            completed_at=stamped,
        )
    else:
        tracker.enter("serialize_artifact")
        payload = result.model_dump(mode="json")
        payload["outcome"] = ZERO_SHOT_SUCCESS_CODE
        tracker.enter("write_artifact")
        return _write(
            store,
            invocation=invocation,
            payload=payload,
            schema_version=result.schema_version,
            outcome=ZERO_SHOT_SUCCESS_CODE,
            decision_outcome=result.decision.outcome.value,
            stage=tracker,
        )

    tracker.enter("serialize_artifact")
    failure_payload = failure.model_dump(mode="json")
    tracker.enter("write_artifact")
    return _write(
        store,
        invocation=invocation,
        payload=failure_payload,
        schema_version=failure.schema_version,
        outcome=failure.outcome,
        decision_outcome=None,
        stage=tracker,
    )
