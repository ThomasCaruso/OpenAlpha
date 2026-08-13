"""One immutable terminal artifact per base-study invocation.

Its own root, its own object name, its own success and failure schemas. A base
key cannot be produced from a mini run identifier and a mini key cannot be
produced from a base one, so the two studies cannot overwrite, shadow or be
mistaken for each other even by accident.

As in the mini study, an operational failure records the exception class and a
fixed message. Exception text can carry a provider URL, a credential or a slice
of a response, and an artifact is durable and readable by anyone who can read
the bucket.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any, Final, Literal

from openalpha_research.failures import FailureCategory, ResearchFailure, ResearchFailureError
from openalpha_research.objectstore import ObjectStore, get_json, put_json
from pydantic import BaseModel, ConfigDict

from openalpha_kronos.studies.provenance import EvidenceClass
from openalpha_kronos.studies.structural_validity.common.conclusion import DiagnosticConclusion
from openalpha_kronos.studies.structural_validity.mini.safe_logging import StageTracker

from .invocation import BaseStudyInvocation
from .runner import KronosBaseDiagnosticArtifact
from .spec import (
    BASE_CLAIM_BOUNDARY,
    BASE_EXPERIMENT_ID,
    BASE_FAILURE_SCHEMA_VERSION,
    BASE_SPECIFICATION_NAME,
    BASE_SPECIFICATION_SHA256,
    BASE_SUCCESS_SCHEMA_VERSION,
    base_artifact_key,
)

__all__ = [
    "BASE_FAILURE_CODE",
    "BASE_OPERATIONAL_FAILURE_CODE",
    "BASE_OPERATIONAL_FAILURE_MESSAGE",
    "BASE_SUCCESS_CODE",
    "BaseStudyFailure",
    "BaseStudyTerminalArtifact",
    "find_existing_base_artifact",
    "publish_base_artifact",
    "verify_existing_base_artifact",
]

BASE_SUCCESS_CODE: Final[str] = "KRONOS_BASE_DIAGNOSTIC_COMPLETED"
BASE_FAILURE_CODE: Final[str] = "KRONOS_BASE_DIAGNOSTIC_FAILED"
BASE_OPERATIONAL_FAILURE_CODE: Final[str] = "KRONOS_BASE_DIAGNOSTIC_OPERATIONAL_FAILURE"

BASE_OPERATIONAL_FAILURE_MESSAGE: Final[str] = (
    "the base replication did not reach a conclusion; see the worker logs for "
    "detail, which are not reproduced here because exception text can carry "
    "credentials, provider responses, or local paths"
)

_SCHEMA_FOR_OUTCOME: Final[dict[str, str]] = {
    BASE_SUCCESS_CODE: BASE_SUCCESS_SCHEMA_VERSION,
    BASE_FAILURE_CODE: BASE_FAILURE_SCHEMA_VERSION,
    BASE_OPERATIONAL_FAILURE_CODE: BASE_FAILURE_SCHEMA_VERSION,
}


class BaseStudyFailure(BaseModel):
    """A base replication that did not reach a conclusion."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    schema_version: Literal["openalpha.bridge.base_study.kronos_base_failure.v1"] = (
        BASE_FAILURE_SCHEMA_VERSION
    )
    claim_boundary: Literal["DEVELOPMENT DIAGNOSTIC - NOT HOLDOUT OR TRADING EVIDENCE"] = (
        BASE_CLAIM_BOUNDARY
    )
    outcome: Literal[
        "KRONOS_BASE_DIAGNOSTIC_FAILED",
        "KRONOS_BASE_DIAGNOSTIC_OPERATIONAL_FAILURE",
    ]
    conclusion: DiagnosticConclusion | None = None
    evidence_class: EvidenceClass = EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY
    experiment_id: Literal["openalpha-kronos-base-replication-v1"] = BASE_EXPERIMENT_ID
    model_family: Literal["kronos-base"] = "kronos-base"
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
    authorizes_training: Literal[False] = False
    authorizes_stage_b: Literal[False] = False
    authorizes_stage_c: Literal[False] = False
    authorizes_test_opening: Literal[False] = False
    authorizes_production_inference: Literal[False] = False
    authorizes_trading_claims: Literal[False] = False
    scientific_result_available: Literal[False] = False


class BaseStudyTerminalArtifact(BaseModel):
    """What was written, and whether this invocation wrote it."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    key: str
    content_sha256: str
    outcome: str
    conclusion: str | None
    evidence_class: Literal[EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY] = (
        EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY
    )
    already_existed: bool
    payload: dict[str, Any]


def _corrupt(key: str, detail: str) -> ResearchFailureError:
    return ResearchFailureError(
        ResearchFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code="BASE_TERMINAL_ARTIFACT_CORRUPT",
            message=f"the existing base artifact at {key} {detail}",
        )
    )


def _mismatch(code: str, key: str, detail: str) -> ResearchFailureError:
    return ResearchFailureError(
        ResearchFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            message=f"the existing base artifact at {key} {detail}",
        )
    )


def find_existing_base_artifact(
    store: ObjectStore, *, invocation: BaseStudyInvocation
) -> BaseStudyTerminalArtifact | None:
    """The verified artifact for this run, if one exists. Checked first."""
    if not store.exists(base_artifact_key(invocation.run_id)):
        return None
    return verify_existing_base_artifact(store, invocation=invocation)


def verify_existing_base_artifact(
    store: ObjectStore, *, invocation: BaseStudyInvocation
) -> BaseStudyTerminalArtifact:
    """Read the stored artifact and check it end to end before returning it."""
    key = base_artifact_key(invocation.run_id)
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

    if existing.get("experiment_id") != BASE_EXPERIMENT_ID:
        raise _mismatch(
            "BASE_TERMINAL_ARTIFACT_EXPERIMENT_MISMATCH",
            key,
            f"was produced under experiment {existing.get('experiment_id')!r}",
        )
    if existing.get("model_family") != "kronos-base":
        raise _mismatch(
            "BASE_TERMINAL_ARTIFACT_MODEL_FAMILY_MISMATCH",
            key,
            f"records model family {existing.get('model_family')!r}, not kronos-base",
        )
    if existing.get("specification_name") != BASE_SPECIFICATION_NAME:
        raise _mismatch(
            "BASE_TERMINAL_ARTIFACT_SPECIFICATION_MISMATCH",
            key,
            f"records specification {existing.get('specification_name')!r}",
        )
    if existing.get("specification_sha256") != BASE_SPECIFICATION_SHA256:
        raise _mismatch(
            "BASE_TERMINAL_ARTIFACT_SPECIFICATION_MISMATCH",
            key,
            "records a different specification digest",
        )
    if existing.get("evidence_class") != EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY.value:
        raise _mismatch(
            "BASE_TERMINAL_ARTIFACT_EVIDENCE_CLASS_MISMATCH",
            key,
            "is not development diagnostic evidence",
        )
    if stored.metadata.evidence_class != EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY.value:
        raise _mismatch(
            "BASE_TERMINAL_ARTIFACT_EVIDENCE_CLASS_MISMATCH",
            key,
            "carries object metadata for a different evidence class",
        )
    if existing.get("run_id") != invocation.run_id or stored.metadata.run_id != invocation.run_id:
        raise _mismatch("BASE_TERMINAL_ARTIFACT_RUN_ID_MISMATCH", key, "belongs to a different run")
    if existing.get("source_commit") != invocation.source_commit:
        raise _mismatch(
            "BASE_TERMINAL_ARTIFACT_COMMIT_MISMATCH",
            key,
            "records a different requested source_commit",
        )
    if existing.get("deployed_commit") != invocation.deployed_commit:
        raise _mismatch(
            "BASE_TERMINAL_ARTIFACT_COMMIT_MISMATCH",
            key,
            "was produced by an image built from a different commit",
        )
    if existing.get("claim_boundary") != BASE_CLAIM_BOUNDARY:
        raise _mismatch(
            "BASE_TERMINAL_ARTIFACT_CLAIM_BOUNDARY_MISMATCH",
            key,
            "does not carry the diagnostic claim boundary",
        )

    conclusion = existing.get("conclusion")
    return BaseStudyTerminalArtifact(
        key=key,
        content_sha256=observed,
        outcome=outcome,
        conclusion=conclusion if isinstance(conclusion, str) else None,
        already_existed=True,
        payload=existing,
    )


def _write(
    store: ObjectStore,
    *,
    invocation: BaseStudyInvocation,
    payload: dict[str, Any],
    schema_version: str,
    outcome: str,
    conclusion: str | None,
    stage: StageTracker | None = None,
) -> BaseStudyTerminalArtifact:
    tracker = stage or StageTracker()
    key = base_artifact_key(invocation.run_id)
    tracker.enter("write_artifact")
    try:
        metadata = put_json(
            store,
            key,
            payload,
            schema_version=schema_version,
            run_id=invocation.run_id,
            experiment_hash=BASE_SPECIFICATION_SHA256,
            evidence_class=EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY,
            immutable=True,
        )
    except ResearchFailureError as error:
        if error.failures[0].code != "OBJECT_ALREADY_EXISTS":
            raise
        tracker.enter("verify_existing_artifact")
        return verify_existing_base_artifact(store, invocation=invocation)
    return BaseStudyTerminalArtifact(
        key=key,
        content_sha256=metadata.content_sha256,
        outcome=outcome,
        conclusion=conclusion,
        already_existed=False,
        payload=payload,
    )


def publish_base_artifact(
    store: ObjectStore,
    *,
    invocation: BaseStudyInvocation,
    execute: Any,
    now: datetime | None = None,
    stage: StageTracker | None = None,
) -> BaseStudyTerminalArtifact:
    """Return the existing artifact, or run the replication and write one.

    The existence check comes first, so a duplicate invocation issues no
    provider request and loads no weight. ``stage`` is write-only here, exactly
    as in the mini publication: nothing reads it, so it cannot influence a
    payload, a digest, a conclusion, or which branch runs.
    """
    tracker = stage or StageTracker()

    tracker.enter("verify_existing_artifact")
    already = find_existing_base_artifact(store, invocation=invocation)
    if already is not None:
        return already

    stamped = now or datetime.now(UTC)
    try:
        result: KronosBaseDiagnosticArtifact = execute()
    except ResearchFailureError as error:
        tracker.enter("serialize_artifact")
        first = error.failures[0]
        failure = BaseStudyFailure(
            outcome=BASE_FAILURE_CODE,
            conclusion=None,
            run_id=invocation.run_id,
            source_commit=invocation.source_commit,
            deployed_commit=invocation.deployed_commit,
            specification_name=BASE_SPECIFICATION_NAME,
            specification_sha256=BASE_SPECIFICATION_SHA256,
            failure_stage="kronos_base_diagnostic",
            failure_code=first.code,
            exception_class=type(error).__name__,
            message=first.message[:2000],
            completed_at=stamped,
        )
    except Exception as error:  # noqa: BLE001 - nothing may escape unrecorded
        tracker.enter("serialize_artifact")
        failure = BaseStudyFailure(
            outcome=BASE_OPERATIONAL_FAILURE_CODE,
            conclusion=DiagnosticConclusion.DIAGNOSTIC_OPERATIONAL_FAILURE,
            run_id=invocation.run_id,
            source_commit=invocation.source_commit,
            deployed_commit=invocation.deployed_commit,
            specification_name=BASE_SPECIFICATION_NAME,
            specification_sha256=BASE_SPECIFICATION_SHA256,
            failure_stage="kronos_base_diagnostic",
            failure_code="BASE_STUDY_OPERATIONAL_FAILURE",
            # The class, and nothing else.
            exception_class=type(error).__name__,
            message=BASE_OPERATIONAL_FAILURE_MESSAGE,
            completed_at=stamped,
        )
    else:
        tracker.enter("serialize_artifact")
        payload = result.model_dump(mode="json")
        payload["outcome"] = BASE_SUCCESS_CODE
        success_conclusion = result.conclusion.value
        tracker.enter("write_artifact")
        return _write(
            store,
            invocation=invocation,
            payload=payload,
            schema_version=result.schema_version,
            outcome=BASE_SUCCESS_CODE,
            conclusion=success_conclusion,
            stage=tracker,
        )

    tracker.enter("serialize_artifact")
    failure_payload = failure.model_dump(mode="json")
    failure_conclusion = failure.conclusion.value if failure.conclusion else None
    tracker.enter("write_artifact")
    return _write(
        store,
        invocation=invocation,
        payload=failure_payload,
        schema_version=failure.schema_version,
        outcome=failure.outcome,
        conclusion=failure_conclusion,
        stage=tracker,
    )
