"""One immutable terminal artifact per diagnostic invocation.

Stored under a diagnostic-only root, distinct from the Stage A canary's own
terminal artifact, so a compatibility result and a diagnostic result can never
be mistaken for one another even though both are development evidence.

Success, typed failure and operational failure are preserved distinctly. An
operational failure records the exception class and a fixed message: exception
text can carry a provider URL, a credential or a slice of a response, and an
artifact is durable and readable by anyone who can read the bucket.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict

from ..cloud.objectstore import ObjectStore, get_json, put_json, run_prefix
from ..errors import BridgeFailure, BridgeTransformError, FailureCategory
from ..phase2.identity import EXPERIMENT_SHA256
from ..phase2.invocation import WorkerInvocation
from ..phase2.states import EvidenceClass
from .conclusion import DiagnosticConclusion
from .runner import DIAGNOSTIC_SCHEMA_VERSION, DiagnosticArtifact
from .spec import CLAIM_BOUNDARY, V2_SPECIFICATION_SHA256

__all__ = [
    "DIAGNOSTIC_ARTIFACT_NAME",
    "DIAGNOSTIC_FAILURE_CODE",
    "DIAGNOSTIC_OPERATIONAL_FAILURE_CODE",
    "DIAGNOSTIC_SUCCESS_CODE",
    "OPERATIONAL_FAILURE_MESSAGE",
    "DiagnosticFailure",
    "DiagnosticTerminalArtifact",
    "diagnostic_artifact_key",
    "publish_diagnostic_artifact",
]

DIAGNOSTIC_SUCCESS_CODE: Final[str] = "FROZEN_INFERENCE_DIAGNOSTIC_COMPLETED"
DIAGNOSTIC_FAILURE_CODE: Final[str] = "FROZEN_INFERENCE_DIAGNOSTIC_FAILED"
DIAGNOSTIC_OPERATIONAL_FAILURE_CODE: Final[str] = "FROZEN_INFERENCE_DIAGNOSTIC_OPERATIONAL_FAILURE"

#: Its own object name under its own subtree, so a diagnostic artifact and a
#: Stage A canary artifact can never collide or be confused.
DIAGNOSTIC_ARTIFACT_NAME: Final[str] = "frozen_inference_diagnostic_terminal.json"
DIAGNOSTIC_SUBTREE: Final[str] = "frozen-inference-diagnostic"

OPERATIONAL_FAILURE_MESSAGE: Final[str] = (
    "the diagnostic did not reach a conclusion; see the worker logs for detail, "
    "which are not reproduced here because exception text can carry credentials, "
    "provider responses, or local paths"
)

FAILURE_SCHEMA_VERSION: Final[str] = "openalpha.bridge.diagnostic.frozen_inference_failure.v1"

_SCHEMA_FOR_OUTCOME: Final[dict[str, str]] = {
    DIAGNOSTIC_SUCCESS_CODE: DIAGNOSTIC_SCHEMA_VERSION,
    DIAGNOSTIC_FAILURE_CODE: FAILURE_SCHEMA_VERSION,
    DIAGNOSTIC_OPERATIONAL_FAILURE_CODE: FAILURE_SCHEMA_VERSION,
}


def diagnostic_artifact_key(run_id: str) -> str:
    """The only key this module writes."""
    prefix = run_prefix(run_id, EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY)
    return f"{prefix}/{DIAGNOSTIC_SUBTREE}/{DIAGNOSTIC_ARTIFACT_NAME}"


class DiagnosticFailure(BaseModel):
    """A diagnostic that did not reach a conclusion, preserved."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    schema_version: Literal["openalpha.bridge.diagnostic.frozen_inference_failure.v1"] = (
        "openalpha.bridge.diagnostic.frozen_inference_failure.v1"
    )
    claim_boundary: Literal["DEVELOPMENT DIAGNOSTIC - NOT HOLDOUT OR TRADING EVIDENCE"] = (
        "DEVELOPMENT DIAGNOSTIC - NOT HOLDOUT OR TRADING EVIDENCE"
    )
    outcome: Literal[
        "FROZEN_INFERENCE_DIAGNOSTIC_FAILED",
        "FROZEN_INFERENCE_DIAGNOSTIC_OPERATIONAL_FAILURE",
    ]
    #: An operational failure has no diagnostic conclusion. A typed failure
    #: carries the one the rules would assign, which is the operational label.
    conclusion: DiagnosticConclusion | None = None
    evidence_class: EvidenceClass = EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY
    run_id: str
    source_commit: str
    deployed_commit: str
    experiment_sha256: str
    specification_v2_sha256: str
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


class DiagnosticTerminalArtifact(BaseModel):
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


def _corrupt(key: str, detail: str) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code="DIAGNOSTIC_TERMINAL_ARTIFACT_CORRUPT",
            message=f"the existing diagnostic artifact at {key} {detail}",
        )
    )


def _mismatch(code: str, key: str, detail: str) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            message=f"the existing diagnostic artifact at {key} {detail}",
        )
    )


def find_existing_artifact(
    store: ObjectStore, *, invocation: WorkerInvocation
) -> DiagnosticTerminalArtifact | None:
    """The verified artifact for this run, if one exists. Checked first."""
    if not store.exists(diagnostic_artifact_key(invocation.run_id)):
        return None
    return verify_existing_artifact(store, invocation=invocation)


def verify_existing_artifact(
    store: ObjectStore, *, invocation: WorkerInvocation
) -> DiagnosticTerminalArtifact:
    """Read the stored artifact and check it end to end before returning it."""
    key = diagnostic_artifact_key(invocation.run_id)
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

    schema_version = existing.get("schema_version")
    outcome = existing.get("outcome")
    if not isinstance(outcome, str) or outcome not in _SCHEMA_FOR_OUTCOME:
        raise _corrupt(key, f"records an unrecognised outcome {outcome!r}")
    if _SCHEMA_FOR_OUTCOME[outcome] != schema_version:
        raise _corrupt(
            key, f"pairs outcome {outcome!r} with schema {schema_version!r}, which disagree"
        )

    if existing.get("evidence_class") != EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY.value:
        raise _mismatch(
            "DIAGNOSTIC_TERMINAL_ARTIFACT_EVIDENCE_CLASS_MISMATCH",
            key,
            "is not development diagnostic evidence",
        )
    if stored.metadata.evidence_class is not EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY:
        raise _mismatch(
            "DIAGNOSTIC_TERMINAL_ARTIFACT_EVIDENCE_CLASS_MISMATCH",
            key,
            "carries object metadata for a different evidence class",
        )
    if existing.get("run_id") != invocation.run_id or stored.metadata.run_id != invocation.run_id:
        raise _mismatch(
            "DIAGNOSTIC_TERMINAL_ARTIFACT_RUN_ID_MISMATCH", key, "belongs to a different run"
        )
    if existing.get("experiment_sha256") != EXPERIMENT_SHA256:
        raise _mismatch(
            "DIAGNOSTIC_TERMINAL_ARTIFACT_EXPERIMENT_MISMATCH",
            key,
            "was produced under a different experiment",
        )
    if existing.get("specification_v2_sha256") != V2_SPECIFICATION_SHA256:
        raise _mismatch(
            "DIAGNOSTIC_TERMINAL_ARTIFACT_SPECIFICATION_MISMATCH",
            key,
            "was produced under a different diagnostic specification",
        )
    if existing.get("source_commit") != invocation.source_commit:
        raise _mismatch(
            "DIAGNOSTIC_TERMINAL_ARTIFACT_COMMIT_MISMATCH",
            key,
            "records a different requested source_commit",
        )
    if existing.get("deployed_commit") != invocation.deployed_commit:
        raise _mismatch(
            "DIAGNOSTIC_TERMINAL_ARTIFACT_COMMIT_MISMATCH",
            key,
            "was produced by an image built from a different commit",
        )
    if existing.get("claim_boundary") != CLAIM_BOUNDARY:
        raise _mismatch(
            "DIAGNOSTIC_TERMINAL_ARTIFACT_CLAIM_BOUNDARY_MISMATCH",
            key,
            "does not carry the diagnostic claim boundary",
        )

    conclusion = existing.get("conclusion")
    return DiagnosticTerminalArtifact(
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
    invocation: WorkerInvocation,
    payload: dict[str, Any],
    schema_version: str,
    outcome: str,
    conclusion: str | None,
) -> DiagnosticTerminalArtifact:
    key = diagnostic_artifact_key(invocation.run_id)
    try:
        metadata = put_json(
            store,
            key,
            payload,
            schema_version=schema_version,
            run_id=invocation.run_id,
            experiment_hash=EXPERIMENT_SHA256,
            evidence_class=EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY,
            immutable=True,
        )
    except BridgeTransformError as error:
        if error.failures[0].code != "OBJECT_ALREADY_EXISTS":
            raise
        return verify_existing_artifact(store, invocation=invocation)
    return DiagnosticTerminalArtifact(
        key=key,
        content_sha256=metadata.content_sha256,
        outcome=outcome,
        conclusion=conclusion,
        already_existed=False,
        payload=payload,
    )


def publish_diagnostic_artifact(
    store: ObjectStore,
    *,
    invocation: WorkerInvocation,
    execute: Any,
    now: datetime | None = None,
) -> DiagnosticTerminalArtifact:
    """Return the existing artifact, or run the diagnostic and write one.

    The existence check comes first, so a duplicate invocation issues no
    provider request and loads no model.
    """
    already = find_existing_artifact(store, invocation=invocation)
    if already is not None:
        return already

    stamped = now or datetime.now(UTC)
    try:
        result: DiagnosticArtifact = execute()
    except BridgeTransformError as error:
        first = error.failures[0]
        failure = DiagnosticFailure(
            outcome=DIAGNOSTIC_FAILURE_CODE,
            conclusion=None,
            run_id=invocation.run_id,
            source_commit=invocation.source_commit,
            deployed_commit=invocation.deployed_commit,
            experiment_sha256=EXPERIMENT_SHA256,
            specification_v2_sha256=V2_SPECIFICATION_SHA256,
            failure_stage="frozen_inference_diagnostic",
            failure_code=first.code,
            exception_class=type(error).__name__,
            message=first.message[:2000],
            completed_at=stamped,
        )
    except Exception as error:  # noqa: BLE001 - nothing may escape unrecorded
        failure = DiagnosticFailure(
            outcome=DIAGNOSTIC_OPERATIONAL_FAILURE_CODE,
            conclusion=DiagnosticConclusion.DIAGNOSTIC_OPERATIONAL_FAILURE,
            run_id=invocation.run_id,
            source_commit=invocation.source_commit,
            deployed_commit=invocation.deployed_commit,
            experiment_sha256=EXPERIMENT_SHA256,
            specification_v2_sha256=V2_SPECIFICATION_SHA256,
            failure_stage="frozen_inference_diagnostic",
            failure_code="DIAGNOSTIC_OPERATIONAL_FAILURE",
            # The class, and nothing else. No str(error), no traceback, no
            # filename, no line number, no path, no provider response.
            exception_class=type(error).__name__,
            message=OPERATIONAL_FAILURE_MESSAGE,
            completed_at=stamped,
        )
    else:
        payload = result.model_dump(mode="json")
        payload["outcome"] = DIAGNOSTIC_SUCCESS_CODE
        payload["experiment_sha256"] = EXPERIMENT_SHA256
        return _write(
            store,
            invocation=invocation,
            payload=payload,
            schema_version=result.schema_version,
            outcome=DIAGNOSTIC_SUCCESS_CODE,
            conclusion=result.conclusion.value,
        )

    return _write(
        store,
        invocation=invocation,
        payload=failure.model_dump(mode="json"),
        schema_version=failure.schema_version,
        outcome=failure.outcome,
        conclusion=failure.conclusion.value if failure.conclusion else None,
    )
