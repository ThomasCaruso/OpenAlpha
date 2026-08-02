"""Terminal artifact for one canary invocation.

Every invocation ends in exactly one immutable object under the compatibility
evidence root, whether it succeeded, failed a contract, or died operationally.
A canary that runs and leaves nothing behind is a canary that can be re-run
until it happens to pass, so the negative result is preserved with the same
force as the positive one.

Nothing here can be talked into writing somewhere else. The evidence class and
the outcome vocabulary are ``Literal`` types, the storage root comes from
``run_prefix`` rather than a formatted string, and the write is conditional.
"""

from __future__ import annotations

import traceback
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from ..cloud.objectstore import ObjectStore, get_json, put_json, run_prefix
from ..cloud.redaction import redact
from ..errors import BridgeFailure, BridgeTransformError, FailureCategory
from .canary import (
    AMENDMENTS,
    CANARY_EVIDENCE_CLASS,
    CANARY_SUCCESS_CODE,
    CanaryFailure,
    CanaryReport,
)
from .identity import EXPERIMENT_SHA256
from .states import EvidenceClass

__all__ = [
    "CANARY_FAILURE_CODE",
    "CANARY_OPERATIONAL_FAILURE_CODE",
    "TERMINAL_ARTIFACT_NAME",
    "TerminalArtifact",
    "publish_terminal_artifact",
    "terminal_artifact_key",
]

#: The two non-success terminal outcomes. A contract violation is a result; an
#: operational death is not, and conflating them would let infrastructure noise
#: read as evidence about the official path.
CANARY_FAILURE_CODE: Literal["STAGE_A_OFFICIAL_CANARY_FAILED"] = "STAGE_A_OFFICIAL_CANARY_FAILED"
CANARY_OPERATIONAL_FAILURE_CODE: Literal["STAGE_A_OFFICIAL_CANARY_OPERATIONAL_FAILURE"] = (
    "STAGE_A_OFFICIAL_CANARY_OPERATIONAL_FAILURE"
)

#: One object per invocation, named identically for every outcome so a missing
#: artifact is unambiguous rather than merely a different filename.
TERMINAL_ARTIFACT_NAME = "stage_a_canary_terminal.json"


def terminal_artifact_key(run_id: str) -> str:
    """The only key this module ever writes.

    Derived from ``run_prefix`` with the compatibility evidence class, so the
    object cannot land under the real or synthetic root even by mistake.
    """
    return f"{run_prefix(run_id, EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY)}/{TERMINAL_ARTIFACT_NAME}"


class TerminalArtifact(BaseModel):
    """What was written, and whether this invocation is the one that wrote it."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    key: str
    content_sha256: str
    outcome: str
    #: Fixed by type. A caller cannot reclassify canary evidence as anything else.
    evidence_class: Literal[EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY] = (
        EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY
    )
    authorizes_stage_b: Literal[False] = False
    authorizes_real_run: Literal[False] = False
    #: True when the artifact already existed and this invocation verified it
    #: rather than writing it.
    already_existed: bool
    payload: dict[str, Any]


def _failure(
    *,
    run_id: str,
    source_commit: str,
    deployed_commit: str,
    outcome: Literal[
        "STAGE_A_OFFICIAL_CANARY_FAILED", "STAGE_A_OFFICIAL_CANARY_OPERATIONAL_FAILURE"
    ],
    failure_stage: str,
    failure_code: str,
    message: str,
    now: datetime,
) -> CanaryFailure:
    return CanaryFailure(
        outcome=outcome,
        run_id=run_id,
        source_commit=source_commit,
        deployed_commit=deployed_commit,
        experiment_sha256=EXPERIMENT_SHA256,
        amendment_sha256=AMENDMENTS,
        failure_stage=failure_stage,
        failure_code=failure_code,
        # Secrets are scrubbed before anything reaches durable storage.
        message=str(redact(message))[:2000],
        completed_at=now,
    )


def _typed_failure(
    error: BridgeTransformError,
    *,
    run_id: str,
    source_commit: str,
    deployed_commit: str,
    now: datetime,
) -> CanaryFailure:
    first: BridgeFailure = error.failures[0]
    return _failure(
        run_id=run_id,
        source_commit=source_commit,
        deployed_commit=deployed_commit,
        outcome=CANARY_FAILURE_CODE,
        failure_stage="stage_a_official_canary",
        failure_code=first.code,
        message=first.message,
        now=now,
    )


def _operational_failure(
    error: BaseException,
    *,
    run_id: str,
    source_commit: str,
    deployed_commit: str,
    now: datetime,
) -> CanaryFailure:
    # The type name and last frame are enough to act on, and keep an unbounded
    # traceback out of the artifact.
    frames = traceback.extract_tb(error.__traceback__)
    where = f"{frames[-1].filename}:{frames[-1].lineno}" if frames else "unknown"
    return _failure(
        run_id=run_id,
        source_commit=source_commit,
        deployed_commit=deployed_commit,
        outcome=CANARY_OPERATIONAL_FAILURE_CODE,
        failure_stage="stage_a_official_canary",
        failure_code="CANARY_OPERATIONAL_FAILURE",
        message=f"{type(error).__name__} at {where}: {error}",
        now=now,
    )


def _publish(
    store: ObjectStore,
    *,
    run_id: str,
    payload: dict[str, Any],
    schema_version: str,
    outcome: str,
) -> TerminalArtifact:
    """Write once. If it already exists, verify rather than replace."""
    key = terminal_artifact_key(run_id)
    try:
        metadata = put_json(
            store,
            key,
            payload,
            schema_version=schema_version,
            run_id=run_id,
            experiment_hash=EXPERIMENT_SHA256,
            evidence_class=EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY,
            immutable=True,
        )
    except BridgeTransformError as error:
        if error.failures[0].code != "OBJECT_ALREADY_EXISTS":
            raise
        return _verify_existing(store, key=key, run_id=run_id)
    return TerminalArtifact(
        key=key,
        content_sha256=metadata.content_sha256,
        outcome=outcome,
        already_existed=False,
        payload=payload,
    )


def _verify_existing(store: ObjectStore, *, key: str, run_id: str) -> TerminalArtifact:
    """A duplicate invocation reads what is there and confirms it.

    It does not overwrite, and it does not report its own result. The first
    terminal artifact for a run id is the terminal artifact for that run id.
    """
    stored = store.get(key)
    existing = get_json(store, key)
    if not isinstance(existing, dict):
        raise BridgeTransformError(
            BridgeFailure(
                category=FailureCategory.INVALID_CONFIGURATION,
                code="CANARY_TERMINAL_ARTIFACT_CORRUPT",
                message=f"the existing terminal artifact at {key} is not a JSON object",
            )
        )
    outcome = existing.get("outcome")
    if not isinstance(outcome, str):
        raise BridgeTransformError(
            BridgeFailure(
                category=FailureCategory.INVALID_CONFIGURATION,
                code="CANARY_TERMINAL_ARTIFACT_CORRUPT",
                message=f"the existing terminal artifact at {key} records no outcome",
            )
        )
    if existing.get("evidence_class") != EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY.value:
        raise BridgeTransformError(
            BridgeFailure(
                category=FailureCategory.INVALID_CONFIGURATION,
                code="CANARY_TERMINAL_ARTIFACT_EVIDENCE_CLASS_MISMATCH",
                message=(
                    f"the existing artifact at {key} is not compatibility-canary evidence; "
                    "refusing to treat it as this run's terminal result"
                ),
            )
        )
    if existing.get("run_id", run_id) != run_id:
        raise BridgeTransformError(
            BridgeFailure(
                category=FailureCategory.INVALID_CONFIGURATION,
                code="CANARY_TERMINAL_ARTIFACT_RUN_ID_MISMATCH",
                message=f"the existing artifact at {key} belongs to a different run",
            )
        )
    return TerminalArtifact(
        key=key,
        content_sha256=stored.metadata.content_sha256,
        outcome=outcome,
        already_existed=True,
        payload=existing,
    )


def publish_terminal_artifact(
    store: ObjectStore,
    *,
    run_id: str,
    source_commit: str,
    deployed_commit: str,
    execute: Any,
    now: datetime | None = None,
) -> TerminalArtifact:
    """Run the canary and write exactly one immutable terminal artifact.

    ``execute`` is a zero-argument callable returning a ``CanaryReport``. Three
    terminal shapes, one of which always happens:

    * success, the verified report;
    * typed failure, a contract the official path did not satisfy;
    * operational failure, something that broke before a contract was tested.

    Only the first is evidence that the path works. None of the three authorize
    Stage B, a real run, or opening any held-out partition.
    """
    stamped = now or datetime.now(UTC)
    try:
        report: CanaryReport = execute()
    except BridgeTransformError as error:
        failure = _typed_failure(
            error,
            run_id=run_id,
            source_commit=source_commit,
            deployed_commit=deployed_commit,
            now=stamped,
        )
    except Exception as error:  # noqa: BLE001 - deliberate: nothing may escape unrecorded
        failure = _operational_failure(
            error,
            run_id=run_id,
            source_commit=source_commit,
            deployed_commit=deployed_commit,
            now=stamped,
        )
    else:
        if report.outcome != CANARY_SUCCESS_CODE:
            raise BridgeTransformError(
                BridgeFailure(
                    category=FailureCategory.INVALID_CONFIGURATION,
                    code="CANARY_UNEXPECTED_SUCCESS_OUTCOME",
                    message=(
                        f"a returned report must carry {CANARY_SUCCESS_CODE}, got {report.outcome}"
                    ),
                )
            )
        if report.evidence_class is not CANARY_EVIDENCE_CLASS:
            raise BridgeTransformError(
                BridgeFailure(
                    category=FailureCategory.INVALID_CONFIGURATION,
                    code="CANARY_UNEXPECTED_EVIDENCE_CLASS",
                    message="a returned report must carry the compatibility evidence class",
                )
            )
        return _publish(
            store,
            run_id=run_id,
            payload=report.model_dump(mode="json"),
            schema_version=report.schema_version,
            outcome=report.outcome,
        )

    return _publish(
        store,
        run_id=run_id,
        payload=failure.model_dump(mode="json"),
        schema_version=failure.schema_version,
        outcome=failure.outcome,
    )
