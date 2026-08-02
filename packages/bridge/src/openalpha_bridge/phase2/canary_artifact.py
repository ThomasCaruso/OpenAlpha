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

import hashlib
from datetime import UTC, datetime
from typing import Any, Final, Literal

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
from .invocation import WorkerInvocation
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
    invocation: WorkerInvocation,
    outcome: Literal[
        "STAGE_A_OFFICIAL_CANARY_FAILED", "STAGE_A_OFFICIAL_CANARY_OPERATIONAL_FAILURE"
    ],
    failure_stage: str,
    failure_code: str,
    message: str | None = None,
    exception_class: str | None = None,
    now: datetime,
) -> CanaryFailure:
    return CanaryFailure(
        outcome=outcome,
        run_id=invocation.run_id,
        source_commit=invocation.source_commit,
        deployed_commit=invocation.deployed_commit,
        experiment_sha256=EXPERIMENT_SHA256,
        amendment_sha256=AMENDMENTS,
        failure_stage=failure_stage,
        failure_code=failure_code,
        exception_class=exception_class,
        # Typed failures carry our own message, which is written here and
        # contains no external text. It is still scrubbed, because a code path
        # can interpolate a value that came from outside.
        message=str(redact(message))[:2000] if message is not None else OPERATIONAL_FAILURE_MESSAGE,
        completed_at=now,
    )


def _typed_failure(
    error: BridgeTransformError,
    *,
    invocation: WorkerInvocation,
    now: datetime,
) -> CanaryFailure:
    first: BridgeFailure = error.failures[0]
    return _failure(
        invocation=invocation,
        outcome=CANARY_FAILURE_CODE,
        failure_stage="stage_a_official_canary",
        failure_code=first.code,
        message=first.message,
        exception_class=type(error).__name__,
        now=now,
    )


#: Every operational failure records this and nothing else. An exception
#: message can carry a provider URL, a signed request, a local path, or a slice
#: of the response body, and an artifact is durable and readable by anyone who
#: can read the bucket. The exception class is what an operator acts on; the
#: message adds nothing that is safe to keep.
OPERATIONAL_FAILURE_MESSAGE: Final[str] = (
    "the canary did not reach a contract check; see the worker logs for detail, "
    "which are not reproduced here because exception text can carry credentials, "
    "provider responses, or local paths"
)


def _operational_failure(
    error: BaseException,
    *,
    invocation: WorkerInvocation,
    now: datetime,
) -> CanaryFailure:
    """Record the exception class and a fixed message. Never str(error).

    No traceback, no filename, no line number, no local path, no provider URL
    or response, and no raw data reaches the artifact.
    """
    return _failure(
        invocation=invocation,
        outcome=CANARY_OPERATIONAL_FAILURE_CODE,
        failure_stage="stage_a_official_canary",
        failure_code="CANARY_OPERATIONAL_FAILURE",
        exception_class=type(error).__name__,
        now=now,
    )


#: Outcome vocabulary this worker can produce, and the schema each belongs to.
#: A stored artifact whose pair disagrees is corrupt, not merely unexpected.
_SCHEMA_FOR_OUTCOME: Final[dict[str, str]] = {
    CANARY_SUCCESS_CODE: "openalpha.bridge.phase2.stage_a_canary.v1",
    CANARY_FAILURE_CODE: "openalpha.bridge.phase2.stage_a_canary_failure.v1",
    CANARY_OPERATIONAL_FAILURE_CODE: "openalpha.bridge.phase2.stage_a_canary_failure.v1",
}


def _publish(
    store: ObjectStore,
    *,
    invocation: WorkerInvocation,
    payload: dict[str, Any],
    schema_version: str,
    outcome: str,
) -> TerminalArtifact:
    """Write once. If the write loses a race, verify rather than replace."""
    key = terminal_artifact_key(invocation.run_id)
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
        # Another worker wrote between our existence check and our write. Its
        # artifact is the terminal one.
        return verify_existing_artifact(store, invocation=invocation)
    return TerminalArtifact(
        key=key,
        content_sha256=metadata.content_sha256,
        outcome=outcome,
        already_existed=False,
        payload=payload,
    )


def _corrupt(key: str, detail: str) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code="CANARY_TERMINAL_ARTIFACT_CORRUPT",
            message=f"the existing terminal artifact at {key} {detail}",
        )
    )


def _mismatch(code: str, key: str, detail: str) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            message=f"the existing artifact at {key} {detail}",
        )
    )


def find_existing_artifact(
    store: ObjectStore, *, invocation: WorkerInvocation
) -> TerminalArtifact | None:
    """The verified terminal artifact for this run, if one already exists.

    Called before anything executes. A run id that already has a terminal
    artifact must not retrieve market data, resolve assets, or load a model
    again: the first artifact is the result, so repeating the work can only
    cost money and risk producing something that disagrees with it.
    """
    if not store.exists(terminal_artifact_key(invocation.run_id)):
        return None
    return verify_existing_artifact(store, invocation=invocation)


def verify_existing_artifact(
    store: ObjectStore, *, invocation: WorkerInvocation
) -> TerminalArtifact:
    """Read the stored artifact and check it end to end before returning it.

    Content hash, run id, experiment identity, evidence class, requested
    commit, deployed commit, outcome and schema are all verified. Returning a
    stored object unchecked would let a corrupted or foreign artifact stand in
    for this run's result.
    """
    key = terminal_artifact_key(invocation.run_id)
    stored = store.get(key)

    # Content hash, recomputed from the bytes rather than trusted from metadata.
    observed_sha256 = hashlib.sha256(stored.body).hexdigest()
    if observed_sha256 != stored.metadata.content_sha256:
        raise _corrupt(
            key,
            (
                f"hashes to {observed_sha256} but its metadata records "
                f"{stored.metadata.content_sha256}"
            ),
        )

    existing = get_json(store, key)
    if not isinstance(existing, dict):
        raise _corrupt(key, "is not a JSON object")

    # Schema.
    schema_version = existing.get("schema_version")
    if schema_version not in _SCHEMA_FOR_OUTCOME.values():
        raise _corrupt(key, f"declares an unrecognised schema_version {schema_version!r}")

    # Outcome, and that it belongs to the vocabulary this worker can produce.
    outcome = existing.get("outcome")
    if not isinstance(outcome, str):
        raise _corrupt(key, "records no outcome")
    if outcome not in _SCHEMA_FOR_OUTCOME:
        raise _corrupt(key, f"records an unrecognised outcome {outcome!r}")
    if _SCHEMA_FOR_OUTCOME[outcome] != schema_version:
        raise _corrupt(
            key,
            f"pairs outcome {outcome!r} with schema {schema_version!r}, which do not agree",
        )

    # Evidence class, in the payload and in the object metadata.
    if existing.get("evidence_class") != EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY.value:
        raise _mismatch(
            "CANARY_TERMINAL_ARTIFACT_EVIDENCE_CLASS_MISMATCH",
            key,
            (
                "is not compatibility-canary evidence; refusing to treat it as "
                "this run's terminal result"
            ),
        )
    if stored.metadata.evidence_class is not EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY:
        raise _mismatch(
            "CANARY_TERMINAL_ARTIFACT_EVIDENCE_CLASS_MISMATCH",
            key,
            "carries object metadata for a different evidence class",
        )

    # Run identity, in the payload and in the object metadata.
    if existing.get("run_id") != invocation.run_id:
        raise _mismatch(
            "CANARY_TERMINAL_ARTIFACT_RUN_ID_MISMATCH", key, "belongs to a different run"
        )
    if stored.metadata.run_id != invocation.run_id:
        raise _mismatch(
            "CANARY_TERMINAL_ARTIFACT_RUN_ID_MISMATCH",
            key,
            "carries object metadata for a different run",
        )

    # Experiment identity.
    if existing.get("experiment_sha256") != EXPERIMENT_SHA256:
        raise _mismatch(
            "CANARY_TERMINAL_ARTIFACT_EXPERIMENT_MISMATCH",
            key,
            "was produced under a different experiment",
        )
    if tuple(existing.get("amendment_sha256") or ()) != AMENDMENTS:
        raise _mismatch(
            "CANARY_TERMINAL_ARTIFACT_EXPERIMENT_MISMATCH",
            key,
            "was produced under a different amendment chain",
        )

    # The commit the caller asked for, and the commit the image was built from.
    if existing.get("source_commit") != invocation.source_commit:
        raise _mismatch(
            "CANARY_TERMINAL_ARTIFACT_COMMIT_MISMATCH",
            key,
            "records a different requested source_commit",
        )
    if existing.get("deployed_commit") != invocation.deployed_commit:
        raise _mismatch(
            "CANARY_TERMINAL_ARTIFACT_COMMIT_MISMATCH",
            key,
            "was produced by an image built from a different commit",
        )

    return TerminalArtifact(
        key=key,
        content_sha256=observed_sha256,
        outcome=outcome,
        already_existed=True,
        payload=existing,
    )


def publish_terminal_artifact(
    store: ObjectStore,
    *,
    invocation: WorkerInvocation,
    execute: Any,
    now: datetime | None = None,
) -> TerminalArtifact:
    """Return the existing terminal artifact, or run the canary and write one.

    The existence check comes first. If this run id already has a verified
    terminal artifact, ``execute`` is never called: no provider request, no
    asset resolution, no model load.

    Otherwise ``execute`` runs and ends in exactly one of three shapes:

    * success, the verified report;
    * typed failure, a contract the official path did not satisfy;
    * operational failure, something that broke before a contract was tested.

    Only the first is evidence that the path works. None of the three authorize
    Stage B, a real run, or opening any held-out partition.
    """
    already = find_existing_artifact(store, invocation=invocation)
    if already is not None:
        return already

    stamped = now or datetime.now(UTC)
    try:
        report: CanaryReport = execute()
    except BridgeTransformError as error:
        failure = _typed_failure(error, invocation=invocation, now=stamped)
    except Exception as error:  # noqa: BLE001 - deliberate: nothing may escape unrecorded
        failure = _operational_failure(error, invocation=invocation, now=stamped)
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
            invocation=invocation,
            payload=report.model_dump(mode="json"),
            schema_version=report.schema_version,
            outcome=report.outcome,
        )

    return _publish(
        store,
        invocation=invocation,
        payload=failure.model_dump(mode="json"),
        schema_version=failure.schema_version,
        outcome=failure.outcome,
    )
