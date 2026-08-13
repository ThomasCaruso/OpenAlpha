"""Two immutable artifacts per probe run: the fit, then the terminal result.

The fit artifact is published BEFORE the test partition is opened and carries
the selected hyperparameters, the preprocessing state and the coefficients. The
test phase then re-reads it, verifies its digest, and refuses to run against
anything else. That is what turns "we did not tune on test" from a promise into
a record: the numbers used at test time were public before test data was seen.

Failure artifacts record the exception class and a fixed message. Exception text
can carry a provider URL, a credential or a slice of a response.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict

from ..cloud.objectstore import ObjectStore, get_json, put_json
from ..diagnostic.safe_logging import StageTracker
from ..errors import BridgeFailure, BridgeTransformError, FailureCategory
from ..phase2.identity import canonical_json
from ..phase2.states import EvidenceClass
from .invocation import ProbeInvocation
from .spec import (
    PROBE_CLAIM_BOUNDARY,
    PROBE_EXPERIMENT_ID,
    PROBE_FAILURE_SCHEMA_VERSION,
    PROBE_FIT_SCHEMA_VERSION,
    PROBE_SPECIFICATION_NAME,
    PROBE_SPECIFICATION_SHA256,
    PROBE_SUCCESS_SCHEMA_VERSION,
    probe_fit_key,
    probe_test_key,
)

__all__ = [
    "PROBE_FAILURE_CODE",
    "PROBE_FIT_CODE",
    "PROBE_OPERATIONAL_FAILURE_CODE",
    "PROBE_OPERATIONAL_FAILURE_MESSAGE",
    "PROBE_SUCCESS_CODE",
    "ProbeFailure",
    "ProbeTerminalArtifact",
    "find_existing_fit_artifact",
    "load_verified_fit_artifact",
    "publish_probe_fit",
    "publish_probe_test",
]

PROBE_FIT_CODE: Final[str] = "KRONOS_REPRESENTATION_PROBE_FIT_COMPLETED"
PROBE_SUCCESS_CODE: Final[str] = "KRONOS_REPRESENTATION_PROBE_COMPLETED"
PROBE_FAILURE_CODE: Final[str] = "KRONOS_REPRESENTATION_PROBE_FAILED"
PROBE_OPERATIONAL_FAILURE_CODE: Final[str] = "KRONOS_REPRESENTATION_PROBE_OPERATIONAL_FAILURE"

PROBE_OPERATIONAL_FAILURE_MESSAGE: Final[str] = (
    "the representation probe did not reach a conclusion; see the worker logs "
    "for detail, which are not reproduced here because exception text can carry "
    "credentials, provider responses, or local paths"
)

_SCHEMA_FOR_OUTCOME: Final[dict[str, str]] = {
    PROBE_FIT_CODE: PROBE_FIT_SCHEMA_VERSION,
    PROBE_SUCCESS_CODE: PROBE_SUCCESS_SCHEMA_VERSION,
    PROBE_FAILURE_CODE: PROBE_FAILURE_SCHEMA_VERSION,
    PROBE_OPERATIONAL_FAILURE_CODE: PROBE_FAILURE_SCHEMA_VERSION,
}


def _fail(code: str, message: str) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(category=FailureCategory.INVALID_CONFIGURATION, code=code, message=message)
    )


class ProbeFailure(BaseModel):
    """A probe phase that did not reach a conclusion."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    schema_version: Literal["openalpha.bridge.representation_probe.kronos_frozen_failure.v1"] = (
        PROBE_FAILURE_SCHEMA_VERSION
    )
    claim_boundary: Literal["DEVELOPMENT PROBE - NOT TRADING EVIDENCE"] = PROBE_CLAIM_BOUNDARY
    outcome: Literal[
        "KRONOS_REPRESENTATION_PROBE_FAILED",
        "KRONOS_REPRESENTATION_PROBE_OPERATIONAL_FAILURE",
    ]
    phase: Literal["fit", "test"]
    evidence_class: EvidenceClass = EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY
    experiment_id: Literal["openalpha-kronos-frozen-representation-probe-v1"] = (
        PROBE_EXPERIMENT_ID
    )
    study_type: Literal["frozen_representation_probe"] = "frozen_representation_probe"
    run_id: str
    source_commit: str
    deployed_commit: str
    specification_name: str = PROBE_SPECIFICATION_NAME
    specification_sha256: str = PROBE_SPECIFICATION_SHA256
    failure_stage: str
    failure_code: str
    exception_class: str | None = None
    message: str
    completed_at: datetime
    training_performed: Literal[False] = False
    kronos_parameters_updated: Literal[False] = False
    test_partition_opened: bool = False
    authorizes_training: Literal[False] = False
    authorizes_fine_tuning: Literal[False] = False
    authorizes_supervised_adaptation: Literal[False] = False
    authorizes_production_inference: Literal[False] = False
    authorizes_trading_claims: Literal[False] = False
    scientific_result_available: Literal[False] = False


class ProbeTerminalArtifact(BaseModel):
    """What was written, and whether this invocation wrote it."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    key: str
    content_sha256: str
    outcome: str
    phase: Literal["fit", "test"]
    decision_outcome: str | None = None
    evidence_class: Literal[EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY] = (
        EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY
    )
    already_existed: bool
    payload: dict[str, Any]


def _write(
    store: ObjectStore,
    *,
    key: str,
    invocation: ProbeInvocation,
    payload: dict[str, Any],
    schema_version: str,
    outcome: str,
    phase: Literal["fit", "test"],
    decision_outcome: str | None = None,
    stage: StageTracker | None = None,
) -> ProbeTerminalArtifact:
    tracker = stage or StageTracker()
    tracker.enter("write_artifact")
    try:
        metadata = put_json(
            store,
            key,
            payload,
            schema_version=schema_version,
            run_id=invocation.run_id,
            experiment_hash=PROBE_SPECIFICATION_SHA256,
            evidence_class=EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY,
            immutable=True,
        )
    except BridgeTransformError as error:
        if error.failures[0].code != "OBJECT_ALREADY_EXISTS":
            raise
        tracker.enter("verify_existing_artifact")
        return _verify_existing(store, key=key, invocation=invocation, phase=phase)
    return ProbeTerminalArtifact(
        key=key,
        content_sha256=metadata.content_sha256,
        outcome=outcome,
        phase=phase,
        decision_outcome=decision_outcome,
        already_existed=False,
        payload=payload,
    )


def _verify_existing(
    store: ObjectStore,
    *,
    key: str,
    invocation: ProbeInvocation,
    phase: Literal["fit", "test"],
) -> ProbeTerminalArtifact:
    """Read a stored artifact and check it end to end before returning it."""
    stored = store.get(key)
    observed = hashlib.sha256(stored.body).hexdigest()
    if observed != stored.metadata.content_sha256:
        raise _fail(
            "PROBE_ARTIFACT_CORRUPT",
            f"{key} hashes to {observed} but its metadata records "
            f"{stored.metadata.content_sha256}",
        )
    existing = get_json(store, key)
    if not isinstance(existing, dict):
        raise _fail("PROBE_ARTIFACT_CORRUPT", f"{key} is not a JSON object")

    outcome = existing.get("outcome")
    if not isinstance(outcome, str) or outcome not in _SCHEMA_FOR_OUTCOME:
        raise _fail("PROBE_ARTIFACT_CORRUPT", f"{key} records unrecognised outcome {outcome!r}")
    if _SCHEMA_FOR_OUTCOME[outcome] != existing.get("schema_version"):
        raise _fail(
            "PROBE_ARTIFACT_CORRUPT",
            f"{key} pairs outcome {outcome!r} with a disagreeing schema",
        )
    for field, expected in (
        ("experiment_id", PROBE_EXPERIMENT_ID),
        ("specification_name", PROBE_SPECIFICATION_NAME),
        ("specification_sha256", PROBE_SPECIFICATION_SHA256),
        ("run_id", invocation.run_id),
        ("source_commit", invocation.source_commit),
        ("deployed_commit", invocation.deployed_commit),
        ("claim_boundary", PROBE_CLAIM_BOUNDARY),
    ):
        if existing.get(field) != expected:
            raise _fail(
                "PROBE_ARTIFACT_MISMATCH",
                f"{key} records {field}={existing.get(field)!r}, expected {expected!r}",
            )
    if stored.metadata.run_id != invocation.run_id:
        raise _fail("PROBE_ARTIFACT_MISMATCH", f"{key} metadata belongs to a different run")

    decision = existing.get("decision")
    decision_outcome = (
        decision.get("outcome") if isinstance(decision, dict) else None
    )
    return ProbeTerminalArtifact(
        key=key,
        content_sha256=observed,
        outcome=outcome,
        phase=phase,
        decision_outcome=decision_outcome if isinstance(decision_outcome, str) else None,
        already_existed=True,
        payload=existing,
    )


def find_existing_fit_artifact(
    store: ObjectStore, *, invocation: ProbeInvocation
) -> ProbeTerminalArtifact | None:
    """The verified fit artifact for this run, if one exists."""
    key = probe_fit_key(invocation.run_id)
    if not store.exists(key):
        return None
    return _verify_existing(store, key=key, invocation=invocation, phase="fit")


def load_verified_fit_artifact(
    store: ObjectStore, *, invocation: ProbeInvocation, expected_digest: str | None = None
) -> ProbeTerminalArtifact:
    """Load the fit artifact the test phase must run against, or fail closed.

    The test phase may not proceed on a fit that does not exist, does not verify,
    belongs to another run, or -- when a digest is supplied -- is not byte-for-byte
    the fit the caller intended.
    """
    key = probe_fit_key(invocation.run_id)
    if not store.exists(key):
        raise _fail(
            "PROBE_FIT_ARTIFACT_MISSING",
            (
                f"no fit artifact at {key}; the test phase may not run before the fit "
                "phase has published its selected hyperparameters"
            ),
        )
    artifact = _verify_existing(store, key=key, invocation=invocation, phase="fit")
    if artifact.outcome != PROBE_FIT_CODE:
        raise _fail(
            "PROBE_FIT_ARTIFACT_NOT_SUCCESSFUL",
            f"the fit artifact records outcome {artifact.outcome!r}, not {PROBE_FIT_CODE}",
        )
    if expected_digest is not None and artifact.content_sha256 != expected_digest:
        raise _fail(
            "PROBE_FIT_ARTIFACT_DIGEST_MISMATCH",
            (
                f"the stored fit artifact hashes to {artifact.content_sha256}, but the "
                f"test phase was asked to run against {expected_digest}"
            ),
        )
    if artifact.payload.get("test_partition_opened") is not False:
        raise _fail(
            "PROBE_FIT_ARTIFACT_TOUCHED_TEST",
            "the fit artifact does not record test_partition_opened = false",
        )
    return artifact


def publish_probe_fit(
    store: ObjectStore,
    *,
    invocation: ProbeInvocation,
    execute: Any,
    now: datetime | None = None,
    stage: StageTracker | None = None,
) -> ProbeTerminalArtifact:
    """Return the existing fit, or run the fit phase and write one."""
    tracker = stage or StageTracker()
    key = probe_fit_key(invocation.run_id)

    tracker.enter("verify_existing_artifact")
    already = find_existing_fit_artifact(store, invocation=invocation)
    if already is not None:
        return already

    stamped = now or datetime.now(UTC)
    try:
        payload: dict[str, Any] = execute()
    except BridgeTransformError as error:
        failure = ProbeFailure(
            outcome=PROBE_FAILURE_CODE,
            phase="fit",
            run_id=invocation.run_id,
            source_commit=invocation.source_commit,
            deployed_commit=invocation.deployed_commit,
            failure_stage="representation_probe_fit",
            failure_code=error.failures[0].code,
            exception_class=type(error).__name__,
            message=error.failures[0].message[:2000],
            completed_at=stamped,
        )
    except Exception as error:  # noqa: BLE001 - nothing may escape unrecorded
        failure = ProbeFailure(
            outcome=PROBE_OPERATIONAL_FAILURE_CODE,
            phase="fit",
            run_id=invocation.run_id,
            source_commit=invocation.source_commit,
            deployed_commit=invocation.deployed_commit,
            failure_stage="representation_probe_fit",
            failure_code="PROBE_OPERATIONAL_FAILURE",
            exception_class=type(error).__name__,
            message=PROBE_OPERATIONAL_FAILURE_MESSAGE,
            completed_at=stamped,
        )
    else:
        payload["outcome"] = PROBE_FIT_CODE
        return _write(
            store,
            key=key,
            invocation=invocation,
            payload=payload,
            schema_version=PROBE_FIT_SCHEMA_VERSION,
            outcome=PROBE_FIT_CODE,
            phase="fit",
            stage=tracker,
        )

    return _write(
        store,
        key=key,
        invocation=invocation,
        payload=failure.model_dump(mode="json"),
        schema_version=failure.schema_version,
        outcome=failure.outcome,
        phase="fit",
        stage=tracker,
    )


def publish_probe_test(
    store: ObjectStore,
    *,
    invocation: ProbeInvocation,
    execute: Any,
    now: datetime | None = None,
    stage: StageTracker | None = None,
) -> ProbeTerminalArtifact:
    """Return the existing terminal artifact, or run the test phase and write one."""
    tracker = stage or StageTracker()
    key = probe_test_key(invocation.run_id)

    tracker.enter("verify_existing_artifact")
    if store.exists(key):
        return _verify_existing(store, key=key, invocation=invocation, phase="test")

    stamped = now or datetime.now(UTC)
    try:
        payload: dict[str, Any] = execute()
    except BridgeTransformError as error:
        failure = ProbeFailure(
            outcome=PROBE_FAILURE_CODE,
            phase="test",
            run_id=invocation.run_id,
            source_commit=invocation.source_commit,
            deployed_commit=invocation.deployed_commit,
            failure_stage="representation_probe_test",
            failure_code=error.failures[0].code,
            exception_class=type(error).__name__,
            message=error.failures[0].message[:2000],
            completed_at=stamped,
        )
    except Exception as error:  # noqa: BLE001 - nothing may escape unrecorded
        failure = ProbeFailure(
            outcome=PROBE_OPERATIONAL_FAILURE_CODE,
            phase="test",
            run_id=invocation.run_id,
            source_commit=invocation.source_commit,
            deployed_commit=invocation.deployed_commit,
            failure_stage="representation_probe_test",
            failure_code="PROBE_OPERATIONAL_FAILURE",
            exception_class=type(error).__name__,
            message=PROBE_OPERATIONAL_FAILURE_MESSAGE,
            completed_at=stamped,
        )
    else:
        payload["outcome"] = PROBE_SUCCESS_CODE
        decision = payload.get("decision")
        return _write(
            store,
            key=key,
            invocation=invocation,
            payload=payload,
            schema_version=PROBE_SUCCESS_SCHEMA_VERSION,
            outcome=PROBE_SUCCESS_CODE,
            phase="test",
            decision_outcome=(
                decision.get("outcome") if isinstance(decision, dict) else None
            ),
            stage=tracker,
        )

    return _write(
        store,
        key=key,
        invocation=invocation,
        payload=failure.model_dump(mode="json"),
        schema_version=failure.schema_version,
        outcome=failure.outcome,
        phase="test",
        stage=tracker,
    )


def payload_digest(payload: dict[str, Any]) -> str:
    """The digest the store will record for this payload."""
    return hashlib.sha256(canonical_json(payload)).hexdigest()


__all__ += ["payload_digest"]
