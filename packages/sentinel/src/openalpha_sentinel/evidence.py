from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta
from typing import Any, Literal, Self

from openalpha_research import (
    ArtifactKind,
    ArtifactRef,
    EnvironmentMetadata,
    GitMetadata,
    LocalArtifactStore,
    ManifestArtifact,
    ManifestProfile,
    MethodologyStatus,
    RunManifest,
    RunState,
    RunStateEvent,
    RunStateJournal,
    publish_manifest,
)
from openalpha_research.artifacts import Sha256
from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrozenModel(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
    )

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        if update:
            raise TypeError("model_copy updates bypass validation")
        return super().model_copy(update=None, deep=deep)


class EvidenceContext(FrozenModel):
    run_id: str = Field(min_length=1, max_length=128)
    attempt_id: str = Field(min_length=1, max_length=128)
    code_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    code_dirty: bool
    diff_sha256: Sha256 | None
    dependency_lock_sha256: Sha256
    environment: EnvironmentMetadata

    @model_validator(mode="after")
    def require_consistent_environment(self) -> EvidenceContext:
        if self.environment.dependency_lock_sha256 != self.dependency_lock_sha256:
            raise ValueError("environment lock hash does not match evidence context")
        if self.code_dirty != (self.diff_sha256 is not None):
            raise ValueError("dirty evidence requires exactly one diff hash")
        return self


class LedgerEvent(FrozenModel):
    sequence: int = Field(ge=1)
    event_type: Literal["FORECAST_CREATED", "DIAGNOSTICS_COMPUTED", "OUTCOME_RESOLVED"]
    payload_sha256: Sha256
    previous_record_sha256: Sha256 | None
    occurred_at: datetime
    record_sha256: Sha256


class JournalEventSnapshot(FrozenModel):
    sequence: int = Field(ge=1)
    state: str = Field(min_length=1, max_length=32)
    occurred_at: datetime
    reason: str | None


class CreationEvidence(FrozenModel):
    schema_version: Literal["sentinel-creation-evidence-v0"]
    run_id: str
    attempt_id: str
    experiment_id: str
    canonical_spec_ref: ArtifactRef
    data_snapshot_ref: ArtifactRef
    data_quality_ref: ArtifactRef
    forecast_ref: ArtifactRef
    diagnostics_ref: ArtifactRef
    ledger_events: tuple[LedgerEvent, LedgerEvent]
    journal_events: tuple[
        JournalEventSnapshot,
        JournalEventSnapshot,
        JournalEventSnapshot,
        JournalEventSnapshot,
    ]
    seal_ref: ArtifactRef


class ResolvedEvidence(FrozenModel):
    schema_version: Literal["sentinel-resolved-evidence-v0"]
    creation: CreationEvidence
    outcome_ref: ArtifactRef
    audit_ref: ArtifactRef
    outcome_event: LedgerEvent
    manifest_ref: ArtifactRef


def publish_creation(
    *,
    store: LocalArtifactStore,
    context: EvidenceContext,
    canonical_spec: dict[str, object],
    data_snapshot: dict[str, object],
    data_quality: dict[str, object],
    forecast: dict[str, object],
    diagnostics: dict[str, object],
    created_at: datetime,
) -> CreationEvidence:
    spec_ref = _put_json(store, canonical_spec)
    experiment_id = f"exp_{spec_ref.sha256}"
    data_ref = _put_json(store, data_snapshot)
    quality_ref = _put_json(store, data_quality)
    forecast_ref = _put_json(store, forecast)
    diagnostics_ref = _put_json(store, diagnostics)
    first = _ledger_event(
        sequence=1,
        event_type="FORECAST_CREATED",
        payload_sha256=forecast_ref.sha256,
        previous_record_sha256=None,
        occurred_at=created_at,
    )
    second = _ledger_event(
        sequence=2,
        event_type="DIAGNOSTICS_COMPUTED",
        payload_sha256=diagnostics_ref.sha256,
        previous_record_sha256=first.record_sha256,
        occurred_at=created_at + timedelta(microseconds=1),
    )
    journal = RunStateJournal.start(
        run_id=context.run_id,
        experiment_id=experiment_id,
        attempt_id=context.attempt_id,
        occurred_at=created_at - timedelta(microseconds=4),
    )
    for offset, state in enumerate(
        (RunState.VALIDATED, RunState.QUEUED, RunState.RUNNING),
        start=1,
    ):
        journal = journal.transition(
            state,
            occurred_at=created_at - timedelta(microseconds=4 - offset),
        )
    journal_snapshots = tuple(
        JournalEventSnapshot(
            sequence=event.sequence,
            state=event.state.value,
            occurred_at=event.occurred_at,
            reason=event.reason,
        )
        for event in journal.events
    )
    seal_without_ref = {
        "schema_version": "sentinel-creation-evidence-v0",
        "run_id": context.run_id,
        "attempt_id": context.attempt_id,
        "experiment_id": experiment_id,
        "canonical_spec_ref": spec_ref.model_dump(mode="json"),
        "data_snapshot_ref": data_ref.model_dump(mode="json"),
        "data_quality_ref": quality_ref.model_dump(mode="json"),
        "forecast_ref": forecast_ref.model_dump(mode="json"),
        "diagnostics_ref": diagnostics_ref.model_dump(mode="json"),
        "ledger_events": [event.model_dump(mode="json") for event in (first, second)],
        "journal_events": [event.model_dump(mode="json") for event in journal_snapshots],
    }
    seal_ref = _put_json(store, seal_without_ref)
    return CreationEvidence(
        schema_version="sentinel-creation-evidence-v0",
        run_id=context.run_id,
        attempt_id=context.attempt_id,
        experiment_id=experiment_id,
        canonical_spec_ref=spec_ref,
        data_snapshot_ref=data_ref,
        data_quality_ref=quality_ref,
        forecast_ref=forecast_ref,
        diagnostics_ref=diagnostics_ref,
        ledger_events=(first, second),
        journal_events=(
            journal_snapshots[0],
            journal_snapshots[1],
            journal_snapshots[2],
            journal_snapshots[3],
        ),
        seal_ref=seal_ref,
    )


def verify_creation(store: LocalArtifactStore, creation: CreationEvidence) -> bool:
    refs = (
        creation.canonical_spec_ref,
        creation.data_snapshot_ref,
        creation.data_quality_ref,
        creation.forecast_ref,
        creation.diagnostics_ref,
        creation.seal_ref,
    )
    if not all(store.verify(ref) for ref in refs):
        return False
    if creation.experiment_id != f"exp_{creation.canonical_spec_ref.sha256}":
        return False
    first, second = creation.ledger_events
    if first.payload_sha256 != creation.forecast_ref.sha256:
        return False
    if second.payload_sha256 != creation.diagnostics_ref.sha256:
        return False
    if second.previous_record_sha256 != first.record_sha256:
        return False
    if any(event.record_sha256 != _ledger_hash(event) for event in creation.ledger_events):
        return False
    expected_seal = _creation_seal_payload(creation)
    return store.read_bytes(creation.seal_ref) == _canonical_json_bytes(expected_seal)


def append_outcome_and_complete(
    *,
    store: LocalArtifactStore,
    context: EvidenceContext,
    creation: CreationEvidence,
    outcome_loader: Callable[[], dict[str, object]],
    methodology_audit: dict[str, object],
    methodology_status: MethodologyStatus,
    test_evidence: tuple[str, ...],
    completed_at: datetime,
    manifest_parameters: tuple[tuple[str, str], ...] = (
        ('phase', '2'),
        ('origin', 'SPY-2024-07-05'),
    ),
) -> ResolvedEvidence:
    if not verify_creation(store, creation):
        raise ValueError("immutable forecast creation seal failed verification")
    outcome = outcome_loader()
    outcome_body_sha = hashlib.sha256(_canonical_json_bytes(outcome)).hexdigest()
    outcome_event = _ledger_event(
        sequence=3,
        event_type="OUTCOME_RESOLVED",
        payload_sha256=outcome_body_sha,
        previous_record_sha256=creation.ledger_events[-1].record_sha256,
        occurred_at=completed_at,
    )
    outcome_record = {
        "schema_version": "1.0",
        "forecast_seal_sha256": creation.seal_ref.sha256,
        "outcome": outcome,
        "ledger_event": outcome_event.model_dump(mode="json"),
    }
    outcome_ref = _put_json(store, outcome_record)
    restored_journal = RunStateJournal(
        events=tuple(
            RunStateEvent(
                sequence=event.sequence,
                run_id=creation.run_id,
                experiment_id=creation.experiment_id,
                attempt_id=creation.attempt_id,
                state=RunState(event.state),
                occurred_at=event.occurred_at,
                reason=event.reason,
            )
            for event in creation.journal_events
        )
    )
    completed_journal = restored_journal.transition(
        RunState.COMPLETED,
        occurred_at=completed_at + timedelta(microseconds=1),
    )
    audit_payload = {
        **methodology_audit,
        "ledger_chain": [
            *[event.model_dump(mode="json") for event in creation.ledger_events],
            outcome_event.model_dump(mode="json"),
        ],
        "run_state_journal": [
            {
                "sequence": event.sequence,
                "state": event.state.value,
                "occurred_at": event.occurred_at.isoformat(),
                "reason": event.reason,
            }
            for event in completed_journal.events
        ],
    }
    audit_ref = _put_json(store, audit_payload)
    artifacts = _manifest_artifacts(
        context=context,
        creation=creation,
        outcome_ref=outcome_ref,
        audit_ref=audit_ref,
        creation_time=creation.ledger_events[0].occurred_at,
        completed_at=completed_at,
        manifest_parameters=manifest_parameters,
    )
    manifest = RunManifest(
        schema_version="1.0",
        profile=ManifestProfile.SENTINEL_ORIGIN,
        run_id=context.run_id,
        attempt_id=context.attempt_id,
        experiment_id=creation.experiment_id,
        canonical_spec_schema_version="1.0",
        state=RunState.COMPLETED,
        methodology_status=methodology_status,
        artifacts=artifacts,
        git=GitMetadata(
            commit_sha=context.code_commit,
            dirty=context.code_dirty,
            diff_sha256=context.diff_sha256,
        ),
        environment=context.environment,
        test_evidence=test_evidence,
        created_at=completed_at + timedelta(microseconds=2),
    )
    manifest_ref = publish_manifest(store, manifest)
    return ResolvedEvidence(
        schema_version="sentinel-resolved-evidence-v0",
        creation=creation,
        outcome_ref=outcome_ref,
        audit_ref=audit_ref,
        outcome_event=outcome_event,
        manifest_ref=manifest_ref,
    )


def verify_resolved_evidence(store: LocalArtifactStore, evidence: ResolvedEvidence) -> bool:
    if not verify_creation(store, evidence.creation):
        return False
    if not all(
        store.verify(ref)
        for ref in (evidence.outcome_ref, evidence.audit_ref, evidence.manifest_ref)
    ):
        return False
    event = evidence.outcome_event
    if event.sequence != 3 or event.event_type != "OUTCOME_RESOLVED":
        return False
    if event.previous_record_sha256 != evidence.creation.ledger_events[-1].record_sha256:
        return False
    if event.record_sha256 != _ledger_hash(event):
        return False
    outcome_record = json.loads(store.read_bytes(evidence.outcome_ref))
    if not isinstance(outcome_record, dict):
        return False
    if outcome_record.get('forecast_seal_sha256') != evidence.creation.seal_ref.sha256:
        return False
    if outcome_record.get('ledger_event') != event.model_dump(mode='json'):
        return False
    if not isinstance(outcome_record, dict) or not isinstance(outcome_record.get("outcome"), dict):
        return False
    observed_body_sha = hashlib.sha256(
        _canonical_json_bytes(outcome_record["outcome"])
    ).hexdigest()
    return observed_body_sha == event.payload_sha256


def _manifest_artifacts(
    *,
    context: EvidenceContext,
    creation: CreationEvidence,
    outcome_ref: ArtifactRef,
    audit_ref: ArtifactRef,
    creation_time: datetime,
    completed_at: datetime,
    manifest_parameters: tuple[tuple[str, str], ...],
) -> tuple[ManifestArtifact, ...]:
    definitions = (
        (ArtifactKind.CANONICAL_SPEC, creation.canonical_spec_ref, (), creation_time),
        (
            ArtifactKind.DATA_SNAPSHOT,
            creation.data_snapshot_ref,
            (creation.canonical_spec_ref.sha256,),
            creation_time,
        ),
        (
            ArtifactKind.DATA_QUALITY,
            creation.data_quality_ref,
            (creation.data_snapshot_ref.sha256,),
            creation_time,
        ),
        (
            ArtifactKind.FORECASTS,
            creation.forecast_ref,
            (creation.canonical_spec_ref.sha256, creation.data_snapshot_ref.sha256),
            creation_time,
        ),
        (
            ArtifactKind.DIAGNOSTICS,
            creation.diagnostics_ref,
            (creation.forecast_ref.sha256,),
            creation_time,
        ),
        (
            ArtifactKind.FORECAST_ORIGINS,
            creation.seal_ref,
            (creation.forecast_ref.sha256, creation.diagnostics_ref.sha256),
            creation_time,
        ),
        (
            ArtifactKind.FORECAST_METRICS,
            outcome_ref,
            (creation.seal_ref.sha256,),
            completed_at,
        ),
        (
            ArtifactKind.METHODOLOGY_AUDIT,
            audit_ref,
            (outcome_ref.sha256,),
            completed_at,
        ),
    )
    return tuple(
        ManifestArtifact(
            kind=kind,
            ref=ref,
            run_id=context.run_id,
            experiment_id=creation.experiment_id,
            schema_version="1.0",
            producer="openalpha-sentinel",
            producer_version="0.1.0",
            input_sha256=inputs,
            created_at=created_at,
            code_commit=context.code_commit,
            code_dirty=context.code_dirty,
            dependency_lock_sha256=context.dependency_lock_sha256,
            parameters=manifest_parameters,
            random_seed=None,
        )
        for kind, ref, inputs, created_at in definitions
    )


def _ledger_event(
    *,
    sequence: int,
    event_type: Literal[
        "FORECAST_CREATED",
        "DIAGNOSTICS_COMPUTED",
        "OUTCOME_RESOLVED",
    ],
    payload_sha256: str,
    previous_record_sha256: str | None,
    occurred_at: datetime,
) -> LedgerEvent:
    hash_body = {
        "sequence": sequence,
        "event_type": event_type,
        "payload_sha256": payload_sha256,
        "previous_record_sha256": previous_record_sha256,
        "occurred_at": occurred_at.isoformat(),
    }
    return LedgerEvent(
        sequence=sequence,
        event_type=event_type,
        payload_sha256=payload_sha256,
        previous_record_sha256=previous_record_sha256,
        occurred_at=occurred_at,
        record_sha256=hashlib.sha256(_canonical_json_bytes(hash_body)).hexdigest(),
    )


def _ledger_hash(event: LedgerEvent) -> str:
    body = {
        "sequence": event.sequence,
        "event_type": event.event_type,
        "payload_sha256": event.payload_sha256,
        "previous_record_sha256": event.previous_record_sha256,
        "occurred_at": event.occurred_at.isoformat(),
    }
    return hashlib.sha256(_canonical_json_bytes(body)).hexdigest()


def _creation_seal_payload(creation: CreationEvidence) -> dict[str, object]:
    return creation.model_dump(mode="json", exclude={"seal_ref"})


def _put_json(store: LocalArtifactStore, payload: dict[str, object]) -> ArtifactRef:
    return store.put_bytes(_canonical_json_bytes(payload), media_type="application/json")


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
