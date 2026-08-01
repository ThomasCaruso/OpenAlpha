"""Framework-agnostic Phase 2 control service.

All control-plane behaviour lives here as plain Python so it is testable without
a web framework or a cloud SDK. ``http.py`` binds it to FastAPI and the Modal
application binds it to web endpoints; neither adds logic.

There is deliberately no remote-shell or arbitrary-command operation.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Protocol

from ..errors import BridgeFailure, BridgeTransformError, FailureCategory
from ..phase2.gates import TerminalConclusion
from ..phase2.identity import EXPERIMENT_SHA256, canonical_sha256
from ..phase2.states import EvidenceClass, Phase2State
from .identity import CloudRunIdentity, assert_idempotent_match
from .journal import CloudJournal
from .lease import current_lease
from .models import (
    ArtifactEntry,
    ArtifactManifestResponse,
    CancelRunRequest,
    CreateRunRequest,
    ExecutionMode,
    LogEntry,
    LogsResponse,
    ResumeRunRequest,
    RunCreatedResponse,
    RunStatusResponse,
)
from .objectstore import ObjectStore, get_json, get_model, put_json, run_prefix
from .redaction import redact

__all__ = ["ComputeBackend", "Phase2ControlService", "ServiceConfig"]

_LOGGER = logging.getLogger("openalpha.bridge.cloud.service")

_RESUMABLE_STATES = frozenset({Phase2State.FAILED, Phase2State.BLOCKED})
_TERMINAL_STATES = frozenset({Phase2State.FINALIZED})


def _fail(code: str, message: str, *, status: int = 400) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            observed_value=status,
            message=message,
        )
    )


class ComputeBackend(Protocol):
    """The managed-compute backend. Modal is the only implementation today.

    Kept narrow deliberately so another backend could be added later without
    touching the control service or the scientific engine.
    """

    @property
    def name(self) -> str: ...

    @property
    def image_digest(self) -> str: ...

    @property
    def app_version(self) -> str: ...

    def spawn_real_run(self, run_id: str, payload: dict[str, Any]) -> str:
        """Start the GPU worker in the background and return a cloud execution id."""
        ...

    def spawn_synthetic_run(self, run_id: str, payload: dict[str, Any]) -> str: ...

    def execution_status(self, cloud_execution_id: str) -> str: ...

    def cancel(self, cloud_execution_id: str) -> None: ...


class ServiceConfig:
    """Deployment-scoped configuration. Holds no secret values."""

    def __init__(
        self,
        *,
        base_url: str,
        object_store_namespace: str,
        dependency_lock_sha256: str,
        kronos_revision: str,
        provider_identity: str,
        approved_source_commits: frozenset[str] | None = None,
        rate_limit_per_hour: int = 12,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.object_store_namespace = object_store_namespace
        self.dependency_lock_sha256 = dependency_lock_sha256
        self.kronos_revision = kronos_revision
        self.provider_identity = provider_identity
        self.approved_source_commits = approved_source_commits
        self.rate_limit_per_hour = rate_limit_per_hour


class Phase2ControlService:
    """Create, observe, resume, and cancel managed Phase 2 runs."""

    def __init__(
        self,
        *,
        store: ObjectStore,
        backend: ComputeBackend,
        config: ServiceConfig,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._store = store
        self._backend = backend
        self._config = config
        self._clock = clock or (lambda: datetime.now(UTC))
        self._recent_creations: list[datetime] = []

    # ------------------------------------------------------------- helpers

    def _identity_key(self, run_id: str, evidence_class: EvidenceClass) -> str:
        return f"{run_prefix(run_id, evidence_class)}/identity/run_identity.json"

    def _idempotency_key(self, key: str, evidence_class: EvidenceClass) -> str:
        digest = canonical_sha256({"idempotency_key": key})
        namespace = (
            "openalpha/bridge-phase2"
            if evidence_class is EvidenceClass.REAL_PHASE2
            else "openalpha-synthetic/bridge-phase2"
        )
        return f"{namespace}/idempotency/{digest}.json"

    def _load_identity(self, run_id: str) -> CloudRunIdentity:
        for evidence_class in (EvidenceClass.REAL_PHASE2, EvidenceClass.SYNTHETIC_PIPELINE_VALIDATION):
            key = self._identity_key(run_id, evidence_class)
            if self._store.exists(key):
                return get_model(self._store, key, CloudRunIdentity)
        raise _fail("RUN_NOT_FOUND", f"no run with id {run_id}", status=404)

    def _journal(self, identity: CloudRunIdentity) -> CloudJournal:
        return CloudJournal(
            self._store,
            run_id=identity.run_id,
            experiment_hash=identity.active_experiment_sha256,
            evidence_class=identity.evidence_class,
        )

    def _enforce_rate_limit(self, now: datetime) -> None:
        window_start = now.timestamp() - 3600
        self._recent_creations = [
            stamp for stamp in self._recent_creations if stamp.timestamp() >= window_start
        ]
        if len(self._recent_creations) >= self._config.rate_limit_per_hour:
            raise _fail(
                "RATE_LIMIT_EXCEEDED",
                f"at most {self._config.rate_limit_per_hour} runs may be created per hour",
                status=429,
            )

    # -------------------------------------------------------------- create

    def create_run(self, request: CreateRunRequest) -> RunCreatedResponse:
        now = self._clock()

        if request.experiment_hash != EXPERIMENT_SHA256:
            raise _fail(
                "UNKNOWN_EXPERIMENT_HASH",
                "experiment_hash does not match the locked experiment",
            )
        approved = self._config.approved_source_commits
        if approved is not None and request.source_commit not in approved:
            raise _fail(
                "UNAPPROVED_SOURCE_COMMIT",
                "source_commit is not in the approved deployment set",
                status=403,
            )
        if request.execution_mode is ExecutionMode.REAL and not request.confirm_real_evidence:
            raise _fail(
                "REAL_EVIDENCE_NOT_CONFIRMED",
                "a real run requires confirm_real_evidence=true",
            )

        evidence_class = request.execution_mode.evidence_class
        identity = CloudRunIdentity.derive(
            source_commit=request.source_commit,
            provider_identity=self._config.provider_identity,
            kronos_revision=self._config.kronos_revision,
            cloud_image_digest=self._backend.image_digest,
            modal_app_version=self._backend.app_version,
            dependency_lock_sha256=self._config.dependency_lock_sha256,
            object_store_namespace=self._config.object_store_namespace,
            operator=request.operator,
            evidence_class=evidence_class,
            created_at=now,
            idempotency_key=request.idempotency_key,
        )

        # Idempotent replay: identical inputs return the same run.
        if request.idempotency_key:
            slot = self._idempotency_key(request.idempotency_key, evidence_class)
            if self._store.exists(slot):
                recorded = get_model(self._store, slot, CloudRunIdentity)
                assert_idempotent_match(recorded, identity)
                return self._created_response(recorded, replay=True)

        if self._store.exists(self._identity_key(identity.run_id, evidence_class)):
            existing = get_model(
                self._store,
                self._identity_key(identity.run_id, evidence_class),
                CloudRunIdentity,
            )
            return self._created_response(existing, replay=True)

        self._enforce_rate_limit(now)
        self._recent_creations.append(now)

        payload = {
            "run_id": identity.run_id,
            "evidence_class": evidence_class.value,
            "source_commit": request.source_commit,
            "operator": request.operator,
            "experiment_hash": request.experiment_hash,
            "confirm_open_test_partition": request.confirm_open_test_partition,
        }
        spawn = (
            self._backend.spawn_real_run
            if request.execution_mode is ExecutionMode.REAL
            else self._backend.spawn_synthetic_run
        )
        cloud_execution_id = spawn(identity.run_id, payload)

        bound = identity.model_copy(update={"cloud_execution_id": cloud_execution_id})
        put_json(
            self._store,
            self._identity_key(identity.run_id, evidence_class),
            bound.model_dump(mode="json"),
            schema_version=bound.schema_version,
            run_id=bound.run_id,
            experiment_hash=bound.active_experiment_sha256,
            evidence_class=evidence_class,
            immutable=True,
            created_at=now,
        )
        CloudJournal(
            self._store,
            run_id=bound.run_id,
            experiment_hash=bound.active_experiment_sha256,
            evidence_class=evidence_class,
        ).append(Phase2State.CREATED, occurred_at=now, stage="create_run")

        if request.idempotency_key:
            put_json(
                self._store,
                self._idempotency_key(request.idempotency_key, evidence_class),
                bound.model_dump(mode="json"),
                schema_version="openalpha.bridge.phase2.idempotency.v1",
                run_id=bound.run_id,
                experiment_hash=bound.active_experiment_sha256,
                evidence_class=evidence_class,
                immutable=True,
                created_at=now,
            )

        return self._created_response(bound, replay=False)

    def _created_response(
        self, identity: CloudRunIdentity, *, replay: bool
    ) -> RunCreatedResponse:
        journal = self._journal(identity)
        return RunCreatedResponse(
            run_id=identity.run_id,
            state=journal.current_state(),
            created_at=identity.created_at,
            experiment_hash=identity.active_experiment_sha256,
            status_url=f"{self._config.base_url}/v1/bridge/phase2/runs/{identity.run_id}",
            artifact_url=(
                f"{self._config.base_url}/v1/bridge/phase2/runs/{identity.run_id}/artifacts"
            ),
            cloud_execution_id=identity.cloud_execution_id,
            evidence_class=identity.evidence_class,
            idempotent_replay=replay,
        )

    # -------------------------------------------------------------- status

    def get_run(self, run_id: str) -> RunStatusResponse:
        from .testgate import is_cloud_test_partition_opened

        identity = self._load_identity(run_id)
        journal = self._journal(identity)
        entries = journal.entries()
        latest = entries[-1] if entries else None
        state = latest.state if latest else Phase2State.CREATED

        conclusion: TerminalConclusion | None = None
        final_key = f"{run_prefix(run_id, identity.evidence_class)}/final/gate_table.json"
        if self._store.exists(final_key):
            payload = get_json(self._store, final_key)
            raw = payload.get("conclusion")
            if raw:
                conclusion = TerminalConclusion(raw)

        cloud_status: str | None = None
        if identity.cloud_execution_id:
            try:
                cloud_status = self._backend.execution_status(identity.cloud_execution_id)
            except Exception:  # noqa: BLE001 - status is best effort
                cloud_status = "unknown"

        opened = is_cloud_test_partition_opened(
            self._store, run_id=run_id, evidence_class=identity.evidence_class
        )
        blocker = latest.reason if latest and latest.state is Phase2State.BLOCKED else None
        failure = latest.reason if latest and latest.state is Phase2State.FAILED else None

        return RunStatusResponse(
            run_id=run_id,
            state=state,
            current_stage=latest.stage if latest else None,
            latest_journal_sequence=latest.sequence if latest else 0,
            latest_journal_sha256=latest.entry_sha256 if latest else None,
            created_at=identity.created_at,
            updated_at=latest.occurred_at if latest else None,
            blocker_code=blocker,
            failure_code=failure,
            progress=redact(latest.progress if latest else {}),
            cloud_execution_id=identity.cloud_execution_id,
            cloud_execution_status=cloud_status,
            artifacts_available=bool(
                self._store.list_keys(f"{run_prefix(run_id, identity.evidence_class)}/final/")
            ),
            test_partition_opened=opened,
            test_sealed=not opened,
            final_conclusion=conclusion,
            evidence_class=identity.evidence_class,
        )

    # -------------------------------------------------------------- resume

    def resume_run(self, run_id: str, request: ResumeRunRequest) -> RunCreatedResponse:
        identity = self._load_identity(run_id)
        journal = self._journal(identity)
        state = journal.current_state()

        if state in _TERMINAL_STATES:
            raise _fail(
                "RUN_ALREADY_FINALIZED",
                f"run {run_id} is finalized and cannot be resumed",
                status=409,
            )
        if state not in _RESUMABLE_STATES:
            raise _fail(
                "RUN_NOT_RESUMABLE",
                (
                    f"run {run_id} is in {state.value}; only a verified BLOCKED or FAILED "
                    "run may be resumed"
                ),
                status=409,
            )

        lease = current_lease(
            self._store, run_id=run_id, evidence_class=identity.evidence_class
        )
        now = self._clock()
        if lease is not None and not lease.is_expired(now):
            raise _fail(
                "RUN_LEASE_HELD",
                f"run {run_id} is still leased by {lease.holder}",
                status=409,
            )

        payload = {
            "run_id": run_id,
            "evidence_class": identity.evidence_class.value,
            "source_commit": identity.source_commit,
            "operator": request.operator,
            "experiment_hash": identity.active_experiment_sha256,
            "resume": True,
        }
        spawn = (
            self._backend.spawn_real_run
            if identity.is_real_evidence
            else self._backend.spawn_synthetic_run
        )
        cloud_execution_id = spawn(run_id, payload)
        journal.append(
            state,
            occurred_at=now,
            stage="resume_requested",
            reason=f"resume requested by {request.operator}",
        )
        return self._created_response(
            identity.model_copy(update={"cloud_execution_id": cloud_execution_id}),
            replay=False,
        )

    # -------------------------------------------------------------- cancel

    def cancel_run(self, run_id: str, request: CancelRunRequest) -> RunStatusResponse:
        identity = self._load_identity(run_id)
        journal = self._journal(identity)
        now = self._clock()

        if journal.current_state() in _TERMINAL_STATES:
            raise _fail(
                "RUN_ALREADY_FINALIZED",
                f"run {run_id} is finalized and cannot be cancelled",
                status=409,
            )

        # Record cancellation BEFORE terminating the cloud execution, so an
        # interrupted cancel still leaves the intent durably recorded.
        journal.append(
            Phase2State.BLOCKED,
            occurred_at=now,
            stage="cancel",
            reason=redact(request.reason) or f"cancelled by {request.operator}",
        )
        if identity.cloud_execution_id:
            try:
                self._backend.cancel(identity.cloud_execution_id)
            except Exception as error:  # noqa: BLE001 - cancellation is best effort
                # The intent is already durably journaled above, so a failed
                # termination downgrades to a warning rather than losing it.
                _LOGGER.warning(
                    "cancel of %s failed: %s", run_id, redact(type(error).__name__)
                )
        return self.get_run(run_id)

    # ----------------------------------------------------------- artifacts

    def list_artifacts(self, run_id: str, *, signed: bool = False) -> ArtifactManifestResponse:
        identity = self._load_identity(run_id)
        prefix = run_prefix(run_id, identity.evidence_class)
        entries: list[ArtifactEntry] = []

        for key in self._store.list_keys(prefix):
            # Never expose model assets, feature shards, or raw provider data.
            if _is_private_artifact(key):
                continue
            stored = self._store.get(key)
            url: str | None = None
            if signed and hasattr(self._store, "signed_url"):
                url = self._store.signed_url(key)  # type: ignore[attr-defined]
            entries.append(
                ArtifactEntry(
                    key=key,
                    category=_category_for(key, prefix),
                    content_sha256=stored.metadata.content_sha256,
                    size_bytes=len(stored.body),
                    created_at=stored.metadata.created_at,
                    download_url=url,
                )
            )

        manifest = tuple(sorted(entries, key=lambda entry: entry.key))
        return ArtifactManifestResponse(
            run_id=run_id,
            evidence_class=identity.evidence_class,
            artifacts=manifest,
            manifest_sha256=canonical_sha256([e.model_dump(mode="json") for e in manifest]),
        )

    # ---------------------------------------------------------------- logs

    def get_logs(self, run_id: str, *, limit: int = 500) -> LogsResponse:
        identity = self._load_identity(run_id)
        journal = self._journal(identity)
        entries = journal.entries()
        truncated = len(entries) > limit
        selected = entries[-limit:] if truncated else entries

        return LogsResponse(
            run_id=run_id,
            entries=tuple(
                LogEntry(
                    sequence=entry.sequence,
                    occurred_at=entry.occurred_at,
                    state=entry.state,
                    stage=entry.stage,
                    message=redact(entry.reason or entry.state.value),
                    progress=redact(entry.progress),
                )
                for entry in selected
            ),
            truncated=truncated,
        )


def _is_private_artifact(key: str) -> bool:
    """Model weights, feature shards, and raw provider data are never listed."""
    private_markers = (
        "/checkpoint/weights",
        "/cache/",
        "/raw/",
        "/features/",
        "/identity/lease.json",
    )
    private_suffixes = (".pt", ".pth", ".safetensors", ".npz", ".npy", ".csv", ".parquet")
    return key.endswith(private_suffixes) or any(m in key for m in private_markers)


def _category_for(key: str, prefix: str) -> str:
    relative = key[len(prefix) :].lstrip("/")
    return relative.split("/", 1)[0] if "/" in relative else "root"
