from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from .artifacts import ArtifactRef, LocalArtifactStore, Sha256
from .errors import ArtifactIntegrityError, ManifestIntegrityError
from .run_state import RunState

Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")]
ExperimentId = Annotated[str, StringConstraints(pattern=r"^exp_[0-9a-f]{64}$")]
CommitSha = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}([0-9a-f]{24})?$")]


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
            raise TypeError("model_copy updates bypass validation; validate a new manifest model")
        return super().model_copy(update=None, deep=deep)


class ArtifactKind(StrEnum):
    CANONICAL_SPEC = "canonical_spec"
    DATA_SNAPSHOT = "data_snapshot"
    DATA_QUALITY = "data_quality"
    FORECAST_ORIGINS = "forecast_origins"
    FORECASTS = "forecasts"
    FORECAST_METRICS = "forecast_metrics"
    SIGNALS = "signals"
    ORDERS = "orders"
    FILLS = "fills"
    ACCOUNTING_LEDGER = "accounting_ledger"
    RISK_METRICS = "risk_metrics"
    METHODOLOGY_AUDIT = "methodology_audit"
    RESEARCH_REPORT = "research_report"
    DIAGNOSTICS = "diagnostics"


class MethodologyStatus(StrEnum):
    PASSED = "passed"
    PASSED_WITH_WARNINGS = "passed_with_warnings"
    FAILED = "failed"


class ManifestProfile(StrEnum):
    LEGACY_RESEARCH_RUN = "legacy_research_run"
    SENTINEL_ORIGIN = "sentinel_origin"


class GitMetadata(FrozenModel):
    commit_sha: CommitSha
    dirty: bool
    diff_sha256: Sha256 | None = None

    @model_validator(mode="after")
    def disclose_dirty_tree(self) -> GitMetadata:
        if self.dirty and self.diff_sha256 is None:
            raise ValueError("diff_sha256 is required when the Git tree is dirty")
        if not self.dirty and self.diff_sha256 is not None:
            raise ValueError("diff_sha256 must be absent when the Git tree is clean")
        return self


class EnvironmentMetadata(FrozenModel):
    os_name: str = Field(min_length=1, max_length=128)
    os_version: str = Field(min_length=1, max_length=256)
    architecture: str = Field(min_length=1, max_length=128)
    python_version: str = Field(min_length=1, max_length=64)
    node_version: str | None = Field(default=None, min_length=1, max_length=64)
    dependency_lock_sha256: Sha256
    container_image: str | None = Field(default=None, min_length=1, max_length=512)
    hardware: str = Field(min_length=1, max_length=2_000)


class ManifestArtifact(FrozenModel):
    kind: ArtifactKind
    ref: ArtifactRef
    run_id: Identifier
    experiment_id: ExperimentId
    schema_version: str = Field(min_length=1, max_length=32)
    producer: str = Field(min_length=1, max_length=128)
    producer_version: str = Field(min_length=1, max_length=128)
    input_sha256: tuple[Sha256, ...] = ()
    created_at: datetime
    code_commit: CommitSha
    code_dirty: bool
    dependency_lock_sha256: Sha256
    parameters: tuple[tuple[str, str], ...] = ()
    random_seed: int | None = Field(default=None, ge=0, le=2**32 - 1)

    @field_validator("created_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value

    @field_validator("input_sha256")
    @classmethod
    def require_unique_inputs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("input_sha256 values must be unique")
        return value

    @field_validator("parameters")
    @classmethod
    def require_unique_parameter_names(
        cls, value: tuple[tuple[str, str], ...]
    ) -> tuple[tuple[str, str], ...]:
        names = [name for name, _ in value]
        if len(names) != len(set(names)):
            raise ValueError("artifact parameter names must be unique")
        return value


class RunManifest(FrozenModel):
    schema_version: Literal["1.0"]
    profile: ManifestProfile = ManifestProfile.LEGACY_RESEARCH_RUN
    run_id: Identifier
    attempt_id: Identifier
    experiment_id: ExperimentId
    canonical_spec_schema_version: str = Field(min_length=1, max_length=32)
    state: RunState
    methodology_status: MethodologyStatus
    artifacts: tuple[ManifestArtifact, ...] = Field(min_length=1)
    git: GitMetadata
    environment: EnvironmentMetadata
    test_evidence: tuple[str, ...] = Field(min_length=1)
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value

    @field_validator("test_evidence")
    @classmethod
    def require_nonempty_evidence(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError("test evidence entries cannot be blank")
        return value

    @model_validator(mode="after")
    def require_unique_artifact_paths(self) -> RunManifest:
        paths = [artifact.ref.relative_path for artifact in self.artifacts]
        if len(paths) != len(set(paths)):
            raise ValueError("artifact inventory paths must be unique")
        return self


REQUIRED_COMPLETED_ARTIFACT_KINDS = frozenset(
    {
        ArtifactKind.CANONICAL_SPEC,
        ArtifactKind.DATA_SNAPSHOT,
        ArtifactKind.DATA_QUALITY,
        ArtifactKind.FORECAST_ORIGINS,
        ArtifactKind.FORECASTS,
        ArtifactKind.FORECAST_METRICS,
        ArtifactKind.SIGNALS,
        ArtifactKind.ORDERS,
        ArtifactKind.FILLS,
        ArtifactKind.ACCOUNTING_LEDGER,
        ArtifactKind.RISK_METRICS,
        ArtifactKind.METHODOLOGY_AUDIT,
    }
)
REQUIRED_SENTINEL_ORIGIN_ARTIFACT_KINDS = frozenset(
    {
        ArtifactKind.CANONICAL_SPEC,
        ArtifactKind.DATA_SNAPSHOT,
        ArtifactKind.DATA_QUALITY,
        ArtifactKind.FORECAST_ORIGINS,
        ArtifactKind.FORECASTS,
        ArtifactKind.DIAGNOSTICS,
        ArtifactKind.FORECAST_METRICS,
        ArtifactKind.METHODOLOGY_AUDIT,
    }
)
REQUIRED_COMPLETED_ARTIFACT_KINDS_BY_PROFILE = {
    ManifestProfile.LEGACY_RESEARCH_RUN: REQUIRED_COMPLETED_ARTIFACT_KINDS,
    ManifestProfile.SENTINEL_ORIGIN: REQUIRED_SENTINEL_ORIGIN_ARTIFACT_KINDS,
}
MANIFEST_MEDIA_TYPE = "application/vnd.openalpha.run-manifest+json"


def canonical_manifest_bytes(manifest: RunManifest) -> bytes:
    payload = manifest.model_dump(mode="json", exclude_none=False)
    artifacts = payload["artifacts"]
    assert isinstance(artifacts, list)
    payload["artifacts"] = sorted(
        artifacts,
        key=lambda artifact: (
            artifact["kind"],
            artifact["ref"]["relative_path"],
        ),
    )
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def publish_manifest(store: LocalArtifactStore, manifest: RunManifest) -> ArtifactRef:
    if manifest.state is not RunState.COMPLETED:
        raise ManifestIntegrityError("only a completed run can publish a completed manifest")
    if manifest.methodology_status is MethodologyStatus.FAILED:
        raise ManifestIntegrityError("methodology failed; completed manifest publication refused")

    artifacts_by_kind: dict[ArtifactKind, list[ManifestArtifact]] = {}
    inventory_hashes = {artifact.ref.sha256 for artifact in manifest.artifacts}
    for artifact in manifest.artifacts:
        artifacts_by_kind.setdefault(artifact.kind, []).append(artifact)

    required_kinds = REQUIRED_COMPLETED_ARTIFACT_KINDS_BY_PROFILE[manifest.profile]
    missing_kinds = required_kinds - artifacts_by_kind.keys()
    if missing_kinds:
        missing = ", ".join(sorted(kind.value for kind in missing_kinds))
        raise ManifestIntegrityError(f"missing required artifact kinds: {missing}")
    for unique_kind in required_kinds:
        if len(artifacts_by_kind[unique_kind]) != 1:
            raise ManifestIntegrityError(
                f"completed manifest requires exactly one {unique_kind.value} artifact"
            )

    spec = artifacts_by_kind[ArtifactKind.CANONICAL_SPEC][0]
    if manifest.experiment_id != f"exp_{spec.ref.sha256}":
        raise ManifestIntegrityError(
            "experiment identity does not match the canonical specification hash"
        )
    if spec.schema_version != manifest.canonical_spec_schema_version:
        raise ManifestIntegrityError(
            "canonical specification schema version does not match the manifest"
        )
    _verify_json_artifact(
        store,
        spec,
        label="canonical specification",
        require_canonical=True,
    )

    for artifact in manifest.artifacts:
        unknown_inputs = set(artifact.input_sha256) - inventory_hashes
        if unknown_inputs:
            raise ManifestIntegrityError(
                f"artifact {artifact.kind.value} references unknown input hashes: "
                f"{', '.join(sorted(unknown_inputs))}"
            )
        try:
            store.read_bytes(artifact.ref)
        except ArtifactIntegrityError as error:
            raise ManifestIntegrityError(str(error)) from error
        if artifact.run_id != manifest.run_id:
            raise ManifestIntegrityError(
                f"artifact lineage run_id mismatch for {artifact.kind.value}"
            )
        if artifact.experiment_id != manifest.experiment_id:
            raise ManifestIntegrityError(
                f"artifact lineage experiment_id mismatch for {artifact.kind.value}"
            )
        if artifact.code_commit != manifest.git.commit_sha:
            raise ManifestIntegrityError(
                f"artifact lineage code commit mismatch for {artifact.kind.value}"
            )
        if artifact.code_dirty is not manifest.git.dirty:
            raise ManifestIntegrityError(
                f"artifact lineage dirty-tree mismatch for {artifact.kind.value}"
            )
        if artifact.dependency_lock_sha256 != manifest.environment.dependency_lock_sha256:
            raise ManifestIntegrityError(
                f"artifact lineage dependency lock mismatch for {artifact.kind.value}"
            )
    _verify_acyclic_lineage(manifest.artifacts)

    audit = artifacts_by_kind[ArtifactKind.METHODOLOGY_AUDIT][0]
    _verify_audit_status(store, audit, manifest.methodology_status)
    return store.put_bytes(canonical_manifest_bytes(manifest), media_type=MANIFEST_MEDIA_TYPE)


def _verify_audit_status(
    store: LocalArtifactStore,
    artifact: ManifestArtifact,
    declared_status: MethodologyStatus,
) -> None:
    payload = _verify_json_artifact(store, artifact, label="methodology audit")
    observed = payload.get("methodology_status")
    if observed != declared_status.value:
        raise ManifestIntegrityError(
            "manifest methodology status does not match the verified audit artifact"
        )


def _verify_json_artifact(
    store: LocalArtifactStore,
    artifact: ManifestArtifact,
    *,
    label: str,
    require_canonical: bool = False,
) -> dict[str, object]:
    if artifact.ref.media_type != "application/json":
        raise ManifestIntegrityError(
            f"{artifact.kind.value} artifact requires application/json media type"
        )
    raw = store.read_bytes(artifact.ref)
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ManifestIntegrityError(f"{label} artifact is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ManifestIntegrityError(f"{label} artifact must be a JSON object")
    if payload.get("schema_version") != artifact.schema_version:
        raise ManifestIntegrityError(
            f"{label} artifact schema version does not match its JSON payload"
        )
    if require_canonical:
        try:
            canonical = json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError) as error:
            raise ManifestIntegrityError(
                f"{label} artifact contains non-canonical JSON values"
            ) from error
        if raw != canonical:
            raise ManifestIntegrityError(f"{label} artifact JSON is not canonical")
    return payload


def _verify_acyclic_lineage(artifacts: tuple[ManifestArtifact, ...]) -> None:
    graph = {artifact.ref.sha256: frozenset(artifact.input_sha256) for artifact in artifacts}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(sha256: str) -> None:
        if sha256 in visiting:
            raise ManifestIntegrityError("artifact input lineage contains a cycle")
        if sha256 in visited:
            return
        visiting.add(sha256)
        for input_sha256 in graph[sha256]:
            visit(input_sha256)
        visiting.remove(sha256)
        visited.add(sha256)

    for artifact_sha256 in graph:
        visit(artifact_sha256)
