"""Request and response models for the Phase 2 control API.

Run creation accepts only approved inputs. There is deliberately no field for
architecture, metrics, thresholds, partitions, periods, symbols, or training
configuration: those are locked by the experiment and its amendments, and the
API must not become a channel for changing them.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..phase2.gates import TerminalConclusion
from ..phase2.states import EvidenceClass, Phase2State

__all__ = [
    "ArtifactEntry",
    "ArtifactManifestResponse",
    "CreateRunRequest",
    "ExecutionMode",
    "LogEntry",
    "LogsResponse",
    "ResumeRunRequest",
    "RunCreatedResponse",
    "RunStatusResponse",
]


class ExecutionMode(StrEnum):
    """Real empirical evidence, or synthetic pipeline validation."""

    REAL = "real"
    SYNTHETIC = "synthetic"

    @property
    def evidence_class(self) -> EvidenceClass:
        return (
            EvidenceClass.REAL_PHASE2
            if self is ExecutionMode.REAL
            else EvidenceClass.SYNTHETIC_PIPELINE_VALIDATION
        )


class CreateRunRequest(BaseModel):
    """Approved inputs only."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    experiment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_commit: str = Field(min_length=7, max_length=40, pattern=r"^[0-9a-f]+$")
    operator: str = Field(min_length=1, max_length=128)
    execution_mode: ExecutionMode = ExecutionMode.SYNTHETIC
    idempotency_key: str | None = Field(default=None, min_length=8, max_length=128)
    confirm_real_evidence: bool = False
    confirm_open_test_partition: bool = False


class ResumeRunRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    operator: str = Field(min_length=1, max_length=128)
    idempotency_key: str | None = Field(default=None, min_length=8, max_length=128)


class CancelRunRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    operator: str = Field(min_length=1, max_length=128)
    reason: str | None = Field(default=None, max_length=512)
    idempotency_key: str | None = Field(default=None, min_length=8, max_length=128)


class RunCreatedResponse(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    schema_version: Literal["openalpha.bridge.phase2.run_created.v1"] = (
        "openalpha.bridge.phase2.run_created.v1"
    )
    run_id: str
    state: Phase2State
    created_at: datetime
    experiment_hash: str
    status_url: str
    artifact_url: str
    cloud_execution_id: str | None
    evidence_class: EvidenceClass
    idempotent_replay: bool = False


class RunStatusResponse(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    schema_version: Literal["openalpha.bridge.phase2.run_status.v1"] = (
        "openalpha.bridge.phase2.run_status.v1"
    )
    run_id: str
    state: Phase2State
    current_stage: str | None
    latest_journal_sequence: int
    latest_journal_sha256: str | None
    created_at: datetime | None
    updated_at: datetime | None
    blocker_code: str | None
    failure_code: str | None
    progress: dict[str, Any]
    cloud_execution_id: str | None
    cloud_execution_status: str | None
    artifacts_available: bool
    test_partition_opened: bool
    test_sealed: bool
    final_conclusion: TerminalConclusion | None
    evidence_class: EvidenceClass


class ArtifactEntry(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    key: str
    category: str
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    created_at: datetime
    download_url: str | None = None
    download_expires_at: datetime | None = None


class ArtifactManifestResponse(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    schema_version: Literal["openalpha.bridge.phase2.artifact_manifest.v1"] = (
        "openalpha.bridge.phase2.artifact_manifest.v1"
    )
    run_id: str
    evidence_class: EvidenceClass
    artifacts: tuple[ArtifactEntry, ...]
    manifest_sha256: str


class LogEntry(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    sequence: int
    occurred_at: datetime
    state: Phase2State
    stage: str | None
    message: str
    progress: dict[str, Any]


class LogsResponse(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    schema_version: Literal["openalpha.bridge.phase2.logs.v1"] = (
        "openalpha.bridge.phase2.logs.v1"
    )
    run_id: str
    entries: tuple[LogEntry, ...]
    truncated: bool = False


class ApiError(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    error: str
    code: str
    message: str
    status_code: int
