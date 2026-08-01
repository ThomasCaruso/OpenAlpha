"""Cloud run identity.

Binds everything that could change a scientific result: source commit, locked
experiment and amendment hashes, score mask, provider, Kronos revision, cloud
image digest, Modal application version, dependency-lock hash, object-store
namespace, operator, and evidence class.

Any change to those inputs produces a different ``run_id``. Idempotency is
keyed on (idempotency_key, identity): the same key with identical inputs returns
the same run; the same key with conflicting inputs fails.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..errors import BridgeFailure, BridgeTransformError, FailureCategory
from ..phase2.identity import (
    AMENDMENT_1_SHA256,
    AMENDMENT_2_SHA256,
    EXPERIMENT_SHA256,
    canonical_sha256,
)
from ..phase2.states import EvidenceClass
from ..windowing import score_mask_sha256

__all__ = ["CloudRunIdentity", "assert_idempotent_match"]


def _fail(code: str, message: str) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            message=message,
        )
    )


class CloudRunIdentity(BaseModel):
    """Content-addressed identity of one managed cloud run."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    schema_version: Literal["openalpha.bridge.phase2.cloud_identity.v1"] = (
        "openalpha.bridge.phase2.cloud_identity.v1"
    )
    run_id: str = Field(min_length=8, max_length=64)
    source_commit: str = Field(min_length=7, max_length=40)
    original_experiment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    amendment_1_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    amendment_2_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    active_experiment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    score_mask_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_identity: str = Field(min_length=1)
    kronos_revision: str = Field(min_length=1)
    cloud_image_digest: str = Field(min_length=1)
    modal_app_version: str = Field(min_length=1)
    dependency_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    object_store_namespace: str = Field(min_length=1)
    operator: str = Field(min_length=1)
    evidence_class: EvidenceClass
    cloud_execution_id: str | None = None
    idempotency_key: str | None = None
    created_at: datetime

    @property
    def binding_sha256(self) -> str:
        """Hash of every result-affecting input, excluding mutable bookkeeping."""
        return canonical_sha256(
            {
                "source_commit": self.source_commit,
                "original_experiment": self.original_experiment_sha256,
                "amendment_1": self.amendment_1_sha256,
                "amendment_2": self.amendment_2_sha256,
                "active_experiment": self.active_experiment_sha256,
                "score_mask": self.score_mask_sha256,
                "provider": self.provider_identity,
                "kronos_revision": self.kronos_revision,
                "cloud_image_digest": self.cloud_image_digest,
                "modal_app_version": self.modal_app_version,
                "dependency_lock": self.dependency_lock_sha256,
                "object_store_namespace": self.object_store_namespace,
                "operator": self.operator,
                "evidence_class": self.evidence_class.value,
            }
        )

    @classmethod
    def derive(
        cls,
        *,
        source_commit: str,
        provider_identity: str,
        kronos_revision: str,
        cloud_image_digest: str,
        modal_app_version: str,
        dependency_lock_sha256: str,
        object_store_namespace: str,
        operator: str,
        evidence_class: EvidenceClass,
        created_at: datetime,
        active_experiment_sha256: str = EXPERIMENT_SHA256,
        cloud_execution_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> CloudRunIdentity:
        binding = canonical_sha256(
            {
                "source_commit": source_commit,
                "original_experiment": EXPERIMENT_SHA256,
                "amendment_1": AMENDMENT_1_SHA256,
                "amendment_2": AMENDMENT_2_SHA256,
                "active_experiment": active_experiment_sha256,
                "score_mask": score_mask_sha256(),
                "provider": provider_identity,
                "kronos_revision": kronos_revision,
                "cloud_image_digest": cloud_image_digest,
                "modal_app_version": modal_app_version,
                "dependency_lock": dependency_lock_sha256,
                "object_store_namespace": object_store_namespace,
                "operator": operator,
                "evidence_class": evidence_class.value,
            }
        )
        prefix = "syn" if evidence_class is EvidenceClass.SYNTHETIC_PIPELINE_VALIDATION else "run"
        return cls(
            run_id=f"{prefix}_{binding[:24]}",
            source_commit=source_commit,
            original_experiment_sha256=EXPERIMENT_SHA256,
            amendment_1_sha256=AMENDMENT_1_SHA256,
            amendment_2_sha256=AMENDMENT_2_SHA256,
            active_experiment_sha256=active_experiment_sha256,
            score_mask_sha256=score_mask_sha256(),
            provider_identity=provider_identity,
            kronos_revision=kronos_revision,
            cloud_image_digest=cloud_image_digest,
            modal_app_version=modal_app_version,
            dependency_lock_sha256=dependency_lock_sha256,
            object_store_namespace=object_store_namespace,
            operator=operator,
            evidence_class=evidence_class,
            cloud_execution_id=cloud_execution_id,
            idempotency_key=idempotency_key,
            created_at=created_at,
        )

    @property
    def is_real_evidence(self) -> bool:
        return self.evidence_class is EvidenceClass.REAL_PHASE2


def assert_idempotent_match(existing: CloudRunIdentity, candidate: CloudRunIdentity) -> None:
    """Same key plus identical inputs returns the run; conflicting inputs fail."""
    if existing.binding_sha256 != candidate.binding_sha256:
        raise _fail(
            "IDEMPOTENCY_KEY_CONFLICT",
            (
                f"idempotency key {candidate.idempotency_key!r} was already used with "
                "different inputs; use a new key or submit identical inputs"
            ),
        )
