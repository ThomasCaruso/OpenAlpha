"""Run identity and configuration hashing.

A run's identity is derived from the locked experiment hash plus the resolved
execution configuration. Changing any locked input starts a new run identity, so
completed stages are content-addressed and cannot be silently reused across
different configurations.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..errors import BridgeFailure, BridgeTransformError, FailureCategory
from .states import EvidenceClass

__all__ = [
    "AMENDMENT_1_SHA256",
    "AMENDMENT_2_SHA256",
    "AMENDMENT_3_SHA256",
    "EXPERIMENT_SHA256",
    "RunIdentity",
    "canonical_json",
    "canonical_sha256",
    "file_sha256",
    "verify_locked_hashes",
]

#: Committed locks. Verified before any stage runs.
EXPERIMENT_SHA256 = "d52a9be733f4ec331e081346e64ab7415ac0f9510e494733746f975f114f47b2"
AMENDMENT_1_SHA256 = "4c60846e8cb791d922c29a5e704fe3a473740171dd2bbb7b3390796ceeeef3c8"
AMENDMENT_2_SHA256 = "66f3c8171c2805bccf4b924125bd206edad27c957dfff40a0abf3864fbab10c1"
AMENDMENT_3_SHA256 = "d9484020a22df42c93edc19941374c628ea891f08c8bee8151b82264c00bb12b"


def canonical_json(value: Any) -> bytes:
    """Deterministic JSON encoding used for every content address."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_locked_hashes(research_root: Path) -> dict[str, str]:
    """Verify the experiment and both amendments are byte-identical.

    Returns the observed digests. Raises on any mismatch so a drifted lock can
    never reach a stage.
    """
    expected = {
        "experiment.yaml": EXPERIMENT_SHA256,
        "phase2-preregistration-amendment.yaml": AMENDMENT_1_SHA256,
        "phase2-amendment-2-context-prefix.yaml": AMENDMENT_2_SHA256,
        "phase2-amendment-3-scale-features.yaml": AMENDMENT_3_SHA256,
    }
    observed: dict[str, str] = {}
    for name, want in expected.items():
        path = research_root / name
        if not path.is_file():
            raise BridgeTransformError(
                BridgeFailure(
                    category=FailureCategory.INVALID_CONFIGURATION,
                    code="MISSING_LOCK_FILE",
                    field=name,
                    message=f"locked file not found: {path}",
                )
            )
        got = file_sha256(path)
        observed[name] = got
        if got != want:
            raise BridgeTransformError(
                BridgeFailure(
                    category=FailureCategory.INVALID_CONFIGURATION,
                    code="EXPERIMENT_HASH_MISMATCH",
                    field=name,
                    observed_value=got,
                    message=f"{name} expected {want}, observed {got}",
                )
            )
    return observed


class RunIdentity(BaseModel):
    """Content-addressed identity for one Phase 2 execution."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    schema_version: Literal["openalpha.bridge.phase2.identity.v1"] = (
        "openalpha.bridge.phase2.identity.v1"
    )
    run_id: str = Field(min_length=8, max_length=64)
    experiment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    amendment_1_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    amendment_2_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_class: EvidenceClass
    provider_mode: str
    kronos_mode: str
    source_commit: str | None = None

    @classmethod
    def derive(
        cls,
        *,
        configuration: dict[str, Any],
        evidence_class: EvidenceClass,
        provider_mode: str,
        kronos_mode: str,
        source_commit: str | None = None,
    ) -> RunIdentity:
        configuration_sha256 = canonical_sha256(configuration)
        seed = canonical_sha256(
            {
                "experiment": EXPERIMENT_SHA256,
                "amendment_1": AMENDMENT_1_SHA256,
                "amendment_2": AMENDMENT_2_SHA256,
                "configuration": configuration_sha256,
                "evidence_class": evidence_class.value,
                "provider_mode": provider_mode,
                "kronos_mode": kronos_mode,
            }
        )
        prefix = "syn" if evidence_class is EvidenceClass.SYNTHETIC_PIPELINE_VALIDATION else "run"
        return cls(
            run_id=f"{prefix}_{seed[:24]}",
            experiment_sha256=EXPERIMENT_SHA256,
            amendment_1_sha256=AMENDMENT_1_SHA256,
            amendment_2_sha256=AMENDMENT_2_SHA256,
            configuration_sha256=configuration_sha256,
            evidence_class=evidence_class,
            provider_mode=provider_mode,
            kronos_mode=kronos_mode,
            source_commit=source_commit,
        )

    @property
    def is_real_evidence(self) -> bool:
        return self.evidence_class is EvidenceClass.REAL_PHASE2
