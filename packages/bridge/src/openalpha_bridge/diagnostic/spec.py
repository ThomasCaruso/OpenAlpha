"""Constants mirroring phase2-frozen-inference-diagnostic.yaml.

The YAML is the specification; this module is how code reads it. Both are
hashed, and ``verify_diagnostic_specification`` refuses to run against a file
whose bytes have changed, so the two cannot drift apart unnoticed.

Nothing here amends the sealed experiment or its three amendments. Their bytes
and digests are untouched; this is a separately hashed development document.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final, Literal

from ..errors import BridgeFailure, BridgeTransformError, FailureCategory

__all__ = [
    "CLAIM_BOUNDARY",
    "CONTEXT_CANDLES",
    "DIAGNOSTIC_EVIDENCE_CLASS_VALUE",
    "DIAGNOSTIC_SPECIFICATION_NAME",
    "DIAGNOSTIC_SPECIFICATION_SHA256",
    "KRONOS_MINI_SPEC",
    "OFFICIAL_INFERENCE_SETTINGS",
    "PRIMARY_METRIC",
    "ROLLOUT_COUNT",
    "ROLLOUT_SEEDS",
    "TARGET_CANDLES",
    "THRESHOLDS",
    "TOTAL_CANDLES",
    "WINDOW",
    "DiagnosticThresholds",
    "ForecastModelSpec",
    "InferenceSettings",
    "verify_diagnostic_specification",
]

from pydantic import BaseModel, ConfigDict, Field

DIAGNOSTIC_SPECIFICATION_NAME: Final[str] = "phase2-frozen-inference-diagnostic.yaml"

#: sha256 of the specification file. Separately hashed; it is not part of the
#: sealed experiment chain and does not alter any sealed digest.
DIAGNOSTIC_SPECIFICATION_SHA256: Final[str] = (
    "c909e156a61ac5b5323de9e794aad20521839c0f43f491fb356c1968f730101b"
)

CLAIM_BOUNDARY: Final[str] = "DEVELOPMENT DIAGNOSTIC - NOT HOLDOUT OR TRADING EVIDENCE"

#: Reuses the Stage A canary window exactly, so no new data access is created.
TOTAL_CANDLES: Final[int] = 512
CONTEXT_CANDLES: Final[int] = 448
TARGET_CANDLES: Final[int] = 64

DIAGNOSTIC_EVIDENCE_CLASS_VALUE: Final[str] = "development_compatibility_canary"

PRIMARY_METRIC: Final[str] = "close_return_mae"


class DiagnosticWindow(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    symbol: Literal["SPY"] = "SPY"
    interval: Literal["1d"] = "1d"
    start_inclusive: str = "2015-05-07"
    end_exclusive: str = "2017-05-18"
    required_candles: int = TOTAL_CANDLES
    context_candles: int = CONTEXT_CANDLES
    target_candles: int = TARGET_CANDLES
    provider_retrievals_permitted: Literal[1] = 1


WINDOW: Final[DiagnosticWindow] = DiagnosticWindow()


class ForecastModelSpec(BaseModel):
    """The pinned Kronos-mini forecasting model."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    name: Literal["Kronos-mini"] = "Kronos-mini"
    repository: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    weights_file: str
    weights_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    weights_size_bytes: int = Field(gt=0)
    coarse_vocabulary: int = Field(gt=0)
    fine_vocabulary: int = Field(gt=0)
    decoder_layers: int = Field(gt=0)
    model_dimension: int = Field(gt=0)


KRONOS_MINI_SPEC: Final[ForecastModelSpec] = ForecastModelSpec(
    repository="NeoQuasar/Kronos-mini",
    revision="f4e68697d9d5aed55cef5c96aabc3376bcad9f81",
    config_sha256="70daca2cb11e3a979dd6b8ac12ee08e2aace877acf28f5b8dfb4fe5609736201",
    weights_file="model.safetensors",
    weights_sha256="a7d5f37e2e9fbd9891f7d7d4f72574512dd1f704fee14223e0a8cd0fbf54197c",
    weights_size_bytes=16_440_776,
    # s1_bits = s2_bits = 10 in the pinned config, so 2**10 per stream.
    coarse_vocabulary=1024,
    fine_vocabulary=1024,
    decoder_layers=4,
    model_dimension=256,
)


class InferenceSettings(BaseModel):
    """Exactly what is passed to the frozen model. Recorded in the artifact."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 0.9
    sample_count_per_rollout: int = 1
    prediction_length: int = TARGET_CANDLES
    maximum_context: int = TOTAL_CANDLES
    context_length_used: int = CONTEXT_CANDLES
    parameters_frozen: Literal[True] = True
    gradient_mode: Literal["inference_only"] = "inference_only"


OFFICIAL_INFERENCE_SETTINGS: Final[InferenceSettings] = InferenceSettings()

ROLLOUT_COUNT: Final[int] = 64

#: Fixed by the specification, not derived at execution time.
ROLLOUT_SEEDS: Final[tuple[int, ...]] = tuple(20_150_507 + index for index in range(ROLLOUT_COUNT))


class DiagnosticThresholds(BaseModel):
    """Every number the conclusion rules compare against."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    roundtrip_invalid_fraction_maximum: float = 0.01
    minimum_relative_improvement: float = 0.05
    valid_rollout_support_fraction_minimum: float = 0.25
    accuracy_equivalence_margin: float = 0.05


THRESHOLDS: Final[DiagnosticThresholds] = DiagnosticThresholds()


def verify_diagnostic_specification(research_root: Path | str) -> str:
    """Confirm the specification file is byte-identical, or fail closed."""
    path = Path(research_root) / DIAGNOSTIC_SPECIFICATION_NAME
    if not path.is_file():
        raise BridgeTransformError(
            BridgeFailure(
                category=FailureCategory.INVALID_CONFIGURATION,
                code="MISSING_DIAGNOSTIC_SPECIFICATION",
                field=DIAGNOSTIC_SPECIFICATION_NAME,
                message=f"diagnostic specification not found: {path}",
            )
        )
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != DIAGNOSTIC_SPECIFICATION_SHA256:
        raise BridgeTransformError(
            BridgeFailure(
                category=FailureCategory.INVALID_CONFIGURATION,
                code="DIAGNOSTIC_SPECIFICATION_HASH_MISMATCH",
                field=DIAGNOSTIC_SPECIFICATION_NAME,
                message=(
                    f"the diagnostic specification hashes to {observed}, expected "
                    f"{DIAGNOSTIC_SPECIFICATION_SHA256}"
                ),
            )
        )
    return observed
