"""Constants mirroring the diagnostic specifications.

v1 (``phase2-frozen-inference-diagnostic.yaml``) is preserved byte-identical
and superseded. v2 (``-v2.yaml``) is the operative document. Both are verified,
so a v1 file that drifted would still be caught even though nothing executes
under it.

Neither amends the sealed experiment or its three amendments; their bytes and
digests are untouched.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..errors import BridgeFailure, BridgeTransformError, FailureCategory

__all__ = [
    "CLAIM_BOUNDARY",
    "CONTEXT_CANDLES",
    "CONTROL_REPETITIONS",
    "CONTROL_SEED",
    "DIAGNOSTIC_EVIDENCE_CLASS_VALUE",
    "KRONOS_MINI_SPEC",
    "OFFICIAL_INFERENCE_SETTINGS",
    "OFFICIAL_SOURCE_FILES",
    "PRIMARY_METRIC",
    "ROLLOUT_COUNT",
    "ROLLOUT_SEEDS",
    "TARGET_CANDLES",
    "THRESHOLDS",
    "TOTAL_CANDLES",
    "V1_SPECIFICATION_NAME",
    "V1_SPECIFICATION_SHA256",
    "V2_SPECIFICATION_NAME",
    "V2_SPECIFICATION_SHA256",
    "V3_SPECIFICATION_NAME",
    "V3_SPECIFICATION_SHA256",
    "V4_SPECIFICATION_NAME",
    "V4_SPECIFICATION_SHA256",
    "WINDOW",
    "DiagnosticThresholds",
    "ForecastModelSpec",
    "InferenceSettings",
    "OfficialSourceFile",
    "verify_diagnostic_specifications",
]

V1_SPECIFICATION_NAME: Final[str] = "phase2-frozen-inference-diagnostic.yaml"
V1_SPECIFICATION_SHA256: Final[str] = (
    "c909e156a61ac5b5323de9e794aad20521839c0f43f491fb356c1968f730101b"
)

V2_SPECIFICATION_NAME: Final[str] = "phase2-frozen-inference-diagnostic-v2.yaml"
V2_SPECIFICATION_SHA256: Final[str] = (
    "c39fff4541afcc948cd80fcc545398312897efe723deca26554143989e4a175e"
)

V3_SPECIFICATION_NAME: Final[str] = "phase2-frozen-inference-diagnostic-v3.yaml"
V3_SPECIFICATION_SHA256: Final[str] = (
    "f10076b6676a72552b1c9c96720d0087c009fc939e4667509bfcfccf7929bcb6"
)

#: The operative document. v1, v2 and v3 are preserved and still verified.
V4_SPECIFICATION_NAME: Final[str] = "phase2-frozen-inference-diagnostic-v4.yaml"
V4_SPECIFICATION_SHA256: Final[str] = (
    "bd407722adfc3ebf92eb187828d42c2c9cfa57b2fdc0406121f27d39a5c44977"
)
OPERATIVE_SPECIFICATION_NAME: Final[str] = V4_SPECIFICATION_NAME

CLAIM_BOUNDARY: Final[str] = "DEVELOPMENT DIAGNOSTIC - NOT HOLDOUT OR TRADING EVIDENCE"

TOTAL_CANDLES: Final[int] = 512
CONTEXT_CANDLES: Final[int] = 448
TARGET_CANDLES: Final[int] = 64

DIAGNOSTIC_EVIDENCE_CLASS_VALUE: Final[str] = "development_compatibility_canary"

PRIMARY_METRIC: Final[str] = "close_return_mae"


class DiagnosticWindow(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    symbol: Literal["SPY"] = "SPY"
    frequency: Literal["1d"] = "1d"
    calendar: Literal["XNYS"] = "XNYS"
    start_inclusive: str = "2015-05-07"
    end_exclusive: str = "2017-05-18"
    required_candles: int = TOTAL_CANDLES
    context_candles: int = CONTEXT_CANDLES
    target_candles: int = TARGET_CANDLES
    provider_retrievals_permitted: Literal[1] = 1


WINDOW: Final[DiagnosticWindow] = DiagnosticWindow()


class OfficialSourceFile(BaseModel):
    """Both digests for one pinned source file.

    The sealed experiment records the CRLF-normalized digest. model/module.py
    is committed with CRLF so the two coincide; model/kronos.py is committed
    with LF so they differ. Recording both keeps the sealed value authoritative
    without pretending the upstream bytes are something they are not.
    """

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    relative_path: str
    sealed_sha256_crlf_normalized: str = Field(pattern=r"^[0-9a-f]{64}$")
    as_committed_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    as_committed_line_ending: Literal["LF", "CRLF"]
    as_committed_bytes: int = Field(gt=0)


OFFICIAL_SOURCE_FILES: Final[tuple[OfficialSourceFile, ...]] = (
    OfficialSourceFile(
        relative_path="model/kronos.py",
        sealed_sha256_crlf_normalized=(
            "638a56e035856c600c9848b368be087cb706a61603a0790124968c95b8c69f3a"
        ),
        as_committed_sha256=("0a5f90282e2039c2de0771473419715c845def154896dbd0f5747837e6241032"),
        as_committed_line_ending="LF",
        as_committed_bytes=30133,
    ),
    OfficialSourceFile(
        relative_path="model/module.py",
        sealed_sha256_crlf_normalized=(
            "a07edbadc0e96804c8158c021bbc6063bb7cc43b34d7fc470d5c8ff2005a409f"
        ),
        as_committed_sha256=("a07edbadc0e96804c8158c021bbc6063bb7cc43b34d7fc470d5c8ff2005a409f"),
        as_committed_line_ending="CRLF",
        as_committed_bytes=23426,
    ),
)


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
    coarse_vocabulary=1024,
    fine_vocabulary=1024,
    decoder_layers=4,
    model_dimension=256,
)


class InferenceSettings(BaseModel):
    """Exactly what the official predictor receives.

    Derived from ``KronosPredictor.predict`` defaults in the pinned source.
    ``auto_regressive_inference`` declares different defaults (top_p 0.99,
    sample_count 5); predict() is the entry point and its defaults govern.
    """

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 0.9
    sample_count_per_call: int = 1
    prediction_length: int = TARGET_CANDLES
    max_context: int = TOTAL_CANDLES
    clip: float = 5.0
    context_length_used: int = CONTEXT_CANDLES
    verbose: Literal[False] = False
    parameters_frozen: Literal[True] = True
    gradient_mode: Literal["inference_mode"] = "inference_mode"


OFFICIAL_INFERENCE_SETTINGS: Final[InferenceSettings] = InferenceSettings()

ROLLOUT_COUNT: Final[int] = 64

#: Fixed by the specification, not derived at execution time.
ROLLOUT_SEEDS: Final[tuple[int, ...]] = tuple(20_150_507 + index for index in range(ROLLOUT_COUNT))

#: Size-matched control sampling, fixed by v3. The seed is the window's
#: exclusive end date, distinct from the rollout base seed so the two streams
#: cannot be confused.
CONTROL_SEED: Final[int] = 20_170_518
CONTROL_REPETITIONS: Final[int] = 200


class DiagnosticThresholds(BaseModel):
    """Every number the conclusion rules compare against."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    #: Materiality only. Never permission to call a nonzero-invalidity round
    #: trip clean.
    roundtrip_material_invalid_fraction: float = 0.01
    minimum_relative_improvement: float = 0.05
    valid_rollout_support_fraction_minimum: float = 0.25
    #: Clipped input rows above this make the tokenizer reading inconclusive,
    #: because the clip distorted the input before the tokenizer saw it.
    material_clipping_row_fraction: float = 0.01
    #: Skill against persistence must exceed this to count as skill at all.
    minimum_persistence_skill: float = 0.0


THRESHOLDS: Final[DiagnosticThresholds] = DiagnosticThresholds()


def _verify_one(research_root: Path, name: str, expected: str) -> str:
    path = Path(research_root) / name
    if not path.is_file():
        raise BridgeTransformError(
            BridgeFailure(
                category=FailureCategory.INVALID_CONFIGURATION,
                code="MISSING_DIAGNOSTIC_SPECIFICATION",
                field=name,
                message=f"diagnostic specification not found: {path}",
            )
        )
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != expected:
        raise BridgeTransformError(
            BridgeFailure(
                category=FailureCategory.INVALID_CONFIGURATION,
                code="DIAGNOSTIC_SPECIFICATION_HASH_MISMATCH",
                field=name,
                message=f"{name} hashes to {observed}, expected {expected}",
            )
        )
    return observed


def verify_diagnostic_specifications(research_root: Path | str) -> dict[str, str]:
    """Verify both documents, or fail closed.

    v1, v2 and v3 are checked even though all three are superseded: they are
    preserved evidence of what was specified before, and a superseded document
    that drifted would make the supersession record meaningless. v4 is
    operative.
    """
    root = Path(research_root)
    return {
        V1_SPECIFICATION_NAME: _verify_one(root, V1_SPECIFICATION_NAME, V1_SPECIFICATION_SHA256),
        V2_SPECIFICATION_NAME: _verify_one(root, V2_SPECIFICATION_NAME, V2_SPECIFICATION_SHA256),
        V3_SPECIFICATION_NAME: _verify_one(root, V3_SPECIFICATION_NAME, V3_SPECIFICATION_SHA256),
        V4_SPECIFICATION_NAME: _verify_one(root, V4_SPECIFICATION_NAME, V4_SPECIFICATION_SHA256),
    }
