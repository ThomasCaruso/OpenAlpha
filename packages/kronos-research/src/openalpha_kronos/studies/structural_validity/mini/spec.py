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

from openalpha_research.failures import FailureCategory, ResearchFailure, ResearchFailureError
from pydantic import BaseModel, ConfigDict

from ....model.assets import (
    KRONOS_MINI_SPEC,
    OFFICIAL_SOURCE_FILES,
    ForecastModelSpec,
    OfficialSourceFile,
)
from ....model.contracts import OFFICIAL_TOKEN_VOCABULARY

__all__ = [
    "CLAIM_BOUNDARY",
    "CONTEXT_CANDLES",
    "CONTROL_REPETITIONS",
    "CONTROL_SEED",
    "DIAGNOSTIC_EVIDENCE_CLASS_VALUE",
    "KRONOS_MINI_SPEC",
    "OFFICIAL_INFERENCE_SETTINGS",
    "OFFICIAL_SNAPSHOT_ALLOW_PATTERNS",
    "OFFICIAL_SOURCE_FILES",
    "OFFICIAL_TOKEN_VOCABULARY",
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

#: The only files either official repository is fetched for. Both the
#: diagnostic and the runtime probe use this exact set, and every one of
#: them is hash-verified before a weight is read. Fetching more would pull
#: files nothing verifies.
OFFICIAL_SNAPSHOT_ALLOW_PATTERNS: Final[tuple[str, ...]] = (
    "config.json",
    "model.safetensors",
)

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
        raise ResearchFailureError(
            ResearchFailure(
                category=FailureCategory.INVALID_CONFIGURATION,
                code="MISSING_DIAGNOSTIC_SPECIFICATION",
                field=name,
                message=f"diagnostic specification not found: {path}",
            )
        )
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != expected:
        raise ResearchFailureError(
            ResearchFailure(
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
