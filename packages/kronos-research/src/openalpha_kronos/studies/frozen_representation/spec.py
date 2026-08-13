"""Pinned identity for the frozen-representation probe.

Every constant here mirrors a value in the sealed preregistration. Nothing is
computed from a result, and nothing may be changed after the seal: the
specification digest is verified inside the container before any work begins,
so a drifted document fails the run closed rather than producing a number.

The frozen pair is imported from the base study so there is one source of truth
for what Kronos-base is, and the digests this probe expects are ALSO written out
here as literals and checked against the import by :func:`verify_pinned_pair`.
"""

from __future__ import annotations

import hashlib
import re
from datetime import date, datetime
from pathlib import Path
from typing import Final, Literal

from openalpha_research.failures import FailureCategory, ResearchFailure, ResearchFailureError
from pydantic import BaseModel, ConfigDict, Field

from openalpha_kronos.model.assets import KRONOS_BASE_SPEC, KRONOS_BASE_TOKENIZER_SPEC

__all__ = [
    "ASSET_PANEL",
    "CONTEXT_CANDLES",
    "EMBARGO_ORDINALS",
    "ENGINEERED_FEATURE_NAMES",
    "FEATURE_SET_IDS",
    "HORIZON_CANDLES",
    "LOGISTIC_C_GRID",
    "MINIMUM_TEST_ORIGINS_PER_ASSET",
    "MINIMUM_TEST_SAMPLES",
    "MINIMUM_TEST_SESSIONS",
    "PROBE_ARTIFACT_ROOT",
    "PROBE_CLAIM_BOUNDARY",
    "PROBE_EXPERIMENT_ID",
    "PROBE_FAILURE_SCHEMA_VERSION",
    "PROBE_FIT_OBJECT_NAME",
    "PROBE_FIT_SCHEMA_VERSION",
    "PROBE_PROBE_SCHEMA_VERSION",
    "PROBE_RUN_ID_PATTERN",
    "PROBE_SPECIFICATION_NAME",
    "PROBE_SPECIFICATION_SHA256",
    "PROBE_SUCCESS_SCHEMA_VERSION",
    "REPRESENTATION_DIMENSION",
    "RIDGE_ALPHA_GRID",
    "SEALED_AT_UTC",
    "SEALING_COMMIT",
    "STRIDE",
    "TEST_START_INCLUSIVE",
    "TRAIN_ORDINALS",
    "TRAIN_VALIDATION_END_EXCLUSIVE",
    "TRAIN_VALIDATION_ORIGIN_OFFSETS",
    "TRAIN_VALIDATION_SESSIONS",
    "TRAIN_VALIDATION_START_INCLUSIVE",
    "VALIDATION_ORDINALS",
    "ProbeThresholds",
    "probe_fit_key",
    "probe_test_key",
    "prove_probe_budget",
    "verify_pinned_pair",
    "verify_probe_specification",
]

PROBE_EXPERIMENT_ID: Final = "openalpha-kronos-frozen-representation-probe-v1"

PROBE_SPECIFICATION_NAME: Final[str] = "kronos-frozen-representation-probe-v1.yaml"
PROBE_SPECIFICATION_SHA256: Final[str] = (
    "3c3656f03e6e0abaef7001cec95f71ca4678943aeea6567aba33a051edb95ece"
)

#: The commit that first contained the corrected design. The test boundary is
#: derived from its timestamp, not chosen.
SEALING_COMMIT: Final[str] = "1a6fad7a57cbddbdaeecdb1713dfb7b93f971de0"
SEALED_AT_UTC: Final[str] = "2026-08-04T22:06:45+00:00"

#: A fourth root. Shares no prefix with any completed study's subtree.
PROBE_ARTIFACT_ROOT: Final[str] = "openalpha-compatibility/kronos-representation-probe"
PROBE_FIT_OBJECT_NAME: Final[str] = "kronos_representation_probe_fit.json"
PROBE_TEST_OBJECT_NAME: Final[str] = "kronos_representation_probe_terminal.json"

#: ``frp_`` plus 8 to 32 lowercase hex. Disjoint from canary_, base_ and zsb_.
PROBE_RUN_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^frp_[0-9a-f]{8,32}$")

PROBE_FIT_SCHEMA_VERSION: Final = "openalpha.bridge.representation_probe.kronos_frozen_fit.v1"
PROBE_SUCCESS_SCHEMA_VERSION: Final = (
    "openalpha.bridge.representation_probe.kronos_frozen_probe.v1"
)
PROBE_FAILURE_SCHEMA_VERSION: Final = (
    "openalpha.bridge.representation_probe.kronos_frozen_failure.v1"
)
PROBE_PROBE_SCHEMA_VERSION: Final = "openalpha.bridge.representation_probe.runtime_probe.v1"

PROBE_CLAIM_BOUNDARY: Final = "DEVELOPMENT PROBE - NOT TRADING EVIDENCE"

# --------------------------------------------------------------- frozen pair

PROBE_MODEL_SPEC: Final = KRONOS_BASE_SPEC
PROBE_TOKENIZER_SPEC: Final = KRONOS_BASE_TOKENIZER_SPEC

EXPECTED_MODEL_REPOSITORY: Final[str] = "NeoQuasar/Kronos-base"
EXPECTED_MODEL_REVISION: Final[str] = "2b554741eca47781b64468546e77fef3e85130e6"
EXPECTED_MODEL_CONFIG_SHA256: Final[str] = (
    "77ebc3038b647709b92be002f801d72e1a385f4c8c2c5aa1cc6cf21fcfe44eb2"
)
EXPECTED_MODEL_WEIGHTS_SHA256: Final[str] = (
    "abff193acab6db1a0368e9773e75799d11403b6d054ee6d5f0a11aeabc5f4b83"
)
EXPECTED_TOKENIZER_REPOSITORY: Final[str] = "NeoQuasar/Kronos-Tokenizer-base"
EXPECTED_TOKENIZER_REVISION: Final[str] = "0e0117387f39004a9016484a186a908917e22426"
EXPECTED_TOKENIZER_CONFIG_SHA256: Final[str] = (
    "2366e7ccfec76cbc19cf3c4c1b9c5d901be336ca1e83f2d2292c9bff381b77a2"
)
EXPECTED_TOKENIZER_WEIGHTS_SHA256: Final[str] = (
    "59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee"
)
EXPECTED_TOTAL_PARAMETERS: Final[int] = 106_268_634
REQUIRED_TRAINABLE_PARAMETERS: Final[int] = 0

#: Reused, never duplicated.
PROBE_CACHE_VOLUME: Final[str] = "openalpha-kronos-base-cache"

# ------------------------------------------------------------------ geometry

ASSET_PANEL: Final[tuple[str, ...]] = ("SPY", "QQQ", "IWM", "DIA")
FREQUENCY: Final[str] = "1d"
CALENDAR: Final[str] = "XNYS"

CONTEXT_CANDLES: Final[int] = 40
HORIZON_CANDLES: Final[int] = 12
WINDOW_CANDLES: Final[int] = CONTEXT_CANDLES + HORIZON_CANDLES
MAXIMUM_CONTEXT: Final[int] = 512

#: Non-overlapping: stride equals the horizon, so target windows are disjoint.
STRIDE: Final[int] = HORIZON_CANDLES

REPRESENTATION_DIMENSION: Final[int] = 832

# ------------------------------------------------------- train / validation

TRAIN_VALIDATION_START_INCLUSIVE: Final[str] = "2025-01-02"
TRAIN_VALIDATION_END_EXCLUSIVE: Final[str] = "2026-05-13"
TRAIN_VALIDATION_SESSIONS: Final[int] = 340
TRAIN_VALIDATION_ORIGINS_PER_ASSET: Final[int] = 25

TRAIN_VALIDATION_ORIGIN_OFFSETS: Final[tuple[int, ...]] = tuple(
    CONTEXT_CANDLES + STRIDE * index for index in range(TRAIN_VALIDATION_ORIGINS_PER_ASSET)
)

TRAIN_ORDINALS: Final[tuple[int, ...]] = tuple(range(16))
EMBARGO_ORDINALS: Final[tuple[int, ...]] = tuple(range(16, 20))
VALIDATION_ORDINALS: Final[tuple[int, ...]] = tuple(range(20, 25))

#: ceil(context / stride) = ceil(40 / 12) = 4. A training origin's context
#: reaches back ~3.3 strides, so four dropped origins guarantee no training
#: context touches a validation target.
EMBARGO_LENGTH: Final[int] = -(-CONTEXT_CANDLES // STRIDE)

# ---------------------------------------------------------------- test gate

#: Derived from the sealing timestamp, never chosen. The first XNYS session
#: whose 09:30 ET open is strictly after 2026-08-04T22:06:45Z.
TEST_START_INCLUSIVE: Final[str] = "2026-08-05"
EXCLUDED_GAP_START_INCLUSIVE: Final[str] = "2026-05-13"
EXCLUDED_GAP_END_EXCLUSIVE: Final[str] = "2026-08-05"

MINIMUM_TEST_ORIGINS_PER_ASSET: Final[int] = 16
MINIMUM_TEST_SAMPLES: Final[int] = MINIMUM_TEST_ORIGINS_PER_ASSET * len(ASSET_PANEL)
MINIMUM_TEST_SESSIONS: Final[int] = (
    CONTEXT_CANDLES + MINIMUM_TEST_ORIGINS_PER_ASSET * STRIDE
)

# --------------------------------------------------------------- downstream

RIDGE_ALPHA_GRID: Final[tuple[float, ...]] = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
LOGISTIC_C_GRID: Final[tuple[float, ...]] = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)

FEATURE_SET_IDS: Final[tuple[str, ...]] = (
    "kronos_hidden_state",
    "raw_ohlcv",
    "engineered_features",
    "intercept_only",
)
CANDIDATE_FEATURE_SET: Final[str] = "kronos_hidden_state"
CONTROL_FEATURE_SETS: Final[tuple[str, ...]] = (
    "raw_ohlcv",
    "engineered_features",
    "intercept_only",
)

FEATURE_SET_DIMENSIONS: Final[dict[str, int]] = {
    "kronos_hidden_state": REPRESENTATION_DIMENSION,
    "raw_ohlcv": CONTEXT_CANDLES * 6,
    "engineered_features": 12,
    "intercept_only": 0,
}

#: Exactly the twelve the specification enumerates, in this order.
ENGINEERED_RETURN_LAGS: Final[tuple[int, ...]] = (1, 2, 3, 5, 10, 20)
ENGINEERED_VOLATILITY_WINDOWS: Final[tuple[int, ...]] = (5, 10, 20)
ENGINEERED_RANGE_WINDOWS: Final[tuple[int, ...]] = (5, 20)
ENGINEERED_VOLUME_ZSCORE_WINDOW: Final[int] = 20

ENGINEERED_FEATURE_NAMES: Final[tuple[str, ...]] = (
    *(f"log_return_lag_{lag}" for lag in ENGINEERED_RETURN_LAGS),
    *(f"realized_volatility_{w}" for w in ENGINEERED_VOLATILITY_WINDOWS),
    *(f"range_mean_{w}" for w in ENGINEERED_RANGE_WINDOWS),
    f"volume_zscore_{ENGINEERED_VOLUME_ZSCORE_WINDOW}",
)

# ------------------------------------------------------------------ metrics

PRIMARY_METRIC: Final[str] = "horizon_return_mae"
SECONDARY_METRICS: Final[tuple[str, ...]] = (
    "horizon_return_rmse",
    "out_of_sample_r2",
    "directional_accuracy",
    "directional_auc",
)

BOOTSTRAP_METHOD: Final[str] = "paired_percentile_moving_block_bootstrap_over_origin_clusters"
BOOTSTRAP_BLOCK_LENGTH: Final[int] = -(-CONTEXT_CANDLES // STRIDE)
BOOTSTRAP_RESAMPLES: Final[int] = 2000
BOOTSTRAP_CONFIDENCE_LEVEL: Final[float] = 0.95
BOOTSTRAP_SEED: Final[int] = 20260803


class ProbeThresholds(BaseModel):
    """Every threshold the decision rules use. Fixed before execution."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    minimum_relative_improvement_over_every_control: float = 0.05
    bootstrap_confidence_level: float = BOOTSTRAP_CONFIDENCE_LEVEL
    minimum_test_samples: int = MINIMUM_TEST_SAMPLES
    minimum_supporting_assets: int = 3
    total_assets: int = len(ASSET_PANEL)


PROBE_THRESHOLDS: Final[ProbeThresholds] = ProbeThresholds()


def probe_fit_key(run_id: str) -> str:
    """The fit artifact key. Published before the test partition is opened."""
    return f"{PROBE_ARTIFACT_ROOT}/runs/{run_id}/{PROBE_FIT_OBJECT_NAME}"


def probe_test_key(run_id: str) -> str:
    """The terminal artifact key."""
    return f"{PROBE_ARTIFACT_ROOT}/runs/{run_id}/{PROBE_TEST_OBJECT_NAME}"


def _fail(code: str, message: str, *, field: str | None = None) -> ResearchFailureError:
    return ResearchFailureError(
        ResearchFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            field=field,
            message=message,
        )
    )


def prove_probe_budget(
    *,
    context_candles: int = CONTEXT_CANDLES,
    horizon_candles: int = HORIZON_CANDLES,
    maximum_context: int = MAXIMUM_CONTEXT,
) -> dict[str, int | bool]:
    """The extraction window fits the budget with nothing truncated.

    Only the 40 context rows are ever encoded here -- the probe generates
    nothing -- but the horizon is included so the recorded window matches the
    geometry the zero-shot benchmark used.
    """
    total = context_candles + horizon_candles
    if context_candles <= 0 or horizon_candles <= 0:
        raise _fail(
            "PROBE_BUDGET_INVALID",
            f"context and horizon must both be positive, got {context_candles} and "
            f"{horizon_candles}",
        )
    if total > maximum_context:
        raise _fail(
            "PROBE_BUDGET_EXCEEDED",
            f"{context_candles} + {horizon_candles} = {total} exceeds {maximum_context}",
        )
    return {
        "context_candles": context_candles,
        "horizon_candles": horizon_candles,
        "maximum_context": maximum_context,
        "sum": total,
        "encoded_rows": context_candles,
        "truncation_occurs": False,
    }


def verify_pinned_pair() -> dict[str, str]:
    """The imported pin must still be the pair this probe preregistered."""
    checks: tuple[tuple[str, object, object], ...] = (
        ("model_repository", PROBE_MODEL_SPEC.repository, EXPECTED_MODEL_REPOSITORY),
        ("model_revision", PROBE_MODEL_SPEC.revision, EXPECTED_MODEL_REVISION),
        ("model_config_sha256", PROBE_MODEL_SPEC.config_sha256, EXPECTED_MODEL_CONFIG_SHA256),
        ("model_weights_sha256", PROBE_MODEL_SPEC.weights_sha256, EXPECTED_MODEL_WEIGHTS_SHA256),
        (
            "tokenizer_repository",
            PROBE_TOKENIZER_SPEC.repository,
            EXPECTED_TOKENIZER_REPOSITORY,
        ),
        ("tokenizer_revision", PROBE_TOKENIZER_SPEC.revision, EXPECTED_TOKENIZER_REVISION),
        (
            "tokenizer_config_sha256",
            PROBE_TOKENIZER_SPEC.config_sha256,
            EXPECTED_TOKENIZER_CONFIG_SHA256,
        ),
        (
            "tokenizer_weights_sha256",
            PROBE_TOKENIZER_SPEC.weights_sha256,
            EXPECTED_TOKENIZER_WEIGHTS_SHA256,
        ),
    )
    for field, observed, expected in checks:
        if observed != expected:
            raise _fail(
                "PROBE_PINNED_PAIR_MISMATCH",
                f"{field} is {observed!r}, but this probe preregistered {expected!r}",
                field=field,
            )
    if PROBE_MODEL_SPEC.paired_tokenizer_repository != EXPECTED_TOKENIZER_REPOSITORY:
        raise _fail(
            "PROBE_PINNED_PAIR_MISMATCH",
            "the model records a different paired tokenizer than this probe expects",
            field="paired_tokenizer_repository",
        )
    return {
        "model_repository": EXPECTED_MODEL_REPOSITORY,
        "model_revision": EXPECTED_MODEL_REVISION,
        "tokenizer_repository": EXPECTED_TOKENIZER_REPOSITORY,
        "tokenizer_revision": EXPECTED_TOKENIZER_REVISION,
    }


def verify_probe_specification(research_root: Path | str) -> dict[str, str]:
    """Verify this probe's preregistration, or fail closed."""
    path = Path(research_root) / PROBE_SPECIFICATION_NAME
    if not path.is_file():
        raise _fail(
            "MISSING_PROBE_SPECIFICATION",
            f"probe specification not found: {path}",
            field=PROBE_SPECIFICATION_NAME,
        )
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != PROBE_SPECIFICATION_SHA256:
        raise _fail(
            "PROBE_SPECIFICATION_HASH_MISMATCH",
            f"{PROBE_SPECIFICATION_NAME} hashes to {observed}, expected "
            f"{PROBE_SPECIFICATION_SHA256}",
            field=PROBE_SPECIFICATION_NAME,
        )
    return {PROBE_SPECIFICATION_NAME: observed}


def verify_partition_geometry() -> dict[str, int]:
    """The split arithmetic is self-consistent before anything is fitted."""
    ordinals = (*TRAIN_ORDINALS, *EMBARGO_ORDINALS, *VALIDATION_ORDINALS)
    if ordinals != tuple(range(TRAIN_VALIDATION_ORIGINS_PER_ASSET)):
        raise _fail(
            "PROBE_PARTITION_GEOMETRY_INVALID",
            "train, embargo and validation ordinals must tile 0..24 exactly once",
        )
    if len(EMBARGO_ORDINALS) != EMBARGO_LENGTH:
        raise _fail(
            "PROBE_PARTITION_GEOMETRY_INVALID",
            f"embargo is {len(EMBARGO_ORDINALS)} origins, expected {EMBARGO_LENGTH}",
        )
    if STRIDE < HORIZON_CANDLES:
        raise _fail(
            "PROBE_PARTITION_GEOMETRY_INVALID",
            "stride shorter than the horizon would overlap target windows",
        )
    needed = TRAIN_VALIDATION_ORIGIN_OFFSETS[-1] + HORIZON_CANDLES
    if needed != TRAIN_VALIDATION_SESSIONS:
        raise _fail(
            "PROBE_PARTITION_GEOMETRY_INVALID",
            f"last target row ends at {needed}, but {TRAIN_VALIDATION_SESSIONS} declared",
        )
    return {
        "train_origins": len(TRAIN_ORDINALS),
        "embargo_origins": len(EMBARGO_ORDINALS),
        "validation_origins": len(VALIDATION_ORDINALS),
        "train_samples": len(TRAIN_ORDINALS) * len(ASSET_PANEL),
        "validation_samples": len(VALIDATION_ORDINALS) * len(ASSET_PANEL),
        "stride": STRIDE,
        "embargo_length": EMBARGO_LENGTH,
    }


class ProbeWindow(BaseModel):
    """The declared windows, as a checkable object."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    assets: tuple[str, ...] = ASSET_PANEL
    frequency: Literal["1d"] = "1d"
    calendar: Literal["XNYS"] = "XNYS"
    train_validation_start_inclusive: str = TRAIN_VALIDATION_START_INCLUSIVE
    train_validation_end_exclusive: str = TRAIN_VALIDATION_END_EXCLUSIVE
    train_validation_sessions: int = Field(default=TRAIN_VALIDATION_SESSIONS, gt=0)
    test_start_inclusive: str = TEST_START_INCLUSIVE
    minimum_test_sessions: int = Field(default=MINIMUM_TEST_SESSIONS, gt=0)
    context_candles: int = CONTEXT_CANDLES
    horizon_candles: int = HORIZON_CANDLES
    stride: int = STRIDE


PROBE_WINDOW: Final[ProbeWindow] = ProbeWindow()


def test_boundary_date() -> date:
    return date.fromisoformat(TEST_START_INCLUSIVE)


def sealed_at() -> datetime:
    return datetime.fromisoformat(SEALED_AT_UTC)


__all__ += [
    "BOOTSTRAP_BLOCK_LENGTH",
    "BOOTSTRAP_CONFIDENCE_LEVEL",
    "BOOTSTRAP_METHOD",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "CALENDAR",
    "CANDIDATE_FEATURE_SET",
    "CONTROL_FEATURE_SETS",
    "EMBARGO_LENGTH",
    "EXCLUDED_GAP_END_EXCLUSIVE",
    "EXCLUDED_GAP_START_INCLUSIVE",
    "EXPECTED_TOTAL_PARAMETERS",
    "FEATURE_SET_DIMENSIONS",
    "FREQUENCY",
    "PRIMARY_METRIC",
    "PROBE_CACHE_VOLUME",
    "PROBE_MODEL_SPEC",
    "PROBE_TEST_OBJECT_NAME",
    "PROBE_THRESHOLDS",
    "PROBE_TOKENIZER_SPEC",
    "PROBE_WINDOW",
    "REQUIRED_TRAINABLE_PARAMETERS",
    "SECONDARY_METRICS",
    "TRAIN_VALIDATION_ORIGINS_PER_ASSET",
    "WINDOW_CANDLES",
    "ProbeWindow",
    "sealed_at",
    "test_boundary_date",
    "verify_partition_geometry",
]
