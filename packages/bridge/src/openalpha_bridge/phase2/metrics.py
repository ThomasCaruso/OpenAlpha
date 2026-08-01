"""Locked Phase 2 reconstruction metrics.

Every metric is computed strictly on the 64 scored suffix candles. The 448
warm-up context candles are excluded by the immutable score mask and cannot
influence any numerator or denominator.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from ..errors import BridgeFailure, BridgeTransformError, FailureCategory
from ..windowing import CONTEXT_PREFIX_LENGTH, EXAMPLE_LENGTH, SCORED_SUFFIX_LENGTH, score_mask

__all__ = [
    "OHLC_FIELDS",
    "ReconstructionComparison",
    "ReconstructionMethod",
    "ReconstructionMetrics",
    "SliceMetrics",
    "compute_reconstruction_metrics",
    "scored_view",
]

OHLC_FIELDS: tuple[str, ...] = ("open", "high", "low", "close")


class ReconstructionMethod(StrEnum):
    OFFICIAL_DECODER = "OFFICIAL_DECODER"
    TERMINAL_PROJECTION = "TERMINAL_PROJECTION"
    RESIDUAL_PROJECTION_BASELINE = "RESIDUAL_PROJECTION_BASELINE"
    OPENALPHA_BRIDGE_2K = "OPENALPHA_BRIDGE_2K"


def _fail(code: str, message: str) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(
            category=FailureCategory.INVALID_INPUT,
            code=code,
            message=message,
        )
    )


def scored_view(
    values: NDArray[np.float64],
    *,
    mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    """Select only the scored suffix rows.

    Accepts a full 512-row example or an already-sliced 64-row suffix. A full
    example is reduced by the locked mask, so warm-up rows can never survive
    into a metric.
    """
    if values.ndim != 2:
        raise _fail("INVALID_METRIC_RANK", f"metric input must be rank 2, got rank {values.ndim}")

    rows = values.shape[0]
    if rows == SCORED_SUFFIX_LENGTH:
        return values
    if rows != EXAMPLE_LENGTH:
        raise _fail(
            "INVALID_METRIC_LENGTH",
            f"metric input must have {EXAMPLE_LENGTH} or {SCORED_SUFFIX_LENGTH} rows, got {rows}",
        )

    applied = np.asarray(score_mask(), dtype=bool) if mask is None else mask
    if applied.shape != (EXAMPLE_LENGTH,):
        raise _fail("INVALID_METRIC_MASK", "score mask must have 512 entries")
    if bool(applied[:CONTEXT_PREFIX_LENGTH].any()) or not bool(
        applied[CONTEXT_PREFIX_LENGTH:].all()
    ):
        raise _fail("INVALID_METRIC_MASK", "score mask is not the locked 448/64 mask")
    return values[applied]


class ReconstructionMetrics(BaseModel):
    """All locked reconstruction metrics for one method over one slice."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    schema_version: Literal["openalpha.bridge.phase2.metrics.v1"] = (
        "openalpha.bridge.phase2.metrics.v1"
    )
    method: ReconstructionMethod
    scored_candles: int = Field(ge=0)
    scored_sequences: int = Field(ge=0)
    structurally_invalid_candle_fraction: float
    structurally_invalid_sequence_fraction: float
    open_mae: float
    high_mae: float
    low_mae: float
    close_mae: float
    full_ohlc_mae: float
    high_low_range_mae: float
    body_mae: float
    upper_wick_mae: float
    lower_wick_mae: float
    close_return_mae: float
    path_shape_rms_log_close: float
    error_by_suffix_step: tuple[float, ...]
    prefix_length: int = CONTEXT_PREFIX_LENGTH
    suffix_length: int = SCORED_SUFFIX_LENGTH
    score_mask_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SliceMetrics(BaseModel):
    """Per-slice reporting envelope."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    schema_version: Literal["openalpha.bridge.phase2.slice_metrics.v1"] = (
        "openalpha.bridge.phase2.slice_metrics.v1"
    )
    slice_id: str
    partition: str
    symbol: str | None
    interval: str
    volatility_regime: str | None
    metrics: ReconstructionMetrics


class ReconstructionComparison(BaseModel):
    """Paired comparison of two methods over the identical scored suffixes."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    metric: str
    method_a: ReconstructionMethod
    method_b: ReconstructionMethod
    ratio: float


def _structural_invalidity(candles: NDArray[np.float64]) -> NDArray[np.bool_]:
    open_, high, low, close = (candles[:, index] for index in range(4))
    finite = np.isfinite(candles).all(axis=1)
    positive = (candles[:, :4] > 0.0).all(axis=1)
    ordered = (
        (high >= open_) & (high >= close) & (high >= low) & (low <= open_) & (low <= close)
    )
    return ~(finite & positive & ordered)


def compute_reconstruction_metrics(
    *,
    method: ReconstructionMethod,
    predicted: NDArray[np.float64],
    actual: NDArray[np.float64],
    previous_close: NDArray[np.float64] | None = None,
    scored_sequences: int = 1,
    mask: NDArray[np.bool_] | None = None,
    score_mask_hash: str,
) -> ReconstructionMetrics:
    """Compute every locked metric on the scored suffix only.

    ``predicted`` and ``actual`` may be full 512-row examples; the mask reduces
    them to the 64 scored rows before any arithmetic occurs.
    """
    scored_predicted = scored_view(np.asarray(predicted, dtype=np.float64), mask=mask)
    scored_actual = scored_view(np.asarray(actual, dtype=np.float64), mask=mask)

    if scored_predicted.shape != scored_actual.shape:
        raise _fail(
            "METRIC_SHAPE_MISMATCH",
            f"predicted {scored_predicted.shape} does not match actual {scored_actual.shape}",
        )
    if scored_predicted.shape[1] < 4:
        raise _fail("METRIC_FIELD_COUNT", "reconstruction metrics need at least OHLC columns")

    # Every price error is divided by the target row's own close.
    denominator = scored_actual[:, 3]
    if not np.all(denominator > 0.0):
        raise _fail("NON_POSITIVE_METRIC_DENOMINATOR", "target close must be strictly positive")

    absolute = np.abs(scored_predicted[:, :4] - scored_actual[:, :4])
    relative = absolute / denominator[:, None]

    predicted_range = scored_predicted[:, 1] - scored_predicted[:, 2]
    actual_range = scored_actual[:, 1] - scored_actual[:, 2]
    range_mae = float(np.mean(np.abs(predicted_range - actual_range) / denominator))

    predicted_body = np.abs(scored_predicted[:, 3] - scored_predicted[:, 0])
    actual_body = np.abs(scored_actual[:, 3] - scored_actual[:, 0])
    body_mae = float(np.mean(np.abs(predicted_body - actual_body) / denominator))

    predicted_upper = scored_predicted[:, 1] - np.maximum(
        scored_predicted[:, 0], scored_predicted[:, 3]
    )
    actual_upper = scored_actual[:, 1] - np.maximum(scored_actual[:, 0], scored_actual[:, 3])
    predicted_lower = np.minimum(scored_predicted[:, 0], scored_predicted[:, 3]) - scored_predicted[:, 2]
    actual_lower = np.minimum(scored_actual[:, 0], scored_actual[:, 3]) - scored_actual[:, 2]

    if previous_close is None:
        anchor = np.concatenate([scored_actual[:1, 3], scored_actual[:-1, 3]])
    else:
        anchor = np.asarray(previous_close, dtype=np.float64)
        if anchor.shape != (scored_actual.shape[0],):
            raise _fail("INVALID_ANCHOR_SHAPE", "previous_close must align with the scored suffix")

    predicted_return = np.log(scored_predicted[:, 3] / anchor)
    actual_return = np.log(scored_actual[:, 3] / anchor)
    close_return_mae = float(np.mean(np.abs(predicted_return - actual_return)))

    log_ratio = np.log(scored_predicted[:, 3] / scored_actual[:, 3])
    path_shape = float(np.sqrt(np.mean(np.square(log_ratio))))

    invalid = _structural_invalidity(scored_predicted)
    per_step = np.mean(relative, axis=1)

    return ReconstructionMetrics(
        method=method,
        scored_candles=int(scored_actual.shape[0]),
        scored_sequences=scored_sequences,
        structurally_invalid_candle_fraction=float(np.mean(invalid)),
        structurally_invalid_sequence_fraction=float(bool(invalid.any())),
        open_mae=float(np.mean(relative[:, 0])),
        high_mae=float(np.mean(relative[:, 1])),
        low_mae=float(np.mean(relative[:, 2])),
        close_mae=float(np.mean(relative[:, 3])),
        full_ohlc_mae=float(np.mean(relative)),
        high_low_range_mae=range_mae,
        body_mae=body_mae,
        upper_wick_mae=float(np.mean(np.abs(predicted_upper - actual_upper) / denominator)),
        lower_wick_mae=float(np.mean(np.abs(predicted_lower - actual_lower) / denominator)),
        close_return_mae=close_return_mae,
        path_shape_rms_log_close=path_shape,
        error_by_suffix_step=tuple(float(value) for value in per_step),
        score_mask_sha256=score_mask_hash,
    )
