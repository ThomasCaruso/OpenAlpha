from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence

import numpy as np
from pydantic import Field

from .contracts import FrozenModel
from .structural_validity import AuditCandle


class PathQualityMetrics(FrozenModel):
    normalized_ohlc_mae: float = Field(ge=0.0)
    open_mae: float = Field(ge=0.0)
    high_mae: float = Field(ge=0.0)
    low_mae: float = Field(ge=0.0)
    close_mae: float = Field(ge=0.0)
    high_low_range_mae: float = Field(ge=0.0)
    candle_body_mae: float = Field(ge=0.0)
    upper_wick_mae: float = Field(ge=0.0)
    lower_wick_mae: float = Field(ge=0.0)
    range_direction_accuracy: float = Field(ge=0.0, le=1.0)
    path_shape_distance: float = Field(ge=0.0)
    error_by_horizon_step: tuple[float, ...]
    parkinson_volatility_error: float = Field(ge=0.0)
    cumulative_range_error: float = Field(ge=0.0)
    five_session_close_return_error: float = Field(ge=0.0)
    direction_correct: bool


class BarrierEventMetric(FrozenModel):
    barrier_fraction: float = Field(gt=0.0)
    predicted_positive_touch: bool
    realized_positive_touch: bool
    positive_touch_correct: bool
    predicted_negative_touch: bool
    realized_negative_touch: bool
    negative_touch_correct: bool
    predicted_either_touch: bool
    realized_either_touch: bool
    either_touch_correct: bool


class DiversityMetrics(FrozenModel):
    mean_pairwise_path_distance: float = Field(ge=0.0)
    final_return_variance: float = Field(ge=0.0)
    repeated_path_rate: float = Field(ge=0.0, le=1.0)


def path_quality_metrics(
    *,
    predicted: Sequence[AuditCandle],
    realized: Sequence[AuditCandle],
    cutoff_close: float,
    cutoff_high: float,
    cutoff_low: float,
) -> PathQualityMetrics:
    forecast = tuple(predicted)
    actual = tuple(realized)
    if len(forecast) != 5 or len(actual) != 5:
        raise ValueError("path quality requires two five-candle paths")
    if not math.isfinite(cutoff_close) or cutoff_close <= 0.0:
        raise ValueError("cutoff close must be finite and positive")
    forecast_values = _ohlc_array(forecast)
    actual_values = _ohlc_array(actual)
    absolute = np.abs(forecast_values - actual_values) / cutoff_close
    forecast_range = forecast_values[:, 1] - forecast_values[:, 2]
    actual_range = actual_values[:, 1] - actual_values[:, 2]
    cutoff_range = cutoff_high - cutoff_low
    forecast_body = forecast_values[:, 3] - forecast_values[:, 0]
    actual_body = actual_values[:, 3] - actual_values[:, 0]
    forecast_upper = forecast_values[:, 1] - np.maximum(
        forecast_values[:, 0], forecast_values[:, 3]
    )
    actual_upper = actual_values[:, 1] - np.maximum(
        actual_values[:, 0], actual_values[:, 3]
    )
    forecast_lower = np.minimum(forecast_values[:, 0], forecast_values[:, 3]) - (
        forecast_values[:, 2]
    )
    actual_lower = np.minimum(actual_values[:, 0], actual_values[:, 3]) - actual_values[:, 2]
    range_directions = np.sign(forecast_range - cutoff_range) == np.sign(
        actual_range - cutoff_range
    )
    predicted_return = math.log(forecast[-1].close / cutoff_close)
    realized_return = math.log(actual[-1].close / cutoff_close)
    return PathQualityMetrics(
        normalized_ohlc_mae=float(np.mean(absolute)),
        open_mae=float(np.mean(absolute[:, 0])),
        high_mae=float(np.mean(absolute[:, 1])),
        low_mae=float(np.mean(absolute[:, 2])),
        close_mae=float(np.mean(absolute[:, 3])),
        high_low_range_mae=float(np.mean(np.abs(forecast_range - actual_range)) / cutoff_close),
        candle_body_mae=float(np.mean(np.abs(forecast_body - actual_body)) / cutoff_close),
        upper_wick_mae=float(np.mean(np.abs(forecast_upper - actual_upper)) / cutoff_close),
        lower_wick_mae=float(np.mean(np.abs(forecast_lower - actual_lower)) / cutoff_close),
        range_direction_accuracy=float(np.mean(range_directions)),
        path_shape_distance=float(np.sqrt(np.mean(np.square(absolute)))),
        error_by_horizon_step=tuple(float(value) for value in np.mean(absolute, axis=1)),
        parkinson_volatility_error=abs(
            _parkinson_volatility(forecast_values) - _parkinson_volatility(actual_values)
        ),
        cumulative_range_error=float(abs(np.sum(forecast_range) - np.sum(actual_range)) / cutoff_close),
        five_session_close_return_error=abs(predicted_return - realized_return),
        direction_correct=_direction(predicted_return) == _direction(realized_return),
    )


def barrier_event_metrics(
    *,
    predicted: Sequence[AuditCandle],
    realized: Sequence[AuditCandle],
    cutoff_close: float,
    barriers: Sequence[float],
) -> tuple[BarrierEventMetric, ...]:
    forecast = tuple(predicted)
    actual = tuple(realized)
    if len(forecast) != 5 or len(actual) != 5:
        raise ValueError("barrier metrics require two five-candle paths")
    results = []
    for barrier in barriers:
        if not math.isfinite(barrier) or barrier <= 0.0:
            raise ValueError("barriers must be finite and positive")
        upper = cutoff_close * (1.0 + barrier)
        lower = cutoff_close * (1.0 - barrier)
        predicted_positive = any(item.high >= upper for item in forecast)
        realized_positive = any(item.high >= upper for item in actual)
        predicted_negative = any(item.low <= lower for item in forecast)
        realized_negative = any(item.low <= lower for item in actual)
        results.append(
            BarrierEventMetric(
                barrier_fraction=barrier,
                predicted_positive_touch=predicted_positive,
                realized_positive_touch=realized_positive,
                positive_touch_correct=predicted_positive == realized_positive,
                predicted_negative_touch=predicted_negative,
                realized_negative_touch=realized_negative,
                negative_touch_correct=predicted_negative == realized_negative,
                predicted_either_touch=predicted_positive or predicted_negative,
                realized_either_touch=realized_positive or realized_negative,
                either_touch_correct=(predicted_positive or predicted_negative)
                == (realized_positive or realized_negative),
            )
        )
    return tuple(results)


def diversity_metrics(
    *,
    paths: Sequence[Sequence[AuditCandle]],
    cutoff_close: float,
) -> DiversityMetrics:
    materialized = tuple(tuple(path) for path in paths)
    if len(materialized) < 2 or any(len(path) != 5 for path in materialized):
        raise ValueError("diversity requires at least two five-candle paths")
    arrays = tuple(_ohlc_array(path) / cutoff_close for path in materialized)
    distances = [
        float(np.sqrt(np.mean(np.square(arrays[left] - arrays[right]))))
        for left in range(len(arrays))
        for right in range(left + 1, len(arrays))
    ]
    returns = np.asarray(
        [math.log(path[-1].close / cutoff_close) for path in materialized],
        dtype=float,
    )
    hashes = tuple(_path_hash(path) for path in materialized)
    repeated = len(hashes) - len(set(hashes))
    return DiversityMetrics(
        mean_pairwise_path_distance=float(np.mean(distances)),
        final_return_variance=float(np.var(returns)),
        repeated_path_rate=repeated / len(hashes),
    )


def path_distance(
    *,
    left: Sequence[AuditCandle],
    right: Sequence[AuditCandle],
    cutoff_close: float,
) -> float:
    left_path = tuple(left)
    right_path = tuple(right)
    if len(left_path) != 5 or len(right_path) != 5:
        raise ValueError("path distance requires two five-candle paths")
    if not math.isfinite(cutoff_close) or cutoff_close <= 0.0:
        raise ValueError("cutoff close must be finite and positive")
    difference = (_ohlc_array(left_path) - _ohlc_array(right_path)) / cutoff_close
    return float(np.sqrt(np.mean(np.square(difference))))


def _ohlc_array(path: Sequence[AuditCandle]) -> np.ndarray:
    values = np.asarray(
        [[item.open, item.high, item.low, item.close] for item in path],
        dtype=float,
    )
    if not np.isfinite(values).all():
        raise ValueError("path metric inputs must be finite")
    return values


def _parkinson_volatility(values: np.ndarray) -> float:
    if np.any(values[:, 1] <= 0.0) or np.any(values[:, 2] <= 0.0):
        raise ValueError("Parkinson volatility requires positive high and low")
    return float(
        np.sqrt(np.sum(np.square(np.log(values[:, 1] / values[:, 2]))) / (4 * len(values) * np.log(2)))
    )


def _direction(value: float) -> int:
    return 1 if value > 0.0 else -1 if value < 0.0 else 0


def _path_hash(path: Sequence[AuditCandle]) -> str:
    body = [
        {
            "session": item.session.isoformat() if item.session is not None else None,
            "open": item.open,
            "high": item.high,
            "low": item.low,
            "close": item.close,
            "volume": item.volume,
        }
        for item in path
    ]
    encoded = json.dumps(body, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
