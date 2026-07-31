from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
from numpy.typing import ArrayLike, NDArray
from openalpha_sentinel.structural_validity import (
    AuditCandle,
    PathValidity,
    validate_forecast_path,
)

from .config import BridgeRepresentationConfig, VolumeMode

Timestamp = date | datetime


@dataclass(frozen=True, slots=True)
class BridgeAuditViolation:
    code: str
    sequence_index: int | None
    candle_index: int | None
    timestamp: Timestamp | None
    field: str | None
    observed_value: float | str | None
    message: str


@dataclass(frozen=True, slots=True)
class BridgeAuditResult:
    schema_version: str
    valid: bool
    sequence_count: int
    candle_count: int
    expected_sequence_length: int | None
    volume_present_count: int
    volume_missing_count: int
    timestamps_supplied: bool
    violations: tuple[BridgeAuditViolation, ...]
    sentinel_results: tuple[PathValidity, ...]
    representation_version: str
    configuration_sha256: str


def _violation(
    code: str,
    message: str,
    *,
    sequence_index: int | None = None,
    candle_index: int | None = None,
    timestamp: Timestamp | None = None,
    field: str | None = None,
    observed_value: float | str | None = None,
) -> BridgeAuditViolation:
    return BridgeAuditViolation(
        code=code,
        sequence_index=sequence_index,
        candle_index=candle_index,
        timestamp=timestamp,
        field=field,
        observed_value=observed_value,
        message=message,
    )


def _finalize(
    *,
    config: BridgeRepresentationConfig,
    sequence_count: int,
    candle_count: int,
    expected_sequence_length: int | None,
    volume_present_count: int,
    volume_missing_count: int,
    timestamps_supplied: bool,
    violations: list[BridgeAuditViolation],
    sentinel_results: list[PathValidity],
) -> BridgeAuditResult:
    return BridgeAuditResult(
        schema_version="openalpha.bridge.validation.v1",
        valid=not violations and all(result.valid for result in sentinel_results),
        sequence_count=sequence_count,
        candle_count=candle_count,
        expected_sequence_length=expected_sequence_length,
        volume_present_count=volume_present_count,
        volume_missing_count=volume_missing_count,
        timestamps_supplied=timestamps_supplied,
        violations=tuple(violations),
        sentinel_results=tuple(sentinel_results),
        representation_version=config.representation_version,
        configuration_sha256=config.canonical_sha256,
    )


def _normalize_timestamp_matrix(
    value: object,
    *,
    single_sequence: bool,
    batch_size: int,
    time_steps: int,
    code: str,
    violations: list[BridgeAuditViolation],
) -> NDArray[np.object_] | None:
    raw = np.asarray(value, dtype=object)
    expected_shape = (time_steps,) if single_sequence else (batch_size, time_steps)
    if raw.shape != expected_shape:
        violations.append(
            _violation(
                code,
                f"timestamps must have exact shape {expected_shape}",
                field="timestamps",
                observed_value=str(raw.shape),
            )
        )
        return None
    matrix = raw[np.newaxis, ...] if single_sequence else raw
    for sequence_index in range(batch_size):
        for candle_index in range(time_steps):
            item = matrix[sequence_index, candle_index]
            if not isinstance(item, (date, datetime)):
                violations.append(
                    _violation(
                        "INVALID_TIMESTAMP_TYPE",
                        "timestamp must be a date or datetime",
                        sequence_index=sequence_index,
                        candle_index=candle_index,
                        field="timestamp",
                        observed_value=type(item).__name__,
                    )
                )
                return None
    return matrix


def _strict_timestamp_comparison(current: Timestamp, previous: Timestamp) -> bool | None:
    try:
        return current > previous
    except TypeError:
        return None


def audit_bridge_output(
    *,
    config: BridgeRepresentationConfig,
    candles: ArrayLike,
    volume_present: ArrayLike | None = None,
    timestamps: object | None = None,
    expected_timestamps: object | None = None,
    expected_sequence_length: int | None = None,
) -> BridgeAuditResult:
    """Assert Bridge output independently without changing any supplied value."""

    violations: list[BridgeAuditViolation] = []
    sentinel_results: list[PathValidity] = []
    raw = np.asarray(candles)
    if raw.dtype.kind not in "fiu":
        violations.append(
            _violation(
                "NONNUMERIC_CANDLES",
                "candle output must contain real numeric values",
                field="candles",
            )
        )
        return _finalize(
            config=config,
            sequence_count=0,
            candle_count=0,
            expected_sequence_length=expected_sequence_length,
            volume_present_count=0,
            volume_missing_count=0,
            timestamps_supplied=timestamps is not None,
            violations=violations,
            sentinel_results=sentinel_results,
        )
    if raw.ndim not in (2, 3):
        violations.append(
            _violation(
                "CANDLE_RANK_MISMATCH",
                "candle output must have rank two or three",
                field="candles",
                observed_value=float(raw.ndim),
            )
        )
        return _finalize(
            config=config,
            sequence_count=0,
            candle_count=0,
            expected_sequence_length=expected_sequence_length,
            volume_present_count=0,
            volume_missing_count=0,
            timestamps_supplied=timestamps is not None,
            violations=violations,
            sentinel_results=sentinel_results,
        )

    single_sequence = raw.ndim == 2
    values = np.asarray(raw[np.newaxis, ...] if single_sequence else raw, dtype=np.float64)
    batch_size, time_steps, feature_count = values.shape
    expected_features = len(config.supported_feature_order)
    if feature_count != expected_features:
        violations.append(
            _violation(
                "FEATURE_COUNT_MISMATCH",
                f"expected {expected_features} output features, observed {feature_count}",
                field="candles",
                observed_value=float(feature_count),
            )
        )
        return _finalize(
            config=config,
            sequence_count=batch_size,
            candle_count=batch_size * time_steps,
            expected_sequence_length=expected_sequence_length,
            volume_present_count=0,
            volume_missing_count=0,
            timestamps_supplied=timestamps is not None,
            violations=violations,
            sentinel_results=sentinel_results,
        )
    if expected_sequence_length is not None and time_steps != expected_sequence_length:
        violations.append(
            _violation(
                "SEQUENCE_LENGTH_MISMATCH",
                f"expected {expected_sequence_length} candles per sequence, observed {time_steps}",
                field="time",
                observed_value=float(time_steps),
            )
        )

    expected_mask_shape = (time_steps,) if single_sequence else (batch_size, time_steps)
    presence: NDArray[np.bool_] | None
    if config.volume_mode is VolumeMode.PRICE_ONLY:
        presence = None
        if volume_present is not None:
            violations.append(
                _violation(
                    "VOLUME_MASK_UNEXPECTED",
                    "PRICE_ONLY output must not carry volume presence metadata",
                    field="volume_present",
                )
            )
    elif config.volume_mode is VolumeMode.VOLUME_REQUIRED:
        if volume_present is None:
            presence = np.ones((batch_size, time_steps), dtype=np.bool_)
        else:
            supplied = np.asarray(volume_present)
            if supplied.dtype.kind != "b" or supplied.shape != expected_mask_shape:
                violations.append(
                    _violation(
                        "VOLUME_MASK_SHAPE",
                        f"volume mask must be boolean with exact shape {expected_mask_shape}",
                        field="volume_present",
                        observed_value=str(supplied.shape),
                    )
                )
                presence = np.ones((batch_size, time_steps), dtype=np.bool_)
            else:
                presence = np.asarray(
                    supplied[np.newaxis, ...] if single_sequence else supplied,
                    dtype=np.bool_,
                )
                if not presence.all():
                    violations.append(
                        _violation(
                            "REQUIRED_VOLUME_MISSING",
                            "VOLUME_REQUIRED output must mark every candle present",
                            field="volume_present",
                        )
                    )
    else:
        if volume_present is None:
            violations.append(
                _violation(
                    "VOLUME_MASK_REQUIRED",
                    "VOLUME_OPTIONAL output requires explicit presence metadata",
                    field="volume_present",
                )
            )
            presence = np.zeros((batch_size, time_steps), dtype=np.bool_)
        else:
            supplied = np.asarray(volume_present)
            if supplied.dtype.kind != "b" or supplied.shape != expected_mask_shape:
                violations.append(
                    _violation(
                        "VOLUME_MASK_SHAPE",
                        f"volume mask must be boolean with exact shape {expected_mask_shape}",
                        field="volume_present",
                        observed_value=str(supplied.shape),
                    )
                )
                presence = np.zeros((batch_size, time_steps), dtype=np.bool_)
            else:
                presence = np.asarray(
                    supplied[np.newaxis, ...] if single_sequence else supplied,
                    dtype=np.bool_,
                )

    if presence is not None and feature_count == 5:
        for sequence_index in range(batch_size):
            for candle_index in range(time_steps):
                if not presence[sequence_index, candle_index]:
                    stored_volume = float(values[sequence_index, candle_index, 4])
                    if not math.isnan(stored_volume):
                        violations.append(
                            _violation(
                                "MISSING_VOLUME_MUST_USE_NAN",
                                "missing optional volume must use NaN storage plus a false mask",
                                sequence_index=sequence_index,
                                candle_index=candle_index,
                                field="volume",
                                observed_value=stored_volume,
                            )
                        )

    actual_matrix = None
    if timestamps is not None:
        actual_matrix = _normalize_timestamp_matrix(
            timestamps,
            single_sequence=single_sequence,
            batch_size=batch_size,
            time_steps=time_steps,
            code="TIMESTAMP_SHAPE",
            violations=violations,
        )
    expected_matrix = None
    if expected_timestamps is not None:
        expected_matrix = _normalize_timestamp_matrix(
            expected_timestamps,
            single_sequence=single_sequence,
            batch_size=batch_size,
            time_steps=time_steps,
            code="EXPECTED_TIMESTAMP_SHAPE",
            violations=violations,
        )

    if actual_matrix is not None:
        for sequence_index in range(batch_size):
            row = tuple(actual_matrix[sequence_index])
            counts = Counter(row)
            for candle_index, timestamp in enumerate(row):
                if counts[timestamp] > 1:
                    violations.append(
                        _violation(
                            "DUPLICATE_TIMESTAMP",
                            "timestamps must be unique within a sequence",
                            sequence_index=sequence_index,
                            candle_index=candle_index,
                            timestamp=timestamp,
                            field="timestamp",
                        )
                    )
                comparison = None
                if candle_index > 0:
                    comparison = _strict_timestamp_comparison(timestamp, row[candle_index - 1])
                    if comparison is None:
                        violations.append(
                            _violation(
                                "INCOMPARABLE_TIMESTAMPS",
                                "adjacent timestamps must use mutually comparable types",
                                sequence_index=sequence_index,
                                candle_index=candle_index,
                                timestamp=timestamp,
                                field="timestamp",
                            )
                        )
                if comparison is False:
                    violations.append(
                        _violation(
                            "TIMESTAMP_NOT_STRICTLY_INCREASING",
                            "timestamps must be strictly increasing",
                            sequence_index=sequence_index,
                            candle_index=candle_index,
                            timestamp=timestamp,
                            field="timestamp",
                        )
                    )

    synthetic_start = date(2000, 1, 1)
    for sequence_index in range(batch_size):
        if actual_matrix is None:
            sessions: tuple[Timestamp, ...] = tuple(
                synthetic_start + timedelta(days=index) for index in range(time_steps)
            )
        else:
            sessions = tuple(actual_matrix[sequence_index])
        if expected_matrix is not None:
            expected_sessions: tuple[Timestamp, ...] = tuple(expected_matrix[sequence_index])
        elif expected_sequence_length is not None and expected_sequence_length != time_steps:
            expected_sessions = tuple(
                synthetic_start + timedelta(days=index) for index in range(expected_sequence_length)
            )
        else:
            expected_sessions = sessions
        if not expected_sessions:
            violations.append(
                _violation(
                    "EMPTY_SEQUENCE",
                    "validator requires at least one expected candle",
                    sequence_index=sequence_index,
                    field="time",
                )
            )
            continue

        audit_rows: list[AuditCandle] = []
        for candle_index in range(time_steps):
            volume: float | None
            if presence is None or not presence[sequence_index, candle_index]:
                volume = None
            else:
                volume = float(values[sequence_index, candle_index, 4])
            audit_rows.append(
                AuditCandle(
                    session=sessions[candle_index],
                    open=float(values[sequence_index, candle_index, 0]),
                    high=float(values[sequence_index, candle_index, 1]),
                    low=float(values[sequence_index, candle_index, 2]),
                    close=float(values[sequence_index, candle_index, 3]),
                    volume=volume,
                )
            )
        cutoff_close = float(values[sequence_index, 0, 3]) if time_steps else 1.0
        if not math.isfinite(cutoff_close) or cutoff_close <= 0.0:
            cutoff_close = 1.0
        sentinel = validate_forecast_path(
            path_id=f"bridge-sequence-{sequence_index}",
            candles=audit_rows,
            expected_sessions=expected_sessions,
            cutoff_close=cutoff_close,
            cutoff_volume=1.0,
        )
        sentinel_results.append(sentinel)
        for item in sentinel.violations:
            violations.append(
                _violation(
                    item.code,
                    "Sentinel structural validation violation",
                    sequence_index=sequence_index,
                    candle_index=item.step - 1 if item.step > 0 else None,
                    timestamp=item.timestamp,
                    observed_value=item.observed_gap,
                )
            )

    volume_present_count = int(presence.sum()) if presence is not None else 0
    volume_missing_count = int(presence.size - presence.sum()) if presence is not None else 0
    return _finalize(
        config=config,
        sequence_count=batch_size,
        candle_count=batch_size * time_steps,
        expected_sequence_length=expected_sequence_length,
        volume_present_count=volume_present_count,
        volume_missing_count=volume_missing_count,
        timestamps_supplied=timestamps is not None,
        violations=violations,
        sentinel_results=sentinel_results,
    )
