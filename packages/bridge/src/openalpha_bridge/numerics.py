from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class RoundTripTolerance:
    absolute: float
    relative: float
    log_space: float
    recursive_relative: float


@dataclass(frozen=True, slots=True)
class FieldNumericalError:
    field: str
    compared_count: int
    maximum_absolute_error: float
    maximum_relative_error: float
    maximum_log_error: float
    tolerance: RoundTripTolerance
    passes: bool


@dataclass(frozen=True, slots=True)
class NumericalAudit:
    schema_version: str
    dtype: Literal["float32", "float64"]
    sequence_count: int
    sequence_length: int
    field_errors: tuple[FieldNumericalError, ...]
    maximum_absolute_error: float
    maximum_relative_error: float
    maximum_log_error: float
    maximum_recursive_close_drift: float
    final_recursive_close_drift: float
    passes: bool


def roundtrip_tolerance(dtype: str, field: str) -> RoundTripTolerance:
    if dtype == "float64":
        return RoundTripTolerance(
            absolute=1.0e-12,
            relative=1.0e-12,
            log_space=1.0e-12,
            recursive_relative=1.0e-12,
        )
    if dtype != "float32":
        raise ValueError("round-trip dtype must be float32 or float64")
    if field == "volume":
        return RoundTripTolerance(
            absolute=1.0e-3,
            relative=5.0e-5,
            log_space=1.0e-4,
            recursive_relative=1.0e-4,
        )
    return RoundTripTolerance(
        absolute=1.0e-6,
        relative=2.0e-5,
        log_space=1.0e-4,
        recursive_relative=1.0e-4,
    )


def _normalized_arrays(
    reference: ArrayLike,
    reconstructed: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], bool, str]:
    reference_raw = np.asarray(reference)
    reconstructed_raw = np.asarray(reconstructed)
    if reference_raw.dtype.kind not in "fi" or reconstructed_raw.dtype.kind not in "fi":
        raise TypeError("round-trip arrays must have real floating dtype")
    if reference_raw.shape != reconstructed_raw.shape:
        raise ValueError("reference and reconstructed shapes must match exactly")
    if reference_raw.ndim not in (2, 3):
        raise ValueError("round-trip arrays must have rank two or three")
    if reference_raw.shape[-1] not in (4, 5):
        raise ValueError("round-trip arrays must contain ordered OHLC or OHLCV")
    dtype = reconstructed_raw.dtype.name
    if dtype not in ("float32", "float64"):
        raise ValueError("reconstructed array dtype must be float32 or float64")
    single_sequence = reference_raw.ndim == 2
    reference_values = np.asarray(
        reference_raw[np.newaxis, ...] if single_sequence else reference_raw,
        dtype=np.float64,
    )
    reconstructed_values = np.asarray(
        reconstructed_raw[np.newaxis, ...] if single_sequence else reconstructed_raw,
        dtype=np.float64,
    )
    return reference_values, reconstructed_values, single_sequence, dtype


def numerical_roundtrip_audit(
    *,
    reference: ArrayLike,
    reconstructed: ArrayLike,
    volume_present: ArrayLike | None = None,
) -> NumericalAudit:
    reference_values, reconstructed_values, single_sequence, dtype = _normalized_arrays(
        reference,
        reconstructed,
    )
    batch_size, time_steps, feature_count = reference_values.shape
    mask: NDArray[np.bool_] | None = None
    if feature_count == 5:
        if volume_present is None:
            raise ValueError("OHLCV round-trip audit requires an explicit volume presence mask")
        raw_mask = np.asarray(volume_present)
        expected = (time_steps,) if single_sequence else (batch_size, time_steps)
        if raw_mask.dtype.kind != "b" or raw_mask.shape != expected:
            raise ValueError(f"volume presence mask must be boolean with exact shape {expected}")
        mask = np.asarray(
            raw_mask[np.newaxis, ...] if single_sequence else raw_mask, dtype=np.bool_
        )

    field_names = ("open", "high", "low", "close", "volume")[:feature_count]
    errors: list[FieldNumericalError] = []
    for field_index, field in enumerate(field_names):
        reference_field = reference_values[..., field_index]
        reconstructed_field = reconstructed_values[..., field_index]
        if field == "volume":
            assert mask is not None
            reference_field = reference_field[mask]
            reconstructed_field = reconstructed_field[mask]
        if reference_field.size == 0:
            errors.append(
                FieldNumericalError(
                    field=field,
                    compared_count=0,
                    maximum_absolute_error=0.0,
                    maximum_relative_error=0.0,
                    maximum_log_error=0.0,
                    tolerance=roundtrip_tolerance(dtype, field),
                    passes=True,
                )
            )
            continue
        if not np.isfinite(reference_field).all() or not np.isfinite(reconstructed_field).all():
            raise ValueError(f"nonfinite present value in round-trip field {field}")
        absolute = np.abs(reconstructed_field - reference_field)
        denominator = np.maximum(np.abs(reference_field), np.finfo(np.float64).tiny)
        relative = absolute / denominator
        if field == "volume":
            log_error = np.abs(np.log1p(reconstructed_field) - np.log1p(reference_field))
        else:
            if not (reference_field > 0.0).all() or not (reconstructed_field > 0.0).all():
                raise ValueError(f"nonpositive price in round-trip field {field}")
            log_error = np.abs(np.log(reconstructed_field) - np.log(reference_field))
        tolerance = roundtrip_tolerance(dtype, field)
        elementwise_pass = absolute <= (
            tolerance.absolute + tolerance.relative * np.abs(reference_field)
        )
        passes = bool(elementwise_pass.all() and (log_error <= tolerance.log_space).all())
        errors.append(
            FieldNumericalError(
                field=field,
                compared_count=int(reference_field.size),
                maximum_absolute_error=float(np.max(absolute)),
                maximum_relative_error=float(np.max(relative)),
                maximum_log_error=float(np.max(log_error)),
                tolerance=tolerance,
                passes=passes,
            )
        )

    reference_close = reference_values[..., 3]
    reconstructed_close = reconstructed_values[..., 3]
    close_drift = np.abs(reconstructed_close - reference_close) / np.maximum(
        np.abs(reference_close), np.finfo(np.float64).tiny
    )
    maximum_close_drift = float(np.max(close_drift))
    final_close_drift = float(np.max(close_drift[:, -1]))
    recursive_limit = roundtrip_tolerance(dtype, "close").recursive_relative
    all_absolute = max(item.maximum_absolute_error for item in errors)
    all_relative = max(item.maximum_relative_error for item in errors)
    all_log = max(item.maximum_log_error for item in errors)
    passes = all(item.passes for item in errors) and maximum_close_drift <= recursive_limit
    return NumericalAudit(
        schema_version="openalpha.bridge.numerical-audit.v1",
        dtype=dtype,  # type: ignore[arg-type]
        sequence_count=batch_size,
        sequence_length=time_steps,
        field_errors=tuple(errors),
        maximum_absolute_error=all_absolute,
        maximum_relative_error=all_relative,
        maximum_log_error=all_log,
        maximum_recursive_close_drift=maximum_close_drift,
        final_recursive_close_drift=final_close_drift,
        passes=passes,
    )
