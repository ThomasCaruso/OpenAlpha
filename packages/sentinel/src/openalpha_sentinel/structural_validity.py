from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from datetime import date, datetime
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from .contracts import FrozenModel

ViolationCode = Literal[
    "NONFINITE_OPEN",
    "NONFINITE_HIGH",
    "NONFINITE_LOW",
    "NONFINITE_CLOSE",
    "NONFINITE_VOLUME",
    "NONPOSITIVE_OPEN",
    "NONPOSITIVE_HIGH",
    "NONPOSITIVE_LOW",
    "NONPOSITIVE_CLOSE",
    "HIGH_BELOW_OPEN",
    "HIGH_BELOW_CLOSE",
    "HIGH_BELOW_LOW",
    "LOW_ABOVE_OPEN",
    "LOW_ABOVE_CLOSE",
    "NEGATIVE_VOLUME",
    "MISSING_TIMESTAMP",
    "UNEXPECTED_TIMESTAMP",
    "DUPLICATE_TIMESTAMP",
    "HORIZON_LENGTH_MISMATCH",
]
DiagnosticName = Literal[
    "INVALID_PATH_FRACTION",
    "INVALID_CANDLE_FRACTION",
    "TOTAL_CONSTRAINT_VIOLATIONS",
    "MAX_CONSTRAINT_VIOLATION_SEVERITY",
    "MEAN_CONSTRAINT_VIOLATION_SEVERITY",
    "EARLIEST_INVALID_HORIZON_STEP",
    "HIGH_LOW_INVERSION_COUNT",
    "HIGH_BELOW_BODY_COUNT",
    "LOW_ABOVE_BODY_COUNT",
    "NONFINITE_OUTPUT_COUNT",
    "NONPOSITIVE_PRICE_COUNT",
]
TimestampValue = datetime | date


class AuditCandle(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=True,
        extra="forbid",
        frozen=True,
        strict=True,
    )

    session: TimestampValue | None
    open: float
    high: float
    low: float
    close: float
    volume: float | None


class StructuralViolation(FrozenModel):
    path_id: str = Field(min_length=1, max_length=256)
    step: int = Field(ge=0)
    timestamp: TimestampValue | None
    code: ViolationCode
    observed_gap: float | None
    normalized_severity: float | None
    existed_before_outcome: Literal[True] = True


class PathValidity(FrozenModel):
    path_id: str
    valid: bool
    candle_count: int = Field(ge=0)
    expected_candle_count: int = Field(ge=1)
    invalid_candle_count: int = Field(ge=0)
    earliest_invalid_horizon_step: int | None = Field(default=None, ge=1)
    violations: tuple[StructuralViolation, ...]


class StructuralDiagnostic(FrozenModel):
    name: DiagnosticName
    status: Literal["available", "not_computable"]
    value: float | int | None
    units: str


class StructuralDiagnostics(FrozenModel):
    values: tuple[StructuralDiagnostic, ...]


class ProjectionAdjustment(FrozenModel):
    step: int = Field(ge=1)
    timestamp: TimestampValue
    field: Literal["high", "low"]
    original_value: float
    projected_value: float
    absolute_adjustment: float = Field(ge=0.0)
    normalized_adjustment: float = Field(ge=0.0)


class ProjectionResult(FrozenModel):
    method: Literal["CONSTRAINT_PROJECTION_V0"]
    path_id: str
    original_candles: tuple[AuditCandle, ...]
    projected_candles: tuple[AuditCandle, ...]
    adjustments: tuple[ProjectionAdjustment, ...]
    original_log_return: float
    projected_log_return: float


def validate_forecast_path(
    *,
    path_id: str,
    candles: Sequence[AuditCandle],
    expected_sessions: Sequence[TimestampValue],
    cutoff_close: float,
    cutoff_volume: float,
) -> PathValidity:
    if not path_id:
        raise ValueError("path_id must be nonempty")
    if not expected_sessions:
        raise ValueError("expected_sessions must be nonempty")
    if not math.isfinite(cutoff_close) or cutoff_close <= 0.0:
        raise ValueError("cutoff_close must be finite and positive")
    if not math.isfinite(cutoff_volume) or cutoff_volume < 0.0:
        raise ValueError("cutoff_volume must be finite and nonnegative")

    rows = tuple(candles)
    expected = tuple(expected_sessions)
    violations: list[StructuralViolation] = []
    if len(rows) != len(expected):
        violations.append(
            _violation(
                path_id,
                0,
                None,
                "HORIZON_LENGTH_MISMATCH",
                float(abs(len(rows) - len(expected))),
                None,
            )
        )

    actual_sessions = tuple(row.session for row in rows)
    counts = Counter(session for session in actual_sessions if session is not None)
    for step, row in enumerate(rows, start=1):
        if row.session is None:
            violations.append(_violation(path_id, step, None, "MISSING_TIMESTAMP", None, None))
        else:
            if counts[row.session] > 1:
                violations.append(
                    _violation(
                        path_id,
                        step,
                        row.session,
                        "DUPLICATE_TIMESTAMP",
                        None,
                        None,
                    )
                )
            expected_at_step = expected[step - 1] if step <= len(expected) else None
            if expected_at_step != row.session:
                violations.append(
                    _violation(
                        path_id,
                        step,
                        row.session,
                        "UNEXPECTED_TIMESTAMP",
                        None,
                        None,
                    )
                )
        _check_prices(
            path_id=path_id,
            step=step,
            row=row,
            cutoff_close=cutoff_close,
            violations=violations,
        )
        _check_volume(
            path_id=path_id,
            step=step,
            row=row,
            cutoff_volume=cutoff_volume,
            violations=violations,
        )

    actual_set = {session for session in actual_sessions if session is not None}
    for step, session in enumerate(expected, start=1):
        if session not in actual_set:
            violations.append(_violation(path_id, step, session, "MISSING_TIMESTAMP", None, None))

    invalid_steps = sorted({item.step for item in violations if item.step > 0})
    return PathValidity(
        path_id=path_id,
        valid=not violations,
        candle_count=len(rows),
        expected_candle_count=len(expected),
        invalid_candle_count=len(invalid_steps),
        earliest_invalid_horizon_step=invalid_steps[0] if invalid_steps else None,
        violations=tuple(violations),
    )


def summarize_structural_validity(
    results: Sequence[PathValidity],
) -> StructuralDiagnostics:
    paths = tuple(results)
    if not paths:
        raise ValueError("at least one path validity result is required")
    violation_list = tuple(violation for path in paths for violation in path.violations)
    severities = tuple(
        item.normalized_severity for item in violation_list if item.normalized_severity is not None
    )
    invalid_steps = tuple(
        path.earliest_invalid_horizon_step
        for path in paths
        if path.earliest_invalid_horizon_step is not None
    )
    total_candles = sum(path.expected_candle_count for path in paths)
    invalid_candles = sum(path.invalid_candle_count for path in paths)
    counts = Counter(item.code for item in violation_list)
    available = "available"
    not_computable = "not_computable"

    return StructuralDiagnostics(
        values=(
            StructuralDiagnostic(
                name="INVALID_PATH_FRACTION",
                status=available,
                value=sum(not path.valid for path in paths) / len(paths),
                units="fraction",
            ),
            StructuralDiagnostic(
                name="INVALID_CANDLE_FRACTION",
                status=available,
                value=invalid_candles / total_candles,
                units="fraction",
            ),
            StructuralDiagnostic(
                name="TOTAL_CONSTRAINT_VIOLATIONS",
                status=available,
                value=len(violation_list),
                units="count",
            ),
            _optional_diagnostic(
                "MAX_CONSTRAINT_VIOLATION_SEVERITY",
                max(severities) if severities else None,
                "cutoff_close_fraction",
            ),
            _optional_diagnostic(
                "MEAN_CONSTRAINT_VIOLATION_SEVERITY",
                sum(severities) / len(severities) if severities else None,
                "cutoff_close_fraction",
            ),
            StructuralDiagnostic(
                name="EARLIEST_INVALID_HORIZON_STEP",
                status=available if invalid_steps else not_computable,
                value=min(invalid_steps) if invalid_steps else None,
                units="one_based_step",
            ),
            StructuralDiagnostic(
                name="HIGH_LOW_INVERSION_COUNT",
                status=available,
                value=counts["HIGH_BELOW_LOW"],
                units="count",
            ),
            StructuralDiagnostic(
                name="HIGH_BELOW_BODY_COUNT",
                status=available,
                value=counts["HIGH_BELOW_OPEN"] + counts["HIGH_BELOW_CLOSE"],
                units="count",
            ),
            StructuralDiagnostic(
                name="LOW_ABOVE_BODY_COUNT",
                status=available,
                value=counts["LOW_ABOVE_OPEN"] + counts["LOW_ABOVE_CLOSE"],
                units="count",
            ),
            StructuralDiagnostic(
                name="NONFINITE_OUTPUT_COUNT",
                status=available,
                value=sum(item.code.startswith("NONFINITE_") for item in violation_list),
                units="count",
            ),
            StructuralDiagnostic(
                name="NONPOSITIVE_PRICE_COUNT",
                status=available,
                value=sum(item.code.startswith("NONPOSITIVE_") for item in violation_list),
                units="count",
            ),
        )
    )


def constraint_projection_v0(
    *,
    path_id: str,
    candles: Sequence[AuditCandle],
    cutoff_close: float,
) -> ProjectionResult:
    if not math.isfinite(cutoff_close) or cutoff_close <= 0.0:
        raise ValueError("cutoff_close must be finite and positive")
    original = tuple(candles)
    if not original:
        raise ValueError("projection requires at least one candle")
    projected: list[AuditCandle] = []
    adjustments: list[ProjectionAdjustment] = []
    for step, row in enumerate(original, start=1):
        if row.session is None:
            raise ValueError("projection requires every timestamp")
        prices = (row.open, row.high, row.low, row.close)
        if not all(math.isfinite(value) and value > 0.0 for value in prices):
            raise ValueError("projection requires finite positive prices")
        repaired_high = max(row.high, row.open, row.close, row.low)
        repaired_low = min(row.low, row.open, row.close, repaired_high)
        if repaired_high != row.high:
            adjustments.append(
                _adjustment(step, row.session, "high", row.high, repaired_high, cutoff_close)
            )
        if repaired_low != row.low:
            adjustments.append(
                _adjustment(step, row.session, "low", row.low, repaired_low, cutoff_close)
            )
        projected.append(
            AuditCandle(
                session=row.session,
                open=row.open,
                high=repaired_high,
                low=repaired_low,
                close=row.close,
                volume=row.volume,
            )
        )
    original_return = math.log(original[-1].close / cutoff_close)
    projected_return = math.log(projected[-1].close / cutoff_close)
    return ProjectionResult(
        method="CONSTRAINT_PROJECTION_V0",
        path_id=path_id,
        original_candles=original,
        projected_candles=tuple(projected),
        adjustments=tuple(adjustments),
        original_log_return=original_return,
        projected_log_return=projected_return,
    )


def _check_prices(
    *,
    path_id: str,
    step: int,
    row: AuditCandle,
    cutoff_close: float,
    violations: list[StructuralViolation],
) -> None:
    price_fields = (
        ("OPEN", row.open),
        ("HIGH", row.high),
        ("LOW", row.low),
        ("CLOSE", row.close),
    )
    for name, value in price_fields:
        if not math.isfinite(value):
            violations.append(
                _violation(
                    path_id,
                    step,
                    row.session,
                    cast(ViolationCode, f"NONFINITE_{name}"),
                    None,
                    None,
                )
            )
        elif value <= 0.0:
            gap = abs(value)
            violations.append(
                _violation(
                    path_id,
                    step,
                    row.session,
                    cast(ViolationCode, f"NONPOSITIVE_{name}"),
                    gap,
                    gap / cutoff_close,
                )
            )
    comparisons: tuple[tuple[ViolationCode, float, float], ...] = (
        ("HIGH_BELOW_OPEN", row.open, row.high),
        ("HIGH_BELOW_CLOSE", row.close, row.high),
        ("HIGH_BELOW_LOW", row.low, row.high),
        ("LOW_ABOVE_OPEN", row.low, row.open),
        ("LOW_ABOVE_CLOSE", row.low, row.close),
    )
    for code, left, right in comparisons:
        if math.isfinite(left) and math.isfinite(right) and left > right:
            gap = left - right
            violations.append(
                _violation(
                    path_id,
                    step,
                    row.session,
                    code,
                    gap,
                    gap / cutoff_close,
                )
            )


def _check_volume(
    *,
    path_id: str,
    step: int,
    row: AuditCandle,
    cutoff_volume: float,
    violations: list[StructuralViolation],
) -> None:
    if row.volume is None:
        return
    if not math.isfinite(row.volume):
        violations.append(_violation(path_id, step, row.session, "NONFINITE_VOLUME", None, None))
    elif row.volume < 0.0:
        gap = abs(row.volume)
        violations.append(
            _violation(
                path_id,
                step,
                row.session,
                "NEGATIVE_VOLUME",
                gap,
                gap / max(cutoff_volume, 1.0),
            )
        )


def _violation(
    path_id: str,
    step: int,
    timestamp: TimestampValue | None,
    code: ViolationCode,
    observed_gap: float | None,
    normalized_severity: float | None,
) -> StructuralViolation:
    return StructuralViolation(
        path_id=path_id,
        step=step,
        timestamp=timestamp,
        code=code,
        observed_gap=observed_gap,
        normalized_severity=normalized_severity,
        existed_before_outcome=True,
    )


def _optional_diagnostic(
    name: Literal[
        "MAX_CONSTRAINT_VIOLATION_SEVERITY",
        "MEAN_CONSTRAINT_VIOLATION_SEVERITY",
    ],
    value: float | None,
    units: str,
) -> StructuralDiagnostic:
    return StructuralDiagnostic(
        name=name,
        status="available" if value is not None else "not_computable",
        value=value,
        units=units,
    )


def _adjustment(
    step: int,
    timestamp: TimestampValue,
    field: Literal["high", "low"],
    original: float,
    projected: float,
    cutoff_close: float,
) -> ProjectionAdjustment:
    absolute = abs(projected - original)
    return ProjectionAdjustment(
        step=step,
        timestamp=timestamp,
        field=field,
        original_value=original,
        projected_value=projected,
        absolute_adjustment=absolute,
        normalized_adjustment=absolute / cutoff_close,
    )
