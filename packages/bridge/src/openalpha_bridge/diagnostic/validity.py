"""Decoded candles, their structural validity, and the terminal projection.

``phase2.provider.Candle`` constrains prices to be finite and strictly
positive, which is right for retrieved market data and wrong here: a container
that refuses to hold invalid output cannot be used to measure how often the
output is invalid. ``DecodedCandle`` accepts anything a decoder emits,
including non-finite and non-positive values, so invalidity is observed rather
than raised.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict

__all__ = [
    "PROJECTION_METHOD",
    "CandleValidity",
    "DecodedCandle",
    "PathValidity",
    "ProjectionOutcome",
    "path_validity",
    "project_path",
    "validate_candle",
]

#: Names the one projection this diagnostic is permitted to apply.
PROJECTION_METHOD: Literal["TERMINAL_PROJECTION_V0"] = "TERMINAL_PROJECTION_V0"


class DecodedCandle(BaseModel):
    """Whatever the decoder produced. Deliberately unconstrained."""

    model_config = ConfigDict(allow_inf_nan=True, extra="forbid", frozen=True, strict=True)

    open: float
    high: float
    low: float
    close: float
    volume: float


class CandleValidity(BaseModel):
    """Which structural rules one candle satisfied, and which it broke."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    index: int
    valid: bool
    violations: tuple[str, ...]


class PathValidity(BaseModel):
    """Validity of a whole decoded path."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    candle_count: int
    invalid_candle_count: int
    invalid_candle_fraction: float
    #: True when at least one candle is invalid. The spec calls this the invalid
    #: path indicator; a path is only valid if every candle in it is.
    path_is_invalid: bool
    violation_counts: dict[str, int]
    per_candle: tuple[CandleValidity, ...]


def validate_candle(index: int, candle: DecodedCandle) -> CandleValidity:
    """Every structural rule in the specification, checked one at a time."""
    violations: list[str] = []
    prices = {
        "open": candle.open,
        "high": candle.high,
        "low": candle.low,
        "close": candle.close,
    }
    for name, value in prices.items():
        if not math.isfinite(value):
            violations.append(f"{name}_not_finite")
        elif value <= 0.0:
            violations.append(f"{name}_not_positive")

    if not math.isfinite(candle.volume):
        violations.append("volume_not_finite")
    elif candle.volume < 0.0:
        violations.append("volume_negative")

    # Ordering rules are only meaningful once the values are finite; comparing
    # against a NaN silently returns False and would read as "no violation".
    if all(math.isfinite(value) for value in prices.values()):
        if candle.high < candle.open:
            violations.append("high_below_open")
        if candle.high < candle.close:
            violations.append("high_below_close")
        if candle.high < candle.low:
            violations.append("high_below_low")
        if candle.low > candle.open:
            violations.append("low_above_open")
        if candle.low > candle.close:
            violations.append("low_above_close")

    return CandleValidity(index=index, valid=not violations, violations=tuple(violations))


def path_validity(candles: tuple[DecodedCandle, ...]) -> PathValidity:
    per_candle = tuple(validate_candle(index, c) for index, c in enumerate(candles))
    invalid = [c for c in per_candle if not c.valid]
    counts: dict[str, int] = {}
    for entry in per_candle:
        for violation in entry.violations:
            counts[violation] = counts.get(violation, 0) + 1
    return PathValidity(
        candle_count=len(per_candle),
        invalid_candle_count=len(invalid),
        invalid_candle_fraction=(len(invalid) / len(per_candle)) if per_candle else 0.0,
        path_is_invalid=bool(invalid),
        violation_counts=counts,
        per_candle=per_candle,
    )


class ProjectionOutcome(BaseModel):
    """The projected path and how much intervention it took."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    method: Literal["TERMINAL_PROJECTION_V0"] = PROJECTION_METHOD
    projected: tuple[DecodedCandle, ...]
    adjusted_candle_count: int
    adjusted_field_count: int
    #: Sum of absolute changes, and the largest single change. Reported so the
    #: intervention can be judged as small or large rather than merely present.
    total_absolute_adjustment: float
    maximum_absolute_adjustment: float
    #: Non-finite inputs cannot be repaired by an order projection. They are
    #: counted, not silently replaced with a substituted value.
    unrepairable_candle_count: int


def project_path(candles: tuple[DecodedCandle, ...]) -> ProjectionOutcome:
    """Apply only the terminal projection baseline. Tokens are never touched.

    Open and close are preserved exactly. ``high`` becomes the maximum of the
    raw high, open and close; ``low`` the minimum of the raw low, open and
    close; volume is clamped at zero. Nothing else is altered, and no value is
    invented: a candle whose prices are not finite is left as it is and counted
    as unrepairable, because substituting a number there would be fabrication
    rather than projection.
    """
    projected: list[DecodedCandle] = []
    adjusted_candles = 0
    adjusted_fields = 0
    total = 0.0
    largest = 0.0
    unrepairable = 0

    for candle in candles:
        finite = all(
            math.isfinite(value)
            for value in (candle.open, candle.high, candle.low, candle.close, candle.volume)
        )
        if not finite:
            unrepairable += 1
            projected.append(candle)
            continue

        high = max(candle.high, candle.open, candle.close)
        low = min(candle.low, candle.open, candle.close)
        volume = max(candle.volume, 0.0)

        changes = (
            abs(high - candle.high),
            abs(low - candle.low),
            abs(volume - candle.volume),
        )
        touched = sum(1 for delta in changes if delta > 0.0)
        if touched:
            adjusted_candles += 1
            adjusted_fields += touched
            total += sum(changes)
            largest = max(largest, *changes)

        projected.append(
            DecodedCandle(
                open=candle.open, high=high, low=low, close=candle.close, volume=volume
            )
        )

    return ProjectionOutcome(
        projected=tuple(projected),
        adjusted_candle_count=adjusted_candles,
        adjusted_field_count=adjusted_fields,
        total_absolute_adjustment=total,
        maximum_absolute_adjustment=largest,
        unrepairable_candle_count=unrepairable,
    )
