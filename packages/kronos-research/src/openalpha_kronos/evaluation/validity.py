"""Structural validity over the six official channels, and the projection.

Operates on ``OfficialRow``, so a decoded path keeps its session and its amount
column rather than being reduced on the way in.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict

from ..model.input import OfficialRow

__all__ = [
    "PROJECTION_METHOD",
    "CandleValidity",
    "PathValidity",
    "ProjectionOutcome",
    "path_validity",
    "project_path",
    "validate_candle",
]

PROJECTION_METHOD: Literal["TERMINAL_PROJECTION_V0"] = "TERMINAL_PROJECTION_V0"


class CandleValidity(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    index: int
    valid: bool
    violations: tuple[str, ...]


class PathValidity(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    candle_count: int
    invalid_candle_count: int
    invalid_candle_fraction: float
    #: True when at least one candle is invalid. Any invalid candle is a strict
    #: structural failure observation, independent of the materiality threshold.
    path_is_invalid: bool
    violation_counts: dict[str, int]
    per_candle: tuple[CandleValidity, ...]


def validate_candle(index: int, candle: OfficialRow) -> CandleValidity:
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

    for name, value in (("volume", candle.volume), ("amount", candle.amount)):
        if not math.isfinite(value):
            violations.append(f"{name}_not_finite")
        elif value < 0.0:
            violations.append(f"{name}_negative")

    # Ordering rules need finite values; comparing against NaN returns False
    # and would silently read as "no violation".
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


def path_validity(candles: tuple[OfficialRow, ...]) -> PathValidity:
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
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    method: Literal["TERMINAL_PROJECTION_V0"] = PROJECTION_METHOD
    projected: tuple[OfficialRow, ...]
    adjusted_candle_count: int
    adjusted_field_count: int
    total_absolute_adjustment: float
    maximum_absolute_adjustment: float
    #: Non-finite inputs cannot be repaired by an order projection. They are
    #: counted, not replaced with an invented value.
    unrepairable_candle_count: int


def project_path(candles: tuple[OfficialRow, ...]) -> ProjectionOutcome:
    """Apply only the terminal projection baseline. Tokens are never touched.

    Open and close are preserved exactly. ``high`` becomes the maximum of the
    raw high, open and close; ``low`` the minimum of the raw low, open and
    close; volume and amount are clamped at zero. A candle whose values are not
    finite is left alone and counted, because substituting a number there would
    be fabrication rather than projection.
    """
    projected: list[OfficialRow] = []
    adjusted_candles = 0
    adjusted_fields = 0
    total = 0.0
    largest = 0.0
    unrepairable = 0

    for candle in candles:
        if not all(math.isfinite(value) for value in candle.channels()):
            unrepairable += 1
            projected.append(candle)
            continue

        high = max(candle.high, candle.open, candle.close)
        low = min(candle.low, candle.open, candle.close)
        volume = max(candle.volume, 0.0)
        amount = max(candle.amount, 0.0)

        changes = (
            abs(high - candle.high),
            abs(low - candle.low),
            abs(volume - candle.volume),
            abs(amount - candle.amount),
        )
        touched = sum(1 for delta in changes if delta > 0.0)
        if touched:
            adjusted_candles += 1
            adjusted_fields += touched
            total += sum(changes)
            largest = max(largest, *changes)

        projected.append(
            OfficialRow(
                session=candle.session,
                open=candle.open,
                high=high,
                low=low,
                close=candle.close,
                volume=volume,
                amount=amount,
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
