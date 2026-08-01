"""Locked paired bootstrap over scored evaluation sequences.

The resampling unit is the scored sequence or a declared instrument-time block,
never the individual candle, so dependence induced by overlapping warm-up
context is preserved. Behaviour is fixed by the locked seed and replicate count
and never selected at runtime from observed results.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..errors import BridgeFailure, BridgeTransformError, FailureCategory
from .metrics import ReconstructionMethod

__all__ = [
    "LOCKED_CONFIDENCE_LEVEL",
    "LOCKED_RESAMPLES",
    "LOCKED_SEED",
    "BootstrapResult",
    "ConfirmationDecision",
    "PairedObservation",
    "ResamplingUnit",
    "paired_bootstrap",
]

LOCKED_RESAMPLES = 10_000
LOCKED_SEED = 20_260_731
LOCKED_CONFIDENCE_LEVEL = 0.95
_MINIMUM_UNITS = 2


class ResamplingUnit(StrEnum):
    SCORED_SEQUENCE = "scored_sequence"
    INSTRUMENT_TIME_BLOCK = "instrument_time_block"


class ConfirmationDecision(StrEnum):
    A_BETTER = "A_BETTER"
    B_BETTER = "B_BETTER"
    INCONCLUSIVE = "INCONCLUSIVE"
    NOT_EVALUATED = "NOT_EVALUATED"


def _fail(code: str, message: str) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(
            category=FailureCategory.INVALID_INPUT,
            code=code,
            message=message,
        )
    )


class PairedObservation(BaseModel):
    """One resampling unit's paired metric values for two methods."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    unit_id: str = Field(min_length=1)
    value_a: float
    value_b: float


class BootstrapResult(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    schema_version: Literal["openalpha.bridge.phase2.bootstrap.v1"] = (
        "openalpha.bridge.phase2.bootstrap.v1"
    )
    metric: str
    method_a: ReconstructionMethod
    method_b: ReconstructionMethod
    point_difference: float
    interval_lower: float
    interval_upper: float
    resampling_unit: ResamplingUnit
    replicates: int = Field(ge=0)
    seed: int
    sample_count: int = Field(ge=0)
    confidence_level: float
    confirmation: ConfirmationDecision


def paired_bootstrap(
    *,
    metric: str,
    method_a: ReconstructionMethod,
    method_b: ReconstructionMethod,
    observations: tuple[PairedObservation, ...],
    resampling_unit: ResamplingUnit = ResamplingUnit.SCORED_SEQUENCE,
    replicates: int = LOCKED_RESAMPLES,
    seed: int = LOCKED_SEED,
    confidence_level: float = LOCKED_CONFIDENCE_LEVEL,
) -> BootstrapResult:
    """Paired percentile bootstrap of ``mean(a) - mean(b)``.

    A negative interval upper bound confirms that method A's error is below
    method B's.
    """
    if not observations:
        raise _fail("EMPTY_BOOTSTRAP_SAMPLE", "paired bootstrap requires at least one unit")

    unit_ids = [item.unit_id for item in observations]
    if len(set(unit_ids)) != len(unit_ids):
        raise _fail("DUPLICATE_RESAMPLING_UNIT", "resampling unit identifiers must be unique")

    values_a = np.array([item.value_a for item in observations], dtype=np.float64)
    values_b = np.array([item.value_b for item in observations], dtype=np.float64)
    if not (np.isfinite(values_a).all() and np.isfinite(values_b).all()):
        raise _fail("NON_FINITE_PAIRED_VALUE", "paired values must be finite")

    count = values_a.size
    point = float(np.mean(values_a) - np.mean(values_b))

    if count < _MINIMUM_UNITS:
        return BootstrapResult(
            metric=metric,
            method_a=method_a,
            method_b=method_b,
            point_difference=point,
            interval_lower=point,
            interval_upper=point,
            resampling_unit=resampling_unit,
            replicates=0,
            seed=seed,
            sample_count=count,
            confidence_level=confidence_level,
            confirmation=ConfirmationDecision.NOT_EVALUATED,
        )

    differences = values_a - values_b
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, count, size=(replicates, count))
    replicate_means = differences[indices].mean(axis=1)

    tail = (1.0 - confidence_level) / 2.0
    lower = float(np.quantile(replicate_means, tail))
    upper = float(np.quantile(replicate_means, 1.0 - tail))

    if upper < 0.0:
        confirmation = ConfirmationDecision.A_BETTER
    elif lower > 0.0:
        confirmation = ConfirmationDecision.B_BETTER
    else:
        confirmation = ConfirmationDecision.INCONCLUSIVE

    return BootstrapResult(
        metric=metric,
        method_a=method_a,
        method_b=method_b,
        point_difference=point,
        interval_lower=lower,
        interval_upper=upper,
        resampling_unit=resampling_unit,
        replicates=replicates,
        seed=seed,
        sample_count=count,
        confidence_level=confidence_level,
        confirmation=confirmation,
    )


def pair_by_unit(
    *,
    values_a: dict[str, float],
    values_b: dict[str, float],
) -> tuple[PairedObservation, ...]:
    """Pair two methods' per-unit results, failing on any missing counterpart."""
    missing_b = sorted(set(values_a) - set(values_b))
    missing_a = sorted(set(values_b) - set(values_a))
    if missing_a or missing_b:
        raise _fail(
            "MISSING_PAIRED_UNIT",
            (
                f"unpaired resampling units: missing in A {missing_a}, "
                f"missing in B {missing_b}"
            ),
        )
    return tuple(
        PairedObservation(unit_id=unit, value_a=values_a[unit], value_b=values_b[unit])
        for unit in sorted(values_a)
    )
