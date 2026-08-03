"""Metrics, aggregation and the deterministic paired bootstrap.

Pooled error is reported but never decides anything on its own: a handful of
high-volatility windows can dominate a pooled mean, so every decision quantity
here is either a median, a fraction of origins, or a paired interval.

The bootstrap is seeded from the specification and resamples asset-origins, not
forecast steps. Steps inside one origin are not independent, so resampling them
would manufacture confidence the data does not contain.
"""

from __future__ import annotations

import math
import random
import statistics
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from ..diagnostic.metrics import DirectionalAccuracy, directional_accuracy
from ..diagnostic.official_input import OfficialRow

__all__ = [
    "BootstrapInterval",
    "DistributionSummary",
    "ExtendedForecastMetrics",
    "StepSummary",
    "extended_metrics",
    "paired_bootstrap",
    "relative_skill",
    "step_summaries",
    "summarize",
]

_MINIMUM_BOOTSTRAP_SAMPLE: Final[int] = 2


def _anchored_returns(rows: tuple[OfficialRow, ...], anchor: float) -> list[float] | None:
    """log(close[t] / close[t-1]) with the first transition on the anchor."""
    previous = anchor
    out: list[float] = []
    for row in rows:
        close = row.close
        if not math.isfinite(close) or close <= 0.0 or previous <= 0.0:
            return None
        out.append(math.log(close / previous))
        previous = close
    return out


class ExtendedForecastMetrics(BaseModel):
    """The primary metric and every preregistered secondary metric."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    defined: bool
    close_return_mae: float | None = None
    close_mae: float | None = None
    close_return_rmse: float | None = None
    median_absolute_return_error: float | None = None
    directional: DirectionalAccuracy | None = None
    scored_steps: int = 0
    undefined_reason: str | None = None


def extended_metrics(
    predicted: tuple[OfficialRow, ...],
    actual: tuple[OfficialRow, ...],
    *,
    anchor_close: float,
) -> ExtendedForecastMetrics:
    """Score one path against one target on all five declared metrics."""
    if len(predicted) != len(actual) or not predicted:
        return ExtendedForecastMetrics(
            defined=False,
            undefined_reason=(
                f"predicted {len(predicted)} steps against {len(actual)} target steps"
            ),
        )
    if not math.isfinite(anchor_close) or anchor_close <= 0.0:
        return ExtendedForecastMetrics(
            defined=False,
            scored_steps=len(predicted),
            undefined_reason="anchor close is not usable",
        )

    predicted_returns = _anchored_returns(predicted, anchor_close)
    actual_returns = _anchored_returns(actual, anchor_close)
    if predicted_returns is None or actual_returns is None:
        return ExtendedForecastMetrics(
            defined=False,
            scored_steps=len(predicted),
            undefined_reason="a close is not positive, so its log return is undefined",
        )

    errors = [abs(p - a) for p, a in zip(predicted_returns, actual_returns, strict=True)]
    squared = [(p - a) ** 2 for p, a in zip(predicted_returns, actual_returns, strict=True)]
    steps = len(errors)
    return ExtendedForecastMetrics(
        defined=True,
        close_return_mae=sum(errors) / steps,
        close_mae=sum(abs(p.close - a.close) for p, a in zip(predicted, actual, strict=True))
        / steps,
        close_return_rmse=math.sqrt(sum(squared) / steps),
        median_absolute_return_error=statistics.median(errors),
        directional=directional_accuracy(predicted, actual, anchor_close=anchor_close),
        scored_steps=steps,
    )


def relative_skill(candidate: float | None, baseline: float | None) -> float | None:
    """1 - candidate/baseline. Positive beats the baseline."""
    if candidate is None or baseline is None or baseline == 0.0:
        return None
    return 1.0 - candidate / baseline


class DistributionSummary(BaseModel):
    """Mean, median and spread over a set of per-origin values."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    count: int = Field(ge=0)
    mean: float | None = None
    median: float | None = None
    standard_deviation: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    undefined_reason: str | None = None


def summarize(values: list[float]) -> DistributionSummary:
    """Summarize a per-origin sample. Never silently drops a missing value."""
    if not values:
        return DistributionSummary(count=0, undefined_reason="no values to summarize")
    return DistributionSummary(
        count=len(values),
        mean=sum(values) / len(values),
        median=statistics.median(values),
        # Population standard deviation, matching the convention used elsewhere
        # in this tree. A single value has zero spread, not undefined spread.
        standard_deviation=statistics.pstdev(values),
        minimum=min(values),
        maximum=max(values),
    )


class BootstrapInterval(BaseModel):
    """A deterministic paired percentile bootstrap over asset-origins.

    ``differences`` are persistence minus candidate on the primary metric, so a
    positive value is favorable and an interval strictly above zero is a
    favorable exclusion of zero.
    """

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    defined: bool
    method: str = "paired_percentile_bootstrap_over_asset_origins"
    unit_of_resampling: str = "asset_origin"
    sample_size: int = Field(ge=0)
    resamples: int = Field(ge=0)
    confidence_level: float
    seed: int
    point_estimate: float | None = None
    lower: float | None = None
    upper: float | None = None
    excludes_zero: bool = False
    excludes_zero_favorably: bool = False
    undefined_reason: str | None = None


def paired_bootstrap(
    differences: list[float], *, seed: int, resamples: int, confidence_level: float
) -> BootstrapInterval:
    """Percentile bootstrap of the mean paired difference.

    Deterministic: the generator is seeded from the specification, so the
    interval is reproducible and cannot be re-rolled after the fact.
    """
    size = len(differences)
    if size < _MINIMUM_BOOTSTRAP_SAMPLE:
        return BootstrapInterval(
            defined=False,
            sample_size=size,
            resamples=resamples,
            confidence_level=confidence_level,
            seed=seed,
            undefined_reason="fewer than two paired differences, so no interval exists",
        )
    if resamples <= 0:
        return BootstrapInterval(
            defined=False,
            sample_size=size,
            resamples=resamples,
            confidence_level=confidence_level,
            seed=seed,
            undefined_reason="the resample count is not positive",
        )

    generator = random.Random(seed)
    means: list[float] = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(size):
            total += differences[generator.randrange(size)]
        means.append(total / size)
    means.sort()

    tail = (1.0 - confidence_level) / 2.0
    lower_index = math.floor(tail * (resamples - 1))
    upper_index = math.ceil((1.0 - tail) * (resamples - 1))
    lower = means[lower_index]
    upper = means[upper_index]
    point = sum(differences) / size
    excludes = (lower > 0.0 and upper > 0.0) or (lower < 0.0 and upper < 0.0)
    return BootstrapInterval(
        defined=True,
        sample_size=size,
        resamples=resamples,
        confidence_level=confidence_level,
        seed=seed,
        point_estimate=point,
        lower=lower,
        upper=upper,
        excludes_zero=excludes,
        # Favorable means the whole interval sits above zero: the candidate beat
        # persistence by a margin the resampling did not erase.
        excludes_zero_favorably=lower > 0.0,
    )


class StepSummary(BaseModel):
    """Error at one forecast step, pooled across every asset-origin."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    step: int = Field(ge=1)
    candidate_absolute_return_error: DistributionSummary
    persistence_absolute_return_error: DistributionSummary
    fraction_of_origins_beating_persistence: float | None = None


def step_summaries(
    *,
    candidate_step_errors: list[list[float]],
    persistence_step_errors: list[list[float]],
    horizon: int,
) -> tuple[StepSummary, ...]:
    """Aggregate per-step absolute return error across asset-origins.

    Each inner list is one asset-origin's per-step errors. Forecast skill often
    decays with step, and a pooled number hides that entirely.
    """
    summaries: list[StepSummary] = []
    for step in range(horizon):
        candidate = [row[step] for row in candidate_step_errors if step < len(row)]
        persistence = [row[step] for row in persistence_step_errors if step < len(row)]
        wins = [
            1.0
            for c, p in zip(candidate, persistence, strict=False)
            if c < p
        ]
        fraction = (len(wins) / len(candidate)) if candidate else None
        summaries.append(
            StepSummary(
                step=step + 1,
                candidate_absolute_return_error=summarize(candidate),
                persistence_absolute_return_error=summarize(persistence),
                fraction_of_origins_beating_persistence=fraction,
            )
        )
    return tuple(summaries)


def absolute_return_errors(
    predicted: tuple[OfficialRow, ...],
    actual: tuple[OfficialRow, ...],
    *,
    anchor_close: float,
) -> list[float] | None:
    """Per-step absolute anchored return error, for step-level aggregation."""
    predicted_returns = _anchored_returns(predicted, anchor_close)
    actual_returns = _anchored_returns(actual, anchor_close)
    if predicted_returns is None or actual_returns is None:
        return None
    if len(predicted_returns) != len(actual_returns):
        return None
    return [abs(p - a) for p, a in zip(predicted_returns, actual_returns, strict=True)]


__all__ += ["absolute_return_errors"]
