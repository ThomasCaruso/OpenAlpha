"""Metrics, aggregation and the deterministic paired moving-block bootstrap.

Pooled error is reported but never decides anything on its own: a handful of
high-volatility windows can dominate a pooled mean, so every decision quantity
here is either a median, a fraction of origins, or a paired interval.

The decision-bearing interval has to respect two different dependencies at once.

**Cross-sectional.** SPY, QQQ, IWM and DIA are evaluated over the same
chronological windows, so their errors within an origin move together. The base
resampling unit is therefore the whole origin cluster: all four assets enter or
none do, and no asset is ever drawn, weighted or omitted on its own.

**Temporal.** The origins are sequential and their information windows overlap.
The context is 40 sessions and the stride is 12, so adjacent origins share 28
context sessions, origins two strides apart share 16, and origins three strides
apart share 4. Four strides apart they share none. On top of that, one origin's
twelve target sessions become part of the next origins' context. Twenty-five
origins are therefore nothing like twenty-five independent draws, and resampling
them independently would understate uncertainty for the same reason resampling
the four assets independently would.

So the clusters are resampled in **consecutive four-origin blocks**. Four is
``ceil(context_length / origin_stride) = ceil(40 / 12)``, the smallest block that
spans every directly overlapping context relationship. It is fixed by geometry,
declared before execution, and never estimated or tuned from results.

There is deliberately no flat asset-origin bootstrap and no independent
origin-cluster bootstrap in this module. Omitting them is what makes "the
decision layer cannot read one" a structural fact rather than a convention:
neither function exists, and :class:`MovingBlockBootstrapInterval` is the only
interval type the decision layer accepts.
"""

from __future__ import annotations

import math
import statistics
from typing import Final, Literal

from openalpha_research.failures import FailureCategory, ResearchFailure, ResearchFailureError
from openalpha_research.resampling import moving_block_percentile_interval
from pydantic import BaseModel, ConfigDict, Field

from openalpha_kronos.evaluation.metrics import DirectionalAccuracy, directional_accuracy
from openalpha_kronos.model.input import OfficialRow

from .spec import ASSET_PANEL, CONTEXT_CANDLES, ORIGIN_STRIDE, ORIGINS_PER_ASSET

__all__ = [
    "BOOTSTRAP_ASSETS_PER_CLUSTER",
    "BOOTSTRAP_AVAILABLE_BLOCK_STARTS",
    "BOOTSTRAP_BASE_RESAMPLING_UNIT",
    "BOOTSTRAP_BLOCKS_PER_RESAMPLE",
    "BOOTSTRAP_BLOCK_LENGTH",
    "BOOTSTRAP_CLUSTER_COUNT",
    "BOOTSTRAP_METHOD",
    "BOOTSTRAP_OBSERVATION_COUNT",
    "BOOTSTRAP_TEMPORAL_RESAMPLING_UNIT",
    "DistributionSummary",
    "ExtendedForecastMetrics",
    "MovingBlockBootstrapInterval",
    "OriginCluster",
    "StepSummary",
    "block_starts",
    "extended_metrics",
    "paired_origin_moving_block_bootstrap",
    "relative_skill",
    "step_summaries",
    "summarize",
    "undefined_moving_block_bootstrap",
]

BOOTSTRAP_METHOD: Final[str] = (
    "paired_percentile_moving_block_bootstrap_over_origin_clusters"
)
BOOTSTRAP_BASE_RESAMPLING_UNIT: Final[str] = "origin_ordinal_cluster_all_assets"
BOOTSTRAP_TEMPORAL_RESAMPLING_UNIT: Final[str] = "four_consecutive_origin_clusters"

BOOTSTRAP_CLUSTER_COUNT: Final[int] = ORIGINS_PER_ASSET
BOOTSTRAP_ASSETS_PER_CLUSTER: Final[int] = len(ASSET_PANEL)
BOOTSTRAP_OBSERVATION_COUNT: Final[int] = BOOTSTRAP_CLUSTER_COUNT * BOOTSTRAP_ASSETS_PER_CLUSTER

#: ceil(40 / 12) = 4. The smallest block spanning every pair of origins whose
#: contexts overlap. Derived from the geometry, not fitted to any result.
BOOTSTRAP_BLOCK_LENGTH: Final[int] = -(-CONTEXT_CANDLES // ORIGIN_STRIDE)

#: Contiguous starts with no wraparound: 25 - 4 + 1 = 22, i.e. 0..21.
BOOTSTRAP_AVAILABLE_BLOCK_STARTS: Final[int] = (
    BOOTSTRAP_CLUSTER_COUNT - BOOTSTRAP_BLOCK_LENGTH + 1
)

#: ceil(25 / 4) = 7 blocks, giving 28 positions, truncated back to 25.
BOOTSTRAP_BLOCKS_PER_RESAMPLE: Final[int] = -(
    -BOOTSTRAP_CLUSTER_COUNT // BOOTSTRAP_BLOCK_LENGTH
)


def _fail(code: str, message: str, *, field: str | None = None) -> ResearchFailureError:
    return ResearchFailureError(
        ResearchFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            field=field,
            message=message,
        )
    )


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


class OriginCluster(BaseModel):
    """One chronological origin, carrying every asset evaluated at it.

    The cluster is the resampling unit. It exists as a type so a caller cannot
    hand the bootstrap a flat list of asset-origin rows by accident: the four
    assets of one ordinal arrive together or the model refuses to build.

    ``paired_differences`` are persistence minus candidate on the primary
    metric, positionally aligned with ``assets``. Positive is favorable.
    """

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    ordinal: int = Field(ge=0)
    assets: tuple[str, ...]
    paired_differences: tuple[float, ...]


class MovingBlockBootstrapInterval(BaseModel):
    """A deterministic paired percentile moving-block bootstrap.

    The only interval type the decision layer accepts. Its metadata states both
    resampling units explicitly, so a reader never has to infer whether
    cross-sectional or temporal dependence was preserved.
    """

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    defined: bool
    method: str = BOOTSTRAP_METHOD
    base_resampling_unit: str = BOOTSTRAP_BASE_RESAMPLING_UNIT
    temporal_resampling_unit: str = BOOTSTRAP_TEMPORAL_RESAMPLING_UNIT
    cluster_count: int = Field(ge=0)
    assets_per_cluster: int = Field(ge=0)
    observation_count: int = Field(ge=0)
    block_length: int = Field(ge=0)
    available_block_starts: int = Field(ge=0)
    blocks_drawn_per_resample: int = Field(ge=0)
    clusters_retained_per_resample: int = Field(ge=0)
    resamples: int = Field(ge=0)
    confidence_level: float
    seed: int
    point_estimate: float | None = None
    lower: float | None = None
    upper: float | None = None
    excludes_zero: bool = False
    excludes_zero_favorably: bool = False
    #: Stated so no reader has to read the code to know what was held together.
    assets_resampled_within_cluster: Literal[False] = False
    origins_resampled_independently: Literal[False] = False
    wraparound: Literal[False] = False
    block_length_selected_before_execution: Literal[True] = True
    block_length_tuned_from_results: Literal[False] = False
    undefined_reason: str | None = None


def _validate_clusters(clusters: tuple[OriginCluster, ...]) -> None:
    """Every structural requirement, checked before a single resample is drawn.

    An undefined or malformed input is a typed failure here, never a quietly
    smaller bootstrap sample: shrinking the sample would narrow the interval,
    which is the exact error this whole change exists to remove.
    """
    if len(clusters) != BOOTSTRAP_CLUSTER_COUNT:
        raise _fail(
            "ZERO_SHOT_BOOTSTRAP_CLUSTER_COUNT_INVALID",
            (
                f"the cluster bootstrap needs exactly {BOOTSTRAP_CLUSTER_COUNT} origin "
                f"clusters, got {len(clusters)}"
            ),
            field="clusters",
        )

    expected_ordinals = tuple(range(BOOTSTRAP_CLUSTER_COUNT))
    observed_ordinals = tuple(cluster.ordinal for cluster in clusters)
    if observed_ordinals != expected_ordinals:
        raise _fail(
            "ZERO_SHOT_BOOTSTRAP_ORDINALS_INVALID",
            (
                f"origin ordinals must be exactly {expected_ordinals[0]}.."
                f"{expected_ordinals[-1]} in ascending order, got {observed_ordinals}"
            ),
            field="clusters",
        )

    for cluster in clusters:
        if len(cluster.assets) != BOOTSTRAP_ASSETS_PER_CLUSTER:
            raise _fail(
                "ZERO_SHOT_BOOTSTRAP_CLUSTER_ASSET_COUNT_INVALID",
                (
                    f"ordinal {cluster.ordinal} carries {len(cluster.assets)} assets, "
                    f"expected exactly {BOOTSTRAP_ASSETS_PER_CLUSTER}"
                ),
                field=f"ordinal_{cluster.ordinal}",
            )
        if len(set(cluster.assets)) != len(cluster.assets):
            raise _fail(
                "ZERO_SHOT_BOOTSTRAP_DUPLICATE_ASSET",
                f"ordinal {cluster.ordinal} repeats an asset: {cluster.assets}",
                field=f"ordinal_{cluster.ordinal}",
            )
        if tuple(cluster.assets) != ASSET_PANEL:
            raise _fail(
                "ZERO_SHOT_BOOTSTRAP_ASSET_PANEL_MISMATCH",
                (
                    f"ordinal {cluster.ordinal} carries {cluster.assets}, expected the "
                    f"preregistered panel {ASSET_PANEL} in that order"
                ),
                field=f"ordinal_{cluster.ordinal}",
            )
        if len(cluster.paired_differences) != len(cluster.assets):
            raise _fail(
                "ZERO_SHOT_BOOTSTRAP_MISSING_DIFFERENCE",
                (
                    f"ordinal {cluster.ordinal} has {len(cluster.paired_differences)} paired "
                    f"differences for {len(cluster.assets)} assets"
                ),
                field=f"ordinal_{cluster.ordinal}",
            )
        for asset, difference in zip(cluster.assets, cluster.paired_differences, strict=True):
            if not math.isfinite(difference):
                raise _fail(
                    "ZERO_SHOT_BOOTSTRAP_NON_FINITE_DIFFERENCE",
                    f"ordinal {cluster.ordinal} asset {asset} has a non-finite difference",
                    field=f"ordinal_{cluster.ordinal}",
                )


def block_starts(
    clusters: int = BOOTSTRAP_CLUSTER_COUNT, length: int = BOOTSTRAP_BLOCK_LENGTH
) -> tuple[int, ...]:
    """Every valid contiguous block start. No wraparound, so 0..21."""
    return tuple(range(clusters - length + 1))


def paired_origin_moving_block_bootstrap(
    clusters: tuple[OriginCluster, ...],
    *,
    block_length: int = BOOTSTRAP_BLOCK_LENGTH,
    seed: int,
    resamples: int,
    confidence_level: float,
) -> MovingBlockBootstrapInterval:
    """Percentile moving-block bootstrap of the mean paired difference.

    The algorithm, exactly:

    1. Validate the cluster structure and the block length. Anything malformed
       is a typed failure, never a quietly different computation.
    2. Enumerate every valid contiguous block start: ``0..21`` for 25 clusters
       and a block of 4. There is no wraparound, so ordinal 24 never joins
       ordinal 0 -- those two windows are two years apart and adjoining them
       would invent a transition the data never contained.
    3. Seed one generator once from the specification.
    4. For each of ``resamples`` iterations:
       a. Draw 7 block starts with replacement from ``0..21``.
       b. Expand each start into its 4 consecutive origin clusters, in
          chronological order.
       c. Concatenate to 28 cluster positions and keep the first 25.
       d. Every retained cluster contributes all four of its assets, so the
          denominator is exactly 100 paired differences.
       e. Average those 100 observations into one resampled mean.
    5. Take the percentile interval from the sorted resampled means.

    The point estimate is the mean over all 100 measured observations, not a
    resampled quantity -- equivalently, the mean of the 25 four-asset cluster
    means. Only the uncertainty comes from the block resampling.
    """
    _validate_clusters(clusters)
    if block_length != BOOTSTRAP_BLOCK_LENGTH:
        raise _fail(
            "ZERO_SHOT_BOOTSTRAP_BLOCK_LENGTH_INVALID",
            (
                f"the block length is fixed at {BOOTSTRAP_BLOCK_LENGTH} by the benchmark "
                f"geometry, ceil({CONTEXT_CANDLES} / {ORIGIN_STRIDE}); got {block_length}. "
                "It is predeclared and may not be estimated or tuned from results"
            ),
            field="block_length",
        )
    if resamples <= 0:
        raise _fail(
            "ZERO_SHOT_BOOTSTRAP_RESAMPLES_INVALID",
            f"the resample count must be positive, got {resamples}",
            field="resamples",
        )
    if not 0.0 < confidence_level < 1.0:
        raise _fail(
            "ZERO_SHOT_BOOTSTRAP_CONFIDENCE_INVALID",
            f"the confidence level must lie strictly between 0 and 1, got {confidence_level}",
            field="confidence_level",
        )

    # The arithmetic lives in the study-neutral core. Everything above this line
    # is this study's own contract -- exactly 25 clusters, exactly the four
    # preregistered assets, block length fixed by ceil(40 / 12) -- and stays
    # here, so sharing the estimator cannot relax a guard that a published
    # result depends on. A regression test recomputes the intervals recorded in
    # the terminal artifact and requires them bit-identical.
    core = moving_block_percentile_interval(
        [sum(cluster.paired_differences) for cluster in clusters],
        observations=len(clusters) * BOOTSTRAP_ASSETS_PER_CLUSTER,
        block_length=block_length,
        seed=seed,
        resamples=resamples,
        confidence_level=confidence_level,
    )
    return MovingBlockBootstrapInterval(
        defined=True,
        cluster_count=core.cluster_count,
        assets_per_cluster=BOOTSTRAP_ASSETS_PER_CLUSTER,
        observation_count=core.observation_count,
        block_length=core.block_length,
        available_block_starts=core.available_block_starts,
        blocks_drawn_per_resample=core.blocks_drawn_per_resample,
        clusters_retained_per_resample=core.cluster_count,
        resamples=core.resamples,
        confidence_level=core.confidence_level,
        seed=core.seed,
        point_estimate=core.point_estimate,
        lower=core.lower,
        upper=core.upper,
        excludes_zero=core.excludes_zero,
        # Favorable means the whole interval sits above zero: the candidate beat
        # persistence by a margin the block resampling did not erase.
        excludes_zero_favorably=core.excludes_zero_favorably,
    )


def undefined_moving_block_bootstrap(
    *, seed: int, resamples: int, confidence_level: float, reason: str
) -> MovingBlockBootstrapInterval:
    """An interval that could not be computed, stated rather than faked.

    Used when a paired difference is undefined somewhere in the panel. The run
    still publishes, the interval is explicitly undefined, and the predeclared
    Z5 limitation carries the reason -- rather than the bootstrap silently
    running on a smaller, narrower sample.
    """
    return MovingBlockBootstrapInterval(
        defined=False,
        cluster_count=0,
        assets_per_cluster=BOOTSTRAP_ASSETS_PER_CLUSTER,
        observation_count=0,
        block_length=BOOTSTRAP_BLOCK_LENGTH,
        available_block_starts=BOOTSTRAP_AVAILABLE_BLOCK_STARTS,
        blocks_drawn_per_resample=BOOTSTRAP_BLOCKS_PER_RESAMPLE,
        clusters_retained_per_resample=0,
        resamples=resamples,
        confidence_level=confidence_level,
        seed=seed,
        undefined_reason=reason,
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
