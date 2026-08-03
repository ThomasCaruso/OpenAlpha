"""Metrics, aggregation and the deterministic paired cluster bootstrap.

Pooled error is reported but never decides anything on its own: a handful of
high-volatility windows can dominate a pooled mean, so every decision quantity
here is either a median, a fraction of origins, or a paired interval.

The bootstrap resamples **origin ordinals**, not individual asset-origin rows,
and not forecast steps. SPY, QQQ, IWM and DIA are evaluated over the same
chronological windows, so their errors within an origin are cross-sectionally
dependent. Treating the four as four independent draws would manufacture
confidence the data does not contain and could let a too-narrow interval satisfy
Z1's favorable-exclusion requirement on its own. Each resample therefore draws
whole origin clusters, carrying all four assets together.

There is deliberately no flat asset-origin bootstrap in this module. Omitting it
is what makes "the decision layer cannot read one" a structural fact rather than
a convention: the function does not exist, and :class:`ClusterBootstrapInterval`
is the only interval type the decision layer accepts.
"""

from __future__ import annotations

import math
import random
import statistics
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..diagnostic.metrics import DirectionalAccuracy, directional_accuracy
from ..diagnostic.official_input import OfficialRow
from ..errors import BridgeFailure, BridgeTransformError, FailureCategory
from .spec import ASSET_PANEL, ORIGINS_PER_ASSET

__all__ = [
    "BOOTSTRAP_ASSETS_PER_CLUSTER",
    "BOOTSTRAP_CLUSTER_COUNT",
    "BOOTSTRAP_METHOD",
    "BOOTSTRAP_OBSERVATION_COUNT",
    "BOOTSTRAP_UNIT_OF_RESAMPLING",
    "ClusterBootstrapInterval",
    "DistributionSummary",
    "ExtendedForecastMetrics",
    "OriginCluster",
    "StepSummary",
    "extended_metrics",
    "paired_cluster_bootstrap",
    "relative_skill",
    "step_summaries",
    "summarize",
    "undefined_cluster_bootstrap",
]

BOOTSTRAP_METHOD: Final[str] = "paired_percentile_cluster_bootstrap_over_origin_ordinals"
BOOTSTRAP_UNIT_OF_RESAMPLING: Final[str] = "origin_ordinal_cluster_all_assets"
BOOTSTRAP_CLUSTER_COUNT: Final[int] = ORIGINS_PER_ASSET
BOOTSTRAP_ASSETS_PER_CLUSTER: Final[int] = len(ASSET_PANEL)
BOOTSTRAP_OBSERVATION_COUNT: Final[int] = BOOTSTRAP_CLUSTER_COUNT * BOOTSTRAP_ASSETS_PER_CLUSTER


def _fail(code: str, message: str, *, field: str | None = None) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(
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


class ClusterBootstrapInterval(BaseModel):
    """A deterministic paired percentile cluster bootstrap over origin ordinals.

    The only interval type the decision layer accepts. Its metadata states the
    resampling unit explicitly so a reader never has to infer whether
    cross-asset dependence was preserved.
    """

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    defined: bool
    method: str = BOOTSTRAP_METHOD
    unit_of_resampling: str = BOOTSTRAP_UNIT_OF_RESAMPLING
    cluster_count: int = Field(ge=0)
    assets_per_cluster: int = Field(ge=0)
    observation_count: int = Field(ge=0)
    resamples: int = Field(ge=0)
    confidence_level: float
    seed: int
    point_estimate: float | None = None
    lower: float | None = None
    upper: float | None = None
    excludes_zero: bool = False
    excludes_zero_favorably: bool = False
    #: Stated so no reader has to check the code to know assets were not
    #: resampled independently inside an ordinal.
    assets_resampled_within_cluster: Literal[False] = False
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


def paired_cluster_bootstrap(
    clusters: tuple[OriginCluster, ...],
    *,
    seed: int,
    resamples: int,
    confidence_level: float,
) -> ClusterBootstrapInterval:
    """Percentile bootstrap of the mean paired difference, clustered by ordinal.

    The algorithm, exactly:

    1. Validate the cluster structure. Anything malformed is a typed failure.
    2. For each of ``resamples`` iterations, draw ``cluster_count`` ordinals with
       replacement from a generator seeded once from the specification.
    3. When an ordinal is drawn, take **all four** of its asset paired
       differences. Assets are never drawn, weighted or omitted individually,
       and never selected on their result.
    4. Average the resulting ``cluster_count * assets_per_cluster``
       observations to get one resampled mean.
    5. Take the percentile interval from the sorted resampled means.

    The point estimate is the mean over all observations as actually measured,
    not a resampled quantity. Only the uncertainty comes from the 25 clusters.
    """
    _validate_clusters(clusters)
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

    cluster_count = len(clusters)
    per_cluster_totals = [sum(cluster.paired_differences) for cluster in clusters]
    observations = cluster_count * BOOTSTRAP_ASSETS_PER_CLUSTER

    generator = random.Random(seed)
    means: list[float] = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(cluster_count):
            # One draw selects an ordinal, and an ordinal is indivisible: its
            # four asset differences enter together, which is what preserves the
            # cross-sectional dependence between the four ETFs.
            total += per_cluster_totals[generator.randrange(cluster_count)]
        means.append(total / observations)
    means.sort()

    tail = (1.0 - confidence_level) / 2.0
    lower = means[math.floor(tail * (resamples - 1))]
    upper = means[math.ceil((1.0 - tail) * (resamples - 1))]
    point = sum(per_cluster_totals) / observations
    excludes = (lower > 0.0 and upper > 0.0) or (lower < 0.0 and upper < 0.0)
    return ClusterBootstrapInterval(
        defined=True,
        cluster_count=cluster_count,
        assets_per_cluster=BOOTSTRAP_ASSETS_PER_CLUSTER,
        observation_count=observations,
        resamples=resamples,
        confidence_level=confidence_level,
        seed=seed,
        point_estimate=point,
        lower=lower,
        upper=upper,
        excludes_zero=excludes,
        # Favorable means the whole interval sits above zero: the candidate beat
        # persistence by a margin the cluster resampling did not erase.
        excludes_zero_favorably=lower > 0.0,
    )


def undefined_cluster_bootstrap(
    *, seed: int, resamples: int, confidence_level: float, reason: str
) -> ClusterBootstrapInterval:
    """An interval that could not be computed, stated rather than faked.

    Used when a paired difference is undefined somewhere in the panel. The run
    still publishes, the interval is explicitly undefined, and the predeclared
    Z5 limitation carries the reason -- rather than the bootstrap silently
    running on a smaller, narrower sample.
    """
    return ClusterBootstrapInterval(
        defined=False,
        cluster_count=0,
        assets_per_cluster=BOOTSTRAP_ASSETS_PER_CLUSTER,
        observation_count=0,
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
