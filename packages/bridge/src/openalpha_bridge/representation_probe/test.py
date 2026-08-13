"""The gated test phase: eligibility, scoring, bootstrap and decision.

Nothing here may run until every gate passes. The gates are checked before a
single test observation is retrieved, so a premature invocation fails closed
rather than reading data it is not entitled to see.

The bootstrap is the same algorithm the zero-shot benchmark settled on after two
pre-execution corrections -- clusters of all four assets at an ordinal, resampled
in consecutive four-origin blocks, no wraparound. It is implemented here rather
than imported because that module's validator is hard-coded to that study's
twenty-five clusters, and a completed study must not be generalised to fit a
later one.
"""

from __future__ import annotations

import math
from datetime import date
from enum import StrEnum
from typing import Any, Final

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..errors import BridgeFailure, BridgeTransformError, FailureCategory
from ..resampling import moving_block_percentile_interval
from .spec import (
    ASSET_PANEL,
    BOOTSTRAP_BLOCK_LENGTH,
    BOOTSTRAP_CONFIDENCE_LEVEL,
    BOOTSTRAP_METHOD,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CANDIDATE_FEATURE_SET,
    MINIMUM_TEST_ORIGINS_PER_ASSET,
    MINIMUM_TEST_SAMPLES,
    MINIMUM_TEST_SESSIONS,
    PROBE_THRESHOLDS,
    TEST_START_INCLUSIVE,
    ProbeThresholds,
)

__all__ = [
    "MovingBlockInterval",
    "OriginCluster",
    "ProbeDecision",
    "ProbeFinding",
    "ProbeOutcome",
    "decide",
    "paired_moving_block_bootstrap",
    "verify_test_eligibility",
]

#: The probe scores exactly this many test ordinals, so its bootstrap takes
#: exactly this many clusters. Mirrors the zero-shot study's exact-25 guard.
PROBE_CLUSTER_COUNT: Final[int] = MINIMUM_TEST_ORIGINS_PER_ASSET


def _fail(code: str, message: str, *, field: str | None = None) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            field=field,
            message=message,
        )
    )


# ------------------------------------------------------------- eligibility


class TestEligibility(BaseModel):
    """Proof that the test partition may be opened."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    test_start_inclusive: str = TEST_START_INCLUSIVE
    sessions_available: int = Field(ge=0)
    sessions_required: int = MINIMUM_TEST_SESSIONS
    origins_per_asset: int = Field(ge=0)
    origins_required_per_asset: int = MINIMUM_TEST_ORIGINS_PER_ASSET
    total_test_samples: int = Field(ge=0)
    minimum_test_samples: int = MINIMUM_TEST_SAMPLES
    boundary_verified: bool = True
    specification_verified: bool = True
    fit_artifact_verified: bool = True
    eligible: bool = True


def verify_test_eligibility(
    *,
    sessions_by_asset: dict[str, int],
    first_session_by_asset: dict[str, date],
    origins_by_asset: dict[str, int],
) -> TestEligibility:
    """Fail closed unless every declared opening condition holds.

    Called BEFORE any test observation is scored. The session counts it takes
    come from retrieval metadata, never from a projected calendar date.
    """
    boundary = date.fromisoformat(TEST_START_INCLUSIVE)

    missing = [asset for asset in ASSET_PANEL if asset not in sessions_by_asset]
    if missing:
        raise _fail(
            "PROBE_ASSET_PANEL_INCOMPLETE",
            f"missing assets {sorted(missing)}; the panel may not be shrunk",
        )

    for asset in ASSET_PANEL:
        first = first_session_by_asset.get(asset)
        if first is None or first < boundary:
            raise _fail(
                "PROBE_TEST_BOUNDARY_VIOLATED",
                (
                    f"{asset} test data begins {first}, before the sealed boundary "
                    f"{boundary}; every test session must start on or after it"
                ),
                field=asset,
            )
        available = sessions_by_asset[asset]
        if available < MINIMUM_TEST_SESSIONS:
            raise _fail(
                "PROBE_TEST_PARTITION_NOT_READY",
                (
                    f"{asset} has {available} sessions on or after {boundary}, and "
                    f"{MINIMUM_TEST_SESSIONS} are required. The minimum may not be "
                    "reduced or waived; the study waits."
                ),
                field=asset,
            )
        origins = origins_by_asset.get(asset, 0)
        if origins < MINIMUM_TEST_ORIGINS_PER_ASSET:
            raise _fail(
                "PROBE_TEST_PARTITION_NOT_READY",
                (
                    f"{asset} yields {origins} complete non-overlapping origins, and "
                    f"{MINIMUM_TEST_ORIGINS_PER_ASSET} are required"
                ),
                field=asset,
            )

    total = sum(min(origins_by_asset[a], MINIMUM_TEST_ORIGINS_PER_ASSET) for a in ASSET_PANEL)
    return TestEligibility(
        sessions_available=min(sessions_by_asset[a] for a in ASSET_PANEL),
        origins_per_asset=min(origins_by_asset[a] for a in ASSET_PANEL),
        total_test_samples=total,
    )


# ---------------------------------------------------------------- bootstrap


class OriginCluster(BaseModel):
    """One test ordinal, carrying every asset's paired difference."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    ordinal: int = Field(ge=0)
    assets: tuple[str, ...]
    paired_differences: tuple[float, ...]


class MovingBlockInterval(BaseModel):
    """A deterministic paired percentile moving-block bootstrap interval."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    defined: bool
    method: str = BOOTSTRAP_METHOD
    base_resampling_unit: str = "origin_ordinal_cluster_all_assets"
    temporal_resampling_unit: str = "four_consecutive_origin_clusters"
    cluster_count: int = Field(ge=0)
    assets_per_cluster: int = Field(ge=0)
    observation_count: int = Field(ge=0)
    block_length: int = Field(ge=0)
    available_block_starts: int = Field(ge=0)
    blocks_drawn_per_resample: int = Field(ge=0)
    resamples: int = Field(ge=0)
    confidence_level: float
    seed: int
    point_estimate: float | None = None
    lower: float | None = None
    upper: float | None = None
    excludes_zero: bool = False
    excludes_zero_favorably: bool = False
    assets_resampled_within_cluster: bool = False
    origins_resampled_independently: bool = False
    wraparound: bool = False
    undefined_reason: str | None = None


def _validate_clusters(clusters: tuple[OriginCluster, ...]) -> None:
    if len(clusters) != PROBE_CLUSTER_COUNT:
        raise _fail(
            "PROBE_BOOTSTRAP_CLUSTER_COUNT_INVALID",
            (
                f"the cluster bootstrap needs exactly {PROBE_CLUSTER_COUNT} origin "
                f"clusters, got {len(clusters)}. The probe scores exactly "
                f"{MINIMUM_TEST_ORIGINS_PER_ASSET} test origins per asset, so a different "
                "count means the panel was truncated or extended after the seal"
            ),
        )
    if tuple(c.ordinal for c in clusters) != tuple(range(len(clusters))):
        raise _fail(
            "PROBE_BOOTSTRAP_ORDINALS_INVALID",
            "ordinals must be 0..n-1 in ascending order",
        )
    for cluster in clusters:
        if tuple(cluster.assets) != ASSET_PANEL:
            raise _fail(
                "PROBE_BOOTSTRAP_ASSET_PANEL_MISMATCH",
                f"ordinal {cluster.ordinal} carries {cluster.assets}, expected {ASSET_PANEL}",
            )
        if len(cluster.paired_differences) != len(cluster.assets):
            raise _fail(
                "PROBE_BOOTSTRAP_MISSING_DIFFERENCE",
                f"ordinal {cluster.ordinal} has a missing paired difference",
            )
        for value in cluster.paired_differences:
            if not math.isfinite(value):
                raise _fail(
                    "PROBE_BOOTSTRAP_NON_FINITE_DIFFERENCE",
                    f"ordinal {cluster.ordinal} has a non-finite difference",
                )


def paired_moving_block_bootstrap(
    clusters: tuple[OriginCluster, ...],
    *,
    block_length: int = BOOTSTRAP_BLOCK_LENGTH,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
    confidence_level: float = BOOTSTRAP_CONFIDENCE_LEVEL,
) -> MovingBlockInterval:
    """Resample consecutive blocks of whole origin clusters, no wraparound.

    An ordinal is indivisible: all four assets enter together, preserving
    cross-sectional dependence. Blocks preserve the temporal dependence created
    by overlapping contexts.
    """
    _validate_clusters(clusters)
    if block_length != BOOTSTRAP_BLOCK_LENGTH:
        raise _fail(
            "PROBE_BOOTSTRAP_BLOCK_LENGTH_INVALID",
            f"block length is fixed at {BOOTSTRAP_BLOCK_LENGTH} by ceil(40/12)",
        )
    if resamples <= 0:
        raise _fail("PROBE_BOOTSTRAP_RESAMPLES_INVALID", "resamples must be positive")

    # The arithmetic lives in the study-neutral core, shared with the completed
    # zero-shot benchmark. Everything above this line is this study's own
    # contract -- exactly 16 clusters, exactly the four preregistered assets --
    # so sharing the estimator cannot relax either study's guard.
    core = moving_block_percentile_interval(
        [sum(c.paired_differences) for c in clusters],
        observations=len(clusters) * len(ASSET_PANEL),
        block_length=block_length,
        seed=seed,
        resamples=resamples,
        confidence_level=confidence_level,
    )
    return MovingBlockInterval(
        defined=True,
        cluster_count=core.cluster_count,
        assets_per_cluster=len(ASSET_PANEL),
        observation_count=core.observation_count,
        block_length=core.block_length,
        available_block_starts=core.available_block_starts,
        blocks_drawn_per_resample=core.blocks_drawn_per_resample,
        resamples=core.resamples,
        confidence_level=core.confidence_level,
        seed=core.seed,
        point_estimate=core.point_estimate,
        lower=core.lower,
        upper=core.upper,
        excludes_zero=core.excludes_zero,
        excludes_zero_favorably=core.excludes_zero_favorably,
    )


# ----------------------------------------------------------------- decision


class ProbeFinding(StrEnum):
    PROBE_INCONCLUSIVE = "PROBE_INCONCLUSIVE"
    REPRESENTATION_ADVANTAGE_OBSERVED = "REPRESENTATION_ADVANTAGE_OBSERVED"
    ADVANTAGE_OVER_RAW_ONLY = "ADVANTAGE_OVER_RAW_ONLY"
    NO_REPRESENTATION_ADVANTAGE = "NO_REPRESENTATION_ADVANTAGE"


class ProbeOutcome(StrEnum):
    PROCEED_TO_SUPERVISED_ADAPTATION_DESIGN = "PROCEED_TO_SUPERVISED_ADAPTATION_DESIGN"
    PROCEED_TO_CONFIRMATION_ON_FRESH_ORIGINS = "PROCEED_TO_CONFIRMATION_ON_FRESH_ORIGINS"
    STOP_KRONOS_REPRESENTATION_DIRECTION = "STOP_KRONOS_REPRESENTATION_DIRECTION"
    PROBE_INCONCLUSIVE = "PROBE_INCONCLUSIVE"


class ControlComparison(BaseModel):
    """The candidate measured against one control."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    control: str
    candidate_primary_error: float
    control_primary_error: float
    relative_improvement: float | None = None
    meets_margin: bool = False
    interval: MovingBlockInterval
    interval_favorable: bool = False
    beats_control: bool = False


class RuleEvaluation(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    rule_id: str
    finding: ProbeFinding
    matched: bool
    observed: dict[str, Any]
    detail: str


class ProbeDecision(BaseModel):
    """The complete decision layer for one probe run."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    evaluated_on: str = "untouched_test_partition"
    evaluated_for: str = CANDIDATE_FEATURE_SET
    comparisons: tuple[ControlComparison, ...]
    supporting_assets: tuple[str, ...]
    evaluations: tuple[RuleEvaluation, ...]
    matched_findings: tuple[ProbeFinding, ...]
    outcome: ProbeOutcome
    thresholds: ProbeThresholds
    limitations: tuple[str, ...] = ()
    authorizes_fine_tuning: bool = False
    power_qualification: str = (
        "A negative result means no advantage was detectable at this power, not that "
        "the representation is uninformative."
    )
    structural_validity_used_in_any_rule: bool = False


def relative_improvement(candidate: float, control: float) -> float | None:
    """1 - candidate/control. Positive means the candidate is better."""
    if control == 0.0 or not math.isfinite(control) or not math.isfinite(candidate):
        return None
    return 1.0 - candidate / control


def decide(
    *,
    comparisons: tuple[ControlComparison, ...],
    supporting_assets: tuple[str, ...],
    limitations: tuple[str, ...] = (),
    thresholds: ProbeThresholds = PROBE_THRESHOLDS,
) -> ProbeDecision:
    """Apply R0 to R3 in their preregistered precedence."""
    by_control = {c.control: c for c in comparisons}
    beats_all = bool(comparisons) and all(c.beats_control for c in comparisons)
    enough_assets = len(supporting_assets) >= thresholds.minimum_supporting_assets

    r0 = bool(limitations)
    r1 = (not r0) and beats_all and enough_assets
    beats_raw = by_control.get("raw_ohlcv")
    beats_intercept = by_control.get("intercept_only")
    beats_engineered = by_control.get("engineered_features")
    r2 = (
        not r0
        and not r1
        and beats_raw is not None
        and beats_intercept is not None
        and beats_engineered is not None
        and beats_raw.beats_control
        and beats_intercept.beats_control
        and not beats_engineered.beats_control
    )
    r3 = not r0 and not r1 and not r2

    evaluations = (
        RuleEvaluation(
            rule_id="R0",
            finding=ProbeFinding.PROBE_INCONCLUSIVE,
            matched=r0,
            observed={"limitation_count": float(len(limitations))},
            detail="predeclared limitations that prevent interpretation; evaluated first",
        ),
        RuleEvaluation(
            rule_id="R1",
            finding=ProbeFinding.REPRESENTATION_ADVANTAGE_OBSERVED,
            matched=r1,
            observed={
                "controls_beaten": float(sum(c.beats_control for c in comparisons)),
                "controls_evaluated": float(len(comparisons)),
                "supporting_assets": float(len(supporting_assets)),
                "minimum_relative_improvement": (
                    thresholds.minimum_relative_improvement_over_every_control
                ),
                "minimum_supporting_assets": float(thresholds.minimum_supporting_assets),
            },
            detail=(
                "margin over EVERY control, paired moving-block interval excluding zero "
                "favorably against EVERY control, and at least three of four assets"
            ),
        ),
        RuleEvaluation(
            rule_id="R2",
            finding=ProbeFinding.ADVANTAGE_OVER_RAW_ONLY,
            matched=r2,
            observed={
                "beats_raw_ohlcv": float(bool(beats_raw and beats_raw.beats_control)),
                "beats_intercept_only": float(
                    bool(beats_intercept and beats_intercept.beats_control)
                ),
                "beats_engineered_features": float(
                    bool(beats_engineered and beats_engineered.beats_control)
                ),
            },
            detail="beats raw OHLCV and intercept-only but not the engineered features",
        ),
        RuleEvaluation(
            rule_id="R3",
            finding=ProbeFinding.NO_REPRESENTATION_ADVANTAGE,
            matched=r3,
            observed={"controls_beaten": float(sum(c.beats_control for c in comparisons))},
            detail="neither R1 nor R2 matched",
        ),
    )

    matched = tuple(e.finding for e in evaluations if e.matched)
    if r0:
        outcome = ProbeOutcome.PROBE_INCONCLUSIVE
    elif r1 or r2:
        outcome = ProbeOutcome.PROCEED_TO_CONFIRMATION_ON_FRESH_ORIGINS
    else:
        outcome = ProbeOutcome.STOP_KRONOS_REPRESENTATION_DIRECTION

    return ProbeDecision(
        comparisons=comparisons,
        supporting_assets=supporting_assets,
        evaluations=evaluations,
        matched_findings=matched,
        outcome=outcome,
        thresholds=thresholds,
        limitations=limitations,
    )


def build_comparison(
    *,
    control: str,
    candidate_errors: np.ndarray,
    control_errors: np.ndarray,
    clusters: tuple[OriginCluster, ...],
    thresholds: ProbeThresholds = PROBE_THRESHOLDS,
) -> ControlComparison:
    """Score the candidate against one control and interval the difference."""
    candidate_mae = float(np.mean(candidate_errors))
    control_mae = float(np.mean(control_errors))
    improvement = relative_improvement(candidate_mae, control_mae)
    interval = paired_moving_block_bootstrap(clusters)
    meets = improvement is not None and improvement >= (
        thresholds.minimum_relative_improvement_over_every_control
    )
    favorable = interval.defined and interval.excludes_zero_favorably
    return ControlComparison(
        control=control,
        candidate_primary_error=candidate_mae,
        control_primary_error=control_mae,
        relative_improvement=improvement,
        meets_margin=meets,
        interval=interval,
        interval_favorable=favorable,
        beats_control=meets and favorable,
    )


__all__ += ["ControlComparison", "RuleEvaluation", "TestEligibility", "build_comparison",
            "relative_improvement"]
