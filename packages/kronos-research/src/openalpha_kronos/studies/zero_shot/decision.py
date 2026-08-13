"""The preregistered decision rules Z1 to Z5, and nothing else.

Two properties matter here and are enforced structurally rather than promised.

First, no structural-validity quantity reaches this module. Nothing in the
signature of :func:`decide` can carry an invalid-candle count, so a structural
number cannot influence an outcome even by accident. The completed studies
already answered that question.

Second, Z5 is evaluated first and is decisive. An execution that could not be
interpreted does not get to produce a skill claim in either direction.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

from .aggregation import MovingBlockBootstrapInterval
from .spec import ZERO_SHOT_THRESHOLDS, ZeroShotThresholds

__all__ = [
    "AssetSupport",
    "ConfigurationEvidence",
    "GenerationDirection",
    "RuleEvaluation",
    "ZeroShotDecision",
    "ZeroShotFinding",
    "ZeroShotOutcome",
    "decide",
]


class ZeroShotFinding(StrEnum):
    """Every finding a rule can raise."""

    ZERO_SHOT_SKILL_OBSERVED = "ZERO_SHOT_SKILL_OBSERVED"
    ISOLATED_CONFIGURATION_EFFECT = "ISOLATED_CONFIGURATION_EFFECT"
    ASSET_SPECIFIC_EFFECT = "ASSET_SPECIFIC_EFFECT"
    NO_ZERO_SHOT_SKILL = "NO_ZERO_SHOT_SKILL"
    BENCHMARK_INCONCLUSIVE = "BENCHMARK_INCONCLUSIVE"


class ZeroShotOutcome(StrEnum):
    """The four predeclared decision outcomes."""

    PROCEED_TO_FROZEN_REPRESENTATION_PROBE = "PROCEED_TO_FROZEN_REPRESENTATION_PROBE"
    PROCEED_TO_CONFIGURATION_CONFIRMATION = "PROCEED_TO_CONFIGURATION_CONFIRMATION"
    STOP_KRONOS_ZERO_SHOT_DIRECTION = "STOP_KRONOS_ZERO_SHOT_DIRECTION"
    BENCHMARK_INCONCLUSIVE = "BENCHMARK_INCONCLUSIVE"


class GenerationDirection(StrEnum):
    """What the result says about the zero-shot generation direction itself."""

    CONTINUE_UNDER_CONFIRMATION = "CONTINUE_UNDER_CONFIRMATION"
    STOP_KRONOS_ZERO_SHOT_DIRECTION = "STOP_KRONOS_ZERO_SHOT_DIRECTION"


class AssetSupport(BaseModel):
    """Whether one asset supports one configuration, and on what numbers."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    asset: str
    origins_scored: int = Field(ge=0)
    median_relative_skill: float | None = None
    fraction_beating_persistence: float | None = None
    supports: bool = False


class ConfigurationEvidence(BaseModel):
    """Everything Z1 needs about one temperature configuration."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    label: str
    temperature: float
    origins_scored: int = Field(ge=0)
    pooled_median_relative_skill: float | None = None
    pooled_fraction_beating_persistence: float | None = None
    assets: tuple[AssetSupport, ...]
    bootstrap: MovingBlockBootstrapInterval

    @property
    def supporting_assets(self) -> tuple[str, ...]:
        return tuple(asset.asset for asset in self.assets if asset.supports)

    def z1_conditions(self, thresholds: ZeroShotThresholds) -> dict[str, bool]:
        """The four Z1 conditions, each evaluated on its own."""
        median = self.pooled_median_relative_skill
        fraction = self.pooled_fraction_beating_persistence
        return {
            "median_relative_skill_positive": (
                median is not None and median > thresholds.minimum_median_relative_skill
            ),
            "enough_origins_beat_persistence": (
                fraction is not None
                and fraction >= thresholds.minimum_fraction_of_origins_beating_persistence
            ),
            "enough_supporting_assets": (
                len(self.supporting_assets) >= thresholds.minimum_supporting_assets
            ),
            "moving_block_bootstrap_excludes_zero_favorably": (
                self.bootstrap.defined and self.bootstrap.excludes_zero_favorably
            ),
        }

    def satisfies_z1(self, thresholds: ZeroShotThresholds) -> bool:
        return all(self.z1_conditions(thresholds).values())


class RuleEvaluation(BaseModel):
    """One rule, its verdict, and the numbers it was decided on."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    rule_id: str
    finding: ZeroShotFinding
    matched: bool
    observed: dict[str, Any]
    detail: str


class ZeroShotDecision(BaseModel):
    """The complete decision layer for one benchmark run."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    evaluated_on: str = "primary_candidate"
    primary_candidate: str = "ensemble_mean"
    configurations: tuple[ConfigurationEvidence, ...]
    evaluations: tuple[RuleEvaluation, ...]
    matched_findings: tuple[ZeroShotFinding, ...]
    outcome: ZeroShotOutcome
    zero_shot_generation_direction: GenerationDirection
    thresholds: ZeroShotThresholds
    limitations: tuple[str, ...] = ()
    #: Always true and stated in the specification. Direct generation quality
    #: and representation usefulness are different hypotheses, so a direct
    #: zero-shot failure does not close off the representation probe.
    frozen_representation_probe_permitted: bool = True
    structural_validity_used_in_any_rule: bool = False


_Z5_DETAIL: Final[str] = (
    "predeclared data, provider, execution or support limitations that prevent interpretation"
)


def decide(
    *,
    configurations: tuple[ConfigurationEvidence, ...],
    limitations: tuple[str, ...] = (),
    thresholds: ZeroShotThresholds = ZERO_SHOT_THRESHOLDS,
) -> ZeroShotDecision:
    """Apply Z1 to Z5 in their preregistered precedence.

    ``limitations`` carries the predeclared Z5 triggers the runner observed. It
    is the only way a run can be declared inconclusive: there is no threshold
    here that can be relaxed to reach that verdict after the fact.
    """
    favorable = tuple(c for c in configurations if c.satisfies_z1(thresholds))
    single_asset_configurations = tuple(
        c for c in configurations if len(c.supporting_assets) == 1
    )

    z5_matched = bool(limitations)
    z1_matched = bool(favorable) and not z5_matched
    z2_matched = z1_matched and len(favorable) == 1 and len(configurations) > 1
    z3_matched = (
        not z5_matched and not favorable and bool(single_asset_configurations)
    )
    z4_matched = not z5_matched and not favorable

    evaluations = (
        RuleEvaluation(
            rule_id="Z5",
            finding=ZeroShotFinding.BENCHMARK_INCONCLUSIVE,
            matched=z5_matched,
            observed={"limitation_count": float(len(limitations))},
            detail=_Z5_DETAIL,
        ),
        RuleEvaluation(
            rule_id="Z1",
            finding=ZeroShotFinding.ZERO_SHOT_SKILL_OBSERVED,
            matched=z1_matched,
            observed={
                "favorable_configurations": float(len(favorable)),
                "configurations_evaluated": float(len(configurations)),
                "minimum_median_relative_skill": thresholds.minimum_median_relative_skill,
                "minimum_fraction_of_origins_beating_persistence": (
                    thresholds.minimum_fraction_of_origins_beating_persistence
                ),
                "minimum_supporting_assets": float(thresholds.minimum_supporting_assets),
            },
            detail=(
                "median relative skill, the fraction of origins beating persistence, "
                "per-asset support and the paired MOVING-BLOCK bootstrap interval over "
                "consecutive four-origin blocks, all four required"
            ),
        ),
        RuleEvaluation(
            rule_id="Z2",
            finding=ZeroShotFinding.ISOLATED_CONFIGURATION_EFFECT,
            matched=z2_matched,
            observed={"favorable_configurations": float(len(favorable))},
            detail="exactly one temperature configuration satisfies Z1 and the other does not",
        ),
        RuleEvaluation(
            rule_id="Z3",
            finding=ZeroShotFinding.ASSET_SPECIFIC_EFFECT,
            matched=z3_matched,
            observed={
                "favorable_configurations": float(len(favorable)),
                "single_asset_configurations": float(len(single_asset_configurations)),
            },
            detail=(
                "no configuration satisfies Z1, and for at least one configuration exactly "
                "one asset supports it"
            ),
        ),
        RuleEvaluation(
            rule_id="Z4",
            finding=ZeroShotFinding.NO_ZERO_SHOT_SKILL,
            matched=z4_matched,
            observed={"favorable_configurations": float(len(favorable))},
            detail="no configuration satisfies Z1",
        ),
    )

    matched = tuple(
        evaluation.finding for evaluation in evaluations if evaluation.matched
    )

    if z5_matched:
        outcome = ZeroShotOutcome.BENCHMARK_INCONCLUSIVE
        direction = GenerationDirection.CONTINUE_UNDER_CONFIRMATION
    elif z1_matched or z3_matched:
        outcome = ZeroShotOutcome.PROCEED_TO_CONFIGURATION_CONFIRMATION
        direction = GenerationDirection.CONTINUE_UNDER_CONFIRMATION
    else:
        # Z4 with no asset-specific remnant. The direction stops; the
        # representation probe is a different hypothesis and stays open.
        outcome = ZeroShotOutcome.PROCEED_TO_FROZEN_REPRESENTATION_PROBE
        direction = GenerationDirection.STOP_KRONOS_ZERO_SHOT_DIRECTION

    return ZeroShotDecision(
        configurations=configurations,
        evaluations=evaluations,
        matched_findings=matched,
        outcome=outcome,
        zero_shot_generation_direction=direction,
        thresholds=thresholds,
        limitations=limitations,
    )
