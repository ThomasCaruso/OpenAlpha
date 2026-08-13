"""The preregistered decision rules, v3.

v2 stopped at the first matching rule. That was wrong for this diagnostic: the
properties it measures are not mutually exclusive, and the rule most likely to
match first, round-trip invalidity, is also the least decision-relevant. A run
in which the tokenizer round trip produced a single invalid candle would have
terminated evaluation before rollout support, projection, filtering or baseline
skill were ever considered, and the one origin available would have been spent
without selecting a branch.

v3 evaluates every rule. One of them becomes the ``primary_conclusion`` by a
fixed priority; the rest remain in ``matched_findings``, and every rule,
matched or not, stays in ``evaluations``. ``recommended_next_experiment`` is
derived from the whole set rather than from the primary alone.

Every label is descriptive and local to this origin. None asserts a cause, a
property of the model's token support, or a general property of repair.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict

from ..mini.spec import THRESHOLDS, DiagnosticThresholds
from .methods import MethodAResult, MethodBResult, MethodCResult, MethodDResult

__all__ = [
    "PRIMARY_PRIORITY",
    "RETIRED_LABELS",
    "ConclusionOutcome",
    "DiagnosticConclusion",
    "NextExperiment",
    "ReproducibilityCheck",
    "RuleEvaluation",
    "decide",
]


class DiagnosticConclusion(StrEnum):
    """The complete vocabulary. Descriptive and local, never causal."""

    # Operational
    DIAGNOSTIC_OPERATIONAL_FAILURE = "DIAGNOSTIC_OPERATIONAL_FAILURE"
    REPRODUCIBILITY_FAILURE = "REPRODUCIBILITY_FAILURE"

    # Round trip
    ROUNDTRIP_MATERIAL_INVALIDITY = "ROUNDTRIP_MATERIAL_INVALIDITY"
    ROUNDTRIP_STRUCTURAL_INVALIDITY_OBSERVED = "ROUNDTRIP_STRUCTURAL_INVALIDITY_OBSERVED"
    ROUNDTRIP_INVALIDITY_ON_UNCLIPPED_INPUTS = "ROUNDTRIP_INVALIDITY_ON_UNCLIPPED_INPUTS"
    ROUNDTRIP_INVALIDITY_CONFINED_TO_CLIPPED_INPUTS = (
        "ROUNDTRIP_INVALIDITY_CONFINED_TO_CLIPPED_INPUTS"
    )
    #: Descriptive only. Never primary, never controls the recommendation.
    MATERIAL_CLIPPING_EXPOSURE_OBSERVED = "MATERIAL_CLIPPING_EXPOSURE_OBSERVED"
    #: Only when every invalid reconstruction had a clipped input, so the clip
    #: and the decoder genuinely cannot be told apart.
    ROUNDTRIP_INTERPRETATION_CONFOUNDED_BY_CLIPPING = (
        "ROUNDTRIP_INTERPRETATION_CONFOUNDED_BY_CLIPPING"
    )

    # Rollouts and filtering
    NO_VALID_ROLLOUTS_OBSERVED = "NO_VALID_ROLLOUTS_OBSERVED"
    LOW_VALID_ROLLOUT_FRACTION = "LOW_VALID_ROLLOUT_FRACTION"
    VALID_ONLY_ENSEMBLE_MEETS_IMPROVEMENT_THRESHOLD = (
        "VALID_ONLY_ENSEMBLE_MEETS_IMPROVEMENT_THRESHOLD"
    )
    PROJECTION_RESTORES_VALIDITY_WITHOUT_THRESHOLD_IMPROVEMENT = (
        "PROJECTION_RESTORES_VALIDITY_WITHOUT_THRESHOLD_IMPROVEMENT"
    )

    # Baseline skill
    NO_SKILL_AGAINST_PERSISTENCE = "NO_SKILL_AGAINST_PERSISTENCE"
    SKILL_AGAINST_PERSISTENCE_OBSERVED = "SKILL_AGAINST_PERSISTENCE_OBSERVED"

    # Defaults
    NO_PREREGISTERED_EFFECT_DETECTED = "NO_PREREGISTERED_EFFECT_DETECTED"
    DIAGNOSTIC_INCONCLUSIVE = "DIAGNOSTIC_INCONCLUSIVE"


class NextExperiment(StrEnum):
    """What this origin suggests doing next. A suggestion, not an authorization."""

    REPAIR_REPRODUCIBILITY = "REPAIR_REPRODUCIBILITY"
    INVESTIGATE_CLIPPING_EXPOSURE = "INVESTIGATE_CLIPPING_EXPOSURE"
    CONSTRAINED_OUTPUT_DECODER = "CONSTRAINED_OUTPUT_DECODER"
    CONSTRAINED_TOKEN_SEARCH = "CONSTRAINED_TOKEN_SEARCH"
    REJECTION_SAMPLING_STUDY = "REJECTION_SAMPLING_STUDY"
    ABANDON_STRUCTURAL_VALIDITY_DIRECTION = "ABANDON_STRUCTURAL_VALIDITY_DIRECTION"
    NONE_DIAGNOSTIC_INCONCLUSIVE = "NONE_DIAGNOSTIC_INCONCLUSIVE"


#: Priority for selecting the primary conclusion.
#:
#: Two labels are deliberately absent. R2's bare structural observation does
#: not select a branch on its own. Material clipping exposure is an
#: observation about the input, not about the reconstruction, and letting it
#: become primary would let it displace a finding that actually explains
#: something.
PRIMARY_PRIORITY: tuple[DiagnosticConclusion, ...] = (
    DiagnosticConclusion.DIAGNOSTIC_OPERATIONAL_FAILURE,
    DiagnosticConclusion.REPRODUCIBILITY_FAILURE,
    DiagnosticConclusion.ROUNDTRIP_INTERPRETATION_CONFOUNDED_BY_CLIPPING,
    DiagnosticConclusion.ROUNDTRIP_MATERIAL_INVALIDITY,
    DiagnosticConclusion.NO_VALID_ROLLOUTS_OBSERVED,
    DiagnosticConclusion.VALID_ONLY_ENSEMBLE_MEETS_IMPROVEMENT_THRESHOLD,
    DiagnosticConclusion.LOW_VALID_ROLLOUT_FRACTION,
    DiagnosticConclusion.PROJECTION_RESTORES_VALIDITY_WITHOUT_THRESHOLD_IMPROVEMENT,
    DiagnosticConclusion.NO_SKILL_AGAINST_PERSISTENCE,
    DiagnosticConclusion.DIAGNOSTIC_INCONCLUSIVE,
    DiagnosticConclusion.NO_PREREGISTERED_EFFECT_DETECTED,
)

RETIRED_LABELS: dict[str, str] = {
    "INVALIDITY_NOT_CAUSAL_TO_FORECAST_ERROR": (
        "a causal claim; one origin and 64 stochastic rollouts cannot identify a cause"
    ),
    "GENERATED_TOKEN_SUPPORT_DEFECT": (
        "a claim about the generative distribution's support, measured from one sample"
    ),
    "TOKENIZER_DECODER_DEFECT": (
        "asserts a defect; the observation available is that a round trip was invalid"
    ),
    "VALIDITY_FILTER_IMPROVES_FORECAST": (
        "generalises beyond one window; the replacement states a threshold was met"
    ),
    "VALIDITY_REPAIR_ONLY": ("reads as a general finding about repair rather than one observation"),
}


class ReproducibilityCheck(BaseModel):
    """Method B against Method D rollout zero. Same seed, same settings."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    performed: bool
    agrees: bool
    coarse_tokens_agree: bool | None = None
    fine_tokens_agree: bool | None = None
    sampling_log_probabilities_agree: bool | None = None
    raw_decoded_suffix_agrees: bool | None = None
    validity_agrees: bool | None = None
    forecast_metrics_agree: bool | None = None
    detail: str | None = None


class RuleEvaluation(BaseModel):
    """One rule, its inputs, and whether it fired. Every rule is recorded."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    rule_id: str
    finding: DiagnosticConclusion
    matched: bool
    detail: str
    observed: dict[str, float | None]


class ConclusionOutcome(BaseModel):
    """Several simultaneous findings, one of which is primary."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    primary_conclusion: DiagnosticConclusion
    primary_rule_id: str
    #: Every finding that matched, in rule order. May be empty.
    matched_findings: tuple[DiagnosticConclusion, ...]
    #: Every rule, matched or not.
    evaluations: tuple[RuleEvaluation, ...]
    recommended_next_experiment: NextExperiment
    reproducibility: ReproducibilityCheck
    thresholds: DiagnosticThresholds
    forecast_origins: Literal[1] = 1
    authorizes_training: Literal[False] = False
    authorizes_stage_b: Literal[False] = False
    authorizes_stage_c: Literal[False] = False
    authorizes_test_opening: Literal[False] = False
    authorizes_production_inference: Literal[False] = False
    authorizes_trading_claims: Literal[False] = False


def _primary(error) -> float | None:
    if error is None or not error.defined:
        return None
    return error.close_return_mae


def _skill(comparison) -> float | None:
    if comparison is None or not comparison.defined:
        return None
    return comparison.close_return_skill


def _recommend(
    matched: set[DiagnosticConclusion], thresholds: DiagnosticThresholds
) -> NextExperiment:
    """Derived from the whole finding set, not from the primary alone."""
    C = DiagnosticConclusion
    if C.DIAGNOSTIC_OPERATIONAL_FAILURE in matched:
        return NextExperiment.NONE_DIAGNOSTIC_INCONCLUSIVE
    if C.REPRODUCIBILITY_FAILURE in matched:
        return NextExperiment.REPAIR_REPRODUCIBILITY
    if C.ROUNDTRIP_INTERPRETATION_CONFOUNDED_BY_CLIPPING in matched:
        return NextExperiment.INVESTIGATE_CLIPPING_EXPOSURE
    # MATERIAL_CLIPPING_EXPOSURE_OBSERVED is never consulted here. Exposure on
    # its own must not override a recommendation a substantive finding made.
    # No skill anywhere makes the structural direction moot regardless of what
    # the validity findings say: repairing geometry cannot create signal.
    if C.NO_SKILL_AGAINST_PERSISTENCE in matched:
        return NextExperiment.ABANDON_STRUCTURAL_VALIDITY_DIRECTION
    if C.ROUNDTRIP_INVALIDITY_ON_UNCLIPPED_INPUTS in matched:
        return NextExperiment.CONSTRAINED_OUTPUT_DECODER
    if C.NO_VALID_ROLLOUTS_OBSERVED in matched:
        return NextExperiment.CONSTRAINED_TOKEN_SEARCH
    if C.VALID_ONLY_ENSEMBLE_MEETS_IMPROVEMENT_THRESHOLD in matched:
        return NextExperiment.REJECTION_SAMPLING_STUDY
    if C.LOW_VALID_ROLLOUT_FRACTION in matched:
        return NextExperiment.CONSTRAINED_TOKEN_SEARCH
    if C.DIAGNOSTIC_INCONCLUSIVE in matched:
        return NextExperiment.NONE_DIAGNOSTIC_INCONCLUSIVE
    return NextExperiment.ABANDON_STRUCTURAL_VALIDITY_DIRECTION


def decide(
    *,
    method_a: MethodAResult | None,
    method_b: MethodBResult | None,
    method_c: MethodCResult | None,
    method_d: MethodDResult | None,
    reproducibility: ReproducibilityCheck | None = None,
    operational_failure: bool = False,
    thresholds: DiagnosticThresholds = THRESHOLDS,
) -> ConclusionOutcome:
    """Evaluate every rule, then select a primary conclusion by priority."""
    C = DiagnosticConclusion
    evaluations: list[RuleEvaluation] = []

    def record(
        rule_id: str,
        finding: DiagnosticConclusion,
        matched: bool,
        detail: str,
        observed: dict[str, float | None],
    ) -> None:
        evaluations.append(
            RuleEvaluation(
                rule_id=rule_id,
                finding=finding,
                matched=matched,
                detail=detail,
                observed=observed,
            )
        )

    check = reproducibility or ReproducibilityCheck(
        performed=False, agrees=False, detail="no comparison was supplied"
    )

    missing = [
        name
        for name, value in (("A", method_a), ("B", method_b), ("C", method_c), ("D", method_d))
        if value is None
    ]
    incomplete = operational_failure or bool(missing)
    record(
        "R0",
        C.DIAGNOSTIC_OPERATIONAL_FAILURE,
        incomplete,
        (
            "a method raised an operational failure"
            if operational_failure
            else f"methods did not complete: {', '.join(missing)}"
            if missing
            else "every method completed"
        ),
        {"missing_methods": float(len(missing))},
    )

    if incomplete:
        matched = {C.DIAGNOSTIC_OPERATIONAL_FAILURE}
        return ConclusionOutcome(
            primary_conclusion=C.DIAGNOSTIC_OPERATIONAL_FAILURE,
            primary_rule_id="R0",
            matched_findings=(C.DIAGNOSTIC_OPERATIONAL_FAILURE,),
            evaluations=tuple(evaluations),
            recommended_next_experiment=_recommend(matched, thresholds),
            reproducibility=check,
            thresholds=thresholds,
        )

    assert method_a is not None and method_b is not None
    assert method_c is not None and method_d is not None

    # R0b - the two methods that share a seed must have produced the same path.
    record(
        "R0b",
        C.REPRODUCIBILITY_FAILURE,
        check.performed and not check.agrees,
        "Method B against Method D rollout zero, which share a seed and settings",
        {"performed": float(check.performed), "agrees": float(check.agrees)},
    )

    # --- round trip -------------------------------------------------------
    fraction = method_a.validity_all.invalid_candle_fraction
    invalid_count = method_a.validity_all.invalid_candle_count
    record(
        "R1",
        C.ROUNDTRIP_MATERIAL_INVALIDITY,
        fraction > thresholds.roundtrip_material_invalid_fraction,
        "round-trip invalid candle fraction against the materiality threshold",
        {
            "invalid_candle_fraction": fraction,
            "materiality_threshold": thresholds.roundtrip_material_invalid_fraction,
            "invalid_candle_count": float(invalid_count),
        },
    )
    record(
        "R2",
        C.ROUNDTRIP_STRUCTURAL_INVALIDITY_OBSERVED,
        invalid_count > 0,
        "any invalid round-trip candle at all; a finding, never primary",
        {"invalid_candle_count": float(invalid_count), "invalid_candle_fraction": fraction},
    )

    clip_fraction = method_a.clipping_all.row_clipped_fraction
    unclipped_invalid = method_a.invalid_rows_with_unclipped_input
    clipped_invalid = method_a.invalid_rows_with_clipped_input
    record(
        "R2a",
        C.ROUNDTRIP_INVALIDITY_ON_UNCLIPPED_INPUTS,
        unclipped_invalid > 0,
        "invalid reconstructions whose input was never touched by the clip",
        {
            "invalid_rows_with_unclipped_input": float(unclipped_invalid),
            "unclipped_input_rows": float(method_a.unclipped_input_rows),
        },
    )
    record(
        "R2b",
        C.ROUNDTRIP_INVALIDITY_CONFINED_TO_CLIPPED_INPUTS,
        invalid_count > 0 and unclipped_invalid == 0 and clipped_invalid > 0,
        "every invalid reconstruction had a clipped input, so the clip is confounded",
        {
            "invalid_rows_with_clipped_input": float(clipped_invalid),
            "invalid_rows_with_unclipped_input": float(unclipped_invalid),
        },
    )
    material_clipping = clip_fraction > thresholds.material_clipping_row_fraction
    record(
        "R2c",
        C.MATERIAL_CLIPPING_EXPOSURE_OBSERVED,
        material_clipping,
        "clipped input rows against the materiality threshold; descriptive only",
        {
            "row_clipped_fraction": clip_fraction,
            "threshold": thresholds.material_clipping_row_fraction,
        },
    )
    # The clip only confounds the reading when it could have caused every
    # invalid reconstruction. One invalid row on an entirely unclipped input is
    # direct evidence that it could not have, and that evidence stands however
    # much clipping happened elsewhere in the sequence.
    record(
        "R2d",
        C.ROUNDTRIP_INTERPRETATION_CONFOUNDED_BY_CLIPPING,
        invalid_count > 0 and material_clipping and clipped_invalid > 0 and unclipped_invalid == 0,
        "every invalid reconstruction had a clipped input, under material exposure",
        {
            "invalid_candle_count": float(invalid_count),
            "row_clipped_fraction": clip_fraction,
            "invalid_rows_with_clipped_input": float(clipped_invalid),
            "invalid_rows_with_unclipped_input": float(unclipped_invalid),
        },
    )

    # --- rollouts and filtering -------------------------------------------
    record(
        "R3",
        C.NO_VALID_ROLLOUTS_OBSERVED,
        method_d.valid_rollout_count == 0,
        "valid rollout count among the seeded rollouts at this origin",
        {
            "valid_rollout_count": float(method_d.valid_rollout_count),
            "rollout_count": float(method_d.rollout_count),
        },
    )

    controls = method_d.size_matched_controls
    valid_only = _primary(method_d.valid_only_ensemble_error)
    control_mean = controls.mean_primary_error
    # The de-confounded comparison: k valid paths against k paths drawn from
    # all rollouts, so only the selection differs.
    meets = (
        valid_only is not None
        and control_mean is not None
        and not controls.degenerate
        and valid_only <= control_mean * (1.0 - thresholds.minimum_relative_improvement)
    )
    record(
        "R4",
        C.VALID_ONLY_ENSEMBLE_MEETS_IMPROVEMENT_THRESHOLD,
        meets,
        "valid-only ensemble against size-matched controls of the same k",
        {
            "valid_only_primary_error": valid_only,
            "control_mean_primary_error": control_mean,
            "control_median_primary_error": controls.median_primary_error,
            "valid_only_percentile_rank": controls.valid_only_percentile_rank,
            "k": float(controls.k),
            "degenerate": float(controls.degenerate),
            "minimum_relative_improvement": thresholds.minimum_relative_improvement,
        },
    )
    record(
        "R5",
        C.LOW_VALID_ROLLOUT_FRACTION,
        0 < method_d.valid_rollout_count
        and method_d.valid_rollout_fraction < thresholds.valid_rollout_support_fraction_minimum,
        "valid rollout fraction against the support minimum, at this origin",
        {
            "valid_rollout_fraction": method_d.valid_rollout_fraction,
            "minimum": thresholds.valid_rollout_support_fraction_minimum,
        },
    )

    before = _primary(method_c.forecast_error_before)
    after = _primary(method_c.forecast_error_after)
    record(
        "R6",
        C.PROJECTION_RESTORES_VALIDITY_WITHOUT_THRESHOLD_IMPROVEMENT,
        method_b.validity.path_is_invalid
        and method_c.restores_validity
        and before is not None
        and after is not None
        and after > before * (1.0 - thresholds.minimum_relative_improvement),
        "projection restored validity without meeting the improvement threshold",
        {
            "method_b_path_is_invalid": float(method_b.validity.path_is_invalid),
            "restores_validity": float(method_c.restores_validity),
            "primary_error_before": before,
            "primary_error_after": after,
        },
    )

    # --- baseline skill ----------------------------------------------------
    skills = {
        "method_b": _skill(method_b.persistence),
        "method_c": _skill(method_c.persistence_after),
        "manual_ensemble": _skill(method_d.manual_ensemble_persistence),
        "valid_only_ensemble": _skill(method_d.valid_only_persistence),
    }
    measured = [value for value in skills.values() if value is not None]
    best = max(measured) if measured else None
    record(
        "R7",
        C.SKILL_AGAINST_PERSISTENCE_OBSERVED,
        best is not None and best > thresholds.minimum_persistence_skill,
        "best close-return skill against the zero-return persistence baseline",
        {**skills, "best_skill": best, "threshold": thresholds.minimum_persistence_skill},
    )
    record(
        "R8",
        C.NO_SKILL_AGAINST_PERSISTENCE,
        best is not None and best <= thresholds.minimum_persistence_skill,
        "no candidate beat the persistence baseline by the threshold",
        {"best_skill": best, "threshold": thresholds.minimum_persistence_skill},
    )

    # --- defaults ----------------------------------------------------------
    record(
        "R9",
        C.DIAGNOSTIC_INCONCLUSIVE,
        best is None or (valid_only is None and method_d.valid_rollout_count > 0),
        "a quantity the decision rules require could not be computed",
        {"best_skill": best, "valid_only_primary_error": valid_only},
    )

    matched_set = {entry.finding for entry in evaluations if entry.matched}
    record(
        "R10",
        C.NO_PREREGISTERED_EFFECT_DETECTED,
        not (
            matched_set
            - {
                C.ROUNDTRIP_STRUCTURAL_INVALIDITY_OBSERVED,
                C.MATERIAL_CLIPPING_EXPOSURE_OBSERVED,
            }
        ),
        "no rule other than the bare round-trip observation and clipping exposure matched",
        {"matched_findings": float(len(matched_set))},
    )

    matched_set = {entry.finding for entry in evaluations if entry.matched}
    ordered = tuple(entry.finding for entry in evaluations if entry.matched)
    primary = next(
        (finding for finding in PRIMARY_PRIORITY if finding in matched_set),
        C.NO_PREREGISTERED_EFFECT_DETECTED,
    )
    primary_rule = next(
        (entry.rule_id for entry in evaluations if entry.matched and entry.finding is primary),
        "R10",
    )
    return ConclusionOutcome(
        primary_conclusion=primary,
        primary_rule_id=primary_rule,
        matched_findings=ordered,
        evaluations=tuple(evaluations),
        recommended_next_experiment=_recommend(matched_set, thresholds),
        reproducibility=check,
        thresholds=thresholds,
    )
