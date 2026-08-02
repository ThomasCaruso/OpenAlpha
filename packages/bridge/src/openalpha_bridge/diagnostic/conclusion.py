"""The preregistered decision rules, v2.

Every label states what was observed on this window. None asserts a cause, a
property of the model's token support, or a general property of repair. One
forecast origin and 64 stochastic rollouts cannot establish any of those, so
the vocabulary does not contain words that would claim them.

The conclusion is computed. There is no parameter anywhere that lets a caller
supply one, and the default final rule is descriptive rather than causal.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict

from .methods import MethodAResult, MethodBResult, MethodCResult, MethodDResult
from .spec import THRESHOLDS, DiagnosticThresholds

__all__ = [
    "RETIRED_LABELS",
    "ConclusionOutcome",
    "DiagnosticConclusion",
    "RuleEvaluation",
    "decide",
]


class DiagnosticConclusion(StrEnum):
    """The complete vocabulary. Descriptive and local, never causal."""

    ROUNDTRIP_STRUCTURAL_INVALIDITY_OBSERVED = "ROUNDTRIP_STRUCTURAL_INVALIDITY_OBSERVED"
    ROUNDTRIP_MATERIAL_INVALIDITY = "ROUNDTRIP_MATERIAL_INVALIDITY"
    NO_VALID_ROLLOUTS_OBSERVED = "NO_VALID_ROLLOUTS_OBSERVED"
    VALID_ONLY_ENSEMBLE_MEETS_IMPROVEMENT_THRESHOLD = (
        "VALID_ONLY_ENSEMBLE_MEETS_IMPROVEMENT_THRESHOLD"
    )
    LOW_VALID_ROLLOUT_FRACTION = "LOW_VALID_ROLLOUT_FRACTION"
    PROJECTION_RESTORES_VALIDITY_WITHOUT_THRESHOLD_IMPROVEMENT = (
        "PROJECTION_RESTORES_VALIDITY_WITHOUT_THRESHOLD_IMPROVEMENT"
    )
    NO_PREREGISTERED_EFFECT_DETECTED = "NO_PREREGISTERED_EFFECT_DETECTED"
    DIAGNOSTIC_INCONCLUSIVE = "DIAGNOSTIC_INCONCLUSIVE"
    DIAGNOSTIC_OPERATIONAL_FAILURE = "DIAGNOSTIC_OPERATIONAL_FAILURE"


#: Labels this diagnostic may never emit, and why. Kept in code so a future
#: edit that reintroduces one has to delete the reason first.
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


class RuleEvaluation(BaseModel):
    """One rule, its inputs, and whether it fired."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    rule_id: str
    conclusion: DiagnosticConclusion
    matched: bool
    detail: str
    observed: dict[str, float | None]


class ConclusionOutcome(BaseModel):
    """The computed conclusion and the full evaluation trace."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    conclusion: DiagnosticConclusion
    matched_rule_id: str
    thresholds: DiagnosticThresholds
    evaluations: tuple[RuleEvaluation, ...]
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


def decide(
    *,
    method_a: MethodAResult | None,
    method_b: MethodBResult | None,
    method_c: MethodCResult | None,
    method_d: MethodDResult | None,
    operational_failure: bool = False,
    thresholds: DiagnosticThresholds = THRESHOLDS,
) -> ConclusionOutcome:
    """Evaluate the ordered rules and return the first that matches."""
    evaluations: list[RuleEvaluation] = []

    def record(
        rule_id: str,
        conclusion: DiagnosticConclusion,
        matched: bool,
        detail: str,
        observed: dict[str, float | None],
    ) -> bool:
        evaluations.append(
            RuleEvaluation(
                rule_id=rule_id,
                conclusion=conclusion,
                matched=matched,
                detail=detail,
                observed=observed,
            )
        )
        return matched

    def finish(rule_id: str, conclusion: DiagnosticConclusion) -> ConclusionOutcome:
        return ConclusionOutcome(
            conclusion=conclusion,
            matched_rule_id=rule_id,
            thresholds=thresholds,
            evaluations=tuple(evaluations),
        )

    missing = [
        name
        for name, value in (
            ("A", method_a),
            ("B", method_b),
            ("C", method_c),
            ("D", method_d),
        )
        if value is None
    ]
    if record(
        "R0",
        DiagnosticConclusion.DIAGNOSTIC_OPERATIONAL_FAILURE,
        operational_failure or bool(missing),
        (
            "a method raised an operational failure"
            if operational_failure
            else f"methods did not complete: {', '.join(missing)}"
            if missing
            else "every method completed"
        ),
        {"missing_methods": float(len(missing))},
    ):
        return finish("R0", DiagnosticConclusion.DIAGNOSTIC_OPERATIONAL_FAILURE)

    assert method_a is not None and method_b is not None
    assert method_c is not None and method_d is not None

    # R1 and R2 - the round trip. Materiality first, then the strict
    # observation, so a nonzero-but-immaterial round trip is still reported as
    # structurally invalid rather than described as clean.
    fraction = method_a.validity_all.invalid_candle_fraction
    invalid_count = method_a.validity_all.invalid_candle_count
    if record(
        "R1",
        DiagnosticConclusion.ROUNDTRIP_MATERIAL_INVALIDITY,
        fraction > thresholds.roundtrip_material_invalid_fraction,
        "round-trip invalid candle fraction against the materiality threshold",
        {
            "invalid_candle_fraction": fraction,
            "materiality_threshold": thresholds.roundtrip_material_invalid_fraction,
            "invalid_candle_count": float(invalid_count),
        },
    ):
        return finish("R1", DiagnosticConclusion.ROUNDTRIP_MATERIAL_INVALIDITY)

    if record(
        "R2",
        DiagnosticConclusion.ROUNDTRIP_STRUCTURAL_INVALIDITY_OBSERVED,
        invalid_count > 0,
        "any invalid round-trip candle at all, regardless of materiality",
        {
            "invalid_candle_count": float(invalid_count),
            "invalid_candle_fraction": fraction,
        },
    ):
        return finish("R2", DiagnosticConclusion.ROUNDTRIP_STRUCTURAL_INVALIDITY_OBSERVED)

    if record(
        "R3",
        DiagnosticConclusion.NO_VALID_ROLLOUTS_OBSERVED,
        method_d.valid_rollout_count == 0,
        "valid rollout count among the seeded rollouts at this origin",
        {
            "valid_rollout_count": float(method_d.valid_rollout_count),
            "rollout_count": float(method_d.rollout_count),
        },
    ):
        return finish("R3", DiagnosticConclusion.NO_VALID_ROLLOUTS_OBSERVED)

    # R4 - the valid-only ensemble against the manual seeded ensemble. Those
    # two differ only in which paths are included, which is the comparison the
    # threshold was written for.
    valid_only = _primary(method_d.valid_only_ensemble_error)
    manual = _primary(method_d.manual_seeded_ensemble_error)
    meets = (
        valid_only is not None
        and manual is not None
        and valid_only <= manual * (1.0 - thresholds.minimum_relative_improvement)
    )
    if record(
        "R4",
        DiagnosticConclusion.VALID_ONLY_ENSEMBLE_MEETS_IMPROVEMENT_THRESHOLD,
        meets,
        "valid-only ensemble error against the manual seeded ensemble error",
        {
            "valid_only_primary_error": valid_only,
            "manual_seeded_ensemble_primary_error": manual,
            "minimum_relative_improvement": thresholds.minimum_relative_improvement,
        },
    ):
        return finish("R4", DiagnosticConclusion.VALID_ONLY_ENSEMBLE_MEETS_IMPROVEMENT_THRESHOLD)

    if record(
        "R5",
        DiagnosticConclusion.LOW_VALID_ROLLOUT_FRACTION,
        method_d.valid_rollout_fraction < thresholds.valid_rollout_support_fraction_minimum,
        "valid rollout fraction against the support minimum, at this origin",
        {
            "valid_rollout_fraction": method_d.valid_rollout_fraction,
            "minimum": thresholds.valid_rollout_support_fraction_minimum,
        },
    ):
        return finish("R5", DiagnosticConclusion.LOW_VALID_ROLLOUT_FRACTION)

    before = _primary(method_c.forecast_error_before)
    after = _primary(method_c.forecast_error_after)
    repair_without_improvement = (
        method_b.validity.path_is_invalid
        and method_c.restores_validity
        and before is not None
        and after is not None
        and after > before * (1.0 - thresholds.minimum_relative_improvement)
    )
    if record(
        "R6",
        DiagnosticConclusion.PROJECTION_RESTORES_VALIDITY_WITHOUT_THRESHOLD_IMPROVEMENT,
        repair_without_improvement,
        "projection restored validity without meeting the improvement threshold",
        {
            "method_b_path_is_invalid": float(method_b.validity.path_is_invalid),
            "restores_validity": float(method_c.restores_validity),
            "primary_error_before": before,
            "primary_error_after": after,
        },
    ):
        return finish(
            "R6",
            DiagnosticConclusion.PROJECTION_RESTORES_VALIDITY_WITHOUT_THRESHOLD_IMPROVEMENT,
        )

    # R7 - the comparison R4 needs could not be evaluated, so no statement
    # about it is available either way.
    if record(
        "R7",
        DiagnosticConclusion.DIAGNOSTIC_INCONCLUSIVE,
        valid_only is None or manual is None,
        "the primary error required by R4 is undefined for at least one ensemble",
        {
            "valid_only_primary_error": valid_only,
            "manual_seeded_ensemble_primary_error": manual,
        },
    ):
        return finish("R7", DiagnosticConclusion.DIAGNOSTIC_INCONCLUSIVE)

    # R8 - descriptive default. Not a causal claim, and not a claim that no
    # effect exists: only that none of the preregistered effects was detected
    # on this window.
    record(
        "R8",
        DiagnosticConclusion.NO_PREREGISTERED_EFFECT_DETECTED,
        True,
        "no earlier rule matched on this window",
        {
            "valid_rollout_fraction": method_d.valid_rollout_fraction,
            "valid_group_mean_primary_error": method_d.valid_group_mean_primary_error,
            "invalid_group_mean_primary_error": method_d.invalid_group_mean_primary_error,
        },
    )
    return finish("R8", DiagnosticConclusion.NO_PREREGISTERED_EFFECT_DETECTED)
