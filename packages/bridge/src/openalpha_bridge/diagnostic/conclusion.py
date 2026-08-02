"""The preregistered decision rules.

The conclusion is computed. It is not chosen after the numbers are known, and
there is no code path that lets a caller supply one. The rules are evaluated in
the order the specification fixes and the first match wins; every rule records
whether it matched, so the artifact shows why the others did not.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict

from .methods import MethodAResult, MethodBResult, MethodCResult, MethodDResult
from .spec import THRESHOLDS, DiagnosticThresholds

__all__ = [
    "ConclusionOutcome",
    "DiagnosticConclusion",
    "RuleEvaluation",
    "decide",
]


class DiagnosticConclusion(StrEnum):
    """The complete vocabulary. Nothing outside this set can be produced."""

    TOKENIZER_DECODER_DEFECT = "TOKENIZER_DECODER_DEFECT"
    GENERATED_TOKEN_SUPPORT_DEFECT = "GENERATED_TOKEN_SUPPORT_DEFECT"
    VALIDITY_FILTER_IMPROVES_FORECAST = "VALIDITY_FILTER_IMPROVES_FORECAST"
    VALIDITY_REPAIR_ONLY = "VALIDITY_REPAIR_ONLY"
    NO_VALID_ROLLOUTS = "NO_VALID_ROLLOUTS"
    INVALIDITY_NOT_CAUSAL_TO_FORECAST_ERROR = "INVALIDITY_NOT_CAUSAL_TO_FORECAST_ERROR"
    DIAGNOSTIC_OPERATIONAL_FAILURE = "DIAGNOSTIC_OPERATIONAL_FAILURE"


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
    #: Fixed by type. No outcome of any rule authorizes anything.
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

    # R0 - any method failed operationally, or a method is missing because
    # execution stopped before it ran. Nothing below can be interpreted.
    missing = [
        name
        for name, value in (
            ("A", method_a), ("B", method_b), ("C", method_c), ("D", method_d)
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
        return ConclusionOutcome(
            conclusion=DiagnosticConclusion.DIAGNOSTIC_OPERATIONAL_FAILURE,
            matched_rule_id="R0",
            thresholds=thresholds,
            evaluations=tuple(evaluations),
        )

    assert method_a is not None and method_b is not None
    assert method_c is not None and method_d is not None

    # R1 - the decoder mangles in-distribution tokens. Upstream of generation,
    # so it confounds every statement below it.
    roundtrip = method_a.validity.invalid_candle_fraction
    if record(
        "R1",
        DiagnosticConclusion.TOKENIZER_DECODER_DEFECT,
        roundtrip > thresholds.roundtrip_invalid_fraction_maximum,
        "round-trip invalid candle fraction against its maximum",
        {
            "method_a_invalid_candle_fraction": roundtrip,
            "maximum": thresholds.roundtrip_invalid_fraction_maximum,
        },
    ):
        return ConclusionOutcome(
            conclusion=DiagnosticConclusion.TOKENIZER_DECODER_DEFECT,
            matched_rule_id="R1",
            thresholds=thresholds,
            evaluations=tuple(evaluations),
        )

    # R2 - rejection sampling has nothing to select from.
    if record(
        "R2",
        DiagnosticConclusion.NO_VALID_ROLLOUTS,
        method_d.valid_rollout_count == 0,
        "valid rollout count",
        {
            "valid_rollout_count": float(method_d.valid_rollout_count),
            "rollout_count": float(method_d.rollout_count),
        },
    ):
        return ConclusionOutcome(
            conclusion=DiagnosticConclusion.NO_VALID_ROLLOUTS,
            matched_rule_id="R2",
            thresholds=thresholds,
            evaluations=tuple(evaluations),
        )

    # R3 - conditioning on validity beats the official all-rollout ensemble by
    # the preregistered margin.
    valid_only = _primary(method_d.valid_only_ensemble_error)
    all_rollout = _primary(method_d.all_rollout_ensemble_error)
    improves = (
        valid_only is not None
        and all_rollout is not None
        and valid_only <= all_rollout * (1.0 - thresholds.minimum_relative_improvement)
    )
    if record(
        "R3",
        DiagnosticConclusion.VALIDITY_FILTER_IMPROVES_FORECAST,
        improves,
        "valid-only ensemble error against the all-rollout ensemble error",
        {
            "valid_only_primary_error": valid_only,
            "all_rollout_primary_error": all_rollout,
            "minimum_relative_improvement": thresholds.minimum_relative_improvement,
        },
    ):
        return ConclusionOutcome(
            conclusion=DiagnosticConclusion.VALIDITY_FILTER_IMPROVES_FORECAST,
            matched_rule_id="R3",
            thresholds=thresholds,
            evaluations=tuple(evaluations),
        )

    # R4 - the round trip is clean but most generated token combinations decode
    # outside the valid region, so invalidity enters through generation.
    fraction = method_d.valid_rollout_fraction
    if record(
        "R4",
        DiagnosticConclusion.GENERATED_TOKEN_SUPPORT_DEFECT,
        fraction < thresholds.valid_rollout_support_fraction_minimum,
        "valid rollout fraction against the support minimum",
        {
            "valid_rollout_fraction": fraction,
            "minimum": thresholds.valid_rollout_support_fraction_minimum,
        },
    ):
        return ConclusionOutcome(
            conclusion=DiagnosticConclusion.GENERATED_TOKEN_SUPPORT_DEFECT,
            matched_rule_id="R4",
            thresholds=thresholds,
            evaluations=tuple(evaluations),
        )

    # R5 - projection does restore validity, and buys no accuracy for it.
    before = _primary(method_c.forecast_error_before)
    after = _primary(method_c.forecast_error_after)
    repair_only = (
        method_b.validity.path_is_invalid
        and method_c.restores_validity
        and before is not None
        and after is not None
        and after > before * (1.0 - thresholds.minimum_relative_improvement)
    )
    if record(
        "R5",
        DiagnosticConclusion.VALIDITY_REPAIR_ONLY,
        repair_only,
        "projection restored validity without improving the primary error",
        {
            "method_b_path_is_invalid": float(method_b.validity.path_is_invalid),
            "restores_validity": float(method_c.restores_validity),
            "primary_error_before": before,
            "primary_error_after": after,
        },
    ):
        return ConclusionOutcome(
            conclusion=DiagnosticConclusion.VALIDITY_REPAIR_ONLY,
            matched_rule_id="R5",
            thresholds=thresholds,
            evaluations=tuple(evaluations),
        )

    # R6 - forecasts are structurally acceptable, or their validity is unrelated
    # to their error. Either way invalidity is not what drives forecast quality.
    valid_mean = method_d.valid_group_mean_primary_error
    invalid_mean = method_d.invalid_group_mean_primary_error
    separation = (
        abs(valid_mean - invalid_mean)
        if valid_mean is not None and invalid_mean is not None
        else None
    )
    record(
        "R6",
        DiagnosticConclusion.INVALIDITY_NOT_CAUSAL_TO_FORECAST_ERROR,
        True,
        "no earlier rule matched",
        {
            "valid_group_mean_primary_error": valid_mean,
            "invalid_group_mean_primary_error": invalid_mean,
            "group_separation": separation,
            "accuracy_equivalence_margin": thresholds.accuracy_equivalence_margin,
        },
    )
    return ConclusionOutcome(
        conclusion=DiagnosticConclusion.INVALIDITY_NOT_CAUSAL_TO_FORECAST_ERROR,
        matched_rule_id="R6",
        thresholds=thresholds,
        evaluations=tuple(evaluations),
    )
