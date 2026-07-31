from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

from pydantic import Field

from .contracts import FrozenModel

RootCause = Literal[
    "TOKENIZER_CONSTRAINT_DEFECT",
    "OFF_MANIFOLD_TOKEN_COMBINATIONS",
    "LOW_VALID_PROBABILITY_MASS",
    "CANDIDATE_SEARCH_FAILURE",
    "MINI_OR_TOKENIZER_2K_SPECIFIC",
    "MIXED_OR_UNRESOLVED",
]


class CandidateMass(FrozenModel):
    probability: float = Field(ge=0.0, le=1.0)
    valid: bool
    supported: bool


class ProbabilityMassBounds(FrozenModel):
    considered_mass: float = Field(ge=0.0, le=1.0)
    uncovered_mass: float = Field(ge=0.0, le=1.0)
    valid_lower: float = Field(ge=0.0, le=1.0)
    valid_upper: float = Field(ge=0.0, le=1.0)
    supported_lower: float = Field(ge=0.0, le=1.0)
    supported_upper: float = Field(ge=0.0, le=1.0)
    valid_supported_lower: float = Field(ge=0.0, le=1.0)
    valid_supported_upper: float = Field(ge=0.0, le=1.0)
    valid_constraint_tax: float | None
    support_constraint_tax: float | None
    valid_tax_status: Literal["COMPUTED_LOWER_BOUND", "ZERO_ESTIMATED_MASS"]
    support_tax_status: Literal["COMPUTED_LOWER_BOUND", "ZERO_ESTIMATED_MASS"]
    exact: bool


def probability_mass_bounds(
    *,
    considered: Sequence[CandidateMass],
    considered_mass: float,
) -> ProbabilityMassBounds:
    candidates = tuple(considered)
    if not math.isfinite(considered_mass) or not 0.0 <= considered_mass <= 1.0:
        raise ValueError("considered mass must be finite and in [0, 1]")
    candidate_mass = sum(item.probability for item in candidates)
    if candidate_mass > considered_mass + 1e-12:
        raise ValueError("candidate probability exceeds considered mass")
    uncovered = max(0.0, 1.0 - considered_mass)
    valid = sum(item.probability for item in candidates if item.valid)
    supported = sum(item.probability for item in candidates if item.supported)
    valid_supported = sum(
        item.probability for item in candidates if item.valid and item.supported
    )
    valid_tax, valid_status = _constraint_tax(valid)
    support_tax, support_status = _constraint_tax(valid_supported)
    return ProbabilityMassBounds(
        considered_mass=considered_mass,
        uncovered_mass=uncovered,
        valid_lower=valid,
        valid_upper=min(1.0, valid + uncovered),
        supported_lower=supported,
        supported_upper=min(1.0, supported + uncovered),
        valid_supported_lower=valid_supported,
        valid_supported_upper=min(1.0, valid_supported + uncovered),
        valid_constraint_tax=valid_tax,
        support_constraint_tax=support_tax,
        valid_tax_status=valid_status,
        support_tax_status=support_status,
        exact=uncovered <= 1e-12,
    )


def _constraint_tax(
    probability: float,
) -> tuple[float | None, Literal["COMPUTED_LOWER_BOUND", "ZERO_ESTIMATED_MASS"]]:
    if probability <= 0.0:
        return None, "ZERO_ESTIMATED_MASS"
    return -math.log(probability), "COMPUTED_LOWER_BOUND"


class CompatibilityFacts(FrozenModel):
    tokenizer_2k_material_defect: bool
    tokenizer_base_material_defect: bool
    tokenizer_base_overwhelmingly_valid: bool
    mini_material_raw_invalidity: bool
    mini_invalid_candle_rate: float = Field(ge=0.0, le=1.0)
    larger_model_invalid_candle_rates: tuple[float, ...]
    median_valid_mass_upper: float = Field(ge=0.0, le=1.0)
    low_valid_upper_step_fraction: float = Field(ge=0.0, le=1.0)
    invalid_raw_unsupported_fraction: float = Field(ge=0.0, le=1.0)
    unsupported_pair_invalid_rate: float = Field(ge=0.0, le=1.0)
    supported_pair_invalid_rate: float = Field(ge=0.0, le=1.0)
    unsupported_pair_count: int = Field(ge=0)
    supported_pair_count: int = Field(ge=0)
    median_valid_mass_lower: float = Field(ge=0.0, le=1.0)
    median_valid_supported_mass_upper: float = Field(ge=0.0, le=1.0)
    median_valid_supported_mass_lower: float = Field(ge=0.0, le=1.0)
    median_considered_mass: float = Field(ge=0.0, le=1.0)
    prior_range_gate_failed: bool
    canary_sufficient: bool


class RootCauseClassification(FrozenModel):
    root_cause: RootCause
    matched_rule: str
    conditional_method_authorized: bool


def classify_root_cause(facts: CompatibilityFacts) -> RootCauseClassification:
    larger_rates = facts.larger_model_invalid_candle_rates
    mini_specific = (
        (
            facts.tokenizer_2k_material_defect
            or facts.mini_material_raw_invalidity
        )
        and facts.tokenizer_base_overwhelmingly_valid
        and len(larger_rates) == 2
        and all(
            rate <= 0.05
            and rate <= 0.5 * facts.mini_invalid_candle_rate
            for rate in larger_rates
        )
    )
    if mini_specific:
        return _classification(
            "MINI_OR_TOKENIZER_2K_SPECIFIC",
            "E_MINI_OR_TOKENIZER_2K_SPECIFIC",
        )
    if facts.tokenizer_2k_material_defect or facts.tokenizer_base_material_defect:
        return _classification(
            "TOKENIZER_CONSTRAINT_DEFECT",
            "A_TOKENIZER_CONSTRAINT_DEFECT",
        )
    low_valid_mass = (
        facts.median_valid_mass_upper <= 0.25
        or (
            facts.low_valid_upper_step_fraction >= 0.75
            and facts.median_valid_mass_upper < 0.50
        )
    )
    if low_valid_mass:
        return _classification(
            "LOW_VALID_PROBABILITY_MASS",
            "C_LOW_VALID_PROBABILITY_MASS",
        )
    off_manifold_group = (
        facts.invalid_raw_unsupported_fraction >= 0.70
        and facts.unsupported_pair_count >= 10
        and facts.supported_pair_count >= 10
        and facts.unsupported_pair_invalid_rate
        >= 2.0 * facts.supported_pair_invalid_rate
    )
    off_manifold_mass = (
        facts.median_valid_mass_lower >= 0.25
        and facts.median_valid_supported_mass_upper <= 0.25
    )
    if off_manifold_group or off_manifold_mass:
        return _classification(
            "OFF_MANIFOLD_TOKEN_COMBINATIONS",
            "B_OFF_MANIFOLD_TOKEN_COMBINATIONS",
        )
    if (
        facts.median_valid_supported_mass_lower >= 0.25
        and facts.median_considered_mass >= 0.80
        and facts.prior_range_gate_failed
    ):
        return _classification(
            "CANDIDATE_SEARCH_FAILURE",
            "D_CANDIDATE_SEARCH_FAILURE",
        )
    return _classification(
        "MIXED_OR_UNRESOLVED",
        (
            "F_INSUFFICIENT_CANARY_EVIDENCE"
            if not facts.canary_sufficient
            else "F_MIXED_OR_UNRESOLVED"
        ),
    )


def _classification(
    root_cause: RootCause,
    matched_rule: str,
) -> RootCauseClassification:
    return RootCauseClassification(
        root_cause=root_cause,
        matched_rule=matched_rule,
        conditional_method_authorized=root_cause
        in {
            "OFF_MANIFOLD_TOKEN_COMBINATIONS",
            "LOW_VALID_PROBABILITY_MASS",
            "CANDIDATE_SEARCH_FAILURE",
        },
    )


class NoSupportedCandidateError(ValueError):
    pass


class SupportedCandidate(FrozenModel):
    coarse_token: int = Field(ge=0)
    fine_token: int = Field(ge=0)
    original_probability: float = Field(gt=0.0, le=1.0)
    rank: int = Field(ge=1)
    valid: bool
    exact_pair_count: int = Field(ge=0)


class SupportedSelection(FrozenModel):
    selected: SupportedCandidate
    eligible_candidate_count: int = Field(ge=1)
    rejection_count: int = Field(ge=0)
    auxiliary_fraction: float = Field(ge=0.0, lt=1.0)
    renormalized_probability: float = Field(gt=0.0, le=1.0)


def choose_supported_candidate(
    candidates: Sequence[SupportedCandidate],
    *,
    auxiliary_fraction: float,
    minimum_pair_count: int,
) -> SupportedSelection:
    if not math.isfinite(auxiliary_fraction) or not 0.0 <= auxiliary_fraction < 1.0:
        raise ValueError("auxiliary fraction must be finite and in [0, 1)")
    if minimum_pair_count < 1:
        raise ValueError("minimum pair count must be positive")
    materialized = tuple(sorted(candidates, key=lambda item: item.rank))
    eligible = tuple(
        item
        for item in materialized
        if item.valid and item.exact_pair_count >= minimum_pair_count
    )
    if not eligible:
        raise NoSupportedCandidateError(
            "no valid supported candidate within the declared budget"
        )
    total = sum(item.original_probability for item in eligible)
    threshold = auxiliary_fraction * total
    cumulative = 0.0
    selected = eligible[-1]
    for candidate in eligible:
        cumulative += candidate.original_probability
        if threshold < cumulative:
            selected = candidate
            break
    return SupportedSelection(
        selected=selected,
        eligible_candidate_count=len(eligible),
        rejection_count=len(materialized) - len(eligible),
        auxiliary_fraction=auxiliary_fraction,
        renormalized_probability=selected.original_probability / total,
    )


class ContinuationMeasurements(FrozenModel):
    returned_path_validity_rate: float = Field(ge=0.0, le=1.0)
    hard_failure_count: int = Field(ge=0)
    high_low_range_mae: float = Field(ge=0.0)
    close_mae: float = Field(ge=0.0)
    raw_close_mae: float = Field(ge=0.0)
    pairwise_diversity: float = Field(ge=0.0)
    raw_pairwise_diversity: float = Field(ge=0.0)
    final_return_variance: float = Field(ge=0.0)
    repeated_path_rate: float = Field(ge=0.0, le=1.0)
    mean_raw_path_distance: float = Field(ge=0.0)
    mean_selected_token_rank: float = Field(ge=1.0)
    synthetic_replays_match: bool
    real_replays_match: bool
    median_latency_ms: float = Field(ge=0.0)
    raw_median_latency_ms: float = Field(gt=0.0)
    cache_bytes: int = Field(ge=0)
    peak_process_memory_bytes: int = Field(ge=0)
    raw_peak_process_memory_bytes: int = Field(ge=0)
    weights_unchanged: bool


class ContinuationDecision(FrozenModel):
    gates: dict[str, bool]
    all_passed: bool


def evaluate_continuation(
    measurements: ContinuationMeasurements,
) -> ContinuationDecision:
    close_limit = min(
        1.10 * measurements.raw_close_mae,
        measurements.raw_close_mae + 0.001,
    )
    memory_limit = (
        2 * measurements.raw_peak_process_memory_bytes + 1024**3
    )
    gates = {
        'returned_path_structural_validity': (
            measurements.returned_path_validity_rate == 1.0
        ),
        'hard_failure_rate': measurements.hard_failure_count <= 1,
        'high_low_range_mae': (
            measurements.high_low_range_mae <= 0.009934683599145366
        ),
        'close_mae': measurements.close_mae <= close_limit,
        'diversity': (
            measurements.pairwise_diversity
            >= 0.50 * measurements.raw_pairwise_diversity
            and measurements.final_return_variance > 0.0
            and measurements.repeated_path_rate <= 0.10
        ),
        'distributional_distortion': (
            measurements.mean_raw_path_distance
            <= 0.00894289586146275
            and measurements.mean_selected_token_rank
            <= 9.318181818181818
        ),
        'deterministic_replay': (
            measurements.synthetic_replays_match
            and measurements.real_replays_match
        ),
        'runtime_and_resources': (
            measurements.median_latency_ms
            <= 5.0 * measurements.raw_median_latency_ms
            and measurements.cache_bytes <= 4 * 1024**3
            and measurements.peak_process_memory_bytes <= memory_limit
        ),
        'no_weight_modification': measurements.weights_unchanged,
    }
    return ContinuationDecision(
        gates=gates,
        all_passed=all(gates.values()),
    )
