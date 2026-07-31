import pytest
from openalpha_sentinel.constraint_compatibility import (
    CandidateMass,
    CompatibilityFacts,
    ContinuationMeasurements,
    NoSupportedCandidateError,
    SupportedCandidate,
    choose_supported_candidate,
    classify_root_cause,
    evaluate_continuation,
    probability_mass_bounds,
)


def test_mass_bounds_keep_truncation_honest() -> None:
    result = probability_mass_bounds(
        considered=(
            CandidateMass(probability=0.30, valid=True, supported=True),
            CandidateMass(probability=0.20, valid=False, supported=True),
        ),
        considered_mass=0.50,
    )

    assert result.valid_lower == pytest.approx(0.30)
    assert result.valid_upper == pytest.approx(0.80)
    assert result.supported_lower == pytest.approx(0.50)
    assert result.valid_supported_lower == pytest.approx(0.30)
    assert result.uncovered_mass == pytest.approx(0.50)
    assert result.valid_constraint_tax == pytest.approx(-__import__("math").log(0.30))
    assert result.exact is False


def test_zero_mass_tax_serializes_without_nonfinite_json() -> None:
    result = probability_mass_bounds(considered=(), considered_mass=0.0)

    assert result.valid_constraint_tax is None
    assert result.support_constraint_tax is None
    assert result.valid_tax_status == "ZERO_ESTIMATED_MASS"
    assert result.support_tax_status == "ZERO_ESTIMATED_MASS"
    assert "Infinity" not in result.model_dump_json()


def _facts(**updates: object) -> CompatibilityFacts:
    values: dict[str, object] = {
        "tokenizer_2k_material_defect": False,
        "tokenizer_base_material_defect": False,
        "tokenizer_base_overwhelmingly_valid": False,
        "mini_material_raw_invalidity": False,
        "mini_invalid_candle_rate": 0.60,
        "larger_model_invalid_candle_rates": (),
        "median_valid_mass_upper": 0.70,
        "low_valid_upper_step_fraction": 0.10,
        "invalid_raw_unsupported_fraction": 0.20,
        "unsupported_pair_invalid_rate": 0.20,
        "supported_pair_invalid_rate": 0.20,
        "unsupported_pair_count": 20,
        "supported_pair_count": 20,
        "median_valid_mass_lower": 0.40,
        "median_valid_supported_mass_upper": 0.50,
        "median_valid_supported_mass_lower": 0.10,
        "median_considered_mass": 0.85,
        "prior_range_gate_failed": True,
        "canary_sufficient": True,
    }
    values.update(updates)
    return CompatibilityFacts.model_validate(values)


@pytest.mark.parametrize(
    ("facts", "expected"),
    (
        (
            _facts(
                tokenizer_2k_material_defect=True,
                tokenizer_base_overwhelmingly_valid=True,
                larger_model_invalid_candle_rates=(0.04, 0.05),
            ),
            "MINI_OR_TOKENIZER_2K_SPECIFIC",
        ),
        (
            _facts(tokenizer_2k_material_defect=True),
            "TOKENIZER_CONSTRAINT_DEFECT",
        ),
        (
            _facts(median_valid_mass_upper=0.20),
            "LOW_VALID_PROBABILITY_MASS",
        ),
        (
            _facts(
                invalid_raw_unsupported_fraction=0.75,
                unsupported_pair_invalid_rate=0.60,
                supported_pair_invalid_rate=0.20,
            ),
            "OFF_MANIFOLD_TOKEN_COMBINATIONS",
        ),
        (
            _facts(median_valid_supported_mass_lower=0.30),
            "CANDIDATE_SEARCH_FAILURE",
        ),
        (
            _facts(canary_sufficient=False),
            "MIXED_OR_UNRESOLVED",
        ),
    ),
)
def test_root_cause_classification_is_ordered(
    facts: CompatibilityFacts,
    expected: str,
) -> None:
    result = classify_root_cause(facts)

    assert result.root_cause == expected
    assert result.matched_rule
    assert result.conditional_method_authorized is (
        expected
        in {
            "OFF_MANIFOLD_TOKEN_COMBINATIONS",
            "LOW_VALID_PROBABILITY_MASS",
            "CANDIDATE_SEARCH_FAILURE",
        }
    )


def _candidate(
    coarse: int,
    *,
    probability: float,
    rank: int,
    valid: bool,
    count: int,
) -> SupportedCandidate:
    return SupportedCandidate(
        coarse_token=coarse,
        fine_token=coarse + 10,
        original_probability=probability,
        rank=rank,
        valid=valid,
        exact_pair_count=count,
    )


def test_support_sampler_rejects_invalid_and_rare_pairs() -> None:
    result = choose_supported_candidate(
        (
            _candidate(1, probability=0.50, rank=1, valid=False, count=5),
            _candidate(2, probability=0.30, rank=2, valid=True, count=1),
            _candidate(3, probability=0.15, rank=3, valid=True, count=4),
            _candidate(4, probability=0.05, rank=4, valid=True, count=2),
        ),
        auxiliary_fraction=0.80,
        minimum_pair_count=2,
    )

    assert result.eligible_candidate_count == 2
    assert result.rejection_count == 2
    assert result.selected.coarse_token == 4
    assert result.renormalized_probability == pytest.approx(0.25)


def test_empty_supported_set_is_an_explicit_hard_failure() -> None:
    candidates = (
        _candidate(1, probability=0.7, rank=1, valid=False, count=5),
        _candidate(2, probability=0.3, rank=2, valid=True, count=1),
    )

    with pytest.raises(NoSupportedCandidateError, match="no valid supported"):
        choose_supported_candidate(
            candidates,
            auxiliary_fraction=0.25,
            minimum_pair_count=2,
        )


def _passing_measurements(**updates: object) -> ContinuationMeasurements:
    values: dict[str, object] = {
        "returned_path_validity_rate": 1.0,
        "hard_failure_count": 1,
        "high_low_range_mae": 0.009,
        "close_mae": 0.0105,
        "raw_close_mae": 0.010,
        "pairwise_diversity": 0.011,
        "raw_pairwise_diversity": 0.020,
        "final_return_variance": 0.0001,
        "repeated_path_rate": 0.0,
        "mean_raw_path_distance": 0.008,
        "mean_selected_token_rank": 9.0,
        "synthetic_replays_match": True,
        "real_replays_match": True,
        "median_latency_ms": 400.0,
        "raw_median_latency_ms": 100.0,
        "cache_bytes": 4_000_000_000,
        "peak_process_memory_bytes": 2_500_000_000,
        "raw_peak_process_memory_bytes": 1_000_000_000,
        "weights_unchanged": True,
    }
    values.update(updates)
    return ContinuationMeasurements.model_validate(values)


def test_continuation_requires_every_frozen_gate() -> None:
    passing = evaluate_continuation(_passing_measurements())
    failing = evaluate_continuation(
        _passing_measurements(high_low_range_mae=0.0100)
    )

    assert passing.all_passed is True
    assert all(passing.gates.values())
    assert failing.all_passed is False
    assert failing.gates["high_low_range_mae"] is False
    assert sum(not value for value in failing.gates.values()) == 1
