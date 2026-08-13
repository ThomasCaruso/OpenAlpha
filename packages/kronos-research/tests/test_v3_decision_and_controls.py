"""The v3 corrections: multi-finding decisions, de-confounded controls,
persistence baseline, clipping attribution and the reproducibility check.

Every test executes real code. Nothing here touches a network, an official
asset, Torch, or any partition beyond the already-retrieved window.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

import pytest
from diagnostic_fakes import FakeCodec
from openalpha_kronos.evaluation.metrics import (
    compare_to_persistence,
    directional_accuracy,
    forecast_error,
    persistence_forecast,
)
from openalpha_kronos.model.input import (
    OFFICIAL_COLUMNS,
    ColumnPresence,
    OfficialRow,
    OfficialSeries,
)
from openalpha_kronos.model.normalization import clipping_report, fit_context_state
from openalpha_kronos.studies.structural_validity.common.conclusion import (
    PRIMARY_PRIORITY,
    DiagnosticConclusion,
    NextExperiment,
    ReproducibilityCheck,
    decide,
)
from openalpha_kronos.studies.structural_validity.common.methods import _size_matched_controls
from openalpha_kronos.studies.structural_validity.mini.spec import (
    CONTEXT_CANDLES,
    CONTROL_REPETITIONS,
    CONTROL_SEED,
    TARGET_CANDLES,
    TOTAL_CANDLES,
)
from test_frozen_inference_diagnostic import (
    ALL_ROWS,
    TRUE_TARGET,
    _make_invalid,
    _position,
    _run,
    _shift,
)

START = date(2015, 5, 7)


def _series(rows: tuple[OfficialRow, ...]) -> OfficialSeries:
    return OfficialSeries(
        symbol="SPY",
        frequency="1d",
        calendar="XNYS",
        columns=OFFICIAL_COLUMNS,
        column_presence=ColumnPresence(volume=True, amount=True),
        rows=rows,
        context_target_boundary=CONTEXT_CANDLES,
    )


# ================================================ every rule is evaluated


def test_every_rule_is_evaluated_despite_an_earlier_primary_finding() -> None:
    """Material round-trip invalidity no longer suppresses anything."""
    codec = FakeCodec(corrupt_round_trip=True, corrupt_fraction=0.5)
    artifact, _, _, _ = _run(lambda seed, ctx: _shift(TRUE_TARGET, 1.30), codec=codec)

    ids = [e.rule_id for e in artifact.decision.evaluations]
    assert ids == [
        "R0",
        "R0b",
        "R1",
        "R2",
        "R2a",
        "R2b",
        "R2c",
        "R2d",
        "R3",
        "R4",
        "R5",
        "R6",
        "R7",
        "R8",
        "R9",
        "R10",
    ]
    assert DiagnosticConclusion.ROUNDTRIP_MATERIAL_INVALIDITY in artifact.matched_findings
    # The later scientific findings are present, which under v2 they were not.
    assert DiagnosticConclusion.NO_SKILL_AGAINST_PERSISTENCE in artifact.matched_findings
    assert artifact.method_d.size_matched_controls.k >= 0


def test_several_findings_can_hold_simultaneously() -> None:
    codec = FakeCodec(corrupt_round_trip=True, corrupt_fraction=0.5)
    artifact, _, _, _ = _run(
        lambda seed, ctx: _make_invalid(_shift(TRUE_TARGET, 1.30)), codec=codec
    )
    assert len(artifact.matched_findings) >= 3
    assert DiagnosticConclusion.ROUNDTRIP_MATERIAL_INVALIDITY in artifact.matched_findings
    assert (
        DiagnosticConclusion.ROUNDTRIP_STRUCTURAL_INVALIDITY_OBSERVED in artifact.matched_findings
    )
    assert DiagnosticConclusion.NO_VALID_ROLLOUTS_OBSERVED in artifact.matched_findings


def test_r2_is_never_the_primary_conclusion() -> None:
    assert DiagnosticConclusion.ROUNDTRIP_STRUCTURAL_INVALIDITY_OBSERVED not in PRIMARY_PRIORITY
    codec = FakeCodec(corrupt_round_trip=True, corrupt_count=1)
    artifact, _, _, _ = _run(lambda seed, ctx: _shift(TRUE_TARGET, 1.01), codec=codec)
    assert (
        DiagnosticConclusion.ROUNDTRIP_STRUCTURAL_INVALIDITY_OBSERVED in artifact.matched_findings
    )
    assert artifact.conclusion is not (
        DiagnosticConclusion.ROUNDTRIP_STRUCTURAL_INVALIDITY_OBSERVED
    )


def test_primary_conclusion_follows_the_fixed_priority() -> None:
    codec = FakeCodec(corrupt_round_trip=True, corrupt_fraction=0.5)
    artifact, _, _, _ = _run(
        lambda seed, ctx: _make_invalid(_shift(TRUE_TARGET, 1.30)), codec=codec
    )
    matched = set(artifact.matched_findings)
    expected = next(f for f in PRIMARY_PRIORITY if f in matched)
    assert artifact.conclusion is expected
    assert artifact.conclusion in artifact.matched_findings


def test_a_recommended_next_experiment_is_always_present() -> None:
    artifact, _, _, _ = _run(lambda seed, ctx: _shift(TRUE_TARGET, 1.01))
    assert isinstance(artifact.recommended_next_experiment, NextExperiment)


def test_no_valid_rollouts_recommends_constrained_search() -> None:
    artifact, _, _, _ = _run(lambda seed, ctx: _make_invalid(_shift(TRUE_TARGET, 1.0001)))
    assert DiagnosticConclusion.NO_VALID_ROLLOUTS_OBSERVED in artifact.matched_findings
    assert artifact.recommended_next_experiment is NextExperiment.CONSTRAINED_TOKEN_SEARCH


# ============================================= size-matched controls


def _paths(count: int, spread: float, rng: random.Random) -> list[tuple[OfficialRow, ...]]:
    out = []
    for _ in range(count):
        factor = 1.0 + rng.gauss(0.0, spread)
        out.append(_shift(TRUE_TARGET, factor))
    return out


@pytest.mark.parametrize("k", [0, 1, 7, 32, 64])
def test_controls_handle_every_k(k: int) -> None:
    rng = random.Random(1)
    paths = _paths(64, 0.02, rng)
    sessions = tuple(r.session for r in TRUE_TARGET)
    controls = _size_matched_controls(
        paths=paths,
        k=k,
        sessions=sessions,
        target=TRUE_TARGET,
        anchor=ALL_ROWS[CONTEXT_CANDLES - 1].close,
        valid_only_error=0.01,
    )
    assert controls.k == k
    assert controls.seed == CONTROL_SEED
    assert controls.repetitions == CONTROL_REPETITIONS
    if k == 0:
        assert controls.scored_repetitions == 0
        assert controls.mean_primary_error is None
        assert controls.undefined_reason is not None
        assert not controls.degenerate
    elif k == 64:
        # Every draw is the same set, so one repetition and a degenerate flag.
        assert controls.degenerate is True
        assert controls.scored_repetitions == 1
        assert controls.mean_primary_error is not None
    else:
        assert controls.degenerate is False
        assert controls.scored_repetitions == CONTROL_REPETITIONS
        assert controls.mean_primary_error is not None
        low = controls.minimum_primary_error
        mid = controls.median_primary_error
        high = controls.maximum_primary_error
        spread = controls.standard_deviation_primary_error
        assert low is not None and mid is not None and high is not None
        assert spread is not None
        assert low <= mid <= high
        assert spread >= 0.0


def test_a_k_above_the_rollout_count_fails_softly() -> None:
    controls = _size_matched_controls(
        paths=_paths(8, 0.02, random.Random(2)),
        k=9,
        sessions=tuple(r.session for r in TRUE_TARGET),
        target=TRUE_TARGET,
        anchor=ALL_ROWS[CONTEXT_CANDLES - 1].close,
        valid_only_error=0.01,
    )
    assert controls.scored_repetitions == 0
    assert "exceeds" in (controls.undefined_reason or "")


def test_controls_are_reproducible_under_the_fixed_seed() -> None:
    paths = _paths(64, 0.02, random.Random(3))
    kwargs = {
        "paths": paths,
        "k": 11,
        "sessions": tuple(r.session for r in TRUE_TARGET),
        "target": TRUE_TARGET,
        "anchor": ALL_ROWS[CONTEXT_CANDLES - 1].close,
        "valid_only_error": 0.01,
    }
    first = _size_matched_controls(**kwargs)
    second = _size_matched_controls(**kwargs)
    assert first.mean_primary_error == second.mean_primary_error
    assert first.median_primary_error == second.median_primary_error
    assert first.minimum_primary_error == second.minimum_primary_error

    different = _size_matched_controls(**{**kwargs, "seed": CONTROL_SEED + 1})
    assert different.mean_primary_error != first.mean_primary_error


def test_sampling_is_without_replacement() -> None:
    generator = random.Random(CONTROL_SEED)
    for _ in range(50):
        chosen = generator.sample(list(range(64)), 10)
        assert len(set(chosen)) == 10


def test_random_validity_receives_no_artificial_credit_or_penalty() -> None:
    """The defect v3 exists to remove, checked directly.

    Validity is assigned at random, so it carries no information. Against the
    all-rollout ensemble the valid-only ensemble looks far worse purely because
    it averages fewer paths. Against size-matched controls it sits in the
    middle, which is what "no effect" should look like.
    """
    rng = random.Random(11)
    paths = _paths(64, 0.03, rng)
    sessions = tuple(r.session for r in TRUE_TARGET)
    anchor = ALL_ROWS[CONTEXT_CANDLES - 1].close

    def error_of(subset):
        from openalpha_kronos.studies.structural_validity.common.methods import _ensemble

        ensemble = _ensemble(subset, sessions)
        assert ensemble is not None
        return forecast_error(ensemble, TRUE_TARGET, anchor_close=anchor).close_return_mae

    ranks = []
    for trial in range(25):
        picker = random.Random(1000 + trial)
        valid_index = [i for i in range(64) if picker.random() < 0.25]
        if len(valid_index) < 2:
            continue
        valid_only = error_of([paths[i] for i in valid_index])
        controls = _size_matched_controls(
            paths=paths,
            k=len(valid_index),
            sessions=sessions,
            target=TRUE_TARGET,
            anchor=anchor,
            valid_only_error=valid_only,
            seed=CONTROL_SEED + trial,
            repetitions=60,
        )
        ranks.append(controls.valid_only_percentile_rank)

    assert len(ranks) >= 10
    average_rank = sum(ranks) / len(ranks)
    # Centred, not pinned to either end. Under the old all-rollout comparison
    # the valid-only ensemble was systematically worse regardless of validity.
    assert 0.25 < average_rank < 0.75


def test_the_filtering_rule_uses_the_controls_not_the_all_rollout_ensemble() -> None:
    import inspect

    from openalpha_kronos.studies.structural_validity.common import conclusion as conclusion_module

    source = inspect.getsource(conclusion_module.decide)
    r4 = source[source.index('"R4"') : source.index('"R5"')]
    assert "control_mean" in r4
    assert "manual_seeded_ensemble_error" not in r4


def test_secondary_evidence_is_retained() -> None:
    artifact, _, _, _ = _run(
        lambda seed, ctx: (
            _shift(TRUE_TARGET, 1.001)
            if _position(seed) % 2 == 0
            else _make_invalid(_shift(TRUE_TARGET, 1.20))
        )
    )
    d = artifact.method_d
    assert d.valid_group_mean_primary_error is not None
    assert d.invalid_group_mean_primary_error is not None
    assert d.group_absolute_difference is not None
    assert d.group_relative_difference is not None
    # Spearman over 64 rollouts, descriptive only.
    assert d.invalidity_error_spearman is None or -1.0 <= d.invalidity_error_spearman <= 1.0


def test_no_probability_weighted_ensemble_exists() -> None:
    from openalpha_kronos.studies.structural_validity.common.methods import MethodDResult

    assert not any("weighted" in name for name in MethodDResult.model_fields)


# ================================================= persistence baseline


def test_persistence_predicts_a_flat_close() -> None:
    anchor = ALL_ROWS[CONTEXT_CANDLES - 1].close
    baseline = persistence_forecast(TRUE_TARGET, anchor_close=anchor)
    assert len(baseline) == TARGET_CANDLES
    assert all(row.close == anchor for row in baseline)
    error = forecast_error(baseline, TRUE_TARGET, anchor_close=anchor)
    # Its predicted returns are all zero, so its return MAE is the mean
    # absolute actual return.
    assert error.defined
    assert error.close_return_mae is not None and error.close_return_mae > 0.0


def test_perfect_forecast_has_skill_one() -> None:
    anchor = ALL_ROWS[CONTEXT_CANDLES - 1].close
    comparison = compare_to_persistence(
        TRUE_TARGET, TRUE_TARGET, anchor_close=anchor, label="perfect"
    )
    assert comparison.defined
    assert comparison.candidate_close_return_mae == pytest.approx(0.0, abs=1e-12)
    assert comparison.close_return_skill == pytest.approx(1.0)


def test_persistence_against_itself_has_zero_skill() -> None:
    anchor = ALL_ROWS[CONTEXT_CANDLES - 1].close
    baseline = persistence_forecast(TRUE_TARGET, anchor_close=anchor)
    comparison = compare_to_persistence(
        baseline, TRUE_TARGET, anchor_close=anchor, label="persistence"
    )
    assert comparison.close_return_skill == pytest.approx(0.0)


def test_a_bad_forecast_has_negative_skill() -> None:
    anchor = ALL_ROWS[CONTEXT_CANDLES - 1].close
    comparison = compare_to_persistence(
        _shift(TRUE_TARGET, 1.5), TRUE_TARGET, anchor_close=anchor, label="bad"
    )
    assert comparison.defined
    assert comparison.close_return_skill is not None
    assert comparison.close_return_skill < 0.0


def test_an_absent_candidate_is_undefined_not_zero() -> None:
    comparison = compare_to_persistence(
        None, TRUE_TARGET, anchor_close=ALL_ROWS[CONTEXT_CANDLES - 1].close, label="absent"
    )
    assert comparison.defined is False
    assert comparison.close_return_skill is None


def test_every_candidate_gets_a_persistence_comparison() -> None:
    artifact, _, _, _ = _run(lambda seed, ctx: _shift(TRUE_TARGET, 1.01))
    assert artifact.method_b.persistence.label == "method_b_raw"
    assert artifact.method_c.persistence_after.label == "method_c_projected"
    assert artifact.method_d.manual_ensemble_persistence is not None
    assert artifact.method_d.valid_only_persistence is not None


# ============================================== directional accuracy


def test_directional_accuracy_counts_only_nonzero_pairs() -> None:
    anchor = 100.0
    rows = []
    for index, close in enumerate([101.0, 100.0, 99.0, 100.0]):
        rows.append(
            OfficialRow(
                session=START + timedelta(days=index),
                open=close,
                high=close,
                low=close,
                close=close,
                volume=1.0,
                amount=1.0,
            )
        )
    actual = tuple(rows)
    # Predicted equals actual, so every nonzero step agrees.
    result = directional_accuracy(actual, actual, anchor_close=anchor)
    assert result.defined
    assert result.scored_steps == 4
    assert result.directional_steps == 4
    assert result.agreeing_steps == 4
    assert result.accuracy == pytest.approx(1.0)


def test_directional_accuracy_reports_ties_separately() -> None:
    anchor = 100.0
    flat = tuple(
        OfficialRow(
            session=START + timedelta(days=i),
            open=100.0,
            high=100.0,
            low=100.0,
            close=100.0,
            volume=1.0,
            amount=1.0,
        )
        for i in range(4)
    )
    result = directional_accuracy(flat, flat, anchor_close=anchor)
    assert result.defined
    assert result.directional_steps == 0
    assert result.accuracy is None  # nothing to be right about
    assert result.predicted_zero_steps == 4
    assert result.actual_zero_steps == 4
    assert result.both_zero_steps == 4


def test_directional_accuracy_detects_opposite_signs() -> None:
    anchor = 100.0
    up = tuple(
        OfficialRow(
            session=START + timedelta(days=i),
            open=101.0 + i,
            high=101.0 + i,
            low=101.0 + i,
            close=101.0 + i,
            volume=1.0,
            amount=1.0,
        )
        for i in range(3)
    )
    down = tuple(
        OfficialRow(
            session=START + timedelta(days=i),
            open=99.0 - i,
            high=99.0 - i,
            low=99.0 - i,
            close=99.0 - i,
            volume=1.0,
            amount=1.0,
        )
        for i in range(3)
    )
    result = directional_accuracy(up, down, anchor_close=anchor)
    assert result.directional_steps == 3
    assert result.agreeing_steps == 0
    assert result.accuracy == pytest.approx(0.0)


# ==================================================== clipping analysis


def test_clipping_is_counted_by_channel_and_by_row() -> None:
    rows = list(ALL_ROWS[:CONTEXT_CANDLES])
    state = fit_context_state(tuple(rows))
    report = clipping_report(state, tuple(rows))
    assert report.rows == CONTEXT_CANDLES
    assert report.scalar_values == CONTEXT_CANDLES * 6
    assert set(report.clipped_by_column) == set(OFFICIAL_COLUMNS)
    assert 0.0 <= report.clipped_scalar_fraction <= 1.0


def test_a_clipped_row_is_identified() -> None:
    rows = list(ALL_ROWS[:CONTEXT_CANDLES])
    rows[5] = rows[5].model_copy(update={"volume": 1e12})
    state = fit_context_state(tuple(rows))
    report = clipping_report(state, tuple(rows))
    assert report.rows_with_any_clipped_channel >= 1
    assert 5 in report.clipped_row_indices
    assert report.clipped_by_column["volume"] >= 1


def test_invalidity_is_split_by_clipped_and_unclipped_input() -> None:
    codec = FakeCodec(corrupt_round_trip=True, corrupt_count=3)
    artifact, _, _, _ = _run(lambda seed, ctx: _shift(TRUE_TARGET, 1.01), codec=codec)
    a = artifact.method_a
    assert a.clipped_input_rows + a.unclipped_input_rows == TOTAL_CANDLES
    assert (
        a.invalid_rows_with_clipped_input + a.invalid_rows_with_unclipped_input
        == a.validity_all.invalid_candle_count
    )
    # The fake corrupts the first three rows, which are not clipped, so the
    # decoder reading is the one that applies.
    assert a.invalid_rows_with_unclipped_input == 3
    assert (
        DiagnosticConclusion.ROUNDTRIP_INVALIDITY_ON_UNCLIPPED_INPUTS in artifact.matched_findings
    )


def test_clipping_statistics_cover_the_target_suffix_too() -> None:
    artifact, _, _, _ = _run(lambda seed, ctx: _shift(TRUE_TARGET, 1.01))
    assert artifact.method_a.clipping_target_suffix.rows == TARGET_CANDLES
    assert artifact.method_a.clipping_all.rows == TOTAL_CANDLES


# ============================================ reproducibility check


def test_method_b_and_rollout_zero_agree_exactly() -> None:
    artifact, _, _, _ = _run(lambda seed, ctx: _shift(TRUE_TARGET, 1.01))
    check = artifact.decision.reproducibility
    assert check.performed is True
    assert check.agrees is True
    assert check.coarse_tokens_agree
    assert check.fine_tokens_agree
    assert check.sampling_log_probabilities_agree
    assert check.raw_decoded_suffix_agrees
    assert check.validity_agrees
    assert check.forecast_metrics_agree
    assert DiagnosticConclusion.REPRODUCIBILITY_FAILURE not in artifact.matched_findings


def test_a_deliberate_disagreement_fails_closed() -> None:
    """A run whose two identical-seed paths differ is not scientifically read."""
    artifact, _, _, _ = _run(lambda seed, ctx: _shift(TRUE_TARGET, 1.01))
    tampered_b = artifact.method_b.model_copy(
        update={"coarse_token_ids": tuple(reversed(artifact.method_b.coarse_token_ids))}
    )
    outcome = decide(
        method_a=artifact.method_a,
        method_b=tampered_b,
        method_c=artifact.method_c,
        method_d=artifact.method_d,
        reproducibility=ReproducibilityCheck(
            performed=True,
            agrees=False,
            coarse_tokens_agree=False,
            detail="deliberate disagreement",
        ),
    )
    assert DiagnosticConclusion.REPRODUCIBILITY_FAILURE in outcome.matched_findings
    assert outcome.primary_conclusion is DiagnosticConclusion.REPRODUCIBILITY_FAILURE
    assert outcome.recommended_next_experiment is NextExperiment.REPAIR_REPRODUCIBILITY


def test_a_seed_mismatch_is_not_treated_as_agreement() -> None:
    from openalpha_kronos.studies.structural_validity.mini.runner import _compare_b_to_rollout_zero

    artifact, _, _, _ = _run(lambda seed, ctx: _shift(TRUE_TARGET, 1.01))
    shifted = artifact.method_b.model_copy(update={"seed": artifact.method_b.seed + 1})
    check = _compare_b_to_rollout_zero(shifted, artifact.method_d)
    assert check.performed is False
    assert check.agrees is False
    assert "not comparable" in (check.detail or "")
