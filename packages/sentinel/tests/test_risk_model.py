from datetime import date, timedelta
from typing import Any

from openalpha_sentinel.risk_model import (
    NONSTRUCTURAL_FEATURES,
    STRUCTURAL_FEATURES,
    fit_chronological_oof,
)


def _rows(*, one_class: bool = False) -> tuple[dict[str, Any], ...]:
    start = date(2024, 7, 5)
    rows = []
    for week in range(52):
        cutoff = (start + timedelta(days=7 * week)).isoformat()
        for asset_index, asset in enumerate(('SPY', 'QQQ')):
            error = 0.005 + 0.001 * (week % 8) + 0.0005 * asset_index
            diagnostics = {
                name: float((week + asset_index + feature_index) % 11) / 10.0
                for feature_index, name in enumerate(
                    (*STRUCTURAL_FEATURES, *NONSTRUCTURAL_FEATURES)
                )
            }
            diagnostics['RETURN_DISPERSION'] = float(week) / 52.0
            rows.append(
                {
                    'origin_id': f'{asset}-{cutoff}',
                    'asset': asset,
                    'cutoff': cutoff,
                    'terminal_status': 'completed',
                    'diagnostics': diagnostics,
                    'failure_label': False if one_class else week % 4 == 3,
                    'kronos_absolute_error': error,
                }
            )
    return tuple(rows)


def test_chronological_oof_keeps_assets_paired_and_preprocessing_causal() -> None:
    result = fit_chronological_oof(_rows())

    assert len(result.folds) == 3
    for fold in result.folds:
        assert max(fold.training_cutoffs) < min(fold.validation_cutoffs)
        assert set(fold.training_origin_ids) == {
            row['origin_id']
            for row in _rows()
            if row['cutoff'] in fold.training_cutoffs
        }
        assert set(fold.validation_origin_ids) == {
            row['origin_id']
            for row in _rows()
            if row['cutoff'] in fold.validation_cutoffs
        }
    predicted_ids = {row['origin_id'] for row in result.oof_predictions}
    assert all(
        row['origin_id'] not in predicted_ids
        for row in _rows()
        if row['cutoff'] < result.folds[0].validation_cutoffs[0]
    )
    assert result.selected_family is not None
    first_state = result.family_results[result.selected_family].preprocessing[0]
    assert max(first_state.training_cutoffs) < result.folds[0].validation_cutoffs[0]


def test_all_locked_candidates_and_feature_families_are_reported_deterministically() -> None:
    first = fit_chronological_oof(_rows())
    second = fit_chronological_oof(_rows())

    assert tuple(first.family_results) == ('structural', 'nonstructural', 'combined')
    for family in first.family_results.values():
        assert tuple(item.parameter for item in family.logistic_candidates) == (
            0.01,
            0.1,
            1.0,
            10.0,
        )
        assert tuple(item.parameter for item in family.ridge_candidates) == (
            0.1,
            1.0,
            10.0,
            100.0,
        )
    assert first.selected_family == second.selected_family
    assert first.oof_predictions == second.oof_predictions


def test_one_class_training_folds_make_logistic_candidates_ineligible() -> None:
    result = fit_chronological_oof(_rows(one_class=True))

    assert result.selected_family is None
    assert result.oof_predictions == ()
    assert all(
        not candidate.eligible
        for family in result.family_results.values()
        for candidate in family.logistic_candidates
    )
