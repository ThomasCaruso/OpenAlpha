from datetime import date, timedelta

from openalpha_sentinel.development_analysis import (
    analyze_diagnostics,
    analyze_structural_error,
    build_structural_prevalence,
    moving_block_bootstrap,
)
from openalpha_sentinel.risk_model import NONSTRUCTURAL_FEATURES, STRUCTURAL_FEATURES


def _rows() -> tuple[dict[str, object], ...]:
    rows = []
    start = date(2024, 7, 5)
    for week in range(12):
        cutoff = (start + timedelta(days=7 * week)).isoformat()
        for asset_index, asset in enumerate(('SPY', 'QQQ')):
            invalid = (week + asset_index) % 3 == 0
            diagnostics = {
                name: float(week + asset_index + index) / 10.0
                for index, name in enumerate(
                    (*STRUCTURAL_FEATURES, *NONSTRUCTURAL_FEATURES)
                )
            }
            diagnostics['NONFINITE_OUTPUT_COUNT'] = 0.0
            seed = 1729 if asset == 'SPY' else 2027
            path_id = f'kronos-512-{seed}-req_abc123'
            rows.append(
                {
                    'origin_id': f'{asset}-{cutoff}',
                    'asset': asset,
                    'cutoff': cutoff,
                    'terminal_status': 'completed',
                    'diagnostics': diagnostics,
                    'diagnostic_missingness': {name: False for name in diagnostics},
                    'kronos_absolute_error': 0.01 + 0.002 * week + 0.001 * asset_index,
                    'failure_label': week >= 9,
                    'deployability_label': week % 2 == 0,
                    'direction_correct': week % 3 != 0,
                    'raw_close_return_forecast': 0.001 * week,
                    'individual_path_results': [
                        {
                            'path_id': path_id,
                            'context_length': 512,
                            'sampling_seed': seed,
                            'structural_valid': not invalid,
                            'predicted_log_return': 0.001 * week,
                            'absolute_return_error': (
                                0.02 if invalid else 0.01
                            ),
                            'direction_correct': week % 3 != 0,
                        }
                    ],
                    'individual_structural_validity': [
                        {
                            'path_id': path_id,
                            'valid': not invalid,
                            'invalid_candle_count': 1 if invalid else 0,
                            'earliest_invalid_horizon_step': 3 if invalid else None,
                            'violations': (
                                [
                                    {
                                        'code': 'HIGH_BELOW_CLOSE',
                                        'step': 3,
                                        'normalized_severity': 0.002,
                                    }
                                ]
                                if invalid
                                else []
                            ),
                        }
                    ],
                    'canonical_structural_validity': {'valid': not invalid},
                }
            )
    return tuple(rows)


def test_structural_prevalence_breakdowns_reconcile_to_totals() -> None:
    prevalence = build_structural_prevalence(_rows())

    assert prevalence['total_path_count'] == 24
    assert prevalence['total_candle_count'] == 120
    assert prevalence['invalid_path_count'] == 8
    assert prevalence['invalid_candle_count'] == 8
    assert sum(item['path_count'] for item in prevalence['by_asset'].values()) == 24
    assert sum(item['path_count'] for item in prevalence['by_context'].values()) == 24
    assert sum(item['path_count'] for item in prevalence['by_seed'].values()) == 24
    assert set(prevalence['by_context']) == {'512'}
    assert set(prevalence['by_seed']) == {'1729', '2027'}
    assert prevalence['violation_types']['HIGH_BELOW_CLOSE'] == 8
    assert prevalence['by_horizon_step']['3'] == 8


def test_moving_block_bootstrap_is_reproducible() -> None:
    values = {
        (date(2024, 7, 5) + timedelta(days=7 * index)).isoformat(): (
            float(index),
            float(index + 1),
        )
        for index in range(12)
    }

    first = moving_block_bootstrap(values, statistic='mean')
    second = moving_block_bootstrap(values, statistic='mean')

    assert first == second
    assert first['resamples'] == 1000
    assert first['block_weeks'] == 4
    assert first['seed'] == 314159
    assert first['lower'] <= first['estimate'] <= first['upper']


def test_diagnostic_analysis_reports_all_features_including_constant_failures() -> None:
    analysis = analyze_diagnostics(_rows())

    assert set(analysis) == {*STRUCTURAL_FEATURES, *NONSTRUCTURAL_FEATURES}
    assert analysis['NONFINITE_OUTPUT_COUNT']['status'] == 'unavailable'
    assert analysis['NONFINITE_OUTPUT_COUNT']['reason'] == 'CONSTANT_DIAGNOSTIC'
    assert analysis['RETURN_DISPERSION']['status'] == 'available'
    assert 'future_absolute_error_spearman' in analysis['RETURN_DISPERSION']
    assert set(analysis['RETURN_DISPERSION']['sign_by_asset']) == {'SPY', 'QQQ'}


def test_structural_control_analysis_is_labeled_descriptive() -> None:
    analysis = analyze_structural_error(_rows())

    assert analysis['analysis_role'] == 'DESCRIPTIVE_CONTROL_NOT_PREDICTIVE_EVIDENCE'
    assert set(analysis['controlled_coefficients']) == set(STRUCTURAL_FEATURES)
    assert analysis['predictive_incremental_evidence_source'] == 'CHRONOLOGICAL_OOF_ABLATION'
    assert analysis['individual_valid_count'] == 16
    assert analysis['individual_invalid_count'] == 8
    assert analysis['individual_valid_error_mean'] == 0.01
    assert analysis['individual_invalid_error_mean'] == 0.02
