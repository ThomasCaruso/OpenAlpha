from datetime import date, timedelta

from openalpha_sentinel.interventions import build_risk_coverage, evaluate_policies


def _rows(count: int = 20) -> tuple[dict[str, object], ...]:
    start = date(2024, 7, 5)
    rows = []
    for index in range(count):
        asset = 'SPY' if index % 2 == 0 else 'QQQ'
        cutoff = (start + timedelta(days=7 * (index // 2))).isoformat()
        realized = 0.01 if index % 3 else -0.01
        predicted = realized + 0.001 * (index + 1)
        canonical_error = abs(predicted - realized)
        candidate_error = max(canonical_error - 0.002, 0.0)
        rows.append(
            {
                'origin_id': f'{asset}-{index:03d}',
                'asset': asset,
                'cutoff': cutoff,
                'terminal_status': 'completed',
                'raw_close_return_forecast': predicted,
                'realized_outcome': {'realized_log_return': realized},
                'kronos_absolute_error': canonical_error,
                'baseline_absolute_error': abs(realized),
                'direction_correct': predicted * realized > 0,
                'failure_label': index >= int(count * 0.75),
                'forecast_error': {
                    'valid_path_aggregation_candidate': {
                        'applied': index < 12,
                        'valid_path_count': 2 if index < 12 else 1,
                        'predicted_log_return': realized + candidate_error,
                        'absolute_error': candidate_error,
                    }
                },
            }
        )
    return tuple(rows)


def _oof(rows: tuple[dict[str, object], ...]) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            'origin_id': row['origin_id'],
            'cutoff': row['cutoff'],
            'asset': row['asset'],
            'failure_label': row['failure_label'],
            'kronos_absolute_error': row['kronos_absolute_error'],
            'predicted_failure_probability': index / max(len(rows) - 1, 1),
            'predicted_absolute_error': row['kronos_absolute_error'],
        }
        for index, row in enumerate(rows)
    )


def test_risk_coverage_reports_all_locked_levels_and_counts() -> None:
    rows = _rows(10)
    coverage = build_risk_coverage(rows, _oof(rows))

    assert [item['coverage'] for item in coverage['pooled']] == [1.0, 0.9, 0.8, 0.7, 0.5]
    assert [item['sample_count'] for item in coverage['pooled']] == [10, 9, 8, 7, 5]
    assert all('kronos_mae' in item for item in coverage['pooled'])
    assert all('baseline_mae' in item for item in coverage['pooled'])
    assert all('directional_accuracy' in item for item in coverage['pooled'])
    assert set(coverage['by_asset']) == {'SPY', 'QQQ'}


def test_p2_uses_pooled_50th_and_80th_risk_thresholds() -> None:
    rows = _rows(20)
    policies = evaluate_policies(rows, _oof(rows))
    actions = {item['origin_id']: item['action'] for item in policies['p2']['decisions']}

    assert policies['p2']['blend_weight'] == 0.5
    assert sum(action == 'USE' for action in actions.values()) == 10
    assert sum(action == 'BLEND' for action in actions.values()) == 6
    assert sum(action == 'ABSTAIN' for action in actions.values()) == 4
    assert policies['p2']['asset_specific_thresholds'] is False


def test_p3_retention_requires_support_positive_ci_and_no_asset_worsening() -> None:
    rows = _rows(20)
    policies = evaluate_policies(rows, _oof(rows))

    assert policies['p3']['application_count'] == 12
    assert policies['p3']['pooled_mean_improvement'] > 0
    assert policies['p3']['paired_improvement_ci']['lower'] > 0
    assert all(value >= 0 for value in policies['p3']['mean_improvement_by_asset'].values())
    assert policies['p3']['retain_for_holdout'] is True


def test_p0_and_p3_use_all_completed_rows_not_only_oof_rows() -> None:
    rows = _rows(20)
    partial_oof = _oof(rows[4:])

    policies = evaluate_policies(rows, partial_oof)

    assert policies['p0']['sample_count'] == 20
    assert len(policies['p2']['decisions']) == 16
    assert policies['p3']['application_count'] == 12
