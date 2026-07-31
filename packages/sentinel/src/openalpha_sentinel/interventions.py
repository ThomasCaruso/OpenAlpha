from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from .development_analysis import moving_block_bootstrap

COVERAGE_LEVELS = (1.0, 0.9, 0.8, 0.7, 0.5)


def build_risk_coverage(
    rows: Sequence[Mapping[str, Any]],
    oof_predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    joined = _join_oof(rows, oof_predictions)
    pooled = tuple(_coverage_entry(joined, level) for level in COVERAGE_LEVELS)
    by_asset = {
        asset: tuple(
            _coverage_entry(
                tuple(item for item in joined if item['asset'] == asset),
                level,
            )
            for level in COVERAGE_LEVELS
        )
        for asset in ('SPY', 'QQQ')
    }
    risks = np.asarray([float(item['risk']) for item in joined], dtype=float)
    errors = np.asarray([float(item['kronos_absolute_error']) for item in joined], dtype=float)
    risk_error_spearman = _safe_spearman(risks, errors)
    quintiles = []
    for index, indices in enumerate(np.array_split(np.argsort(risks, kind='stable'), 5), start=1):
        selected = tuple(joined[int(item)] for item in indices)
        quintiles.append(
            {
                'quintile': index,
                'sample_count': len(selected),
                'mean_risk': _mean(float(item['risk']) for item in selected),
                'mean_error': _mean(
                    float(item['kronos_absolute_error']) for item in selected
                ),
            }
        )
    return {
        'schema_version': 'sentinel-phase3a-risk-coverage-v1',
        'risk_error_spearman': risk_error_spearman,
        'error_by_risk_quintile': quintiles,
        'pooled': pooled,
        'by_asset': by_asset,
        'through_time': _quarterly_summary(joined),
        'calibration': _calibration(joined),
        'confusion_matrix_at_0_5': _confusion(joined),
    }


def evaluate_policies(
    rows: Sequence[Mapping[str, Any]],
    oof_predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    joined = _join_oof(rows, oof_predictions)
    completed = _join_completed(rows)
    risks = np.asarray([float(item['risk']) for item in joined], dtype=float)
    threshold_50 = float(np.quantile(risks, 0.50, method='linear'))
    threshold_80 = float(np.quantile(risks, 0.80, method='linear'))
    decisions = []
    p2_errors = []
    for item in joined:
        risk = float(item['risk'])
        if risk <= threshold_50:
            action = 'USE'
            prediction = float(item['predicted_return'])
        elif risk <= threshold_80:
            action = 'BLEND'
            prediction = 0.5 * float(item['predicted_return'])
        else:
            action = 'ABSTAIN'
            prediction = None
        error = (
            abs(float(prediction) - float(item['realized_return']))
            if prediction is not None
            else None
        )
        if error is not None:
            p2_errors.append(error)
        decisions.append(
            {
                'origin_id': item['origin_id'],
                'risk': risk,
                'action': action,
                'predicted_return': prediction,
                'absolute_error': error,
            }
        )
    p3 = _evaluate_p3(completed)
    return {
        'schema_version': 'sentinel-phase3a-interventions-v1',
        'p0': {
            'policy': 'RAW_ACCEPTANCE',
            'coverage': 1.0,
            'sample_count': len(completed),
            'kronos_mae': _mean(
                float(item['kronos_absolute_error']) for item in completed
            ),
            'baseline_mae': _mean(
                float(item['baseline_absolute_error']) for item in completed
            ),
        },
        'p1': {
            'policy': 'SENTINEL_ABSTENTION',
            'risk_coverage': build_risk_coverage(rows, oof_predictions),
        },
        'p2': {
            'policy': 'SENTINEL_BLEND',
            'risk_threshold_50': threshold_50,
            'risk_threshold_80': threshold_80,
            'blend_weight': 0.5,
            'asset_specific_thresholds': False,
            'decisions': decisions,
            'accepted_count': len(p2_errors),
            'accepted_mae': _mean(p2_errors),
        },
        'p3': p3,
    }


def _join_oof(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    by_id = {
        str(row['origin_id']): row
        for row in rows
        if row.get('terminal_status') == 'completed'
    }
    joined = []
    for prediction in predictions:
        origin_id = str(prediction['origin_id'])
        if origin_id not in by_id:
            raise ValueError('OOF prediction references an unknown completed origin')
        row = by_id[origin_id]
        realized = row.get('realized_outcome')
        if not isinstance(realized, Mapping):
            raise TypeError('completed row is missing realized outcome')
        joined.append(
            {
                'origin_id': origin_id,
                'asset': str(row['asset']),
                'cutoff': str(row['cutoff']),
                'risk': float(prediction['predicted_failure_probability']),
                'failure_label': bool(row['failure_label']),
                'kronos_absolute_error': float(row['kronos_absolute_error']),
                'baseline_absolute_error': float(row['baseline_absolute_error']),
                'direction_correct': bool(row['direction_correct']),
                'predicted_return': float(row['raw_close_return_forecast']),
                'realized_return': float(realized['realized_log_return']),
                'forecast_error': row.get('forecast_error'),
            }
        )
    return tuple(sorted(joined, key=lambda item: (item['cutoff'], item['asset'])))


def _join_completed(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    joined = []
    for row in rows:
        if row.get('terminal_status') != 'completed':
            continue
        realized = row.get('realized_outcome')
        if not isinstance(realized, Mapping):
            raise TypeError('completed row is missing realized outcome')
        joined.append(
            {
                'origin_id': str(row['origin_id']),
                'asset': str(row['asset']),
                'cutoff': str(row['cutoff']),
                'failure_label': bool(row['failure_label']),
                'kronos_absolute_error': float(row['kronos_absolute_error']),
                'baseline_absolute_error': float(row['baseline_absolute_error']),
                'direction_correct': bool(row['direction_correct']),
                'predicted_return': float(row['raw_close_return_forecast']),
                'realized_return': float(realized['realized_log_return']),
                'forecast_error': row.get('forecast_error'),
            }
        )
    return tuple(sorted(joined, key=lambda item: (item['cutoff'], item['asset'])))


def _coverage_entry(rows: tuple[dict[str, Any], ...], level: float) -> dict[str, Any]:
    if not rows:
        return {
            'coverage': level,
            'sample_count': 0,
            'accepted_origin_ids': (),
            'kronos_mae': None,
            'baseline_mae': None,
            'directional_accuracy': None,
        }
    count = max(1, math.ceil(level * len(rows) - 1e-12))
    ranked = tuple(
        sorted(rows, key=lambda item: (item['risk'], item['cutoff'], item['asset']))
    )
    accepted = ranked[:count]
    return {
        'coverage': level,
        'sample_count': len(accepted),
        'accepted_origin_ids': tuple(item['origin_id'] for item in accepted),
        'kronos_mae': _mean(
            float(item['kronos_absolute_error']) for item in accepted
        ),
        'baseline_mae': _mean(
            float(item['baseline_absolute_error']) for item in accepted
        ),
        'directional_accuracy': _mean(
            float(bool(item['direction_correct'])) for item in accepted
        ),
    }


def _evaluate_p3(rows: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    applied = []
    for item in rows:
        forecast_error = item.get('forecast_error')
        if not isinstance(forecast_error, Mapping):
            continue
        candidate = forecast_error.get('valid_path_aggregation_candidate')
        if not isinstance(candidate, Mapping) or not bool(candidate.get('applied')):
            continue
        improvement = float(item['kronos_absolute_error']) - float(candidate['absolute_error'])
        applied.append({**item, 'improvement': improvement})
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for item in applied:
        grouped[str(item['cutoff'])].append(float(item['improvement']))
    interval = (
        moving_block_bootstrap(grouped, statistic='mean')
        if grouped
        else {
            'estimate': None,
            'lower': None,
            'upper': None,
            'resamples': 1_000,
            'block_weeks': 4,
            'seed': 314159,
        }
    )
    by_asset = {
        asset: _mean(
            float(item['improvement']) for item in applied if item['asset'] == asset
        )
        for asset in ('SPY', 'QQQ')
    }
    pooled = _mean(float(item['improvement']) for item in applied)
    retain = (
        len(applied) >= 10
        and pooled is not None
        and pooled > 0
        and interval['lower'] is not None
        and float(interval['lower']) > 0
        and all(value is not None and value >= 0 for value in by_asset.values())
    )
    return {
        'policy': 'VALID_PATH_AGGREGATION_DEVELOPMENT_CANDIDATE',
        'application_count': len(applied),
        'pooled_mean_improvement': pooled,
        'paired_improvement_ci': interval,
        'mean_improvement_by_asset': by_asset,
        'retain_for_holdout': retain,
        'retention_rule': (
            'AT_LEAST_10_AND_POSITIVE_POOLED_LOWER_95_CI_AND_NO_ASSET_WORSENING'
        ),
    }


def _calibration(rows: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    bins = []
    ranked = tuple(sorted(rows, key=lambda item: item['risk']))
    for index, indices in enumerate(np.array_split(np.arange(len(ranked)), 5), start=1):
        selected = tuple(ranked[int(item)] for item in indices)
        bins.append(
            {
                'bin': index,
                'sample_count': len(selected),
                'mean_predicted_probability': _mean(float(item['risk']) for item in selected),
                'observed_failure_rate': _mean(
                    float(bool(item['failure_label'])) for item in selected
                ),
            }
        )
    return {'bins': bins}


def _confusion(rows: tuple[dict[str, Any], ...]) -> dict[str, int]:
    predicted = [float(item['risk']) >= 0.5 for item in rows]
    actual = [bool(item['failure_label']) for item in rows]
    return {
        'true_positive': sum(left and right for left, right in zip(predicted, actual)),
        'false_positive': sum(left and not right for left, right in zip(predicted, actual)),
        'true_negative': sum(not left and not right for left, right in zip(predicted, actual)),
        'false_negative': sum(not left and right for left, right in zip(predicted, actual)),
    }


def _quarterly_summary(rows: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in rows:
        month = int(str(item['cutoff'])[5:7])
        year = str(item['cutoff'])[:4]
        quarter = f'{year}-Q{(month - 1) // 3 + 1}'
        grouped[quarter].append(item)
    return {
        quarter: {
            'sample_count': len(items),
            'risk_error_spearman': _safe_spearman(
                [float(item['risk']) for item in items],
                [float(item['kronos_absolute_error']) for item in items],
            ),
        }
        for quarter, items in sorted(grouped.items())
    }


def _safe_spearman(left, right) -> float | None:
    if len(left) < 3 or np.unique(left).size <= 1 or np.unique(right).size <= 1:
        return None
    result: Any = spearmanr(left, right)
    value = float(result.statistic)
    return value if math.isfinite(value) else None


def _mean(values) -> float | None:
    materialized = tuple(values)
    return float(np.mean(materialized)) if materialized else None
