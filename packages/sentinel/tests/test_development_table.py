from collections import OrderedDict

import numpy as np
from openalpha_sentinel.development_manifest import build_development_manifest
from openalpha_sentinel.development_table import build_development_table


def _completed(origin_id: str, error: float) -> dict[str, object]:
    return {
        'schema_version': 'sentinel-phase3a-analysis-row-v1',
        'terminal_status': 'completed',
        'origin_id': origin_id,
        'asset': 'SPY' if '-SPY-' in origin_id else 'QQQ',
        'cutoff': origin_id.split('-h5')[0][-10:],
        'horizon_end': '2024-07-12',
        'data_hash': 'a' * 64,
        'model_revision': 'b' * 40,
        'structural_status': 'PASSED',
        'structural_diagnostics': {
            'INVALID_PATH_FRACTION': {
                'status': 'available',
                'value': 0.0,
                'units': 'fraction',
            }
        },
        'raw_close_return_forecast': 0.01,
        'projected_path': {'implied_return_unchanged': True},
        'projected_path_status': 'PASSED',
        'reliability_features': {
            'RETURN_DISPERSION': {
                'status': 'available',
                'value': 12.0,
                'reason': None,
            },
            'RECENT_MODEL_ERROR': {
                'status': 'not_computable',
                'value': None,
                'reason': 'INSUFFICIENT_PRIOR_RESOLVED_FORECASTS',
            },
        },
        'realized_outcome': {'realized_log_return': 0.02},
        'forecast_error': {
            'kronos_absolute_error': error,
            'baseline_absolute_error': 0.02,
            'direction_correct': True,
            'deployability_label': error > 0.02,
        },
        'runtime_metadata': {'inference_duration_ms': 10.0},
    }


def test_table_is_manifest_ordered_deterministic_and_labeled() -> None:
    origins = build_development_manifest().origins[:4]
    errors = (0.1, 0.2, 0.3, 0.4)
    records = OrderedDict(
        (origin.origin_id, _completed(origin.origin_id, error))
        for origin, error in reversed(tuple(zip(origins, errors)))
    )

    first = build_development_table(origins, records)
    second = build_development_table(origins, dict(records))
    expected = float(np.quantile(errors, 0.75, method='linear'))

    assert first.failure_threshold == expected
    assert [row['origin_id'] for row in first.rows] == [item.origin_id for item in origins]
    assert [row['failure_label'] for row in first.completed_rows] == [
        error >= expected for error in errors
    ]
    assert first.jsonl_bytes.endswith(b'\n')
    assert first.jsonl_bytes == second.jsonl_bytes
    assert first.content_sha256 == second.content_sha256
    assert first.completed_count == 4
    assert first.failed_count == 0


def test_table_preserves_explicit_failure_and_diagnostic_missingness() -> None:
    origins = build_development_manifest().origins[:2]
    records = {
        origins[0].origin_id: _completed(origins[0].origin_id, 0.1),
        origins[1].origin_id: {
            'schema_version': 'sentinel-phase3a-terminal-failure-v1',
            'terminal_status': 'failed',
            'analysis_status': 'unavailable',
            'origin': origins[1].model_dump(mode='json'),
            'code': 'REAL_INFERENCE_FAILURE',
            'stage': 'FORECAST',
            'record_sha256': 'c' * 64,
        },
    }

    table = build_development_table(origins, records)

    assert table.completed_count == 1
    assert table.failed_count == 1
    assert table.rows[1]['failure_label'] is None
    assert table.rows[1]['failure_code'] == 'REAL_INFERENCE_FAILURE'
    assert table.rows[0]['diagnostic_missingness']['RECENT_MODEL_ERROR'] is True
    assert table.rows[0]['diagnostics']['RECENT_MODEL_ERROR'] is None
