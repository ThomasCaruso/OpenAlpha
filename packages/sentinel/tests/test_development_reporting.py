from copy import deepcopy
from typing import Any

import pytest
from openalpha_sentinel.development_reporting import (
    recommend,
    render_development_report,
)
from openalpha_sentinel.development_serialization import (
    canonical_json_bytes,
    sha256_bytes,
)


def _evidence() -> dict[str, Any]:
    pooled = [
        {'coverage': 1.0, 'sample_count': 60, 'kronos_mae': 0.020},
        {'coverage': 0.9, 'sample_count': 54, 'kronos_mae': 0.019},
        {'coverage': 0.8, 'sample_count': 48, 'kronos_mae': 0.018},
        {'coverage': 0.7, 'sample_count': 42, 'kronos_mae': 0.017},
        {'coverage': 0.5, 'sample_count': 30, 'kronos_mae': 0.015},
    ]
    return {
        'operational_feasibility': {
            'request_success_rate': 0.99,
            'median_ensemble_latency_seconds': 300.0,
            'cache_bytes': 100_000_000,
            'hosted_equivalent_cost_usd_per_cutoff': 0.0,
        },
        'freeze_verified': True,
        'failed_origin_count': 0,
        'structural_prevalence': {
            'invalid_path_prevalence': 0.20,
            'invalid_candle_prevalence': 0.05,
            'invalid_canonical_forecast_prevalence': 0.06,
            'violation_types': {'HIGH_BELOW_CLOSE': 12},
            'by_horizon_step': {'1': 2, '2': 3, '3': 2, '4': 4, '5': 3},
        },
        'risk_coverage': {
            'risk_error_spearman': 0.30,
            'pooled': pooled,
            'by_asset': {
                'SPY': [
                    {'coverage': 1.0, 'kronos_mae': 0.020},
                    {'coverage': 0.7, 'kronos_mae': 0.019},
                ],
                'QQQ': [
                    {'coverage': 1.0, 'kronos_mae': 0.025},
                    {'coverage': 0.7, 'kronos_mae': 0.024},
                ],
            },
            'through_time': {
                '2024-Q3': {'risk_error_spearman': 0.25},
            },
            'error_by_risk_quintile': [
                {'quintile': 1, 'mean_error': 0.010},
                {'quintile': 2, 'mean_error': 0.012},
                {'quintile': 3, 'mean_error': 0.011},
                {'quintile': 4, 'mean_error': 0.016},
                {'quintile': 5, 'mean_error': 0.022},
            ],
        },
        'asset_risk_error_spearman': {'SPY': 0.22, 'QQQ': 0.18},
        'diagnostic_analysis': {
            'RETURN_DISPERSION': {
                'status': 'available',
                'future_absolute_error_spearman': 0.20,
            },
            'NONFINITE_OUTPUT_COUNT': {
                'status': 'unavailable',
                'reason': 'CONSTANT_DIAGNOSTIC',
            },
        },
        'model_comparison': {
            'selected_family': 'combined',
            'family_results': {
                'structural': {'selected_logistic_mean_metric': 0.2},
                'nonstructural': {'selected_logistic_mean_metric': 0.1},
                'combined': {'selected_logistic_mean_metric': 0.15},
            },
        },
        'intervention_analysis': {
            'p2': {'accepted_mae': 0.016, 'blend_weight': 0.5},
            'p3': {'retain_for_holdout': False},
        },
        'structural_error': {
            'canonical_valid_error_mean': 0.018,
            'canonical_invalid_error_mean': 0.024,
            'individual_valid_error_mean': 0.019,
            'individual_invalid_error_mean': 0.023,
        },
        'origin_summary': {
            'eligible_origins': 104,
            'completed_origins': 100,
            'failed_origins': 4,
            'generated_paths': 900,
        },
    }


def _artifacts(evidence: dict[str, Any]) -> dict[str, dict[str, Any]]:
    artifacts = {}
    for name, payload in evidence.items():
        artifacts[name] = {
            'payload': payload,
            'sha256': sha256_bytes(canonical_json_bytes(payload)),
        }
    return artifacts


def test_recommendation_mapping_is_applied_in_declared_order() -> None:
    all_pass = _evidence()
    assert recommend(all_pass) == 'PROCEED_TO_LOCKED_HOLDOUT'

    structural_only = deepcopy(all_pass)
    structural_only['risk_coverage']['risk_error_spearman'] = 0.0
    structural_only['risk_coverage']['pooled'][3]['kronos_mae'] = 0.021
    assert recommend(structural_only) == 'PROCEED_AS_STRUCTURAL_CONTRACT_ONLY'

    weak = deepcopy(all_pass)
    weak['risk_coverage']['risk_error_spearman'] = 0.05
    assert recommend(weak) == 'CHANGE_THE_RELIABILITY_APPROACH'

    none = deepcopy(structural_only)
    none['structural_prevalence']['invalid_path_prevalence'] = 0.01
    none['structural_prevalence']['invalid_canonical_forecast_prevalence'] = 0.01
    assert recommend(none) == 'STOP'


def test_report_requires_verified_artifacts_and_answers_thirteen_questions() -> None:
    artifacts = _artifacts(_evidence())
    freeze_hash = 'f' * 64
    report = render_development_report(
        artifacts=artifacts,
        freeze_sha256=freeze_hash,
    )

    assert report.startswith('DEVELOPMENT ANALYSIS - NOT HOLDOUT EVIDENCE')
    assert report.count('\n## Q') == 13
    assert 'NONFINITE_OUTPUT_COUNT: CONSTANT_DIAGNOSTIC' in report
    assert 'Failed origins: 4' in report
    assert freeze_hash in report
    assert 'PROCEED_TO_LOCKED_HOLDOUT' in report
    assert 'Monotonic increase across horizon steps: no.' in report
    assert 'P2 accepted MAE: 0.016' in report
    assert 'Lowest mean logistic log loss family: nonstructural (0.1).' in report
    assert 'decisions' not in report

    tampered = deepcopy(artifacts)
    tampered['risk_coverage']['payload']['risk_error_spearman'] = -1.0
    with pytest.raises(ValueError, match='hash mismatch'):
        render_development_report(artifacts=tampered, freeze_sha256=freeze_hash)
