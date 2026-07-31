from copy import deepcopy
from typing import cast

import numpy as np
from openalpha_sentinel.development_freeze import (
    build_freeze_candidate,
    verify_freeze_candidate,
)
from openalpha_sentinel.risk_model import fit_chronological_oof, fit_final_risk_models

from packages.sentinel.tests.test_risk_model import _rows


def _freeze():
    rows = _rows()
    oof = fit_chronological_oof(rows)
    return build_freeze_candidate(
        rows=rows,
        oof=oof,
        failure_threshold=0.0125,
        artifact_hashes={
            'development_table': 'a' * 64,
            'oof_predictions': 'b' * 64,
            'model_comparison': 'c' * 64,
            'intervention_analysis': 'd' * 64,
        },
        p3_retained=False,
    )


def test_freeze_contains_complete_reproducible_policy() -> None:
    rows = _rows()
    oof = fit_chronological_oof(rows)
    fitted = fit_final_risk_models(rows, oof)
    freeze = _freeze()

    assert freeze.failure_threshold == 0.0125
    assert freeze.selected_feature_family in {'structural', 'nonstructural', 'combined'}
    assert freeze.retained_diagnostics
    assert freeze.feature_order
    assert set(freeze.preprocessing) >= {
        'medians',
        'means',
        'scales',
        'missing_indicators',
    }
    assert set(freeze.logistic_model) >= {'C', 'coefficients', 'intercept'}
    assert set(freeze.ridge_model) >= {'alpha', 'coefficients', 'intercept'}
    assert freeze.reliability_score_formula == 'CLIP_0_100(100*(1-P_FAILURE))'
    assert freeze.action_policy['blend_weight'] == 0.5
    assert (
        freeze.action_policy['threshold_source']
        == 'FINAL_DEVELOPMENT_REFIT_RISK_DISTRIBUTION'
    )
    final_risks = [
        float(item['predicted_failure_probability'])
        for item in fitted['development_predictions']
    ]
    assert freeze.action_policy['risk_quantile_50'] == np.quantile(
        final_risks, 0.50, method='linear'
    )
    assert freeze.action_policy['risk_quantile_80'] == np.quantile(
        final_risks, 0.80, method='linear'
    )
    assert cast(float, freeze.action_policy['risk_quantile_50']) <= cast(
        float, freeze.action_policy['risk_quantile_80']
    )
    assert freeze.structural_gate['raw_path_relabeling_forbidden'] is True
    assert freeze.missing_data_policy['imputation'] == 'TRAINING_MEDIAN'
    assert freeze.software_versions['scikit_learn'] == '1.7.2'
    assert len(freeze.freeze_sha256) == 64
    assert verify_freeze_candidate(freeze) is True


def test_freeze_tamper_and_holdout_or_raw_payload_are_rejected() -> None:
    freeze = _freeze()
    tampered = freeze.model_dump(mode='python')
    tampered['logistic_model'] = deepcopy(tampered['logistic_model'])
    tampered['logistic_model']['coefficients'][0] += 1.0
    assert verify_freeze_candidate(tampered) is False

    forbidden = freeze.model_dump(mode='python')
    forbidden['observations'] = [1, 2, 3]
    assert verify_freeze_candidate(forbidden) is False
    holdout = freeze.model_dump(mode='python')
    holdout['cutoff'] = '2025-07-01'
    assert verify_freeze_candidate(holdout) is False
