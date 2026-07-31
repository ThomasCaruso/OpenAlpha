from __future__ import annotations

import platform
import re
from collections.abc import Mapping, Sequence
from importlib.metadata import version
from typing import Literal

import numpy as np
from openalpha_research.artifacts import Sha256
from pydantic import ValidationError

from .contracts import FrozenModel
from .development_manifest import EXPERIMENT_SHA256
from .development_serialization import canonical_json_bytes, sha256_bytes
from .risk_model import (
    NONSTRUCTURAL_FEATURES,
    STRUCTURAL_FEATURES,
    RiskModelOOFResult,
    fit_final_risk_models,
)

_DATE = re.compile(r'^\d{4}-\d{2}-\d{2}$')


class FreezeCandidate(FrozenModel):
    schema_version: Literal['sentinel-phase3a-freeze-v1']
    experiment_sha256: Sha256
    holdout_accessed: Literal[False]
    failure_threshold: float
    selected_feature_family: str
    retained_diagnostics: tuple[str, ...]
    removed_diagnostics: dict[str, str]
    feature_order: tuple[str, ...]
    preprocessing: dict[str, object]
    logistic_model: dict[str, object]
    ridge_model: dict[str, object]
    reliability_score_formula: Literal['CLIP_0_100(100*(1-P_FAILURE))']
    action_policy: dict[str, object]
    structural_gate: dict[str, object]
    missing_data_policy: dict[str, object]
    failure_reason_thresholds: dict[str, object]
    p3_policy: dict[str, object]
    software_versions: dict[str, str]
    artifact_hashes: dict[str, Sha256]
    freeze_sha256: Sha256


def build_freeze_candidate(
    *,
    rows: Sequence[Mapping[str, object]],
    oof: RiskModelOOFResult,
    failure_threshold: float,
    artifact_hashes: Mapping[str, str],
    p3_retained: bool,
) -> FreezeCandidate:
    if oof.selected_family is None or not oof.oof_predictions:
        raise ValueError('no eligible OOF system can be frozen')
    fitted = fit_final_risk_models(rows, oof)
    preprocessing = fitted['preprocessing']
    if not isinstance(preprocessing, Mapping):
        raise TypeError('final preprocessing descriptor is invalid')
    retained = tuple(str(item) for item in preprocessing['kept_features'])
    feature_order = tuple(str(item) for item in preprocessing['output_feature_order'])
    removed = {
        str(name): str(reason)
        for name, reason in dict(preprocessing['removed_features']).items()
    }
    selected_declared = set(oof.family_results[oof.selected_family].declared_features)
    for name in (*STRUCTURAL_FEATURES, *NONSTRUCTURAL_FEATURES):
        if name not in selected_declared:
            removed[name] = 'FEATURE_FAMILY_NOT_SELECTED'
    predictions = fitted['development_predictions']
    if not isinstance(predictions, Sequence):
        raise TypeError('final development predictions are invalid')
    risks = np.asarray(
        [_number(item['predicted_failure_probability']) for item in predictions],
        dtype=float,
    )
    action_policy = {
        'threshold_source': 'FINAL_DEVELOPMENT_REFIT_RISK_DISTRIBUTION',
        'risk_quantile_50': float(np.quantile(risks, 0.50, method='linear')),
        'risk_quantile_80': float(np.quantile(risks, 0.80, method='linear')),
        'use_maximum_risk_quantile': 0.50,
        'blend_minimum_risk_quantile_exclusive': 0.50,
        'blend_maximum_risk_quantile': 0.80,
        'abstain_minimum_risk_quantile_exclusive': 0.80,
        'blend_weight': 0.5,
        'asset_specific_thresholds': False,
    }
    body = {
        'schema_version': 'sentinel-phase3a-freeze-v1',
        'experiment_sha256': EXPERIMENT_SHA256,
        'holdout_accessed': False,
        'failure_threshold': failure_threshold,
        'selected_feature_family': oof.selected_family,
        'retained_diagnostics': retained,
        'removed_diagnostics': dict(sorted(removed.items())),
        'feature_order': feature_order,
        'preprocessing': {
            'medians': preprocessing['medians'],
            'means': preprocessing['means'],
            'scales': preprocessing['scales'],
            'missing_indicators': preprocessing['missing_indicators'],
            'output_feature_order': preprocessing['output_feature_order'],
        },
        'logistic_model': fitted['logistic_model'],
        'ridge_model': fitted['ridge_model'],
        'reliability_score_formula': 'CLIP_0_100(100*(1-P_FAILURE))',
        'action_policy': action_policy,
        'structural_gate': {
            'finite_nonpositive_or_alignment_failure': 'BLOCK_USE_AND_PERSIST_FAILURE',
            'ohlc_ordering_failure': 'MARK_FAILED_PRESERVE_RAW_AND_PROJECT_SEPARATELY',
            'raw_path_relabeling_forbidden': True,
            'raw_close_return_remains_separately_evaluable': True,
            'reliability_action_determined_independently': True,
        },
        'missing_data_policy': {
            'drop_above_training_fraction': 0.20,
            'imputation': 'TRAINING_MEDIAN',
            'missing_indicator': True,
            'normalization': 'TRAINING_MEAN_AND_POPULATION_SCALE',
        },
        'failure_reason_thresholds': _reason_thresholds(rows, retained),
        'p3_policy': {
            'retained_for_holdout': p3_retained,
            'minimum_valid_512_paths': 2,
            'fallback': 'LOCKED_CANONICAL_WITH_FLAG',
            'raw_canonical_forecast_replaced': False,
        },
        'software_versions': {
            'python': platform.python_version(),
            'operating_system': platform.platform(),
            'numpy': version('numpy'),
            'scikit_learn': version('scikit-learn'),
            'scipy': version('scipy'),
        },
        'artifact_hashes': dict(sorted(artifact_hashes.items())),
    }
    freeze = FreezeCandidate.model_validate(
        {**body, 'freeze_sha256': sha256_bytes(canonical_json_bytes(body))}
    )
    if not verify_freeze_candidate(freeze):
        raise ValueError('newly built freeze candidate failed verification')
    return freeze


def verify_freeze_candidate(candidate: FreezeCandidate | Mapping[str, object]) -> bool:
    try:
        freeze = (
            candidate
            if isinstance(candidate, FreezeCandidate)
            else FreezeCandidate.model_validate(candidate)
        )
    except ValidationError:
        return False
    payload = freeze.model_dump(mode='json')
    if _contains_forbidden(payload):
        return False
    body = {key: value for key, value in payload.items() if key != 'freeze_sha256'}
    return sha256_bytes(canonical_json_bytes(body)) == freeze.freeze_sha256


def _reason_thresholds(
    rows: Sequence[Mapping[str, object]],
    retained: tuple[str, ...],
) -> dict[str, object]:
    negative = {
        'DIRECTIONAL_AGREEMENT',
        'CONTEXT_DIRECTION_AGREEMENT',
        'EARLIEST_INVALID_HORIZON_STEP',
    }
    thresholds: dict[str, object] = {}
    for name in retained:
        values = []
        for row in rows:
            diagnostics = row.get('diagnostics')
            if not isinstance(diagnostics, Mapping):
                continue
            value = diagnostics.get(name)
            if value is not None and np.isfinite(float(value)):
                values.append(float(value))
        if not values:
            continue
        quantile = 0.20 if name in negative else 0.80
        thresholds[name] = {
            'comparison': 'LESS_THAN_OR_EQUAL' if name in negative else 'GREATER_THAN_OR_EQUAL',
            'development_quantile': quantile,
            'threshold': float(np.quantile(values, quantile, method='linear')),
            'explanation_source': 'RECORDED_DIAGNOSTIC_ONLY',
            'existed_before_outcome': True,
        }
    thresholds['STRUCTURALLY_INVALID_MODEL_OUTPUT'] = {
        'comparison': 'CANONICAL_STRUCTURAL_STATUS_EQUALS_FAILED',
        'threshold': None,
        'explanation_source': 'STRUCTURAL_GATE',
        'existed_before_outcome': True,
    }
    return thresholds


def _contains_forbidden(value: object, *, key: str | None = None) -> bool:
    if key in {'observations', 'outcome_ohlcv', 'raw_provider_response'}:
        return True
    if isinstance(value, Mapping):
        return any(_contains_forbidden(item, key=str(name)) for name, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden(item, key=key) for item in value)
    return bool(
        isinstance(value, str)
        and _DATE.fullmatch(value)
        and value >= '2025-07-01'
    )


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError('expected numeric value')
    return float(value)
