from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from pydantic import Field
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from .contracts import FrozenModel

STRUCTURAL_FEATURES = (
    'INVALID_PATH_FRACTION',
    'INVALID_CANDLE_FRACTION',
    'TOTAL_CONSTRAINT_VIOLATIONS',
    'MAX_CONSTRAINT_VIOLATION_SEVERITY',
    'MEAN_CONSTRAINT_VIOLATION_SEVERITY',
    'EARLIEST_INVALID_HORIZON_STEP',
    'HIGH_LOW_INVERSION_COUNT',
    'HIGH_BELOW_BODY_COUNT',
    'LOW_ABOVE_BODY_COUNT',
    'NONFINITE_OUTPUT_COUNT',
    'NONPOSITIVE_PRICE_COUNT',
)
NONSTRUCTURAL_FEATURES = (
    'DIRECTIONAL_AGREEMENT',
    'RETURN_DISPERSION',
    'PATH_DISPERSION',
    'CONTEXT_DIRECTION_AGREEMENT',
    'CONTEXT_RETURN_SPREAD',
    'BASELINE_DISAGREEMENT',
    'RECENT_VOLATILITY',
    'VOLATILITY_CHANGE',
    'TREND_STRENGTH',
    'GAP_OR_OUTLIER_SCORE',
    'HISTORICAL_ANALOGUE_DISTANCE',
    'ANALOGUE_OUTCOME_DISPERSION',
    'RECENT_MODEL_ERROR',
    'HORIZON_PATH_DIVERGENCE',
)
FEATURE_FAMILIES = {
    'structural': STRUCTURAL_FEATURES,
    'nonstructural': NONSTRUCTURAL_FEATURES,
    'combined': (*STRUCTURAL_FEATURES, *NONSTRUCTURAL_FEATURES),
}
LOGISTIC_C_GRID = (0.01, 0.1, 1.0, 10.0)
RIDGE_ALPHA_GRID = (0.1, 1.0, 10.0, 100.0)


class ChronologicalFold(FrozenModel):
    fold_index: int = Field(ge=0)
    training_cutoffs: tuple[str, ...]
    validation_cutoffs: tuple[str, ...]
    training_origin_ids: tuple[str, ...]
    validation_origin_ids: tuple[str, ...]


class PreprocessingState(FrozenModel):
    fold_index: int
    training_cutoffs: tuple[str, ...]
    declared_features: tuple[str, ...]
    kept_features: tuple[str, ...]
    removed_features: dict[str, str]
    missing_indicators: tuple[str, ...]
    output_feature_order: tuple[str, ...]
    medians: dict[str, float]
    means: dict[str, float]
    scales: dict[str, float]


class CandidateResult(FrozenModel):
    model_kind: str
    parameter_name: str
    parameter: float
    eligible: bool
    ineligible_reason: str | None
    fold_metrics: tuple[float, ...]
    mean_metric: float | None
    standard_error: float | None
    predictions: tuple[dict[str, Any], ...]


class FeatureFamilyResult(FrozenModel):
    family: str
    declared_features: tuple[str, ...]
    preprocessing: tuple[PreprocessingState, ...]
    logistic_candidates: tuple[CandidateResult, ...]
    ridge_candidates: tuple[CandidateResult, ...]
    selected_logistic_parameter: float | None
    selected_ridge_parameter: float | None
    selected_logistic_mean_metric: float | None
    selected_logistic_standard_error: float | None


class RiskModelOOFResult(FrozenModel):
    schema_version: str
    folds: tuple[ChronologicalFold, ...]
    family_results: dict[str, FeatureFamilyResult]
    selected_family: str | None
    selection_rule: str
    oof_predictions: tuple[dict[str, Any], ...]


class _FoldMatrices:
    def __init__(
        self,
        *,
        state: PreprocessingState,
        train_x: np.ndarray,
        validation_x: np.ndarray,
        train_rows: tuple[Mapping[str, Any], ...],
        validation_rows: tuple[Mapping[str, Any], ...],
    ) -> None:
        self.state = state
        self.train_x = train_x
        self.validation_x = validation_x
        self.train_rows = train_rows
        self.validation_rows = validation_rows


def _chronological_folds(rows: tuple[Mapping[str, Any], ...]) -> tuple[ChronologicalFold, ...]:
    cutoffs = tuple(sorted({str(row['cutoff']) for row in rows}))
    splitter = TimeSeriesSplit(n_splits=3, gap=1)
    folds = []
    for fold_index, (train_indices, validation_indices) in enumerate(splitter.split(cutoffs)):
        training_cutoffs = tuple(cutoffs[int(index)] for index in train_indices)
        validation_cutoffs = tuple(cutoffs[int(index)] for index in validation_indices)
        folds.append(
            ChronologicalFold(
                fold_index=fold_index,
                training_cutoffs=training_cutoffs,
                validation_cutoffs=validation_cutoffs,
                training_origin_ids=tuple(
                    str(row['origin_id'])
                    for row in rows
                    if str(row['cutoff']) in training_cutoffs
                ),
                validation_origin_ids=tuple(
                    str(row['origin_id'])
                    for row in rows
                    if str(row['cutoff']) in validation_cutoffs
                ),
            )
        )
    return tuple(folds)


def _fit_preprocessing(
    *,
    fold: ChronologicalFold,
    declared_features: tuple[str, ...],
    train_rows: tuple[Mapping[str, Any], ...],
    validation_rows: tuple[Mapping[str, Any], ...],
) -> _FoldMatrices:
    raw_train = _raw_matrix(train_rows, declared_features)
    raw_validation = _raw_matrix(validation_rows, declared_features)
    removed: dict[str, str] = {}
    candidate_indices = []
    for index, name in enumerate(declared_features):
        values = raw_train[:, index]
        finite = values[np.isfinite(values)]
        missing_fraction = 1.0 - len(finite) / len(values)
        if missing_fraction > 0.20:
            removed[name] = 'TRAIN_MISSING_FRACTION_ABOVE_0_20'
        elif len(finite) == 0 or np.unique(finite).size <= 1:
            removed[name] = 'ZERO_TRAIN_VARIANCE'
        else:
            candidate_indices.append(index)
    kept_indices: list[int] = []
    for index in candidate_indices:
        redundant = False
        for earlier in kept_indices:
            left = raw_train[:, earlier]
            right = raw_train[:, index]
            valid = np.isfinite(left) & np.isfinite(right)
            if int(np.sum(valid)) < 3:
                continue
            correlation_result: Any = spearmanr(left[valid], right[valid])
            correlation = float(correlation_result.statistic)
            if math.isfinite(correlation) and abs(correlation) >= 0.95:
                removed[declared_features[index]] = (
                    f'REDUNDANT_WITH_{declared_features[earlier]}'
                )
                redundant = True
                break
        if not redundant:
            kept_indices.append(index)
    kept = tuple(declared_features[index] for index in kept_indices)
    train_kept = raw_train[:, kept_indices] if kept_indices else np.empty((len(train_rows), 0))
    validation_kept = (
        raw_validation[:, kept_indices]
        if kept_indices
        else np.empty((len(validation_rows), 0))
    )
    medians: dict[str, float] = {}
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    missing_indicators = tuple(
        name
        for name, index in zip(kept, range(len(kept)))
        if np.any(~np.isfinite(train_kept[:, index]))
    )
    transformed_train = []
    transformed_validation = []
    for index, name in enumerate(kept):
        train_column = train_kept[:, index]
        validation_column = validation_kept[:, index]
        median = float(np.nanmedian(np.where(np.isfinite(train_column), train_column, np.nan)))
        train_imputed = np.where(np.isfinite(train_column), train_column, median)
        validation_imputed = np.where(np.isfinite(validation_column), validation_column, median)
        mean = float(np.mean(train_imputed))
        scale = float(np.std(train_imputed, ddof=0))
        if scale == 0.0:
            scale = 1.0
        medians[name] = median
        means[name] = mean
        scales[name] = scale
        transformed_train.append((train_imputed - mean) / scale)
        transformed_validation.append((validation_imputed - mean) / scale)
    for name in missing_indicators:
        index = kept.index(name)
        transformed_train.append((~np.isfinite(train_kept[:, index])).astype(float))
        transformed_validation.append((~np.isfinite(validation_kept[:, index])).astype(float))
    train_x = np.column_stack(transformed_train) if transformed_train else np.empty((len(train_rows), 0))
    validation_x = (
        np.column_stack(transformed_validation)
        if transformed_validation
        else np.empty((len(validation_rows), 0))
    )
    state = PreprocessingState(
        fold_index=fold.fold_index,
        training_cutoffs=fold.training_cutoffs,
        declared_features=declared_features,
        kept_features=kept,
        removed_features=removed,
        missing_indicators=missing_indicators,
        output_feature_order=(*kept, *(f'{name}__MISSING' for name in missing_indicators)),
        medians=medians,
        means=means,
        scales=scales,
    )
    return _FoldMatrices(
        state=state,
        train_x=train_x,
        validation_x=validation_x,
        train_rows=train_rows,
        validation_rows=validation_rows,
    )


def _raw_matrix(
    rows: tuple[Mapping[str, Any], ...],
    features: tuple[str, ...],
) -> np.ndarray:
    matrix = np.full((len(rows), len(features)), np.nan, dtype=float)
    for row_index, row in enumerate(rows):
        diagnostics = row.get('diagnostics')
        if not isinstance(diagnostics, Mapping):
            raise TypeError('development row diagnostics must be a mapping')
        for feature_index, name in enumerate(features):
            value = diagnostics.get(name)
            if value is not None:
                matrix[row_index, feature_index] = float(value)
    return matrix


def fit_chronological_oof(
    rows: Sequence[Mapping[str, Any]],
) -> RiskModelOOFResult:
    completed = tuple(
        sorted(
            (row for row in rows if row.get('terminal_status') == 'completed'),
            key=lambda row: (str(row['cutoff']), 0 if row['asset'] == 'SPY' else 1),
        )
    )
    if len({str(row['cutoff']) for row in completed}) < 8:
        raise ValueError('chronological OOF requires at least eight unique cutoffs')
    folds = _chronological_folds(completed)
    family_results: dict[str, FeatureFamilyResult] = {}
    for family, features in FEATURE_FAMILIES.items():
        matrices = tuple(
            _fit_preprocessing(
                fold=fold,
                declared_features=features,
                train_rows=tuple(
                    row for row in completed if str(row['cutoff']) in fold.training_cutoffs
                ),
                validation_rows=tuple(
                    row for row in completed if str(row['cutoff']) in fold.validation_cutoffs
                ),
            )
            for fold in folds
        )
        logistic = tuple(_fit_logistic_candidate(value, matrices) for value in LOGISTIC_C_GRID)
        ridge = tuple(_fit_ridge_candidate(value, matrices) for value in RIDGE_ALPHA_GRID)
        selected_logistic = _select_candidate(logistic)
        selected_ridge = _select_candidate(ridge)
        family_results[family] = FeatureFamilyResult(
            family=family,
            declared_features=features,
            preprocessing=tuple(item.state for item in matrices),
            logistic_candidates=logistic,
            ridge_candidates=ridge,
            selected_logistic_parameter=(
                selected_logistic.parameter if selected_logistic is not None else None
            ),
            selected_ridge_parameter=(
                selected_ridge.parameter if selected_ridge is not None else None
            ),
            selected_logistic_mean_metric=(
                selected_logistic.mean_metric if selected_logistic is not None else None
            ),
            selected_logistic_standard_error=(
                selected_logistic.standard_error if selected_logistic is not None else None
            ),
        )
    selected_family = _select_family(family_results)
    oof_predictions: tuple[dict[str, Any], ...] = ()
    if selected_family is not None:
        family = family_results[selected_family]
        logistic = next(
            item
            for item in family.logistic_candidates
            if item.parameter == family.selected_logistic_parameter
        )
        ridge = next(
            item
            for item in family.ridge_candidates
            if item.parameter == family.selected_ridge_parameter
        )
        logistic_by_id = {
            str(item['origin_id']): item for item in logistic.predictions
        }
        ridge_by_id = {str(item['origin_id']): item for item in ridge.predictions}
        ordered_ids = tuple(
            str(row['origin_id'])
            for row in completed
            if str(row['origin_id']) in logistic_by_id
        )
        oof_predictions = tuple(
            {
                'origin_id': origin_id,
                'cutoff': logistic_by_id[origin_id]['cutoff'],
                'asset': logistic_by_id[origin_id]['asset'],
                'failure_label': logistic_by_id[origin_id]['actual'],
                'kronos_absolute_error': ridge_by_id[origin_id]['actual'],
                'predicted_failure_probability': logistic_by_id[origin_id]['prediction'],
                'predicted_absolute_error': ridge_by_id[origin_id]['prediction'],
                'selected_family': selected_family,
            }
            for origin_id in ordered_ids
        )
    return RiskModelOOFResult(
        schema_version='sentinel-phase3a-oof-v1',
        folds=folds,
        family_results=family_results,
        selected_family=selected_family,
        selection_rule='ONE_STANDARD_ERROR_THEN_FEWEST_DECLARED_FEATURES',
        oof_predictions=oof_predictions,
    )


def _fit_logistic_candidate(
    c_value: float,
    matrices: tuple[_FoldMatrices, ...],
) -> CandidateResult:
    metrics: list[float] = []
    predictions: list[dict[str, Any]] = []
    for fold in matrices:
        labels = np.asarray(
            [bool(row['failure_label']) for row in fold.train_rows],
            dtype=int,
        )
        if fold.train_x.shape[1] == 0:
            return _ineligible('logistic', 'C', c_value, 'NO_ELIGIBLE_FEATURES')
        if np.unique(labels).size < 2:
            return _ineligible('logistic', 'C', c_value, 'ONE_CLASS_TRAINING_FOLD')
        model = LogisticRegression(
            C=c_value,
            penalty='l2',
            solver='lbfgs',
            max_iter=10_000,
            random_state=314159,
        )
        model.fit(fold.train_x, labels)
        probabilities = model.predict_proba(fold.validation_x)[:, 1]
        actual = np.asarray(
            [bool(row['failure_label']) for row in fold.validation_rows],
            dtype=int,
        )
        metrics.append(float(log_loss(actual, probabilities, labels=(0, 1))))
        predictions.extend(
            _prediction_payload(row, float(prediction), bool(label))
            for row, prediction, label in zip(fold.validation_rows, probabilities, actual)
        )
    return _eligible('logistic', 'C', c_value, metrics, predictions)


def _fit_ridge_candidate(
    alpha: float,
    matrices: tuple[_FoldMatrices, ...],
) -> CandidateResult:
    metrics: list[float] = []
    predictions: list[dict[str, Any]] = []
    for fold in matrices:
        if fold.train_x.shape[1] == 0:
            return _ineligible('ridge', 'alpha', alpha, 'NO_ELIGIBLE_FEATURES')
        targets = np.asarray(
            [float(row['kronos_absolute_error']) for row in fold.train_rows],
            dtype=float,
        )
        model = Ridge(alpha=alpha)
        model.fit(fold.train_x, targets)
        predicted = np.maximum(model.predict(fold.validation_x), 0.0)
        actual = np.asarray(
            [float(row['kronos_absolute_error']) for row in fold.validation_rows],
            dtype=float,
        )
        metrics.append(float(mean_absolute_error(actual, predicted)))
        predictions.extend(
            _prediction_payload(row, float(prediction), float(label))
            for row, prediction, label in zip(fold.validation_rows, predicted, actual)
        )
    return _eligible('ridge', 'alpha', alpha, metrics, predictions)


def _prediction_payload(
    row: Mapping[str, Any],
    prediction: float,
    actual: bool | float,
) -> dict[str, Any]:
    return {
        'origin_id': str(row['origin_id']),
        'cutoff': str(row['cutoff']),
        'asset': str(row['asset']),
        'prediction': prediction,
        'actual': actual,
    }


def _eligible(
    kind: str,
    parameter_name: str,
    parameter: float,
    metrics: list[float],
    predictions: list[dict[str, Any]],
) -> CandidateResult:
    mean = float(np.mean(metrics))
    standard_error = (
        float(np.std(metrics, ddof=1) / math.sqrt(len(metrics)))
        if len(metrics) > 1
        else 0.0
    )
    return CandidateResult(
        model_kind=kind,
        parameter_name=parameter_name,
        parameter=parameter,
        eligible=True,
        ineligible_reason=None,
        fold_metrics=tuple(metrics),
        mean_metric=mean,
        standard_error=standard_error,
        predictions=tuple(predictions),
    )


def _ineligible(
    kind: str,
    parameter_name: str,
    parameter: float,
    reason: str,
) -> CandidateResult:
    return CandidateResult(
        model_kind=kind,
        parameter_name=parameter_name,
        parameter=parameter,
        eligible=False,
        ineligible_reason=reason,
        fold_metrics=(),
        mean_metric=None,
        standard_error=None,
        predictions=(),
    )


def _select_candidate(candidates: tuple[CandidateResult, ...]) -> CandidateResult | None:
    eligible = tuple(item for item in candidates if item.eligible and item.mean_metric is not None)
    return min(eligible, key=lambda item: (_candidate_metric(item), item.parameter)) if eligible else None


def _select_family(results: Mapping[str, FeatureFamilyResult]) -> str | None:
    eligible = tuple(
        item
        for item in results.values()
        if item.selected_logistic_mean_metric is not None
        and item.selected_logistic_standard_error is not None
        and item.selected_ridge_parameter is not None
    )
    if not eligible:
        return None
    best = min(eligible, key=_family_metric)
    threshold = _family_metric(best) + _family_standard_error(best)
    within_one_se = tuple(
        item
        for item in eligible
        if _family_metric(item) <= threshold
    )
    family_rank = {'structural': 0, 'nonstructural': 1, 'combined': 2}
    selected = min(
        within_one_se,
        key=lambda item: (len(item.declared_features), family_rank[item.family]),
    )
    return selected.family


def fit_final_risk_models(
    rows: Sequence[Mapping[str, Any]],
    oof: RiskModelOOFResult,
) -> dict[str, Any]:
    if oof.selected_family is None:
        raise ValueError('no eligible OOF feature family can be frozen')
    family = oof.family_results[oof.selected_family]
    if (
        family.selected_logistic_parameter is None
        or family.selected_ridge_parameter is None
    ):
        raise ValueError('selected OOF family is missing model parameters')
    logistic_parameter = family.selected_logistic_parameter
    ridge_parameter = family.selected_ridge_parameter
    completed = tuple(
        sorted(
            (row for row in rows if row.get('terminal_status') == 'completed'),
            key=lambda row: (str(row['cutoff']), 0 if row['asset'] == 'SPY' else 1),
        )
    )
    cutoffs = tuple(sorted({str(row['cutoff']) for row in completed}))
    fold = ChronologicalFold(
        fold_index=3,
        training_cutoffs=cutoffs,
        validation_cutoffs=(),
        training_origin_ids=tuple(str(row['origin_id']) for row in completed),
        validation_origin_ids=(),
    )
    matrices = _fit_preprocessing(
        fold=fold,
        declared_features=family.declared_features,
        train_rows=completed,
        validation_rows=completed,
    )
    labels = np.asarray([bool(row['failure_label']) for row in completed], dtype=int)
    if np.unique(labels).size < 2 or matrices.train_x.shape[1] == 0:
        raise ValueError('final logistic risk model is not identifiable')
    logistic = LogisticRegression(
        C=logistic_parameter,
        penalty='l2',
        solver='lbfgs',
        max_iter=10_000,
        random_state=314159,
    )
    logistic.fit(matrices.train_x, labels)
    targets = np.asarray(
        [float(row['kronos_absolute_error']) for row in completed],
        dtype=float,
    )
    ridge = Ridge(alpha=ridge_parameter)
    ridge.fit(matrices.train_x, targets)
    failure_probabilities = logistic.predict_proba(matrices.train_x)[:, 1]
    predicted_errors = ridge.predict(matrices.train_x)
    return {
        'preprocessing': matrices.state.model_dump(mode='json'),
        'logistic_model': {
            'kind': 'L2_LOGISTIC_REGRESSION',
            'C': logistic_parameter,
            'solver': 'lbfgs',
            'max_iter': 10_000,
            'intercept': float(np.asarray(logistic.intercept_).ravel()[0]),
            'coefficients': [
                float(value) for value in np.asarray(logistic.coef_)[0]
            ],
        },
        'ridge_model': {
            'kind': 'RIDGE_REGRESSION',
            'alpha': ridge_parameter,
            'intercept': float(np.asarray(ridge.intercept_).ravel()[0]),
            'coefficients': [float(value) for value in np.asarray(ridge.coef_)],
        },
        'development_predictions': tuple(
            {
                'origin_id': str(row['origin_id']),
                'asset': str(row['asset']),
                'cutoff': str(row['cutoff']),
                'predicted_failure_probability': float(failure_probability),
                'predicted_absolute_error': float(predicted_error),
            }
            for row, failure_probability, predicted_error in zip(
                completed,
                failure_probabilities,
                predicted_errors,
                strict=True,
            )
        ),
    }


def _candidate_metric(candidate: CandidateResult) -> float:
    if candidate.mean_metric is None:
        raise ValueError('eligible candidate is missing its metric')
    return candidate.mean_metric


def _family_metric(family: FeatureFamilyResult) -> float:
    if family.selected_logistic_mean_metric is None:
        raise ValueError('eligible feature family is missing its metric')
    return family.selected_logistic_mean_metric


def _family_standard_error(family: FeatureFamilyResult) -> float:
    if family.selected_logistic_standard_error is None:
        raise ValueError('eligible feature family is missing its standard error')
    return family.selected_logistic_standard_error
