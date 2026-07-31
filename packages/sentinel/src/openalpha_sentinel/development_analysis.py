from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from .risk_model import NONSTRUCTURAL_FEATURES, STRUCTURAL_FEATURES

ALL_DIAGNOSTICS = (*STRUCTURAL_FEATURES, *NONSTRUCTURAL_FEATURES)
_PATH_CONFIGURATION = re.compile(
    r'.*-(128|256|512)-(1729|2027|7919)(?:-.+)?$'
)


def build_structural_prevalence(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    total_paths = 0
    invalid_paths = 0
    total_candles = 0
    invalid_candles = 0
    canonical_count = 0
    invalid_canonical = 0
    violation_types: Counter[str] = Counter()
    by_step: Counter[str] = Counter()
    severities: list[float] = []
    groups: dict[str, defaultdict[str, list[int]]] = {
        'asset': defaultdict(lambda: [0, 0, 0, 0]),
        'quarter': defaultdict(lambda: [0, 0, 0, 0]),
        'context': defaultdict(lambda: [0, 0, 0, 0]),
        'seed': defaultdict(lambda: [0, 0, 0, 0]),
    }
    for row in rows:
        if row.get('terminal_status') != 'completed':
            continue
        asset = str(row['asset'])
        quarter = _quarter(str(row['cutoff']))
        paths = row.get('individual_structural_validity')
        if not isinstance(paths, list):
            raise TypeError('individual structural validity must be a list')
        canonical = row.get('canonical_structural_validity')
        if not isinstance(canonical, Mapping):
            raise TypeError('canonical structural validity must be a mapping')
        canonical_count += 1
        invalid_canonical += not bool(canonical.get('valid'))
        for path in paths:
            if not isinstance(path, Mapping):
                raise TypeError('path validity entry must be a mapping')
            path_id = str(path['path_id'])
            match = _PATH_CONFIGURATION.fullmatch(path_id)
            context = match.group(1) if match else 'UNKNOWN'
            seed = match.group(2) if match else 'UNKNOWN'
            invalid = not bool(path.get('valid'))
            path_invalid_candles = int(path.get('invalid_candle_count', 0))
            total_paths += 1
            invalid_paths += invalid
            total_candles += 5
            invalid_candles += path_invalid_candles
            for group_name, key in (
                ('asset', asset),
                ('quarter', quarter),
                ('context', context),
                ('seed', seed),
            ):
                counts = groups[group_name][key]
                counts[0] += 1
                counts[1] += int(invalid)
                counts[2] += 5
                counts[3] += path_invalid_candles
            violations = path.get('violations')
            if not isinstance(violations, list):
                raise TypeError('path violations must be a list')
            for violation in violations:
                if not isinstance(violation, Mapping):
                    raise TypeError('structural violation must be a mapping')
                violation_types[str(violation['code'])] += 1
                step = int(violation.get('step', 0))
                if step > 0:
                    by_step[str(step)] += 1
                severity = violation.get('normalized_severity')
                if severity is not None:
                    numeric = float(severity)
                    if math.isfinite(numeric):
                        severities.append(numeric)
    if total_paths == 0 or total_candles == 0 or canonical_count == 0:
        raise ValueError('structural prevalence requires completed paths')
    return {
        'schema_version': 'sentinel-phase3a-structural-prevalence-v1',
        'total_path_count': total_paths,
        'invalid_path_count': invalid_paths,
        'invalid_path_prevalence': invalid_paths / total_paths,
        'total_candle_count': total_candles,
        'invalid_candle_count': invalid_candles,
        'invalid_candle_prevalence': invalid_candles / total_candles,
        'canonical_forecast_count': canonical_count,
        'invalid_canonical_forecast_count': invalid_canonical,
        'invalid_canonical_forecast_prevalence': invalid_canonical / canonical_count,
        'violation_types': dict(sorted(violation_types.items())),
        'severity_distribution': _distribution(severities),
        'by_asset': _group_summary(groups['asset']),
        'through_time': _group_summary(groups['quarter']),
        'by_context': _group_summary(groups['context']),
        'by_seed': _group_summary(groups['seed']),
        'by_horizon_step': dict(sorted(by_step.items(), key=lambda item: int(item[0]))),
    }


def moving_block_bootstrap(
    values_by_cutoff: Mapping[str, Sequence[float]],
    *,
    statistic: str,
    resamples: int = 1_000,
    block_weeks: int = 4,
    seed: int = 314159,
) -> dict[str, Any]:
    if statistic != 'mean':
        raise ValueError('unsupported public moving-block statistic')
    grouped = tuple(
        tuple(float(value) for value in values_by_cutoff[cutoff])
        for cutoff in sorted(values_by_cutoff)
    )
    if not grouped:
        raise ValueError('moving-block bootstrap requires cutoff groups')
    estimate, lower, upper = _block_bootstrap(
        grouped,
        lambda sample: float(np.mean([value for group in sample for value in group])),
        resamples=resamples,
        block_weeks=block_weeks,
        seed=seed,
    )
    return {
        'estimate': estimate,
        'lower': lower,
        'upper': upper,
        'confidence': 0.95,
        'resamples': resamples,
        'block_weeks': block_weeks,
        'seed': seed,
    }


def _block_bootstrap(
    groups: tuple[tuple, ...],
    statistic: Callable[[tuple[tuple, ...]], float],
    *,
    resamples: int,
    block_weeks: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    count = len(groups)
    estimates: list[float] = []
    for _ in range(resamples):
        selected: list[tuple] = []
        while len(selected) < count:
            start = int(rng.integers(0, count))
            selected.extend(groups[(start + offset) % count] for offset in range(block_weeks))
        value = statistic(tuple(selected[:count]))
        if math.isfinite(value):
            estimates.append(value)
    estimate = statistic(groups)
    if not estimates:
        return estimate, estimate, estimate
    lower, upper = np.quantile(estimates, (0.025, 0.975), method='linear')
    return estimate, float(lower), float(upper)


def analyze_diagnostics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    completed = tuple(row for row in rows if row.get('terminal_status') == 'completed')
    analysis: dict[str, dict[str, Any]] = {}
    for name in ALL_DIAGNOSTICS:
        selected = []
        for row in completed:
            diagnostics = row.get('diagnostics')
            if not isinstance(diagnostics, Mapping):
                raise TypeError('diagnostics must be a mapping')
            value = diagnostics.get(name)
            if value is None:
                continue
            numeric = float(value)
            if math.isfinite(numeric):
                selected.append((row, numeric))
        missing_count = len(completed) - len(selected)
        if len(selected) < 3:
            analysis[name] = {
                'status': 'unavailable',
                'reason': 'INSUFFICIENT_FINITE_OBSERVATIONS',
                'sample_count': len(selected),
                'missing_count': missing_count,
            }
            continue
        values = np.asarray([item[1] for item in selected], dtype=float)
        if np.unique(values).size <= 1:
            analysis[name] = {
                'status': 'unavailable',
                'reason': 'CONSTANT_DIAGNOSTIC',
                'sample_count': len(selected),
                'missing_count': missing_count,
            }
            continue
        errors = np.asarray(
            [float(item[0]['kronos_absolute_error']) for item in selected],
            dtype=float,
        )
        failure = np.asarray([bool(item[0]['failure_label']) for item in selected], dtype=float)
        deployability = np.asarray(
            [bool(item[0]['deployability_label']) for item in selected],
            dtype=float,
        )
        direction = np.asarray(
            [bool(item[0]['direction_correct']) for item in selected],
            dtype=float,
        )
        grouped: defaultdict[str, list[tuple[float, float]]] = defaultdict(list)
        for row, value in selected:
            grouped[str(row['cutoff'])].append(
                (value, float(row['kronos_absolute_error']))
            )
        groups = tuple(tuple(grouped[key]) for key in sorted(grouped))
        estimate, lower, upper = _block_bootstrap(
            groups,
            _spearman_group_statistic,
            resamples=1_000,
            block_weeks=4,
            seed=314159,
        )
        sign_by_asset = {
            asset: _safe_spearman(
                [value for row, value in selected if row['asset'] == asset],
                [
                    float(row['kronos_absolute_error'])
                    for row, _ in selected
                    if row['asset'] == asset
                ],
            )
            for asset in ('SPY', 'QQQ')
        }
        analysis[name] = {
            'status': 'available',
            'reason': None,
            'sample_count': len(selected),
            'missing_count': missing_count,
            'future_absolute_error_spearman': _safe_spearman(values, errors),
            'future_absolute_error_spearman_ci': {
                'estimate': estimate,
                'lower': lower,
                'upper': upper,
                'resamples': 1_000,
                'block_weeks': 4,
                'seed': 314159,
            },
            'failure_label_spearman': _safe_spearman(values, failure),
            'deployability_label_spearman': _safe_spearman(values, deployability),
            'direction_correct_spearman': _safe_spearman(values, direction),
            'sign_by_asset': sign_by_asset,
            'chronological_stability': _quarterly_spearman(selected),
        }
    return analysis


def analyze_structural_error(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    completed = tuple(row for row in rows if row.get('terminal_status') == 'completed')
    coefficients: dict[str, Any] = {}
    for feature in STRUCTURAL_FEATURES:
        design_rows: list[list[float]] = []
        targets: list[float] = []
        for row in completed:
            diagnostics = row.get('diagnostics')
            if not isinstance(diagnostics, Mapping):
                continue
            names = (
                feature,
                'RECENT_VOLATILITY',
                'BASELINE_DISAGREEMENT',
                'CONTEXT_RETURN_SPREAD',
            )
            values = [diagnostics.get(name) for name in names]
            if any(value is None for value in values):
                continue
            numeric = [float(value) for value in values if value is not None]
            predicted_magnitude = abs(float(row.get('raw_close_return_forecast', 0.0)))
            if not all(math.isfinite(value) for value in (*numeric, predicted_magnitude)):
                continue
            design_rows.append([1.0, numeric[0], numeric[1], predicted_magnitude, numeric[2], numeric[3]])
            targets.append(float(row['kronos_absolute_error']))
        if len(design_rows) < 8:
            coefficients[feature] = {
                'status': 'unavailable',
                'reason': 'INSUFFICIENT_COMPLETE_CONTROL_ROWS',
                'sample_count': len(design_rows),
            }
            continue
        design = np.asarray(design_rows, dtype=float)
        target = np.asarray(targets, dtype=float)
        fitted, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
        coefficients[feature] = {
            'status': 'available',
            'sample_count': len(design_rows),
            'design_rank': int(rank),
            'structural_coefficient': float(fitted[1]),
            'controls': (
                'RECENT_VOLATILITY',
                'ABS_PREDICTED_RETURN',
                'BASELINE_DISAGREEMENT',
                'CONTEXT_RETURN_SPREAD',
            ),
        }
    canonical_valid = [
        float(row['kronos_absolute_error'])
        for row in completed
        if isinstance(row.get('canonical_structural_validity'), Mapping)
        and bool(row['canonical_structural_validity'].get('valid'))
    ]
    canonical_invalid = [
        float(row['kronos_absolute_error'])
        for row in completed
        if isinstance(row.get('canonical_structural_validity'), Mapping)
        and not bool(row['canonical_structural_validity'].get('valid'))
    ]
    individual_valid: list[float] = []
    individual_invalid: list[float] = []
    for row in completed:
        path_results = row.get('individual_path_results')
        if not isinstance(path_results, Sequence) or isinstance(path_results, (str, bytes)):
            continue
        for raw_result in path_results:
            if not isinstance(raw_result, Mapping):
                continue
            error = raw_result.get('absolute_return_error')
            if error is None or not math.isfinite(float(error)):
                continue
            target = (
                individual_valid
                if bool(raw_result.get('structural_valid'))
                else individual_invalid
            )
            target.append(float(error))
    return {
        'schema_version': 'sentinel-phase3a-structural-error-v1',
        'analysis_role': 'DESCRIPTIVE_CONTROL_NOT_PREDICTIVE_EVIDENCE',
        'predictive_incremental_evidence_source': 'CHRONOLOGICAL_OOF_ABLATION',
        'canonical_valid_error_mean': _mean_or_none(canonical_valid),
        'canonical_invalid_error_mean': _mean_or_none(canonical_invalid),
        'canonical_valid_count': len(canonical_valid),
        'canonical_invalid_count': len(canonical_invalid),
        'individual_valid_error_mean': _mean_or_none(individual_valid),
        'individual_invalid_error_mean': _mean_or_none(individual_invalid),
        'individual_valid_count': len(individual_valid),
        'individual_invalid_count': len(individual_invalid),
        'controlled_coefficients': coefficients,
    }


def analyze_chronological_stability(
    rows: Sequence[Mapping[str, Any]],
    diagnostic: str,
) -> dict[str, Any]:
    selected = []
    for row in rows:
        diagnostics = row.get('diagnostics')
        if row.get('terminal_status') != 'completed' or not isinstance(diagnostics, Mapping):
            continue
        value = diagnostics.get(diagnostic)
        if value is not None:
            selected.append((row, float(value)))
    return {
        'diagnostic': diagnostic,
        'by_quarter': _quarterly_spearman(selected),
        'by_asset': {
            asset: _safe_spearman(
                [value for row, value in selected if row['asset'] == asset],
                [
                    float(row['kronos_absolute_error'])
                    for row, _ in selected
                    if row['asset'] == asset
                ],
            )
            for asset in ('SPY', 'QQQ')
        },
    }


def _spearman_group_statistic(groups: tuple[tuple, ...]) -> float:
    pairs = [pair for group in groups for pair in group]
    return _safe_spearman([pair[0] for pair in pairs], [pair[1] for pair in pairs]) or 0.0


def _safe_spearman(left: Any, right: Any) -> float | None:
    if len(left) < 3 or len(right) < 3:
        return None
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    if np.unique(left_values).size <= 1 or np.unique(right_values).size <= 1:
        return None
    result: Any = spearmanr(left_values, right_values)
    value = float(result.statistic)
    return value if math.isfinite(value) else None


def _quarterly_spearman(
    selected: Sequence[tuple[Mapping[str, Any], float]],
) -> dict[str, Any]:
    grouped: defaultdict[str, list[tuple[float, float]]] = defaultdict(list)
    for row, value in selected:
        grouped[_quarter(str(row['cutoff']))].append(
            (value, float(row['kronos_absolute_error']))
        )
    return {
        quarter: _safe_spearman(
            [item[0] for item in pairs],
            [item[1] for item in pairs],
        )
        for quarter, pairs in sorted(grouped.items())
    }


def _group_summary(groups: Mapping[str, list[int]]) -> dict[str, dict[str, Any]]:
    return {
        key: {
            'path_count': values[0],
            'invalid_path_count': values[1],
            'invalid_path_prevalence': values[1] / values[0],
            'candle_count': values[2],
            'invalid_candle_count': values[3],
            'invalid_candle_prevalence': values[3] / values[2],
        }
        for key, values in sorted(groups.items())
    }


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {'count': 0, 'min': None, 'median': None, 'mean': None, 'max': None}
    array = np.asarray(values, dtype=float)
    return {
        'count': len(values),
        'min': float(np.min(array)),
        'median': float(np.median(array)),
        'mean': float(np.mean(array)),
        'max': float(np.max(array)),
    }


def _quarter(cutoff: str) -> str:
    year = int(cutoff[:4])
    month = int(cutoff[5:7])
    return f'{year}-Q{(month - 1) // 3 + 1}'


def _mean_or_none(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None
