from __future__ import annotations

from collections.abc import Mapping
from itertools import pairwise

from .development_serialization import canonical_json_bytes, sha256_bytes

_POSITIVE_EXPECTED = {
    'INVALID_PATH_FRACTION',
    'INVALID_CANDLE_FRACTION',
    'TOTAL_CONSTRAINT_VIOLATIONS',
    'MAX_CONSTRAINT_VIOLATION_SEVERITY',
    'MEAN_CONSTRAINT_VIOLATION_SEVERITY',
    'HIGH_LOW_INVERSION_COUNT',
    'HIGH_BELOW_BODY_COUNT',
    'LOW_ABOVE_BODY_COUNT',
    'RETURN_DISPERSION',
    'PATH_DISPERSION',
    'CONTEXT_RETURN_SPREAD',
    'BASELINE_DISAGREEMENT',
    'RECENT_VOLATILITY',
    'VOLATILITY_CHANGE',
    'GAP_OR_OUTLIER_SCORE',
    'HISTORICAL_ANALOGUE_DISTANCE',
    'ANALOGUE_OUTCOME_DISPERSION',
    'RECENT_MODEL_ERROR',
    'HORIZON_PATH_DIVERGENCE',
}
_NEGATIVE_EXPECTED = {
    'EARLIEST_INVALID_HORIZON_STEP',
    'DIRECTIONAL_AGREEMENT',
    'CONTEXT_DIRECTION_AGREEMENT',
}


def recommend(evidence: Mapping[str, object]) -> str:
    operational = _mapping(evidence, 'operational_feasibility')
    prevalence = _mapping(evidence, 'structural_prevalence')
    coverage = _mapping(evidence, 'risk_coverage')
    diagnostics = _mapping(evidence, 'diagnostic_analysis')

    s6 = (
        _number(operational['request_success_rate']) >= 0.95
        and _number(operational['median_ensemble_latency_seconds']) <= 600.0
        and _number(operational['cache_bytes']) <= 2_147_483_648
        and _number(operational['hosted_equivalent_cost_usd_per_cutoff']) <= 0.50
    )
    pooled = _coverage_by_level(coverage, 'pooled')
    full_mae = _number(pooled[1.0]['kronos_mae'])
    mae_70 = _number(pooled[0.7]['kronos_mae'])
    rho = _number(coverage['risk_error_spearman'])
    quintiles = tuple(
        _as_mapping(item) for item in _sequence(coverage['error_by_risk_quintile'])
    )
    quintile_errors = [_number(item['mean_error']) for item in quintiles]

    r1 = rho >= 0.20
    r2 = mae_70 <= 0.90 * full_mae
    r3 = (
        quintile_errors[-1] > quintile_errors[0]
        and sum(
            right >= left
            for left, right in pairwise(quintile_errors)
        )
        >= 3
    )
    asset_rho = _mapping(evidence, 'asset_risk_error_spearman')
    by_asset = _mapping(coverage, 'by_asset')
    r4 = all(
        _number(asset_rho[asset]) > 0
        and _number(_coverage_by_level(by_asset, asset)[0.7]['kronos_mae'])
        <= _number(_coverage_by_level(by_asset, asset)[1.0]['kronos_mae'])
        for asset in ('SPY', 'QQQ')
    )
    r5 = _supported_diagnostic(diagnostics)
    material_structural = (
        _number(prevalence['invalid_path_prevalence']) >= 0.05
        or _number(prevalence['invalid_canonical_forecast_prevalence']) >= 0.05
    )

    if s6 and all((r1, r2, r3, r4, r5)) and bool(evidence.get('freeze_verified')):
        return 'PROCEED_TO_LOCKED_HOLDOUT'
    if s6 and material_structural and rho <= 0 and mae_70 >= full_mae:
        return 'PROCEED_AS_STRUCTURAL_CONTRACT_ONLY'
    if s6 and (rho > 0 or mae_70 < full_mae):
        return 'CHANGE_THE_RELIABILITY_APPROACH'
    return 'STOP'


def render_development_report(
    *,
    artifacts: Mapping[str, Mapping[str, object]],
    freeze_sha256: str,
) -> str:
    if len(freeze_sha256) != 64:
        raise ValueError('freeze hash must be SHA-256')
    evidence = _verified_payloads(artifacts)
    decision = recommend(evidence)
    prevalence = _mapping(evidence, 'structural_prevalence')
    coverage = _mapping(evidence, 'risk_coverage')
    diagnostics = _mapping(evidence, 'diagnostic_analysis')
    comparison = _mapping(evidence, 'model_comparison')
    interventions = _mapping(evidence, 'intervention_analysis')
    origins = _mapping(evidence, 'origin_summary')
    structural_error = _mapping(evidence, 'structural_error')
    unavailable = '; '.join(
        f'{name}: {raw.get("reason", "UNSPECIFIED")}'
        for name, raw in sorted(diagnostics.items())
        if isinstance(raw, Mapping) and raw.get('status') != 'available'
    ) or 'None'
    pooled = _coverage_by_level(coverage, 'pooled')
    horizon_steps = _mapping(prevalence, 'by_horizon_step')
    horizon_counts = [
        int(_number(horizon_steps.get(str(step), 0))) for step in range(1, 6)
    ]
    monotonic_horizon = all(
        right >= left for left, right in pairwise(horizon_counts)
    )
    p2 = _mapping(interventions, 'p2')
    p3 = _mapping(interventions, 'p3')
    family_results = _mapping(comparison, 'family_results')
    family_metrics = {
        name: _number(_as_mapping(raw)['selected_logistic_mean_metric'])
        for name, raw in family_results.items()
    }
    lowest_loss_family = min(family_metrics, key=family_metrics.__getitem__)
    invalid_error = _number(structural_error['canonical_invalid_error_mean'])
    valid_error = _number(structural_error['canonical_valid_error_mean'])
    individual_invalid_error = _number(
        structural_error['individual_invalid_error_mean']
    )
    individual_valid_error = _number(
        structural_error['individual_valid_error_mean']
    )
    coverage_summary = '; '.join(
        (
            f'{int(level * 100)}%: n={pooled[level].get("sample_count", "NA")}, '
            f'MAE={pooled[level]["kronos_mae"]}'
        )
        for level in (1.0, 0.9, 0.8, 0.7, 0.5)
    )

    lines = [
        'DEVELOPMENT ANALYSIS - NOT HOLDOUT EVIDENCE',
        '',
        f'Recommendation: {decision}',
        f'Freeze SHA-256: {freeze_sha256}',
        f'Eligible origins: {origins["eligible_origins"]}',
        f'Completed origins: {origins["completed_origins"]}',
        f'Failed origins: {origins["failed_origins"]}',
        f'Generated paths: {origins["generated_paths"]}',
        '',
        '## Q1. How often did official Kronos paths violate OHLC constraints?',
        (
            f'Invalid-path prevalence: {prevalence["invalid_path_prevalence"]}. '
            f'Invalid-candle prevalence: {prevalence["invalid_candle_prevalence"]}. '
            'Invalid canonical-forecast prevalence: '
            f'{prevalence["invalid_canonical_forecast_prevalence"]}.'
        ),
        '',
        '## Q2. Which violations were most common?',
        f'Violation counts: {prevalence["violation_types"]}.',
        '',
        '## Q3. Did violations become more common later in the forecast horizon?',
        f'Violation counts by step 1-5: {horizon_counts}.',
        (
            'Monotonic increase across horizon steps: '
            f'{"yes" if monotonic_horizon else "no"}.'
        ),
        '',
        '## Q4. Did structural violations predict larger return errors?',
        (
            f'Canonical-invalid mean absolute error: {invalid_error}; '
            f'canonical-valid mean absolute error: {valid_error}. '
            f'Invalid was worse: {"yes" if invalid_error > valid_error else "no"}.'
        ),
        (
            f'Individual-invalid mean absolute error: {individual_invalid_error}; '
            f'individual-valid mean absolute error: {individual_valid_error}.'
        ),
        'Controlled structural coefficients are retained in diagnostic_analysis.json.',
        '',
        '## Q5. Did instability diagnostics predict error?',
        (
            f'Pooled chronological OOF risk/error Spearman: '
            f'{coverage["risk_error_spearman"]}. Diagnostic analysis includes '
            f'{len(diagnostics)} declared diagnostics.'
        ),
        '',
        '## Q6. Which feature family performed best out of sample within development?',
        (
            f'Lowest mean logistic log loss family: {lowest_loss_family} '
            f'({family_metrics[lowest_loss_family]}).'
        ),
        f'Selected chronological OOF family: {comparison["selected_family"]}.',
        '',
        '## Q7. Did abstention reduce accepted-forecast error?',
        coverage_summary,
        '',
        '## Q8. Did blending help?',
        (
            f'P2 accepted MAE: {p2.get("accepted_mae")}; '
            f'accepted count: {p2.get("accepted_count", "NA")}; '
            f'fixed blend weight: {p2.get("blend_weight")}.'
        ),
        '',
        '## Q9. Did valid-path aggregation help?',
        (
            f'P3 retained for holdout: {p3.get("retain_for_holdout")}; '
            f'application count: {p3.get("application_count", "NA")}; '
            f'pooled mean improvement: '
            f'{p3.get("pooled_mean_improvement", "NA")}.'
        ),
        '',
        '## Q10. Were effects consistent across SPY and QQQ?',
        f'Per-asset risk/error Spearman: {evidence["asset_risk_error_spearman"]}.',
        '',
        '## Q11. Which diagnostics failed?',
        unavailable,
        '',
        '## Q12. What exactly has been frozen for holdout?',
        (
            f'Verified freeze {freeze_sha256}; selected family '
            f'{comparison["selected_family"]}.'
        ),
        '',
        '## Q13. Is executing the holdout justified?',
        f'{decision}. No holdout origin was accessed by this report.',
        '',
        '## Artifact provenance',
    ]
    lines.extend(
        f'- {name}: {artifact["sha256"]}'
        for name, artifact in sorted(artifacts.items())
    )
    return '\n'.join(lines) + '\n'


def _verified_payloads(
    artifacts: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    payloads: dict[str, object] = {}
    for name, artifact in artifacts.items():
        if set(artifact) != {'payload', 'sha256'}:
            raise ValueError(f'artifact {name} has an invalid envelope')
        payload = artifact['payload']
        expected = str(artifact['sha256'])
        if sha256_bytes(canonical_json_bytes(payload)) != expected:
            raise ValueError(f'artifact {name} hash mismatch')
        payloads[name] = payload
    return payloads


def _supported_diagnostic(diagnostics: Mapping[str, object]) -> bool:
    for name, raw in diagnostics.items():
        if not isinstance(raw, Mapping) or raw.get('status') != 'available':
            continue
        value = raw.get('future_absolute_error_spearman')
        if value is None:
            continue
        rho = _number(value)
        if name in _POSITIVE_EXPECTED and rho >= 0.15:
            return True
        if name in _NEGATIVE_EXPECTED and rho <= -0.15:
            return True
    return False


def _coverage_by_level(
    parent: Mapping[str, object],
    key: str,
) -> dict[float, Mapping[str, object]]:
    raw = parent[key]
    if not isinstance(raw, (list, tuple)):
        raise TypeError(f'{key} coverage must be a sequence')
    return {_number(item['coverage']): item for item in map(_as_mapping, raw)}


def _mapping(parent: Mapping[str, object], key: str) -> Mapping[str, object]:
    return _as_mapping(parent[key])


def _as_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError('expected mapping')
    return value


def _sequence(value: object) -> list[object] | tuple[object, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError('expected sequence')
    return value


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError('expected numeric value')
    return float(value)
