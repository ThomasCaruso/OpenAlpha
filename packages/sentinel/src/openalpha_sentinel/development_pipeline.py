from __future__ import annotations

import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scipy.stats import spearmanr

from .development_analysis import (
    analyze_diagnostics,
    analyze_structural_error,
    build_structural_prevalence,
)
from .development_freeze import (
    FreezeCandidate,
    build_freeze_candidate,
    verify_freeze_candidate,
)
from .development_manifest import DevelopmentManifest, build_development_manifest
from .development_reporting import render_development_report
from .development_runner import (
    OriginExecutionSummary,
    PilotGate,
    verify_offline_state,
)
from .development_serialization import (
    atomic_write_bytes,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    sha256_bytes,
)
from .development_table import DevelopmentTable, build_development_table
from .interventions import build_risk_coverage, evaluate_policies
from .risk_model import RiskModelOOFResult, fit_chronological_oof

_PUBLIC_OUTPUTS = (
    'development_table.jsonl',
    'development_manifest.json',
    'structural_prevalence.json',
    'diagnostic_analysis.json',
    'out_of_fold_predictions.jsonl',
    'model_comparison.json',
    'intervention_analysis.json',
    'risk_coverage.json',
)
_FORBIDDEN_KEYS = {
    'observations',
    'outcome_ohlcv',
    'raw_provider_response',
    'provider_response_payload',
    'yahoo_response',
}
_CREDENTIAL_TOKENS = ('credential', 'api_key', 'api_secret', 'access_token')
_CACHE_PATH_KEYS = ('cache_path', 'model_cache_path', 'hf_cache')


def analyze_development_state(state_root: Path | str) -> dict[str, object]:
    root = Path(state_root)
    verify_offline_state(root)
    manifest = build_development_manifest()
    terminal, markers = _load_terminal_population(root, manifest)
    table = build_development_table(manifest.origins, terminal)
    prevalence = build_structural_prevalence(table.rows)
    diagnostics = analyze_diagnostics(table.rows)
    structural_error = analyze_structural_error(table.rows)
    oof = fit_chronological_oof(table.rows)
    if oof.selected_family is None or not oof.oof_predictions:
        raise ValueError('no eligible chronological OOF reliability system')
    risk_coverage = build_risk_coverage(table.rows, oof.oof_predictions)
    interventions = evaluate_policies(table.rows, oof.oof_predictions)
    comparison = _compact_model_comparison(oof)
    asset_rho = _asset_risk_spearman(table.completed_rows, oof.oof_predictions)
    operational = _operational_summary(root, markers)
    origin_summary = {
        'eligible_origins': len(manifest.origins),
        'completed_origins': table.completed_count,
        'failed_origins': table.failed_count,
        'generated_paths': sum(
            9 for row in table.completed_rows if row.get('terminal_status') == 'completed'
        ),
    }

    output_root = root / 'repository_outputs'
    output_root.mkdir(parents=True, exist_ok=True)
    payloads: dict[str, bytes] = {
        'development_table.jsonl': table.jsonl_bytes,
        'development_manifest.json': canonical_json_bytes(
            manifest.model_dump(mode='json')
        ),
        'structural_prevalence.json': canonical_json_bytes(prevalence),
        'diagnostic_analysis.json': canonical_json_bytes(diagnostics),
        'out_of_fold_predictions.jsonl': canonical_jsonl_bytes(oof.oof_predictions),
        'model_comparison.json': canonical_json_bytes(comparison),
        'intervention_analysis.json': canonical_json_bytes(interventions),
        'risk_coverage.json': canonical_json_bytes(risk_coverage),
    }
    for name, payload in payloads.items():
        parsed: object
        if name.endswith('.jsonl'):
            parsed = tuple(_loads_finite(line) for line in payload.splitlines())
        else:
            parsed = _loads_finite(payload)
        scan_repository_policy(parsed)
        _write_immutable(output_root / name, payload)
    output_hashes = {name: sha256_bytes(payload) for name, payload in payloads.items()}
    body = {
        'schema_version': 'sentinel-phase3a-analysis-receipt-v1',
        'manifest_sha256': manifest.canonical_sha256,
        'table_content_sha256': table.content_sha256,
        'origin_count': len(manifest.origins),
        'completed_count': table.completed_count,
        'failed_count': table.failed_count,
        'output_hashes': output_hashes,
        'analysis_evidence': {
            'operational_feasibility': operational,
            'freeze_verified': False,
            'failed_origin_count': table.failed_count,
            'asset_risk_error_spearman': asset_rho,
            'structural_error': structural_error,
            'origin_summary': origin_summary,
        },
    }
    receipt = {**body, 'receipt_sha256': sha256_bytes(canonical_json_bytes(body))}
    scan_repository_policy(receipt)
    _write_immutable(root / 'analysis-receipt.json', canonical_json_bytes(receipt))
    return receipt


def freeze_development_state(state_root: Path | str) -> dict[str, object]:
    root = Path(state_root)
    analysis = _load_analysis_receipt(root)
    table = _load_table(root)
    oof = fit_chronological_oof(table.rows)
    interventions = _load_public_json(root, 'intervention_analysis.json', analysis)
    if not isinstance(interventions, Mapping):
        raise TypeError('intervention analysis must be a mapping')
    p3 = _require_mapping(interventions, 'p3')
    freeze = build_freeze_candidate(
        rows=table.rows,
        oof=oof,
        failure_threshold=table.failure_threshold,
        artifact_hashes={
            key.removesuffix('.json').removesuffix('.jsonl'): str(value)
            for key, value in _require_mapping(analysis, 'output_hashes').items()
        },
        p3_retained=bool(p3['retain_for_holdout']),
    )
    payload = canonical_json_bytes(freeze.model_dump(mode='json'))
    scan_repository_policy(_loads_finite(payload))
    _write_immutable(root / 'repository_outputs' / 'freeze_candidate.json', payload)
    body = {
        'schema_version': 'sentinel-phase3a-freeze-receipt-v1',
        'analysis_receipt_sha256': analysis['receipt_sha256'],
        'freeze_sha256': freeze.freeze_sha256,
        'public_payload_sha256': sha256_bytes(payload),
        'freeze_verified': verify_freeze_candidate(freeze),
    }
    receipt = {**body, 'receipt_sha256': sha256_bytes(canonical_json_bytes(body))}
    _write_immutable(root / 'freeze-receipt.json', canonical_json_bytes(receipt))
    return receipt


def report_development_state(state_root: Path | str) -> dict[str, object]:
    root = Path(state_root)
    analysis = _load_analysis_receipt(root)
    freeze_receipt, freeze = _load_freeze(root)
    evidence = dict(_require_mapping(analysis, 'analysis_evidence'))
    for key, filename in (
        ('structural_prevalence', 'structural_prevalence.json'),
        ('risk_coverage', 'risk_coverage.json'),
        ('diagnostic_analysis', 'diagnostic_analysis.json'),
        ('model_comparison', 'model_comparison.json'),
        ('intervention_analysis', 'intervention_analysis.json'),
    ):
        evidence[key] = _load_public_json(root, filename, analysis)
    evidence['freeze_verified'] = True
    artifacts = {
        name: {
            'payload': payload,
            'sha256': sha256_bytes(canonical_json_bytes(payload)),
        }
        for name, payload in evidence.items()
    }
    report = render_development_report(
        artifacts=artifacts,
        freeze_sha256=freeze.freeze_sha256,
    )
    payload = report.encode('utf-8')
    scan_repository_policy(report)
    _write_immutable(root / 'repository_outputs' / 'report.md', payload)
    body = {
        'schema_version': 'sentinel-phase3a-report-receipt-v1',
        'analysis_receipt_sha256': analysis['receipt_sha256'],
        'freeze_receipt_sha256': freeze_receipt['receipt_sha256'],
        'report_sha256': sha256_bytes(payload),
        'report_verified': True,
    }
    receipt = {**body, 'receipt_sha256': sha256_bytes(canonical_json_bytes(body))}
    _write_immutable(root / 'report-receipt.json', canonical_json_bytes(receipt))
    return receipt


def verify_development_state(state_root: Path | str) -> dict[str, object]:
    root = Path(state_root)
    verify_offline_state(root)
    manifest = build_development_manifest()
    terminal, _ = _load_terminal_population(root, manifest)
    analysis = _load_analysis_receipt(root)
    rebuilt = build_development_table(manifest.origins, terminal)
    table_path = root / 'repository_outputs' / 'development_table.jsonl'
    if table_path.read_bytes() != rebuilt.jsonl_bytes:
        raise ValueError('deterministic development table hash mismatch')
    freeze_receipt, freeze = _load_freeze(root)
    report_receipt = _load_self_hashed(root / 'report-receipt.json')
    report_path = root / 'repository_outputs' / 'report.md'
    if sha256_bytes(report_path.read_bytes()) != report_receipt.get('report_sha256'):
        raise ValueError('report hash mismatch')
    if report_receipt.get('freeze_receipt_sha256') != freeze_receipt['receipt_sha256']:
        raise ValueError('report provenance does not reference the verified freeze')
    for name in (*_PUBLIC_OUTPUTS, 'freeze_candidate.json'):
        path = root / 'repository_outputs' / name
        if name in _PUBLIC_OUTPUTS:
            expected = _require_mapping(analysis, 'output_hashes')[name]
        else:
            expected = freeze_receipt['public_payload_sha256']
        if sha256_bytes(path.read_bytes()) != expected:
            raise ValueError(f'{name} hash mismatch')
        parsed = (
            tuple(_loads_finite(line) for line in path.read_bytes().splitlines())
            if name.endswith('.jsonl')
            else _loads_finite(path.read_bytes())
        )
        scan_repository_policy(parsed)
    scan_repository_policy(report_path.read_text(encoding='utf-8'))
    return {
        'manifest_verified': True,
        'terminal_origin_count': len(terminal),
        'completed_origin_count': rebuilt.completed_count,
        'failed_origin_count': rebuilt.failed_count,
        'analysis_receipt_sha256': analysis['receipt_sha256'],
        'freeze_sha256': freeze.freeze_sha256,
        'report_sha256': report_receipt['report_sha256'],
        'network_accessed': False,
        'inference_accessed': False,
    }


def scan_repository_policy(value: object, *, key: str | None = None) -> None:
    lowered = key.lower() if key is not None else ''
    if lowered in _FORBIDDEN_KEYS:
        raise ValueError(f'forbidden repository field: {key}')
    if any(token in lowered for token in _CREDENTIAL_TOKENS):
        raise ValueError(f'credential material is prohibited: {key}')
    if any(token in lowered for token in _CACHE_PATH_KEYS):
        raise ValueError(f'repository cache paths are prohibited: {key}')
    if isinstance(value, Mapping):
        for child_key, child in value.items():
            scan_repository_policy(child, key=str(child_key))
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            scan_repository_policy(child, key=key)
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('non-finite values are prohibited')
    if key == 'origin_id' and isinstance(value, str):
        allowed = {item.origin_id for item in build_development_manifest().origins}
        if value not in allowed:
            raise ValueError('origin is outside the locked development manifest')


def _load_terminal_population(
    root: Path,
    manifest: DevelopmentManifest,
) -> tuple[dict[str, Mapping[str, object]], tuple[OriginExecutionSummary, ...]]:
    records: dict[str, Mapping[str, object]] = {}
    markers = []
    for origin in manifest.origins:
        origin_root = root / 'origins' / origin.origin_id
        marker_path = origin_root / 'runner-terminal.json'
        if not marker_path.is_file() or marker_path.is_symlink():
            raise ValueError(f'missing terminal marker for {origin.origin_id}')
        marker = OriginExecutionSummary.model_validate_json(marker_path.read_bytes())
        if marker.origin_id != origin.origin_id:
            raise ValueError('terminal marker identity mismatch')
        markers.append(marker)
        if marker.terminal_status == 'completed':
            path = origin_root / 'analysis-row.json'
            chain_path = origin_root / 'resolved.json'
            if not chain_path.is_file():
                raise ValueError('completed origin is missing resolved chain descriptor')
            chain = _loads_finite(chain_path.read_bytes())
            if not isinstance(chain, Mapping) or not bool(chain.get('chain_verified')):
                raise ValueError('completed origin chain is not verified')
        else:
            path = origin_root / 'terminal-failure.json'
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'missing terminal descriptor for {origin.origin_id}')
        record = _loads_finite(path.read_bytes())
        if not isinstance(record, Mapping):
            raise TypeError('terminal descriptor must be a mapping')
        observed_hash = (
            sha256_bytes(canonical_json_bytes(record))
            if marker.terminal_status == 'completed'
            else record.get('record_sha256')
        )
        if marker.terminal_record_sha256 != observed_hash:
            raise ValueError('terminal descriptor hash mismatch')
        records[origin.origin_id] = record
    return records, tuple(markers)


def _operational_summary(
    root: Path,
    markers: Sequence[OriginExecutionSummary],
) -> dict[str, object]:
    pilot = PilotGate.model_validate_json((root / 'pilot-gate.json').read_bytes())
    requests = sum(item.request_count for item in markers)
    return {
        'request_success_rate': (
            sum(item.request_success_count for item in markers) / requests
        ),
        'median_ensemble_latency_seconds': float(
            statistics.median(item.ensemble_latency_seconds for item in markers)
        ),
        'cache_bytes': max(item.cache_bytes for item in markers),
        'hosted_equivalent_cost_usd_per_cutoff': pilot.hosted_cost_per_cutoff,
        'deterministic_replay_supported': pilot.deterministic_replay_supported,
        'process_stable': all(item.process_stable for item in markers),
    }


def _asset_risk_spearman(
    rows: Sequence[Mapping[str, object]],
    predictions: Sequence[Mapping[str, object]],
) -> dict[str, float | None]:
    by_id = {str(row['origin_id']): row for row in rows}
    result: dict[str, float | None] = {}
    for asset in ('SPY', 'QQQ'):
        pairs = [
            (
                _number(item['predicted_failure_probability']),
                _number(by_id[str(item['origin_id'])]['kronos_absolute_error']),
            )
            for item in predictions
            if str(item['asset']) == asset
        ]
        if len(pairs) < 3:
            result[asset] = None
        else:
            correlation: Any = spearmanr(
                [item[0] for item in pairs],
                [item[1] for item in pairs],
            )
            result[asset] = float(correlation.statistic)
    return result


def _compact_model_comparison(oof: RiskModelOOFResult) -> dict[str, object]:
    payload = oof.model_dump(mode='json')
    payload.pop('oof_predictions', None)
    families = payload.get('family_results')
    if not isinstance(families, dict):
        raise TypeError('OOF family results are invalid')
    for family in families.values():
        if not isinstance(family, dict):
            raise TypeError('OOF family descriptor is invalid')
        for collection in ('logistic_candidates', 'ridge_candidates'):
            candidates = family.get(collection)
            if not isinstance(candidates, list):
                raise TypeError('OOF candidate collection is invalid')
            for candidate in candidates:
                if isinstance(candidate, dict):
                    candidate.pop('predictions', None)
    return payload


def _load_analysis_receipt(root: Path) -> Mapping[str, object]:
    receipt = _load_self_hashed(root / 'analysis-receipt.json')
    output_hashes = _require_mapping(receipt, 'output_hashes')
    if set(output_hashes) != set(_PUBLIC_OUTPUTS):
        raise ValueError('analysis output inventory is incomplete')
    for name, expected in output_hashes.items():
        path = root / 'repository_outputs' / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'missing public analysis artifact: {name}')
        if sha256_bytes(path.read_bytes()) != expected:
            raise ValueError(f'{name} hash mismatch')
    return receipt


def _load_freeze(
    root: Path,
) -> tuple[Mapping[str, object], FreezeCandidate]:
    receipt = _load_self_hashed(root / 'freeze-receipt.json')
    path = root / 'repository_outputs' / 'freeze_candidate.json'
    if sha256_bytes(path.read_bytes()) != receipt.get('public_payload_sha256'):
        raise ValueError('freeze candidate public payload hash mismatch')
    freeze = FreezeCandidate.model_validate_json(path.read_bytes())
    if freeze.freeze_sha256 != receipt.get('freeze_sha256'):
        raise ValueError('freeze receipt hash mismatch')
    if not verify_freeze_candidate(freeze) or not bool(receipt.get('freeze_verified')):
        raise ValueError('freeze candidate failed verification')
    return receipt, freeze


def _load_table(root: Path) -> DevelopmentTable:
    manifest = build_development_manifest()
    terminal, _ = _load_terminal_population(root, manifest)
    return build_development_table(manifest.origins, terminal)


def _load_public_json(
    root: Path,
    name: str,
    analysis: Mapping[str, object],
) -> object:
    path = root / 'repository_outputs' / name
    expected = _require_mapping(analysis, 'output_hashes').get(name)
    if sha256_bytes(path.read_bytes()) != expected:
        raise ValueError(f'{name} hash mismatch')
    return _loads_finite(path.read_bytes())


def _load_self_hashed(path: Path) -> Mapping[str, object]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f'missing immutable receipt: {path.name}')
    payload = _loads_finite(path.read_bytes())
    if not isinstance(payload, Mapping):
        raise TypeError('receipt must be a mapping')
    observed = payload.get('receipt_sha256')
    body = {key: value for key, value in payload.items() if key != 'receipt_sha256'}
    if sha256_bytes(canonical_json_bytes(body)) != observed:
        raise ValueError(f'{path.name} hash mismatch')
    return payload


def _loads_finite(payload: bytes | str) -> object:
    def reject_constant(value: str) -> object:
        raise ValueError(f'non-finite JSON constant: {value}')

    return json.loads(payload, parse_constant=reject_constant)


def _require_mapping(parent: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = parent[key]
    if not isinstance(value, Mapping):
        raise TypeError(f'{key} must be a mapping')
    return value


def _write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.is_symlink() or path.read_bytes() != payload:
            raise ValueError(f'immutable Phase 3A artifact differs: {path.name}')
        return
    atomic_write_bytes(path, payload)


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError('expected numeric value')
    return float(value)
