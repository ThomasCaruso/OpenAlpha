import json
from pathlib import Path

import pytest
from openalpha_sentinel.development_manifest import build_development_manifest
from openalpha_sentinel.development_pipeline import (
    analyze_development_state,
    freeze_development_state,
    report_development_state,
    scan_repository_policy,
    verify_development_state,
)
from openalpha_sentinel.development_runner import (
    OriginExecutionSummary,
    PilotOriginRecord,
    evaluate_pilot,
    preflight,
)
from openalpha_sentinel.development_serialization import (
    atomic_write_bytes,
    canonical_json_bytes,
    sha256_bytes,
)
from openalpha_sentinel.risk_model import NONSTRUCTURAL_FEATURES, STRUCTURAL_FEATURES

_OUTPUTS = {
    'development_table.jsonl',
    'development_manifest.json',
    'structural_prevalence.json',
    'diagnostic_analysis.json',
    'out_of_fold_predictions.jsonl',
    'model_comparison.json',
    'intervention_analysis.json',
    'risk_coverage.json',
    'freeze_candidate.json',
    'report.md',
}


def _seed_terminal_state(root: Path) -> None:
    manifest = build_development_manifest()
    preflight(root)
    for index, origin in enumerate(manifest.origins):
        week = index // 2
        signal = float((week % 4) + (index % 2) * 0.2)
        error = 0.004 + 0.004 * (week % 4) + 0.0001 * (index % 2)
        invalid = week % 3 == 0
        diagnostics = {
            name: {
                'status': 'available',
                'value': signal + feature_index * 0.01,
                'reason': None,
            }
            for feature_index, name in enumerate(STRUCTURAL_FEATURES)
        }
        diagnostics['NONFINITE_OUTPUT_COUNT']['value'] = 0.0
        diagnostics['NONPOSITIVE_PRICE_COUNT']['value'] = 0.0
        reliability = {
            name: {
                'status': 'available',
                'value': signal + feature_index * 0.02,
                'reason': None,
            }
            for feature_index, name in enumerate(NONSTRUCTURAL_FEATURES)
        }
        paths = []
        for context in (128, 256, 512):
            for seed in (1729, 2027, 7919):
                path_invalid = invalid and seed == 1729
                paths.append(
                    {
                        'path_id': f'{origin.origin_id}-{context}-{seed}',
                        'valid': not path_invalid,
                        'invalid_candle_count': int(path_invalid),
                        'earliest_invalid_horizon_step': 3 if path_invalid else None,
                        'violations': (
                            [
                                {
                                    'code': 'HIGH_BELOW_CLOSE',
                                    'step': 3,
                                    'normalized_severity': 0.001,
                                }
                            ]
                            if path_invalid
                            else []
                        ),
                    }
                )
        realized = 0.01 if week % 2 else -0.01
        predicted = realized + error
        individual_path_results = [
            {
                'path_id': str(path['path_id']),
                'context_length': context,
                'sampling_seed': seed,
                'structural_valid': bool(path['valid']),
                'predicted_log_return': predicted,
                'absolute_return_error': error,
                'direction_correct': predicted * realized > 0,
            }
            for path, (context, seed) in zip(
                paths,
                (
                    (context, seed)
                    for context in (128, 256, 512)
                    for seed in (1729, 2027, 7919)
                ),
                strict=True,
            )
        ]
        row = {
            'schema_version': 'sentinel-phase3a-analysis-row-v1',
            'terminal_status': 'completed',
            'origin_id': origin.origin_id,
            'asset': origin.asset,
            'cutoff': origin.cutoff.isoformat(),
            'horizon_end': origin.forecast_sessions[-1].isoformat(),
            'data_hash': sha256_bytes(origin.origin_id.encode()),
            'model_revision': 'b' * 40,
            'structural_status': 'FAILED' if invalid else 'PASSED',
            'structural_diagnostics': diagnostics,
            'individual_structural_validity': paths,
            'individual_path_results': individual_path_results,
            'canonical_structural_validity': {'valid': not invalid},
            'raw_close_return_forecast': predicted,
            'projected_path': {'implied_return_unchanged': True},
            'projected_path_status': 'PASSED',
            'reliability_features': reliability,
            'realized_outcome': {'realized_log_return': realized},
            'forecast_error': {
                'kronos_absolute_error': error,
                'baseline_absolute_error': abs(realized),
                'direction_correct': predicted * realized > 0,
                'deployability_label': error > abs(realized),
                'valid_path_aggregation_candidate': {
                    'applied': invalid,
                    'valid_path_count': 2 if invalid else 3,
                    'absolute_error': max(error - 0.0005, 0.0),
                },
            },
            'runtime_metadata': {
                'cache_hits': 0,
                'cache_misses': 9,
                'inference_duration_ms': 9000.0,
            },
        }
        terminal_hash = sha256_bytes(canonical_json_bytes(row))
        origin_root = root / 'origins' / origin.origin_id
        atomic_write_bytes(origin_root / 'analysis-row.json', canonical_json_bytes(row))
        atomic_write_bytes(
            origin_root / 'resolved.json',
            canonical_json_bytes(
                {
                    'schema_version': 'sentinel-phase3a-synthetic-resolved-v1',
                    'origin_id': origin.origin_id,
                    'chain_verified': True,
                }
            ),
        )
        marker = OriginExecutionSummary(
            schema_version='sentinel-phase3a-origin-execution-v1',
            origin_id=origin.origin_id,
            terminal_status='completed',
            request_success_count=9,
            request_count=9,
            ensemble_latency_seconds=9.0,
            cache_bytes=32_000_000,
            process_stable=True,
            terminal_record_sha256=terminal_hash,
        )
        atomic_write_bytes(
            origin_root / 'runner-terminal.json',
            canonical_json_bytes(marker.model_dump(mode='json')),
        )
    pilot = evaluate_pilot(
        tuple(
            PilotOriginRecord(
                origin_id=origin.origin_id,
                request_success_count=9,
                request_count=9,
                ensemble_latency_seconds=9.0,
                process_stable=True,
            )
            for origin in manifest.origins[:10]
        ),
        cache_bytes=32_000_000,
        hosted_cost_per_cutoff=0.0,
        deterministic_replay_supported=True,
    )
    atomic_write_bytes(
        root / 'pilot-gate.json',
        canonical_json_bytes(pilot.model_dump(mode='json')),
    )


def test_full_offline_pipeline_produces_and_verifies_compact_outputs(
    tmp_path: Path,
) -> None:
    _seed_terminal_state(tmp_path)

    analysis = analyze_development_state(tmp_path)
    freeze = freeze_development_state(tmp_path)
    report = report_development_state(tmp_path)
    verified = verify_development_state(tmp_path)

    output_root = tmp_path / 'repository_outputs'
    assert _OUTPUTS <= {path.name for path in output_root.iterdir()}
    assert analysis['origin_count'] == 104
    assert freeze['freeze_verified'] is True
    assert report['report_verified'] is True
    assert verified['terminal_origin_count'] == 104
    assert verified['network_accessed'] is False
    assert verified['inference_accessed'] is False
    assert (output_root / 'development_table.jsonl').read_bytes().endswith(b'\n')
    assert (output_root / 'report.md').read_text().startswith(
        'DEVELOPMENT ANALYSIS - NOT HOLDOUT EVIDENCE'
    )


def test_policy_scan_rejects_reusable_data_credentials_and_outside_origins() -> None:
    with pytest.raises(ValueError, match='observations'):
        scan_repository_policy({'observations': [{'close': 1.0}]})
    with pytest.raises(ValueError, match='credential'):
        scan_repository_policy({'api_credentials': 'secret'})
    with pytest.raises(ValueError, match='non-finite'):
        scan_repository_policy({'metric': float('nan')})
    with pytest.raises(ValueError, match='outside'):
        scan_repository_policy(
            {'origin_id': 'sentinel-v0-SPY-2025-07-04-h5'}
        )


def test_verify_detects_public_artifact_tampering(tmp_path: Path) -> None:
    _seed_terminal_state(tmp_path)
    analyze_development_state(tmp_path)
    freeze_development_state(tmp_path)
    report_development_state(tmp_path)
    path = tmp_path / 'repository_outputs' / 'risk_coverage.json'
    payload = json.loads(path.read_text())
    payload['risk_error_spearman'] = -1
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match='hash mismatch'):
        verify_development_state(tmp_path)
