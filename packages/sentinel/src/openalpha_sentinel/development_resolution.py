from __future__ import annotations

import math
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from openalpha_research import LocalArtifactStore, MethodologyStatus
from openalpha_research.artifacts import Sha256
from openalpha_research.errors import ArtifactConflictError
from pydantic import Field

from .contracts import MODEL_REVISION, FrozenModel
from .development_manifest import DevelopmentOrigin
from .development_origin import (
    DevelopmentForecastCreation,
    MarketDataProvider,
)
from .development_serialization import (
    atomic_write_bytes,
    canonical_json_bytes,
    sha256_bytes,
)
from .ensemble import average_forecast_paths
from .evidence import (
    EvidenceContext,
    ResolvedEvidence,
    append_outcome_and_complete,
    verify_creation,
    verify_resolved_evidence,
)
from .market_data import MarketDataRequest, MarketDataSnapshot


class DevelopmentResolvedOrigin(FrozenModel):
    schema_version: Literal['sentinel-phase3a-resolved-v1']
    origin: DevelopmentOrigin
    resolved: ResolvedEvidence
    analysis_row: dict[str, object]
    chain_verified: Literal[True]


class TerminalFailureRecord(FrozenModel):
    schema_version: Literal['sentinel-phase3a-terminal-failure-v1']
    origin: DevelopmentOrigin
    terminal_status: Literal['failed']
    analysis_status: Literal['unavailable']
    code: str = Field(min_length=2, max_length=64)
    stage: str = Field(min_length=2, max_length=64)
    message: str = Field(min_length=1, max_length=2_000)
    attempt_sha256: tuple[Sha256, ...]
    occurred_at: datetime
    record_sha256: Sha256


def resolve_development_outcome(
    *,
    creation: DevelopmentForecastCreation,
    state_root: Path,
    market_provider: MarketDataProvider,
    artifact_store: LocalArtifactStore,
    evidence_context: EvidenceContext,
    clock: Callable[[], datetime],
) -> DevelopmentResolvedOrigin:
    if not verify_creation(artifact_store, creation.creation):
        raise ValueError('immutable forecast creation seal failed before outcome access')
    completed_at = clock()
    if completed_at.tzinfo is None or completed_at.utcoffset() is None:
        raise ValueError('resolution clock must be timezone-aware')
    outcome_snapshot: MarketDataSnapshot | None = None
    outcome_payload: dict[str, object] | None = None

    def outcome_loader() -> dict[str, object]:
        nonlocal outcome_snapshot, outcome_payload
        first_session = creation.origin.forecast_sessions[0]
        final_session = creation.origin.forecast_sessions[-1]
        snapshot = market_provider.fetch(
            MarketDataRequest(
                purpose='outcome',
                symbol=creation.origin.asset,
                start_inclusive=first_session,
                end_exclusive=final_session + timedelta(days=1),
                cutoff=final_session,
                minimum_sessions=5,
            )
        )
        _validate_outcome_snapshot(snapshot, creation)
        realized_return = math.log(
            snapshot.observations[-1].close / creation.ensemble.cutoff_close
        )
        predicted_return = creation.raw_close_return_forecast
        kronos_error = abs(predicted_return - realized_return)
        baseline_error = abs(realized_return)
        p3 = _valid_path_candidate(creation, realized_return)
        outcome_snapshot = snapshot
        outcome_payload = {
            'schema_version': 'sentinel-phase3a-outcome-v1',
            'forecast_id': creation.creation.seal_ref.sha256,
            'origin_id': creation.origin.origin_id,
            'resolved_at': completed_at.isoformat(),
            'provider': snapshot.model_dump(mode='json', exclude={'observations'}),
            'outcome_sessions': [row.session.isoformat() for row in snapshot.observations],
            'outcome_ohlcv': [row.model_dump(mode='json') for row in snapshot.observations],
            'realized_log_return': realized_return,
            'canonical_predicted_log_return': predicted_return,
            'kronos_absolute_return_error': kronos_error,
            'direction_correct': _direction(predicted_return) == _direction(realized_return),
            'baseline_predicted_log_return': 0.0,
            'baseline_absolute_return_error': baseline_error,
            'deployability_label': kronos_error > baseline_error,
            'valid_path_aggregation_candidate': p3,
        }
        return outcome_payload

    resolved = append_outcome_and_complete(
        store=artifact_store,
        context=evidence_context,
        creation=creation.creation,
        outcome_loader=outcome_loader,
        methodology_audit={
            'schema_version': '1.0',
            'payload_type': 'sentinel-phase3a-methodology-audit-v1',
            'methodology_status': MethodologyStatus.PASSED_WITH_WARNINGS.value,
            'claim_boundary': 'DEVELOPMENT ANALYSIS - NOT HOLDOUT EVIDENCE',
            'origin_id': creation.origin.origin_id,
            'forecast_sealed_before_outcome': True,
            'structural_status': creation.structural_status,
            'warnings': [
                'UNOFFICIAL_YAHOO_INTERFACE',
                'NOT_POINT_IN_TIME_DATA',
                'RAW_RETURN_OMITS_DIVIDENDS',
                'NO_CROSS_PROVIDER_VERIFICATION',
            ],
        },
        methodology_status=MethodologyStatus.PASSED_WITH_WARNINGS,
        test_evidence=('Phase 3A targeted tests passed',),
        completed_at=completed_at,
        manifest_parameters=(('phase', '3A'), ('origin', creation.origin.origin_id)),
    )
    if outcome_snapshot is None or outcome_payload is None:
        raise ValueError('outcome loader did not produce a resolved snapshot')
    if not verify_resolved_evidence(artifact_store, resolved):
        raise ValueError('complete development origin chain failed verification')
    analysis_row = _analysis_row(creation, outcome_snapshot, outcome_payload, resolved)
    result = DevelopmentResolvedOrigin(
        schema_version='sentinel-phase3a-resolved-v1',
        origin=creation.origin,
        resolved=resolved,
        analysis_row=analysis_row,
        chain_verified=True,
    )
    origin_root = state_root / 'origins' / creation.origin.origin_id
    _write_immutable(
        origin_root / 'resolved.json',
        canonical_json_bytes(result.model_dump(mode='json')),
    )
    _write_immutable(origin_root / 'analysis-row.json', canonical_json_bytes(analysis_row))
    return result


def record_terminal_failure(
    *,
    origin: DevelopmentOrigin,
    state_root: Path,
    code: str,
    stage: str,
    message: str,
    attempt_sha256: tuple[str, ...],
    occurred_at: datetime,
) -> TerminalFailureRecord:
    body = {
        'schema_version': 'sentinel-phase3a-terminal-failure-v1',
        'origin': origin.model_dump(mode='json'),
        'terminal_status': 'failed',
        'analysis_status': 'unavailable',
        'code': code,
        'stage': stage,
        'message': message,
        'attempt_sha256': list(attempt_sha256),
        'occurred_at': occurred_at.isoformat(),
    }
    record = TerminalFailureRecord(
        schema_version='sentinel-phase3a-terminal-failure-v1',
        origin=origin,
        terminal_status='failed',
        analysis_status='unavailable',
        code=code,
        stage=stage,
        message=message,
        attempt_sha256=attempt_sha256,
        occurred_at=occurred_at,
        record_sha256=sha256_bytes(canonical_json_bytes(body)),
    )
    _write_immutable(
        state_root / 'origins' / origin.origin_id / 'terminal-failure.json',
        canonical_json_bytes(record.model_dump(mode='json')),
    )
    return record


def _validate_outcome_snapshot(
    snapshot: MarketDataSnapshot,
    creation: DevelopmentForecastCreation,
) -> None:
    sessions = tuple(row.session for row in snapshot.observations)
    if sessions != creation.origin.forecast_sessions:
        raise ValueError('outcome snapshot does not match the five locked XNYS sessions')
    if snapshot.adjustment != 'raw':
        raise ValueError('outcome resolution requires raw market data')
    if snapshot.row_count != 5:
        raise ValueError('outcome resolution requires exactly five rows')


def _valid_path_candidate(
    creation: DevelopmentForecastCreation,
    realized_return: float,
) -> dict[str, object]:
    valid_paths = tuple(
        member.path
        for member, validity in zip(creation.ensemble.members, creation.individual_validity)
        if member.context_length == 512 and validity.valid
    )
    if len(valid_paths) < 2:
        predicted_return = creation.raw_close_return_forecast
        return {
            'applied': False,
            'reason': 'FEWER_THAN_TWO_VALID_512_PATHS',
            'valid_path_count': len(valid_paths),
            'predicted_log_return': predicted_return,
            'absolute_error': abs(predicted_return - realized_return),
        }
    averaged = average_forecast_paths(
        valid_paths,
        path_id=f'{creation.origin.origin_id}-valid-512-average',
    )
    predicted_return = math.log(
        averaged.observations[-1].close / creation.ensemble.cutoff_close
    )
    return {
        'applied': True,
        'reason': None,
        'valid_path_count': len(valid_paths),
        'path_sha256': averaged.canonical_sha256,
        'predicted_log_return': predicted_return,
        'absolute_error': abs(predicted_return - realized_return),
    }


def _analysis_row(
    creation: DevelopmentForecastCreation,
    snapshot: MarketDataSnapshot,
    outcome: dict[str, object],
    resolved: ResolvedEvidence,
) -> dict[str, object]:
    projected_sha256 = sha256_bytes(
        canonical_json_bytes(
            [item.model_dump(mode='json') for item in creation.projected_path.projected_candles]
        )
    )
    raw_realized_return = outcome.get('realized_log_return')
    if not isinstance(raw_realized_return, (int, float)) or isinstance(
        raw_realized_return, bool
    ):
        raise TypeError('resolved outcome is missing a numeric realized return')
    realized_return = float(raw_realized_return)
    validity_by_path = {
        item.path_id: item
        for item in creation.individual_validity
    }
    predicted_by_configuration = {
        (item.context_length, item.sampling_seed): item.predicted_log_return
        for item in creation.ensemble.individual_predicted_log_returns
    }
    individual_path_results = [
        {
            'path_id': member.path.path_id,
            'context_length': member.context_length,
            'sampling_seed': member.sampling_seed,
            'structural_valid': validity_by_path[member.path.path_id].valid,
            'predicted_log_return': predicted_by_configuration[
                (member.context_length, member.sampling_seed)
            ],
            'absolute_return_error': abs(
                predicted_by_configuration[
                    (member.context_length, member.sampling_seed)
                ]
                - realized_return
            ),
            'direction_correct': (
                _direction(
                    predicted_by_configuration[
                        (member.context_length, member.sampling_seed)
                    ]
                )
                == _direction(realized_return)
            ),
        }
        for member in creation.ensemble.members
    ]
    return {
        'schema_version': 'sentinel-phase3a-analysis-row-v1',
        'terminal_status': 'completed',
        'origin_id': creation.origin.origin_id,
        'asset': creation.origin.asset,
        'cutoff': creation.origin.cutoff.isoformat(),
        'horizon_end': creation.origin.forecast_sessions[-1].isoformat(),
        'data_hash': creation.data_snapshot_metadata['normalized_input_sha256'],
        'model_revision': MODEL_REVISION,
        'structural_status': creation.structural_status,
        'structural_diagnostics': {
            item.name: item.model_dump(mode='json')
            for item in creation.structural_diagnostics.values
        },
        'individual_structural_validity': [
            {
                'path_id': item.path_id,
                'valid': item.valid,
                'invalid_candle_count': item.invalid_candle_count,
                'earliest_invalid_horizon_step': item.earliest_invalid_horizon_step,
                'violations': [entry.model_dump(mode='json') for entry in item.violations],
            }
            for item in creation.individual_validity
        ],
        'individual_path_results': individual_path_results,
        'canonical_structural_validity': creation.canonical_validity.model_dump(mode='json'),
        'raw_close_return_forecast': creation.raw_close_return_forecast,
        'baseline_predicted_return': 0.0,
        'projected_path': {
            'method': creation.projected_path.method,
            'projected_path_sha256': projected_sha256,
            'adjustment_count': len(creation.projected_path.adjustments),
            'total_normalized_adjustment': sum(
                item.normalized_adjustment for item in creation.projected_path.adjustments
            ),
            'implied_return_unchanged': (
                creation.projected_path.original_log_return
                == creation.projected_path.projected_log_return
            ),
        },
        'projected_path_status': creation.projected_path_status,
        'reliability_features': {
            item.name: item.model_dump(mode='json')
            for item in creation.reliability_features.values
        },
        'realized_outcome': {
            'sessions': [row.session.isoformat() for row in snapshot.observations],
            'realized_log_return': outcome['realized_log_return'],
            'resolved_evidence_sha256': resolved.manifest_ref.sha256,
        },
        'forecast_error': {
            'kronos_absolute_error': outcome['kronos_absolute_return_error'],
            'baseline_absolute_error': outcome['baseline_absolute_return_error'],
            'direction_correct': outcome['direction_correct'],
            'deployability_label': outcome['deployability_label'],
            'valid_path_aggregation_candidate': outcome[
                'valid_path_aggregation_candidate'
            ],
        },
        'runtime_metadata': {
            'cache_hits': creation.cache_hits,
            'cache_misses': creation.cache_misses,
            'inference_duration_ms': sum(
                item.inference_duration_ms for item in creation.ensemble.members
            ),
        },
    }


def _direction(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.is_symlink() or path.read_bytes() != payload:
            raise ArtifactConflictError('immutable terminal descriptor already differs')
        return
    atomic_write_bytes(path, payload)
