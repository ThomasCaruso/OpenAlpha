from __future__ import annotations

import math
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal, Protocol

from openalpha_research import LocalArtifactStore
from openalpha_research.errors import ArtifactConflictError
from pydantic import Field

from .contracts import (
    MODEL_REPOSITORY,
    MODEL_REVISION,
    SOURCE_REVISION,
    TOKENIZER_REPOSITORY,
    TOKENIZER_REVISION,
    ForecastPath,
    ForecastRequest,
    ForecastResponse,
    FrozenModel,
)
from .development_diagnostics import (
    ResolvedForecastError,
    compute_development_nonstructural_diagnostics,
)
from .development_manifest import (
    EXPERIMENT_SHA256,
    DevelopmentOrigin,
    build_development_manifest,
    require_development_cutoff,
)
from .development_serialization import (
    atomic_write_bytes,
    canonical_json_bytes,
    sha256_bytes,
)
from .diagnostics import DiagnosticVector
from .ensemble import (
    EnsembleMember,
    EnsembleResult,
    assemble_ensemble,
    average_forecast_paths,
)
from .evidence import CreationEvidence, EvidenceContext, publish_creation, verify_creation
from .inference_cache import InferenceCache, InferenceCacheKey
from .market_data import MarketDataRequest, MarketDataSnapshot
from .structural_validity import (
    AuditCandle,
    PathValidity,
    ProjectionResult,
    StructuralDiagnostics,
    constraint_projection_v0,
    summarize_structural_validity,
    validate_forecast_path,
)

CONTEXTS = (128, 256, 512)
SEEDS = (1729, 2027, 7919)


class MarketDataProvider(Protocol):
    def fetch(self, request: MarketDataRequest) -> MarketDataSnapshot: ...


ForecastMany = Callable[[tuple[ForecastRequest, ...]], tuple[ForecastResponse, ...]]


class DevelopmentForecastCreation(FrozenModel):
    schema_version: Literal['sentinel-phase3a-creation-v1']
    origin: DevelopmentOrigin
    data_snapshot_metadata: dict[str, object]
    individual_paths: tuple[ForecastPath, ...] = Field(min_length=9, max_length=9)
    ensemble: EnsembleResult
    canonical_path: ForecastPath
    individual_validity: tuple[PathValidity, ...] = Field(min_length=9, max_length=9)
    canonical_validity: PathValidity
    structural_status: Literal['PASSED', 'FAILED']
    structural_diagnostics: StructuralDiagnostics
    individual_projections: tuple[ProjectionResult, ...] = Field(min_length=9, max_length=9)
    projected_path: ProjectionResult
    projected_path_status: Literal['PASSED']
    reliability_features: DiagnosticVector
    raw_close_return_forecast: float
    creation: CreationEvidence
    creation_verified: Literal[True]
    cache_hits: int = Field(ge=0, le=9)
    cache_misses: int = Field(ge=0, le=9)


def create_development_forecast(
    *,
    origin: DevelopmentOrigin,
    state_root: Path,
    market_provider: MarketDataProvider,
    forecast_many: ForecastMany,
    cache: InferenceCache,
    artifact_store: LocalArtifactStore,
    evidence_context: EvidenceContext,
    prior_resolved: tuple[ResolvedForecastError, ...],
    clock: Callable[[], datetime],
) -> DevelopmentForecastCreation:
    require_development_cutoff(origin.cutoff)
    manifest = build_development_manifest()
    if origin not in manifest.origins:
        raise ValueError('origin is not in the locked development manifest')
    created_at = clock()
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise ValueError('creation clock must be timezone-aware')
    snapshot = market_provider.fetch(
        MarketDataRequest(
            purpose='forecast_context',
            symbol=origin.asset,
            start_inclusive=date(2022, 6, 1),
            end_exclusive=origin.cutoff + timedelta(days=1),
            cutoff=origin.cutoff,
            minimum_sessions=512,
        )
    )
    _validate_snapshot(snapshot, origin)
    requests = build_development_forecast_requests(
        origin=origin,
        snapshot=snapshot,
        created_at=created_at,
    )
    responses: list[ForecastResponse | None] = [None] * len(requests)
    missing_indices: list[int] = []
    for index, request in enumerate(requests):
        cached = cache.get_verified(request)
        if cached is None:
            missing_indices.append(index)
        else:
            responses[index] = cached
    if missing_indices:
        missing_requests = tuple(requests[index] for index in missing_indices)
        generated = forecast_many(missing_requests)
        if len(generated) != len(missing_requests):
            raise ValueError('forecast provider returned the wrong response count')
        for index, request, response in zip(missing_indices, missing_requests, generated):
            _require_success(response)
            cache.put(request, response)
            responses[index] = response
    if any(response is None for response in responses):
        raise ValueError('inference response assembly is incomplete')
    completed_responses = tuple(response for response in responses if response is not None)
    members = tuple(
        EnsembleMember(
            context_length=request.context_length,
            sampling_seed=request.sampling_seed,
            path=response.generated_paths[0],
            inference_duration_ms=response.inference_duration_ms,
        )
        for request, response in zip(requests, completed_responses)
    )

    ensemble = assemble_ensemble(
        cutoff_close=snapshot.observations[-1].close,
        forecast_sessions=origin.forecast_sessions,
        members=members,
    )
    canonical_path = average_forecast_paths(
        tuple(member.path for member in members if member.context_length == 512),
        path_id=f'{origin.origin_id}-canonical-512-average',
    )
    if tuple(row.close for row in canonical_path.observations) != ensemble.canonical_close_path:
        raise ValueError('canonical OHLCV path closes do not match locked close aggregation')
    individual_validity = tuple(
        _validate_path(member.path, snapshot, origin) for member in members
    )
    canonical_validity = _validate_path(canonical_path, snapshot, origin)
    structural_diagnostics = summarize_structural_validity(individual_validity)
    individual_projections = tuple(
        _project_path(member.path, snapshot) for member in members
    )
    projected_path = _project_path(canonical_path, snapshot)
    projected_validity = validate_forecast_path(
        path_id=f'{canonical_path.path_id}-projected',
        candles=projected_path.projected_candles,
        expected_sessions=origin.forecast_sessions,
        cutoff_close=snapshot.observations[-1].close,
        cutoff_volume=snapshot.observations[-1].volume,
    )
    if not projected_validity.valid:
        raise ValueError('constraint projection did not produce a structurally valid path')
    diagnostic_vector = compute_development_nonstructural_diagnostics(
        symbol=origin.asset,
        cutoff=origin.cutoff,
        context=snapshot.observations,
        ensemble=ensemble,
        created_at=created_at,
        prior_resolved=prior_resolved,
    )
    provider_metadata = snapshot.model_dump(mode='json', exclude={'observations'})
    request_metadata = tuple(
        {
            'cache_key_sha256': InferenceCacheKey.from_request(request).canonical_sha256,
            'context_length': request.context_length,
            'sampling_seed': request.sampling_seed,
            'data_snapshot_sha256': request.data_snapshot_sha256,
        }
        for request in requests
    )
    forecast_payload = {
        'schema_version': '1.0',
        'payload_type': 'sentinel-phase3a-forecast-v1',
        'claim_boundary': 'DEVELOPMENT ANALYSIS - NOT HOLDOUT EVIDENCE',
        'origin_id': origin.origin_id,
        'symbol': origin.asset,
        'cutoff': origin.cutoff.isoformat(),
        'forecast_sessions': [session.isoformat() for session in origin.forecast_sessions],
        'cutoff_close': ensemble.cutoff_close,
        'request_metadata': list(request_metadata),
        'individual_paths': [
            {
                'context_length': member.context_length,
                'sampling_seed': member.sampling_seed,
                'path': member.path.model_dump(mode='json'),
                'predicted_log_return': math.log(
                    member.path.observations[-1].close / ensemble.cutoff_close
                ),
                'inference_duration_ms': member.inference_duration_ms,
            }
            for member in members
        ],
        'canonical_path': canonical_path.model_dump(mode='json'),
        'canonical_predicted_log_return': ensemble.canonical_predicted_log_return,
        'baseline_predicted_log_return': 0.0,
        'created_at': created_at.isoformat(),
    }
    diagnostics_payload = {
        'schema_version': '1.0',
        'payload_type': 'sentinel-phase3a-diagnostics-v1',
        'origin_id': origin.origin_id,
        'structural_status': 'PASSED' if canonical_validity.valid else 'FAILED',
        'individual_validity': [item.model_dump(mode='json') for item in individual_validity],
        'canonical_validity': canonical_validity.model_dump(mode='json'),
        'structural_diagnostics': structural_diagnostics.model_dump(mode='json'),
        'individual_projections': [
            item.model_dump(mode='json') for item in individual_projections
        ],
        'projected_path': projected_path.model_dump(mode='json'),
        'reliability_features': diagnostic_vector.model_dump(mode='json'),
        'action': None,
    }
    creation = publish_creation(
        store=artifact_store,
        context=evidence_context,
        canonical_spec={
            'schema_version': '1.0',
            'payload_type': 'sentinel-phase3a-origin-spec-v1',
            'experiment_sha256': EXPERIMENT_SHA256,
            'development_manifest_sha256': manifest.canonical_sha256,
            'origin': origin.model_dump(mode='json'),
        },
        data_snapshot={
            'schema_version': '1.0',
            'payload_type': 'sentinel-phase3a-data-snapshot-v1',
            **provider_metadata,
        },
        data_quality={
            'schema_version': '1.0',
            'payload_type': 'sentinel-phase3a-data-quality-v1',
            'normalized_input_sha256': snapshot.normalized_input_sha256,
            'row_count': snapshot.row_count,
            'first_session': snapshot.first_session.isoformat(),
            'last_session': snapshot.last_session.isoformat(),
            'quality_summary': list(snapshot.quality_summary),
            'corporate_action_warnings': [
                warning.model_dump(mode='json')
                for warning in snapshot.corporate_action_warnings
            ],
        },
        forecast=forecast_payload,
        diagnostics=diagnostics_payload,
        created_at=created_at,
    )
    if not verify_creation(artifact_store, creation):
        raise ValueError('forecast creation seal did not verify')
    result = DevelopmentForecastCreation(
        schema_version='sentinel-phase3a-creation-v1',
        origin=origin,
        data_snapshot_metadata=provider_metadata,
        individual_paths=tuple(member.path for member in members),
        ensemble=ensemble,
        canonical_path=canonical_path,
        individual_validity=individual_validity,
        canonical_validity=canonical_validity,
        structural_status='PASSED' if canonical_validity.valid else 'FAILED',
        structural_diagnostics=structural_diagnostics,
        individual_projections=individual_projections,
        projected_path=projected_path,
        projected_path_status='PASSED',
        reliability_features=diagnostic_vector,
        raw_close_return_forecast=ensemble.canonical_predicted_log_return,
        creation=creation,
        creation_verified=True,
        cache_hits=len(requests) - len(missing_indices),
        cache_misses=len(missing_indices),
    )
    descriptor = canonical_json_bytes(
        result.model_dump(
            mode='json',
            exclude={'cache_hits', 'cache_misses'},
        )
    )
    _write_immutable(
        state_root / 'origins' / origin.origin_id / 'creation.json',
        descriptor,
    )
    return result


def build_development_forecast_requests(
    *,
    origin: DevelopmentOrigin,
    snapshot: MarketDataSnapshot,
    created_at: datetime,
) -> tuple[ForecastRequest, ...]:
    require_development_cutoff(origin.cutoff)
    if origin not in build_development_manifest().origins:
        raise ValueError('origin is not in the locked development manifest')
    _validate_snapshot(snapshot, origin)
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise ValueError('request creation time must be timezone-aware')
    return tuple(
        _forecast_request(
            origin=origin,
            observations=snapshot.observations[-context:],
            context_length=context,
            sampling_seed=seed,
            created_at=created_at,
        )
        for context in CONTEXTS
        for seed in SEEDS
    )

def _forecast_request(
    *,
    origin: DevelopmentOrigin,
    observations: tuple,
    context_length: int,
    sampling_seed: int,
    created_at: datetime,
) -> ForecastRequest:
    snapshot_sha256 = sha256_bytes(
        canonical_json_bytes([row.model_dump(mode='json') for row in observations])
    )
    return ForecastRequest(
        model_repository=MODEL_REPOSITORY,
        model_revision=MODEL_REVISION,
        tokenizer_repository=TOKENIZER_REPOSITORY,
        tokenizer_revision=TOKENIZER_REVISION,
        source_revision=SOURCE_REVISION,
        symbol=origin.asset,
        cutoff=origin.cutoff,
        forecast_sessions=origin.forecast_sessions,
        observations=observations,
        context_length=context_length,
        forecast_horizon=5,
        sampling_seed=sampling_seed,
        temperature=1.0,
        top_p=0.9,
        sample_count=1,
        data_snapshot_sha256=snapshot_sha256,
        experiment_sha256=EXPERIMENT_SHA256,
        created_at=created_at,
    )


def _validate_snapshot(snapshot: MarketDataSnapshot, origin: DevelopmentOrigin) -> None:
    if snapshot.synthetic is False and snapshot.provider != 'yahoo_finance':
        raise ValueError('Phase 3A provider does not match the locked experiment')
    if snapshot.adjustment != 'raw':
        raise ValueError('Phase 3A requires raw market data')
    if len(snapshot.observations) < 512:
        raise ValueError('development origin requires 512 causal sessions')
    sessions = tuple(row.session for row in snapshot.observations)
    if sessions[-1] != origin.cutoff or any(session > origin.cutoff for session in sessions):
        raise ValueError('forecast snapshot must end exactly at the origin cutoff')
    if snapshot.normalized_schema != ('session', 'open', 'high', 'low', 'close', 'volume'):
        raise ValueError('forecast snapshot schema does not match locked OHLCV order')


def _require_success(response: ForecastResponse) -> None:
    if response.failure is not None or len(response.generated_paths) != 1:
        raise RuntimeError('official inference did not return one preserved path')


def _audit_candles(path: ForecastPath) -> tuple[AuditCandle, ...]:
    return tuple(
        AuditCandle(
            session=row.session,
            open=row.open,
            high=row.high,
            low=row.low,
            close=row.close,
            volume=row.volume,
        )
        for row in path.observations
    )


def _validate_path(
    path: ForecastPath,
    snapshot: MarketDataSnapshot,
    origin: DevelopmentOrigin,
) -> PathValidity:
    return validate_forecast_path(
        path_id=path.path_id,
        candles=_audit_candles(path),
        expected_sessions=origin.forecast_sessions,
        cutoff_close=snapshot.observations[-1].close,
        cutoff_volume=snapshot.observations[-1].volume,
    )


def _project_path(path: ForecastPath, snapshot: MarketDataSnapshot) -> ProjectionResult:
    projected = constraint_projection_v0(
        path_id=path.path_id,
        candles=_audit_candles(path),
        cutoff_close=snapshot.observations[-1].close,
    )
    if projected.original_log_return != projected.projected_log_return:
        raise ValueError('constraint projection changed the implied close return')
    return projected


def _write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.is_symlink() or path.read_bytes() != payload:
            raise ArtifactConflictError('immutable creation descriptor already differs')
        return
    atomic_write_bytes(path, payload)
