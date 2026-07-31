import json
from datetime import UTC, datetime
from pathlib import Path
from shutil import copytree

import pytest
from openalpha_research import ArtifactRef, LocalArtifactStore
from openalpha_sentinel.contracts import OHLCVObservation
from openalpha_sentinel.development_execution import RealDevelopmentExecutor
from openalpha_sentinel.development_manifest import build_development_manifest
from openalpha_sentinel.development_origin import (
    DevelopmentForecastCreation,
    create_development_forecast,
)
from openalpha_sentinel.development_resolution import (
    record_terminal_failure,
    resolve_development_outcome,
)
from openalpha_sentinel.development_runner import OriginExecutionSummary
from openalpha_sentinel.development_serialization import (
    canonical_json_bytes,
    sha256_bytes,
)
from openalpha_sentinel.inference_cache import InferenceCache
from openalpha_sentinel.market_data import MarketDataRequest, MarketDataSnapshot

from packages.sentinel.tests.test_development_origin import (
    FakeInference,
    FakeMarketProvider,
    _evidence_context,
    _snapshot,
)

NOW = datetime(2026, 7, 30, 20, 0, tzinfo=UTC)


class OutcomeProvider:
    def __init__(self, snapshot: MarketDataSnapshot) -> None:
        self.snapshot = snapshot
        self.requests: list[MarketDataRequest] = []

    def fetch(self, request: MarketDataRequest) -> MarketDataSnapshot:
        self.requests.append(request)
        return self.snapshot


def _created(tmp_path: Path) -> tuple[DevelopmentForecastCreation, LocalArtifactStore]:
    origin = build_development_manifest().origins[0]
    state_root = tmp_path / 'state'
    store = LocalArtifactStore(state_root / 'artifacts')
    creation = create_development_forecast(
        origin=origin,
        state_root=state_root,
        market_provider=FakeMarketProvider(_snapshot(origin.cutoff)),
        forecast_many=FakeInference(),
        cache=InferenceCache(tmp_path / 'cache'),
        artifact_store=store,
        evidence_context=_evidence_context(),
        prior_resolved=(),
        clock=lambda: NOW,
    )
    return creation, store


def _outcome_snapshot(creation: DevelopmentForecastCreation) -> MarketDataSnapshot:
    rows = tuple(
        OHLCVObservation(
            session=session,
            timestamp=datetime.combine(session, datetime.min.time(), tzinfo=UTC),
            open=455.0 + index,
            high=456.0 + index,
            low=454.0 + index,
            close=455.5 + index,
            volume=50_000_000.0,
        )
        for index, session in enumerate(creation.origin.forecast_sessions)
    )
    return MarketDataSnapshot(
        provider='yahoo_finance',
        client='yfinance',
        client_version='1.5.2',
        access_class='unofficial_public_interface',
        intended_use='local_research_and_education',
        redistribution='prohibited_by_project_policy',
        adjustment='raw',
        request_parameters=(('interval', '1d'),),
        retrieved_at=NOW,
        normalized_schema=('session', 'open', 'high', 'low', 'close', 'volume'),
        row_count=5,
        first_session=rows[0].session,
        last_session=rows[-1].session,
        normalized_input_sha256='a' * 64,
        observations=rows,
        quality_summary=('COMPLETE_XNYS_WINDOW',),
        corporate_action_warnings=(),
        timezone_normalization='provider_session_label_to_utc_midnight',
        pagination='client_managed_not_exposed',
        synthetic=True,
    )


def test_resolution_appends_outcome_without_mutating_forecast(tmp_path: Path) -> None:
    creation, store = _created(tmp_path)
    provider = OutcomeProvider(_outcome_snapshot(creation))
    forecast_before = store.read_bytes(creation.creation.forecast_ref)

    resolved = resolve_development_outcome(
        creation=creation,
        state_root=tmp_path / 'state',
        market_provider=provider,
        artifact_store=store,
        evidence_context=_evidence_context(),
        clock=lambda: NOW,
    )

    assert provider.requests[0].purpose == 'outcome'
    assert tuple(row.session for row in provider.snapshot.observations) == (
        creation.origin.forecast_sessions
    )
    assert store.read_bytes(creation.creation.forecast_ref) == forecast_before
    assert resolved.chain_verified is True
    assert resolved.analysis_row['structural_status'] in {'PASSED', 'FAILED'}
    required = {
        'structural_status',
        'structural_diagnostics',
        'raw_close_return_forecast',
        'projected_path',
        'projected_path_status',
        'reliability_features',
        'realized_outcome',
        'forecast_error',
    }
    assert required <= resolved.analysis_row.keys()
    assert 'observations' not in json.dumps(resolved.analysis_row)
    projected = resolved.analysis_row['projected_path']
    assert isinstance(projected, dict)
    assert projected['implied_return_unchanged'] is True
    individual = resolved.analysis_row['individual_path_results']
    assert isinstance(individual, list)
    assert len(individual) == 9
    assert {
        'path_id',
        'context_length',
        'sampling_seed',
        'structural_valid',
        'predicted_log_return',
        'absolute_return_error',
        'direction_correct',
    } <= individual[0].keys()
    assert all(item['absolute_return_error'] >= 0.0 for item in individual)


def test_tampered_creation_blocks_outcome_provider_access(tmp_path: Path) -> None:
    creation, store = _created(tmp_path)
    payload = creation.model_dump(mode='python')
    payload['creation']['seal_ref'] = ArtifactRef(
        sha256='0' * 64,
        size_bytes=1,
        media_type='application/json',
        relative_path='sha256/00/' + '0' * 64,
    )
    tampered = DevelopmentForecastCreation.model_validate(payload)
    provider = OutcomeProvider(_outcome_snapshot(creation))

    with pytest.raises(ValueError, match='seal'):
        resolve_development_outcome(
            creation=tampered,
            state_root=tmp_path / 'state',
            market_provider=provider,
            artifact_store=store,
            evidence_context=_evidence_context(),
            clock=lambda: NOW,
        )
    assert provider.requests == []


def test_real_executor_terminal_verifier_fails_closed_on_analysis_tampering(
    tmp_path: Path,
) -> None:
    creation, store = _created(tmp_path)
    resolved = resolve_development_outcome(
        creation=creation,
        state_root=tmp_path / 'state',
        market_provider=OutcomeProvider(_outcome_snapshot(creation)),
        artifact_store=store,
        evidence_context=_evidence_context(),
        clock=lambda: NOW,
    )
    origin_root = tmp_path / 'state' / 'origins' / creation.origin.origin_id
    copytree(tmp_path / 'state' / 'artifacts', origin_root / 'artifacts')
    analysis_path = origin_root / 'analysis-row.json'
    summary = OriginExecutionSummary(
        schema_version='sentinel-phase3a-origin-execution-v1',
        origin_id=creation.origin.origin_id,
        terminal_status='completed',
        request_success_count=9,
        request_count=9,
        ensemble_latency_seconds=1.0,
        cache_bytes=1,
        process_stable=True,
        terminal_record_sha256=sha256_bytes(analysis_path.read_bytes()),
    )
    (origin_root / 'origin-execution.json').write_bytes(
        canonical_json_bytes(summary.model_dump(mode='json'))
    )
    executor = object.__new__(RealDevelopmentExecutor)
    executor.state_root = (tmp_path / 'state').resolve(strict=True)

    assert resolved.chain_verified is True
    assert executor.verify_terminal_summary(creation.origin, summary)

    analysis_path.write_bytes(b'{}')
    assert not executor.verify_terminal_summary(creation.origin, summary)


def test_terminal_failure_is_explicit_and_content_hashed(tmp_path: Path) -> None:
    origin = build_development_manifest().origins[0]
    failure = record_terminal_failure(
        origin=origin,
        state_root=tmp_path,
        code='REAL_INFERENCE_FAILURE',
        stage='FORECAST',
        message='synthetic provider failure',
        attempt_sha256=('a' * 64, 'b' * 64),
        occurred_at=NOW,
    )

    assert failure.terminal_status == 'failed'
    assert failure.analysis_status == 'unavailable'
    assert failure.code == 'REAL_INFERENCE_FAILURE'
    assert failure.stage == 'FORECAST'
    assert failure.attempt_sha256 == ('a' * 64, 'b' * 64)
    assert len(failure.record_sha256) == 64
