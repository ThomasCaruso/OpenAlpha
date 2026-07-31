from datetime import UTC, date, datetime
from pathlib import Path

import exchange_calendars as xcals
from openalpha_research import EnvironmentMetadata, LocalArtifactStore
from openalpha_sentinel.contracts import (
    MODEL_REVISION,
    ForecastPath,
    ForecastResponse,
    OHLCVObservation,
)
from openalpha_sentinel.development_manifest import build_development_manifest
from openalpha_sentinel.development_origin import (
    build_development_forecast_requests,
    create_development_forecast,
)
from openalpha_sentinel.development_serialization import canonical_json_bytes, sha256_bytes
from openalpha_sentinel.evidence import EvidenceContext
from openalpha_sentinel.inference_cache import InferenceCache
from openalpha_sentinel.market_data import MarketDataRequest, MarketDataSnapshot

NOW = datetime(2026, 7, 30, 20, 0, tzinfo=UTC)


class FakeMarketProvider:
    def __init__(self, snapshot: MarketDataSnapshot) -> None:
        self.snapshot = snapshot
        self.requests: list[MarketDataRequest] = []

    def fetch(self, request: MarketDataRequest) -> MarketDataSnapshot:
        self.requests.append(request)
        return self.snapshot


class FakeInference:
    def __init__(self) -> None:
        self.requests = []

    def __call__(self, requests):
        self.requests.extend(requests)
        responses = []
        for request in requests:
            rows = []
            for step, session in enumerate(request.forecast_sessions):
                close = request.observations[-1].close * (
                    1.0 + (request.context_length / 100_000.0) + step / 10_000.0
                )
                invalid = (
                    request.context_length == 512
                    and request.sampling_seed == 1729
                    and step == 1
                )
                rows.append(
                    OHLCVObservation(
                        session=session,
                        timestamp=datetime.combine(session, datetime.min.time(), tzinfo=UTC),
                        open=close,
                        high=close - 10.0 if invalid else close + 1.0,
                        low=close - 11.0 if invalid else close - 1.0,
                        close=close,
                        volume=1_000_000.0,
                    )
                )
            path_sha256 = sha256_bytes(
                canonical_json_bytes([row.model_dump(mode='json') for row in rows])
            )
            responses.append(
                ForecastResponse(
                    provider_id='fake-kronos',
                    checkpoint_id=MODEL_REVISION,
                    request_id=f'req-{request.context_length}-{request.sampling_seed}',
                    generated_paths=(
                        ForecastPath(
                            path_id=f'path-{request.context_length}-{request.sampling_seed}',
                            observations=tuple(rows),
                            canonical_sha256=path_sha256,
                        ),
                    ),
                    inference_duration_ms=10.0,
                    failure=None,
                )
            )
        return tuple(responses)


def _snapshot(cutoff: date) -> MarketDataSnapshot:
    sessions = xcals.get_calendar('XNYS').sessions_in_range('2022-06-01', cutoff.isoformat())
    observations = tuple(
        OHLCVObservation(
            session=timestamp.date(),
            timestamp=datetime.combine(timestamp.date(), datetime.min.time(), tzinfo=UTC),
            open=400.0 + index / 10.0,
            high=401.0 + index / 10.0,
            low=399.0 + index / 10.0,
            close=400.5 + index / 10.0,
            volume=50_000_000.0 + index,
        )
        for index, timestamp in enumerate(sessions)
    )
    normalized_sha256 = sha256_bytes(
        canonical_json_bytes([row.model_dump(mode='json') for row in observations])
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
        row_count=len(observations),
        first_session=observations[0].session,
        last_session=observations[-1].session,
        normalized_input_sha256=normalized_sha256,
        observations=observations,
        quality_summary=('COMPLETE_XNYS_WINDOW',),
        corporate_action_warnings=(),
        timezone_normalization='provider_session_label_to_utc_midnight',
        pagination='client_managed_not_exposed',
        synthetic=True,
    )


def _evidence_context() -> EvidenceContext:
    return EvidenceContext(
        run_id='sentinel-phase3a-spy-20240705',
        attempt_id='attempt-01',
        code_commit='f' * 40,
        code_dirty=True,
        diff_sha256='d' * 64,
        dependency_lock_sha256='e' * 64,
        environment=EnvironmentMetadata(
            os_name='Windows',
            os_version='11',
            architecture='AMD64',
            python_version='3.13.2',
            node_version=None,
            dependency_lock_sha256='e' * 64,
            container_image=None,
            hardware='CPU',
        ),
    )


def test_creation_separates_structural_and_reliability_assurance(tmp_path: Path) -> None:
    origin = build_development_manifest().origins[0]
    provider = FakeMarketProvider(_snapshot(origin.cutoff))
    inference = FakeInference()
    state_root = tmp_path / 'state'
    store = LocalArtifactStore(state_root / 'artifacts')

    result = create_development_forecast(
        origin=origin,
        state_root=state_root,
        market_provider=provider,
        forecast_many=inference,
        cache=InferenceCache(tmp_path / 'cache'),
        artifact_store=store,
        evidence_context=_evidence_context(),
        prior_resolved=(),
        clock=lambda: NOW,
    )

    assert result.structural_status == 'FAILED'
    assert len(result.individual_paths) == 9
    assert result.raw_close_return_forecast == result.ensemble.canonical_predicted_log_return
    assert result.projected_path_status == 'PASSED'
    assert result.projected_path.original_log_return == result.projected_path.projected_log_return
    assert result.creation_verified is True
    assert result.cache_hits == 0
    assert result.cache_misses == 9
    assert len(inference.requests) == 9
    assert [request.purpose for request in provider.requests] == ['forecast_context']


def test_request_builder_exposes_exact_locked_ensemble_for_replay_probe() -> None:
    origin = build_development_manifest().origins[0]
    requests = build_development_forecast_requests(
        origin=origin,
        snapshot=_snapshot(origin.cutoff),
        created_at=NOW,
    )

    assert [(item.context_length, item.sampling_seed) for item in requests] == [
        (context, seed)
        for context in (128, 256, 512)
        for seed in (1729, 2027, 7919)
    ]
    assert all(item.sample_count == 1 for item in requests)
    assert all(item.temperature == 1.0 and item.top_p == 0.9 for item in requests)
    assert all(item.forecast_sessions == origin.forecast_sessions for item in requests)
    assert all(max(row.session for row in item.observations) == origin.cutoff for item in requests)


def test_second_creation_reuses_all_verified_inference(tmp_path: Path) -> None:
    origin = build_development_manifest().origins[0]
    provider = FakeMarketProvider(_snapshot(origin.cutoff))
    inference = FakeInference()
    state_root = tmp_path / 'state'
    cache = InferenceCache(tmp_path / 'cache')
    store = LocalArtifactStore(state_root / 'artifacts')
    arguments = {
        'origin': origin,
        'state_root': state_root,
        'market_provider': provider,
        'forecast_many': inference,
        'cache': cache,
        'artifact_store': store,
        'evidence_context': _evidence_context(),
        'prior_resolved': (),
        'clock': lambda: NOW,
    }

    create_development_forecast(**arguments)
    first_request_count = len(inference.requests)
    second = create_development_forecast(**arguments)

    assert first_request_count == 9
    assert len(inference.requests) == first_request_count
    assert second.cache_hits == 9
    assert second.cache_misses == 0


def test_holdout_cutoff_is_rejected_before_provider_access(tmp_path: Path) -> None:
    origin = build_development_manifest().origins[0]
    payload = origin.model_dump(mode='python')
    payload['cutoff'] = date(2025, 7, 1)
    payload['origin_id'] = 'sentinel-v0-SPY-2025-07-01-h5'
    payload['forecast_sessions'] = (
        date(2025, 7, 2),
        date(2025, 7, 3),
        date(2025, 7, 7),
        date(2025, 7, 8),
        date(2025, 7, 9),
    )
    provider = FakeMarketProvider(_snapshot(origin.cutoff))

    try:
        create_development_forecast(
            origin=type(origin).model_validate(payload),
            state_root=tmp_path / 'state',
            market_provider=provider,
            forecast_many=FakeInference(),
            cache=InferenceCache(tmp_path / 'cache'),
            artifact_store=LocalArtifactStore(tmp_path / 'state' / 'artifacts'),
            evidence_context=_evidence_context(),
            prior_resolved=(),
            clock=lambda: NOW,
        )
    except ValueError as error:
        assert 'holdout' in str(error)
    else:
        raise AssertionError('holdout origin was accepted')
    assert provider.requests == []
