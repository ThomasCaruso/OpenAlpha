from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from openalpha_research.errors import ArtifactIntegrityError
from openalpha_sentinel.contracts import (
    MODEL_REPOSITORY,
    MODEL_REVISION,
    SOURCE_REVISION,
    TOKENIZER_REPOSITORY,
    TOKENIZER_REVISION,
    ForecastPath,
    ForecastRequest,
    ForecastResponse,
    OHLCVObservation,
)
from openalpha_sentinel.development_serialization import canonical_json_bytes, sha256_bytes
from openalpha_sentinel.inference_cache import InferenceCache, InferenceCacheKey

SESSIONS = tuple(date(2024, 7, 8) + timedelta(days=index) for index in range(5))


def _request(
    *,
    created_at: datetime = datetime(2026, 7, 30, tzinfo=UTC),
    experiment_sha256: str = 'b' * 64,
) -> ForecastRequest:
    cutoff = date(2024, 7, 5)
    observations = tuple(
        OHLCVObservation(
            session=cutoff - timedelta(days=511 - index),
            timestamp=datetime.combine(
                cutoff - timedelta(days=511 - index),
                datetime.min.time(),
                tzinfo=UTC,
            ),
            open=100.0 + index,
            high=101.0 + index,
            low=99.0 + index,
            close=100.5 + index,
            volume=1_000_000.0,
        )
        for index in range(512)
    )
    return ForecastRequest(
        model_repository=MODEL_REPOSITORY,
        model_revision=MODEL_REVISION,
        tokenizer_repository=TOKENIZER_REPOSITORY,
        tokenizer_revision=TOKENIZER_REVISION,
        source_revision=SOURCE_REVISION,
        symbol='SPY',
        cutoff=cutoff,
        forecast_sessions=SESSIONS,
        observations=observations,
        context_length=512,
        forecast_horizon=5,
        sampling_seed=1729,
        temperature=1.0,
        top_p=0.9,
        sample_count=1,
        data_snapshot_sha256='a' * 64,
        experiment_sha256=experiment_sha256,
        created_at=created_at,
    )


def _response(request: ForecastRequest) -> ForecastResponse:
    rows = tuple(
        OHLCVObservation(
            session=session,
            timestamp=datetime.combine(session, datetime.min.time(), tzinfo=UTC),
            open=611.5 + step,
            high=612.5 + step,
            low=610.5 + step,
            close=612.0 + step,
            volume=1_000_000.0,
        )
        for step, session in enumerate(request.forecast_sessions)
    )
    path_hash = sha256_bytes(
        canonical_json_bytes([row.model_dump(mode='json') for row in rows])
    )
    return ForecastResponse(
        provider_id='kronos-local',
        checkpoint_id=MODEL_REVISION,
        request_id='request-1',
        generated_paths=(
            ForecastPath(
                path_id='path-512-1729',
                observations=rows,
                canonical_sha256=path_hash,
            ),
        ),
        inference_duration_ms=100.0,
        failure=None,
    )


def test_key_excludes_run_metadata_but_includes_inference_identity() -> None:
    first = InferenceCacheKey.from_request(_request())
    second = InferenceCacheKey.from_request(
        _request(
            created_at=datetime(2026, 7, 31, tzinfo=UTC),
            experiment_sha256='c' * 64,
        )
    )
    assert first.canonical_sha256 == second.canonical_sha256

    for field, value in (
        ('data_snapshot_sha256', 'd' * 64),
        ('forecast_sessions', tuple(date(2024, 7, 9) + timedelta(days=i) for i in range(5))),
        ('context_length', 256),
        ('sampling_seed', 2027),
        ('temperature', 0.95),
        ('top_p', 0.85),
        ('sample_count', 2),
        ('forecast_horizon', 6),
        ('model_revision', 'e' * 40),
    ):
        payload = first.model_dump(mode='python', exclude={'canonical_sha256'})
        changed = InferenceCacheKey.model_validate({**payload, field: value})
        assert changed.canonical_sha256 != first.canonical_sha256


def test_cache_round_trip_detects_corruption(tmp_path: Path) -> None:
    request = _request()
    response = _response(request)
    cache = InferenceCache(tmp_path)

    assert cache.get_verified(request) is None
    descriptor = cache.put(request, response)
    assert cache.get_verified(request) == response

    path = cache.artifact_store.path_for(descriptor.response_ref)
    path.write_bytes(b'corrupt')
    with pytest.raises(ArtifactIntegrityError):
        cache.get_verified(request)


def test_cache_rejects_response_with_wrong_sessions(tmp_path: Path) -> None:
    request = _request()
    response = _response(request)
    payload = response.model_dump(mode='python')
    path_payload = payload['generated_paths'][0]
    observations = list(path_payload['observations'])
    observations[0] = {
        **observations[0],
        'session': date(2024, 7, 7),
        'timestamp': datetime(2024, 7, 7, tzinfo=UTC),
    }
    path_payload['observations'] = tuple(observations)

    with pytest.raises(ValueError, match='sessions'):
        InferenceCache(tmp_path).put(request, ForecastResponse.model_validate(payload))


def test_cache_accepts_only_revision_or_official_qualified_checkpoint(
    tmp_path: Path,
) -> None:
    request = _request()
    payload = _response(request).model_dump(mode='python')
    payload['checkpoint_id'] = f'{MODEL_REPOSITORY}@{MODEL_REVISION}'
    qualified = ForecastResponse.model_validate(payload)
    cache = InferenceCache(tmp_path / 'accepted')

    cache.put(request, qualified)
    assert cache.get_verified(request) == qualified

    payload['checkpoint_id'] = f'Other/Model@{MODEL_REVISION}'
    with pytest.raises(ValueError, match='checkpoint'):
        InferenceCache(tmp_path / 'rejected').put(
            request,
            ForecastResponse.model_validate(payload),
        )
