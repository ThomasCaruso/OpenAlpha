from datetime import UTC, date, datetime, timedelta

import pytest
from openalpha_sentinel.contracts import (
    ForecastFailure,
    ForecastPath,
    ForecastRequest,
    ForecastResponse,
    OHLCVObservation,
)
from pydantic import ValidationError

SHA = "a" * 64
CUTOFF = date(2024, 7, 5)
OUTCOME_SESSIONS = (
    date(2024, 7, 8),
    date(2024, 7, 9),
    date(2024, 7, 10),
    date(2024, 7, 11),
    date(2024, 7, 12),
)


def _observation(session: date, close: float = 100.0) -> OHLCVObservation:
    return OHLCVObservation(
        session=session,
        timestamp=datetime.combine(session, datetime.min.time(), tzinfo=UTC),
        open=close - 0.5,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=1_000_000.0,
    )


def _request() -> ForecastRequest:
    start = CUTOFF - timedelta(days=599)
    observations = tuple(_observation(start + timedelta(days=index)) for index in range(600))
    causal = tuple(observation for observation in observations if observation.session <= CUTOFF)
    return ForecastRequest(
        model_repository="NeoQuasar/Kronos-mini",
        model_revision="f4e68697d9d5aed55cef5c96aabc3376bcad9f81",
        tokenizer_repository="NeoQuasar/Kronos-Tokenizer-2k",
        tokenizer_revision="26966d0035065a0cae0ebad7af8ece35bc1fb51c",
        source_revision="67b630e67f6a18c9e9be918d9b4337c960db1e9a",
        symbol="SPY",
        cutoff=CUTOFF,
        forecast_sessions=OUTCOME_SESSIONS,
        observations=causal[-512:],
        context_length=512,
        forecast_horizon=5,
        sampling_seed=1729,
        temperature=1.0,
        top_p=0.9,
        sample_count=1,
        data_snapshot_sha256=SHA,
        experiment_sha256="b" * 64,
        created_at=datetime(2026, 7, 30, tzinfo=UTC),
    )


def test_request_locks_the_phase_2_inference_contract() -> None:
    request = _request()

    assert len(request.observations) == 512
    assert request.observations[-1].session == CUTOFF
    assert request.forecast_sessions == OUTCOME_SESSIONS
    assert request.sample_count == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("context_length", 64),
        ("forecast_horizon", 4),
        ("sampling_seed", 1),
        ("temperature", 0.8),
        ("top_p", 0.8),
        ("sample_count", 2),
    ],
)
def test_request_rejects_undeclared_inference_settings(field: str, value: object) -> None:
    payload = _request().model_dump(mode="python")
    payload[field] = value

    with pytest.raises(ValidationError):
        ForecastRequest.model_validate(payload)


def test_request_rejects_future_or_unsorted_inputs() -> None:
    payload = _request().model_dump(mode="python")
    rows = list(payload["observations"])
    rows[-1] = _observation(date(2024, 7, 8))
    payload["observations"] = tuple(rows)

    with pytest.raises(ValidationError, match="cutoff"):
        ForecastRequest.model_validate(payload)

    payload = _request().model_dump(mode="python")
    payload["observations"] = tuple(reversed(payload["observations"]))
    with pytest.raises(ValidationError, match="increasing"):
        ForecastRequest.model_validate(payload)


def test_ohlcv_requires_timezone_and_finite_values() -> None:
    payload = _observation(CUTOFF).model_dump(mode="python")
    payload["timestamp"] = datetime(2024, 7, 5, tzinfo=UTC).replace(tzinfo=None)
    with pytest.raises(ValidationError, match="timezone"):
        OHLCVObservation.model_validate(payload)

    payload = _observation(CUTOFF).model_dump(mode="python")
    payload["volume"] = float("nan")
    with pytest.raises(ValidationError):
        OHLCVObservation.model_validate(payload)


def test_request_rejects_invalid_input_ohlcv_but_paths_can_preserve_model_anomalies() -> None:
    payload = _request().model_dump(mode="python")
    rows = list(payload["observations"])
    invalid_input = dict(rows[-1])
    invalid_input["high"] = invalid_input["low"] - 1.0
    rows[-1] = OHLCVObservation.model_validate(invalid_input)
    payload["observations"] = tuple(rows)
    with pytest.raises(ValidationError, match="input.*high"):
        ForecastRequest.model_validate(payload)

    path = ForecastPath(
        path_id="anomalous-model-output",
        observations=tuple(
            OHLCVObservation.model_validate(
                {
                    **invalid_input,
                    "session": session,
                    "timestamp": datetime.combine(
                        session,
                        datetime.min.time(),
                        tzinfo=UTC,
                    ),
                }
            )
            for session in OUTCOME_SESSIONS
        ),
        canonical_sha256="d" * 64,
    )
    assert path.observations[0].high < path.observations[0].low


def test_success_response_has_one_preserved_path_and_no_failure() -> None:
    path = ForecastPath(
        path_id="path-512-1729",
        observations=tuple(_observation(session, 101.0) for session in OUTCOME_SESSIONS),
        canonical_sha256="c" * 64,
    )
    response = ForecastResponse(
        provider_id="kronos-local",
        checkpoint_id="NeoQuasar/Kronos-mini@f4e68697",
        request_id="req-1",
        generated_paths=(path,),
        inference_duration_ms=12.5,
        failure=None,
    )

    assert response.generated_paths == (path,)
    assert response.failure is None

    payload = response.model_dump(mode="python")
    payload["failure"] = ForecastFailure(code="INFERENCE_FAILED", message="failed")
    with pytest.raises(ValidationError, match="failure"):
        ForecastResponse.model_validate(payload)


def test_failure_response_has_no_generated_path() -> None:
    response = ForecastResponse(
        provider_id="kronos-local",
        checkpoint_id="NeoQuasar/Kronos-mini@f4e68697",
        request_id="req-2",
        generated_paths=(),
        inference_duration_ms=3.0,
        failure=ForecastFailure(code="MODEL_LOAD_FAILED", message="not available"),
    )

    assert response.generated_paths == ()
