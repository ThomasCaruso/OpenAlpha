from datetime import UTC, date, datetime
from math import log

import pytest
from openalpha_sentinel.contracts import ForecastPath, OHLCVObservation
from openalpha_sentinel.ensemble import (
    EnsembleMember,
    assemble_ensemble,
    average_forecast_paths,
)
from pydantic import ValidationError

SESSIONS = (
    date(2024, 7, 8),
    date(2024, 7, 9),
    date(2024, 7, 10),
    date(2024, 7, 11),
    date(2024, 7, 12),
)


def _path(context: int, seed: int, final_close: float) -> ForecastPath:
    closes = [100.0 + (final_close - 100.0) * (step + 1) / 5 for step in range(5)]
    return ForecastPath(
        path_id=f"path-{context}-{seed}",
        observations=tuple(
            OHLCVObservation(
                session=session,
                timestamp=datetime.combine(session, datetime.min.time(), tzinfo=UTC),
                open=close,
                high=close + 0.1,
                low=close - 0.1,
                close=close,
                volume=1_000_000.0,
            )
            for session, close in zip(SESSIONS, closes)
        ),
        canonical_sha256=f"{context + seed:064x}"[-64:],
    )


def _members(stress_offset: float = 0.0) -> tuple[EnsembleMember, ...]:
    finals = {
        128: (90.0 + stress_offset, 91.0 + stress_offset, 92.0 + stress_offset),
        256: (105.0 + stress_offset, 106.0 + stress_offset, 107.0 + stress_offset),
        512: (101.0, 102.0, 103.0),
    }
    return tuple(
        EnsembleMember(
            context_length=context,
            sampling_seed=seed,
            path=_path(context, seed, final),
            inference_duration_ms=10.0,
        )
        for context, context_finals in finals.items()
        for seed, final in zip((1729, 2027, 7919), context_finals)
    )


def test_canonical_forecast_uses_only_the_three_512_paths() -> None:
    first = assemble_ensemble(cutoff_close=100.0, forecast_sessions=SESSIONS, members=_members())
    stressed = assemble_ensemble(
        cutoff_close=100.0,
        forecast_sessions=SESSIONS,
        members=_members(stress_offset=50.0),
    )

    assert first.canonical_close_path == stressed.canonical_close_path
    assert first.canonical_close_path[-1] == pytest.approx(102.0)
    assert first.canonical_predicted_log_return == pytest.approx(log(102.0 / 100.0))
    assert first.baseline_close_path == (100.0,) * 5
    assert first.baseline_predicted_log_return == 0.0
    assert len(first.individual_predicted_log_returns) == 9
    assert tuple(member.path for member in first.members if member.context_length == 512) == (
        _members()[6].path,
        _members()[7].path,
        _members()[8].path,
    )


def test_ensemble_rejects_missing_duplicate_or_wrong_sessions() -> None:
    with pytest.raises(ValueError, match="exactly nine"):
        assemble_ensemble(
            cutoff_close=100.0,
            forecast_sessions=SESSIONS,
            members=_members()[:-1],
        )

    duplicate = (*_members()[:-1], _members()[0])
    with pytest.raises(ValueError, match="Cartesian"):
        assemble_ensemble(cutoff_close=100.0, forecast_sessions=SESSIONS, members=duplicate)

    wrong = list(_members())
    wrong[0] = EnsembleMember(
        context_length=128,
        sampling_seed=1729,
        path=_path(128, 1729, 90.0).model_copy(),
        inference_duration_ms=10.0,
    )
    payload = wrong[0].path.model_dump(mode="python")
    rows = list(payload["observations"])
    rows[-1] = rows[-2]
    payload["observations"] = tuple(rows)
    with pytest.raises(ValidationError):
        ForecastPath.model_validate(payload)


def test_average_forecast_paths_uses_named_fields_and_exact_sessions() -> None:
    paths = tuple(
        ForecastPath(
            path_id=f'named-{offset}',
            observations=tuple(
                OHLCVObservation(
                    session=session,
                    timestamp=datetime.combine(session, datetime.min.time(), tzinfo=UTC),
                    open=10.0 + offset + step,
                    high=20.0 + offset + step,
                    low=5.0 + offset + step,
                    close=15.0 + offset + step,
                    volume=100.0 + 10.0 * offset + step,
                )
                for step, session in enumerate(SESSIONS)
            ),
            canonical_sha256=f'{offset + 1:064x}',
        )
        for offset in (0, 3, 6)
    )

    averaged = average_forecast_paths(paths, path_id='offline-average')

    first = averaged.observations[0]
    assert first.open == pytest.approx(13.0)
    assert first.high == pytest.approx(23.0)
    assert first.low == pytest.approx(8.0)
    assert first.close == pytest.approx(18.0)
    assert first.volume == pytest.approx(130.0)
    assert tuple(row.session for row in averaged.observations) == SESSIONS
    assert len(averaged.canonical_sha256) == 64


def test_average_forecast_paths_rejects_timestamp_mismatch() -> None:
    valid = _path(512, 1729, 101.0)
    payload = valid.model_dump(mode='python')
    rows = list(payload['observations'])
    shifted = dict(rows[0])
    shifted['session'] = date(2024, 7, 7)
    shifted['timestamp'] = datetime(2024, 7, 7, tzinfo=UTC)
    rows[0] = OHLCVObservation.model_validate(shifted)
    payload['observations'] = tuple(rows)
    mismatched = ForecastPath.model_validate(payload)

    with pytest.raises(ValueError, match='sessions'):
        average_forecast_paths((valid, mismatched), path_id='bad-average')
