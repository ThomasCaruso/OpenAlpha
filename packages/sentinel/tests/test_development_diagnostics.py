from datetime import UTC, date, datetime, timedelta

import pytest
from openalpha_sentinel.contracts import ForecastPath, OHLCVObservation
from openalpha_sentinel.development_diagnostics import (
    DEVELOPMENT_DIAGNOSTIC_NAMES,
    ResolvedForecastError,
    compute_development_nonstructural_diagnostics,
    recent_model_error,
)
from openalpha_sentinel.ensemble import EnsembleMember, assemble_ensemble

CUTOFF = date(2024, 7, 5)
FUTURE = tuple(date(2024, 7, 8) + timedelta(days=index) for index in range(5))
CREATED_AT = datetime(2026, 7, 30, tzinfo=UTC)


def _history() -> tuple[OHLCVObservation, ...]:
    return tuple(
        OHLCVObservation(
            session=CUTOFF - timedelta(days=511 - index),
            timestamp=datetime.combine(
                CUTOFF - timedelta(days=511 - index),
                datetime.min.time(),
                tzinfo=UTC,
            ),
            open=400.0 + 0.1 * index,
            high=401.0 + 0.1 * index,
            low=399.0 + 0.1 * index,
            close=400.5 + 0.1 * index + 0.2 * ((index % 7) - 3),
            volume=50_000_000.0 + index,
        )
        for index in range(512)
    )


def _ensemble(cutoff_close: float):
    members = []
    for context, context_shift in ((128, -0.01), (256, 0.0), (512, 0.01)):
        for seed, seed_shift in zip((1729, 2027, 7919), (-0.002, 0.0, 0.002)):
            closes = tuple(
                cutoff_close * (1.0 + (context_shift + seed_shift) * (step + 1) / 5)
                for step in range(5)
            )
            rows = tuple(
                OHLCVObservation(
                    session=session,
                    timestamp=datetime.combine(session, datetime.min.time(), tzinfo=UTC),
                    open=close,
                    high=close + 1.0,
                    low=close - 1.0,
                    close=close,
                    volume=50_000_000.0,
                )
                for session, close in zip(FUTURE, closes)
            )
            members.append(
                EnsembleMember(
                    context_length=context,
                    sampling_seed=seed,
                    path=ForecastPath(
                        path_id=f'path-{context}-{seed}',
                        observations=rows,
                        canonical_sha256=f'{context + seed:064x}',
                    ),
                    inference_duration_ms=10.0,
                )
            )
    return assemble_ensemble(
        cutoff_close=cutoff_close,
        forecast_sessions=FUTURE,
        members=tuple(members),
    )


def _prior(index: int, *, outcome_end: date | None = None) -> ResolvedForecastError:
    return ResolvedForecastError(
        origin_id=f'prior-{index}',
        asset='SPY',
        forecast_cutoff=CUTOFF - timedelta(days=80 - 7 * index),
        outcome_end=outcome_end or CUTOFF - timedelta(days=70 - 7 * index),
        resolved_at=CREATED_AT - timedelta(days=10 - index),
        absolute_error=index / 100.0,
        resolved_evidence_sha256=f'{index + 1:064x}',
    )


def test_recent_error_requires_eight_prior_resolved_forecasts() -> None:
    missing = recent_model_error((0.1,) * 7)
    assert missing.status == 'not_computable'
    assert missing.reason == 'INSUFFICIENT_PRIOR_RESOLVED_FORECASTS'
    available = recent_model_error(tuple(float(index) / 100 for index in range(9)))
    assert available.value == pytest.approx(
        sum(float(index) / 100 for index in range(1, 9)) / 8
    )


def test_development_vector_has_exact_locked_nonstructural_diagnostics() -> None:
    history = _history()
    vector = compute_development_nonstructural_diagnostics(
        symbol='SPY',
        cutoff=CUTOFF,
        context=history,
        ensemble=_ensemble(history[-1].close),
        created_at=CREATED_AT,
        prior_resolved=tuple(_prior(index) for index in range(8)),
    )
    by_name = {item.name: item for item in vector.values}

    assert tuple(by_name) == DEVELOPMENT_DIAGNOSTIC_NAMES
    assert len(by_name) == 14
    assert 'UNCERTAINTY_MISCALIBRATION' not in by_name
    assert by_name['RECENT_MODEL_ERROR'].status == 'available'
    assert by_name['RECENT_MODEL_ERROR'].value == pytest.approx(0.035)


def test_recent_error_accepts_cutoff_outcome_and_rejects_future_outcome() -> None:
    history = _history()
    causal = (*tuple(_prior(index) for index in range(7)), _prior(7, outcome_end=CUTOFF))
    diagnostics = compute_development_nonstructural_diagnostics(
        symbol='SPY',
        cutoff=CUTOFF,
        created_at=CREATED_AT,
        context=history,
        ensemble=_ensemble(history[-1].close),
        prior_resolved=causal,
    )
    by_name = {item.name: item for item in diagnostics.values}
    assert by_name['RECENT_MODEL_ERROR'].status == 'available'

    invalid = (
        *tuple(_prior(index) for index in range(7)),
        _prior(7, outcome_end=CUTOFF + timedelta(days=1)),
    )
    with pytest.raises(ValueError, match='no later than the current cutoff'):
        compute_development_nonstructural_diagnostics(
            symbol='SPY',
            cutoff=CUTOFF,
            context=history,
            ensemble=_ensemble(history[-1].close),
            created_at=CREATED_AT,
            prior_resolved=invalid,
        )
