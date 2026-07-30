from datetime import UTC, date, datetime, timedelta

import exchange_calendars as xcals
import numpy as np
from openalpha_sentinel.contracts import ForecastPath, OHLCVObservation
from openalpha_sentinel.diagnostics import compute_phase_2_diagnostics
from openalpha_sentinel.ensemble import EnsembleMember, assemble_ensemble

CUTOFF = date(2024, 7, 5)
FUTURE = (
    date(2024, 7, 8),
    date(2024, 7, 9),
    date(2024, 7, 10),
    date(2024, 7, 11),
    date(2024, 7, 12),
)


def _history() -> tuple[OHLCVObservation, ...]:
    sessions = xcals.get_calendar("XNYS").sessions_in_range("2022-06-01", "2024-07-05")
    observations: list[OHLCVObservation] = []
    close = 400.0
    for index, timestamp in enumerate(sessions):
        close *= float(np.exp(0.0002 + 0.004 * np.sin(index / 7.0)))
        session = timestamp.date()
        observations.append(
            OHLCVObservation(
                session=session,
                timestamp=datetime.combine(session, datetime.min.time(), tzinfo=UTC),
                open=close * (1.0 + 0.001 * np.cos(index / 5.0)),
                high=close * 1.003,
                low=close * 0.997,
                close=close,
                volume=50_000_000.0 + index,
            )
        )
    return tuple(observations)


def _ensemble(cutoff_close: float):
    members: list[EnsembleMember] = []
    for context, context_shift in ((128, -0.01), (256, 0.0), (512, 0.01)):
        for seed, seed_shift in zip((1729, 2027, 7919), (-0.002, 0.0, 0.002)):
            final = cutoff_close * (1.0 + context_shift + seed_shift)
            closes = tuple(
                cutoff_close + (final - cutoff_close) * (step + 1) / 5 for step in range(5)
            )
            path = ForecastPath(
                path_id=f"path-{context}-{seed}",
                observations=tuple(
                    OHLCVObservation(
                        session=session,
                        timestamp=datetime.combine(session, datetime.min.time(), tzinfo=UTC),
                        open=close,
                        high=close * 1.001,
                        low=close * 0.999,
                        close=close,
                        volume=50_000_000.0,
                    )
                    for session, close in zip(FUTURE, closes)
                ),
                canonical_sha256=f"{context + seed:064x}"[-64:],
            )
            members.append(
                EnsembleMember(
                    context_length=context,
                    sampling_seed=seed,
                    path=path,
                    inference_duration_ms=10.0,
                )
            )
    return assemble_ensemble(
        cutoff_close=cutoff_close,
        forecast_sessions=FUTURE,
        members=tuple(members),
    )


def test_phase_2_diagnostics_are_causal_interpretable_and_explicitly_missing() -> None:
    history = _history()
    vector = compute_phase_2_diagnostics(
        symbol="SPY",
        cutoff=CUTOFF,
        context=history,
        ensemble=_ensemble(history[-1].close),
        created_at=datetime(2026, 7, 30, tzinfo=UTC),
    )
    by_name = {item.name: item for item in vector.values}

    assert vector.causal_through == CUTOFF
    assert len(vector.values) == 15
    for name in (
        "DIRECTIONAL_AGREEMENT",
        "RETURN_DISPERSION",
        "PATH_DISPERSION",
        "CONTEXT_DIRECTION_AGREEMENT",
        "CONTEXT_RETURN_SPREAD",
        "BASELINE_DISAGREEMENT",
        "RECENT_VOLATILITY",
        "VOLATILITY_CHANGE",
        "TREND_STRENGTH",
        "GAP_OR_OUTLIER_SCORE",
        "HISTORICAL_ANALOGUE_DISTANCE",
        "ANALOGUE_OUTCOME_DISPERSION",
        "HORIZON_PATH_DIVERGENCE",
    ):
        assert by_name[name].status == "available"
        assert by_name[name].value is not None
        assert by_name[name].available_before_outcome is True

    assert by_name["RECENT_MODEL_ERROR"].status == "not_computable"
    assert by_name["RECENT_MODEL_ERROR"].value is None
    assert by_name["RECENT_MODEL_ERROR"].reason == "NO_PRIOR_RESOLVED_FORECASTS"
    assert by_name["UNCERTAINTY_MISCALIBRATION"].status == "not_computable"
    assert by_name["UNCERTAINTY_MISCALIBRATION"].value is None
    assert by_name["UNCERTAINTY_MISCALIBRATION"].reason == "NO_PRIOR_CALIBRATION_SAMPLE"


def test_diagnostics_do_not_accept_post_cutoff_context() -> None:
    history = _history()
    future_row = history[-1].model_dump(mode="python")
    future_row["session"] = CUTOFF + timedelta(days=3)
    future_row["timestamp"] = datetime(2024, 7, 8, tzinfo=UTC)
    invalid = (*history, OHLCVObservation.model_validate(future_row))

    try:
        compute_phase_2_diagnostics(
            symbol="SPY",
            cutoff=CUTOFF,
            context=invalid,
            ensemble=_ensemble(history[-1].close),
            created_at=datetime(2026, 7, 30, tzinfo=UTC),
        )
    except ValueError as error:
        assert "cutoff" in str(error)
    else:
        raise AssertionError("post-cutoff context was accepted")
