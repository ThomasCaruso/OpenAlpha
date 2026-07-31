import math
from datetime import UTC, date, datetime, timedelta

import pytest
from openalpha_sentinel.structural_validity import (
    AuditCandle,
    constraint_projection_v0,
    summarize_structural_validity,
    validate_forecast_path,
)

SESSIONS = tuple(date(2024, 7, 8) + timedelta(days=offset) for offset in range(5))


def _candle(
    session: datetime | date | None,
    *,
    open_value: float = 100.0,
    high: float = 102.0,
    low: float = 99.0,
    close: float = 101.0,
    volume: float | None = 1_000.0,
) -> AuditCandle:
    return AuditCandle(
        session=session,
        open=open_value,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def _valid_path(path_id: str = "valid"):
    return validate_forecast_path(
        path_id=path_id,
        candles=tuple(_candle(session) for session in SESSIONS),
        expected_sessions=SESSIONS,
        cutoff_close=100.0,
        cutoff_volume=1_000.0,
    )


def test_valid_path_has_no_structural_violations() -> None:
    result = _valid_path()

    assert result.valid is True
    assert result.invalid_candle_count == 0
    assert result.earliest_invalid_horizon_step is None
    assert result.violations == ()


def test_distinct_intraday_timestamps_remain_distinct() -> None:
    timestamps = tuple(
        datetime(2024, 7, 3, 9, 55, tzinfo=UTC) + timedelta(minutes=5 * index) for index in range(5)
    )
    candles = tuple(_candle(timestamp) for timestamp in timestamps)

    result = validate_forecast_path(
        path_id="intraday",
        candles=candles,
        expected_sessions=timestamps,
        cutoff_close=100.0,
        cutoff_volume=1_000.0,
    )

    assert result.valid is True
    assert result.violations == ()


def test_price_constraints_use_cutoff_close_for_severity() -> None:
    candles = tuple(
        _candle(
            session,
            open_value=100.0,
            high=99.0 if index == 0 else 102.0,
            low=101.0 if index == 0 else 99.0,
            close=102.0 if index == 0 else 101.0,
        )
        for index, session in enumerate(SESSIONS)
    )

    result = validate_forecast_path(
        path_id="permuted",
        candles=candles,
        expected_sessions=SESSIONS,
        cutoff_close=100.0,
        cutoff_volume=1_000.0,
    )
    by_code = {item.code: item for item in result.violations}

    assert result.valid is False
    assert result.invalid_candle_count == 1
    assert result.earliest_invalid_horizon_step == 1
    assert set(by_code) == {
        "HIGH_BELOW_OPEN",
        "HIGH_BELOW_CLOSE",
        "HIGH_BELOW_LOW",
        "LOW_ABOVE_OPEN",
    }
    assert by_code["HIGH_BELOW_CLOSE"].observed_gap == pytest.approx(3.0)
    assert by_code["HIGH_BELOW_CLOSE"].normalized_severity == pytest.approx(0.03)
    assert all(item.existed_before_outcome for item in result.violations)


def test_nonfinite_nonpositive_and_negative_volume_remain_explicit() -> None:
    candles = [_candle(session) for session in SESSIONS]
    candles[0] = _candle(
        SESSIONS[0],
        open_value=0.0,
        high=math.nan,
        low=-1.0,
        close=math.inf,
        volume=-10.0,
    )

    result = validate_forecast_path(
        path_id="nonfinite",
        candles=tuple(candles),
        expected_sessions=SESSIONS,
        cutoff_close=100.0,
        cutoff_volume=1_000.0,
    )
    codes = {item.code for item in result.violations}

    assert codes >= {
        "NONPOSITIVE_OPEN",
        "NONFINITE_HIGH",
        "NONPOSITIVE_LOW",
        "NONFINITE_CLOSE",
        "NEGATIVE_VOLUME",
    }
    assert (
        next(
            item for item in result.violations if item.code == "NONFINITE_HIGH"
        ).normalized_severity
        is None
    )
    assert next(
        item for item in result.violations if item.code == "NEGATIVE_VOLUME"
    ).normalized_severity == pytest.approx(0.01)


def test_horizon_and_timestamp_contracts_are_not_positionally_repaired() -> None:
    candles = tuple(_candle(session) for session in SESSIONS[:-1])
    shifted = (*candles[:-1], _candle(SESSIONS[0]))

    result = validate_forecast_path(
        path_id="misaligned",
        candles=shifted,
        expected_sessions=SESSIONS,
        cutoff_close=100.0,
        cutoff_volume=1_000.0,
    )
    codes = [item.code for item in result.violations]

    assert "HORIZON_LENGTH_MISMATCH" in codes
    assert "DUPLICATE_TIMESTAMP" in codes
    assert "UNEXPECTED_TIMESTAMP" in codes
    assert "MISSING_TIMESTAMP" in codes


def test_summary_keeps_path_candle_count_and_severity_distinct() -> None:
    invalid_candles = tuple(
        _candle(
            session,
            high=99.0 if index == 0 else 102.0,
            low=98.0 if index == 0 else 99.0,
            close=99.0 if index == 0 else 101.0,
        )
        for index, session in enumerate(SESSIONS)
    )
    invalid = validate_forecast_path(
        path_id="invalid",
        candles=invalid_candles,
        expected_sessions=SESSIONS,
        cutoff_close=100.0,
        cutoff_volume=1_000.0,
    )

    diagnostics = summarize_structural_validity((_valid_path(), invalid))
    values = {item.name: item.value for item in diagnostics.values}

    assert values == {
        "INVALID_PATH_FRACTION": 0.5,
        "INVALID_CANDLE_FRACTION": 0.1,
        "TOTAL_CONSTRAINT_VIOLATIONS": 1,
        "MAX_CONSTRAINT_VIOLATION_SEVERITY": pytest.approx(0.01),
        "MEAN_CONSTRAINT_VIOLATION_SEVERITY": pytest.approx(0.01),
        "EARLIEST_INVALID_HORIZON_STEP": 1,
        "HIGH_LOW_INVERSION_COUNT": 0,
        "HIGH_BELOW_BODY_COUNT": 1,
        "LOW_ABOVE_BODY_COUNT": 0,
        "NONFINITE_OUTPUT_COUNT": 0,
        "NONPOSITIVE_PRICE_COUNT": 0,
    }


def test_constraint_projection_preserves_open_close_and_return() -> None:
    candles = tuple(
        _candle(
            session,
            open_value=100.0,
            high=99.0 if index == 0 else 110.0,
            low=101.0 if index == 0 else 99.0,
            close=102.0 if index == 0 else 101.0 + index,
        )
        for index, session in enumerate(SESSIONS)
    )

    projection = constraint_projection_v0(
        path_id="repair",
        candles=candles,
        cutoff_close=100.0,
    )
    projected_validity = validate_forecast_path(
        path_id="projected",
        candles=projection.projected_candles,
        expected_sessions=SESSIONS,
        cutoff_close=100.0,
        cutoff_volume=1_000.0,
    )

    assert projected_validity.valid is True
    assert [item.open for item in projection.original_candles] == [
        item.open for item in projection.projected_candles
    ]
    assert [item.close for item in projection.original_candles] == [
        item.close for item in projection.projected_candles
    ]
    assert projection.original_log_return == projection.projected_log_return
    assert {(item.step, item.field) for item in projection.adjustments} == {
        (1, "high"),
        (1, "low"),
    }
    assert projection.adjustments[0].normalized_adjustment > 0.0
