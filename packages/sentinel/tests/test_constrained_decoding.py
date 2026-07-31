from __future__ import annotations

from datetime import date

import pytest
from openalpha_sentinel.constrained_decoding import (
    CandidateTokenPair,
    choose_valid_candidate,
    deterministic_auxiliary_fraction,
    terminal_projection,
)
from openalpha_sentinel.structural_validity import AuditCandle, validate_forecast_path

SESSIONS = tuple(date(2024, 7, day) for day in (8, 9, 10, 11, 12))


def _candle(
    step: int,
    *,
    open_value: float = 100.0,
    high: float = 102.0,
    low: float = 99.0,
    close: float = 101.0,
) -> AuditCandle:
    return AuditCandle(
        session=SESSIONS[step - 1],
        open=open_value,
        high=high,
        low=low,
        close=close,
        volume=1_000.0,
    )


def test_terminal_projection_is_separate_valid_and_preserves_open_close() -> None:
    raw = (
        _candle(1, high=100.5, close=101.0),
        *(_candle(step) for step in range(2, 6)),
    )

    result = terminal_projection(
        path_id="raw-path",
        candles=raw,
        expected_sessions=SESSIONS,
        cutoff_close=100.0,
        cutoff_volume=900.0,
    )

    assert result.method == "TERMINAL_PROJECTION"
    assert result.raw_validity.valid is False
    assert result.projected_validity.valid is True
    assert result.raw_candles == raw
    assert tuple(item.open for item in result.projected_candles) == tuple(
        item.open for item in raw
    )
    assert tuple(item.close for item in result.projected_candles) == tuple(
        item.close for item in raw
    )
    assert result.raw_log_return == result.projected_log_return


def test_valid_candidate_selection_rejects_invalid_pair_and_is_deterministic() -> None:
    candidates = (
        CandidateTokenPair(
            coarse_token=11,
            fine_token=21,
            joint_log_probability=-0.1,
            rank=1,
            decoded_candle=_candle(1, high=100.0, close=101.0),
            valid=False,
        ),
        CandidateTokenPair(
            coarse_token=12,
            fine_token=22,
            joint_log_probability=-1.0,
            rank=2,
            decoded_candle=_candle(1),
            valid=True,
        ),
        CandidateTokenPair(
            coarse_token=13,
            fine_token=23,
            joint_log_probability=-2.0,
            rank=3,
            decoded_candle=_candle(1, close=100.5),
            valid=True,
        ),
    )
    fraction = deterministic_auxiliary_fraction(
        origin_id="sentinel-v1-SPY-2024-07-05-h5",
        seed=1729,
        step=1,
    )

    first = choose_valid_candidate(candidates, auxiliary_fraction=fraction)
    second = choose_valid_candidate(candidates, auxiliary_fraction=fraction)

    assert first == second
    assert first.selected.valid is True
    assert first.selected.coarse_token in {12, 13}
    assert first.candidates_considered == 3
    assert first.rejection_count == 1
    assert first.valid_candidate_count == 2


def test_candidate_selection_requires_explicit_fallback_when_none_valid() -> None:
    candidates = (
        CandidateTokenPair(
            coarse_token=11,
            fine_token=21,
            joint_log_probability=-0.1,
            rank=1,
            decoded_candle=_candle(1, high=100.0, close=101.0),
            valid=False,
        ),
    )

    with pytest.raises(ValueError, match="no valid candidate"):
        choose_valid_candidate(candidates, auxiliary_fraction=0.5)


def test_existing_grammar_rejects_timestamp_shift_and_nonpositive_price() -> None:
    shifted = [_candle(step) for step in range(1, 6)]
    shifted[0] = AuditCandle(
        session=date(2024, 7, 9),
        open=0.0,
        high=102.0,
        low=99.0,
        close=101.0,
        volume=1_000.0,
    )

    result = validate_forecast_path(
        path_id="shifted",
        candles=shifted,
        expected_sessions=SESSIONS,
        cutoff_close=100.0,
        cutoff_volume=900.0,
    )

    assert result.valid is False
    assert {item.code for item in result.violations} >= {
        "NONPOSITIVE_OPEN",
        "UNEXPECTED_TIMESTAMP",
        "MISSING_TIMESTAMP",
        "DUPLICATE_TIMESTAMP",
    }
