from __future__ import annotations

from datetime import date

import pytest
from openalpha_sentinel.decoding_metrics import (
    barrier_event_metrics,
    diversity_metrics,
    path_distance,
    path_quality_metrics,
)
from openalpha_sentinel.structural_validity import AuditCandle

SESSIONS = tuple(date(2024, 7, day) for day in (8, 9, 10, 11, 12))


def _path(offset: float = 0.0) -> tuple[AuditCandle, ...]:
    return tuple(
        AuditCandle(
            session=session,
            open=100.0 + step + offset,
            high=102.0 + step + offset,
            low=99.0 + step + offset,
            close=101.0 + step + offset,
            volume=1_000.0,
        )
        for step, session in enumerate(SESSIONS)
    )


def test_identical_path_has_zero_errors_and_perfect_barrier_accuracy() -> None:
    actual = _path()

    quality = path_quality_metrics(
        predicted=actual,
        realized=actual,
        cutoff_close=100.0,
        cutoff_high=101.0,
        cutoff_low=99.0,
    )
    barriers = barrier_event_metrics(
        predicted=actual,
        realized=actual,
        cutoff_close=100.0,
        barriers=(0.005, 0.01, 0.02),
    )

    assert quality.normalized_ohlc_mae == 0.0
    assert quality.high_low_range_mae == 0.0
    assert quality.parkinson_volatility_error == 0.0
    assert quality.error_by_horizon_step == (0.0,) * 5
    assert all(item.positive_touch_correct for item in barriers)
    assert all(item.negative_touch_correct for item in barriers)
    assert all(item.either_touch_correct for item in barriers)


def test_diversity_detects_collapse_and_retained_variance() -> None:
    raw = (_path(), _path(1.0), _path(2.0))
    collapsed = (_path(), _path(), _path())

    raw_result = diversity_metrics(paths=raw, cutoff_close=100.0)
    collapsed_result = diversity_metrics(paths=collapsed, cutoff_close=100.0)

    assert raw_result.mean_pairwise_path_distance > 0.0
    assert raw_result.final_return_variance > 0.0
    assert raw_result.repeated_path_rate == 0.0
    assert collapsed_result.mean_pairwise_path_distance == 0.0
    assert collapsed_result.final_return_variance == 0.0
    assert collapsed_result.repeated_path_rate == pytest.approx(2 / 3)


def test_path_distance_is_zero_for_identity_and_positive_for_changed_path() -> None:
    assert path_distance(left=_path(), right=_path(), cutoff_close=100.0) == 0.0
    assert path_distance(left=_path(), right=_path(1.0), cutoff_close=100.0) > 0.0
