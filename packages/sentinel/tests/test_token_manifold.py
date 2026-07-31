import math

import pytest
from openalpha_sentinel.token_manifold import (
    build_token_support,
    causal_volatility_regimes,
    normalized_reconstruction_errors,
    select_support_windows,
    summarize_roundtrip_rows,
    wilson_interval,
)


def test_right_aligned_windows_are_complete_and_nonoverlapping() -> None:
    windows = select_support_windows(tuple(range(1600)), window_length=512, count=3)

    assert windows[0] == tuple(range(64, 576))
    assert windows[1] == tuple(range(576, 1088))
    assert windows[2] == tuple(range(1088, 1600))


def test_support_windows_reject_insufficient_rows() -> None:
    with pytest.raises(ValueError, match="1,536"):
        select_support_windows(tuple(range(1535)), window_length=512, count=3)


def test_exact_pair_support_uses_floor_two_and_jeffreys_smoothing() -> None:
    support = build_token_support(
        ((1, 2), (1, 2), (1, 3)),
        vocab_s1=1024,
        vocab_s2=1024,
    )

    frequent = support.describe(1, 2)
    rare = support.describe(1, 3)
    absent = support.describe(9, 9)

    assert frequent.coarse_count == 3
    assert frequent.fine_count == 2
    assert frequent.exact_pair_count == 2
    assert frequent.empirically_supported is True
    assert frequent.smoothed_probability == pytest.approx(
        2.5 / (3 + 0.5 * 1024 * 1024)
    )
    assert rare.empirically_supported is False
    assert absent.exact_pair_count == 0
    assert math.isfinite(absent.smoothed_surprisal)


def test_roundtrip_summary_uses_locked_wilson_thresholds() -> None:
    rows = tuple(
        {"valid": index != 0, "violation_codes": ("HIGH_BELOW_CLOSE",) if index == 0 else ()}
        for index in range(100)
    )

    summary = summarize_roundtrip_rows(rows)
    lower, upper = wilson_interval(successes=1, observations=100)

    assert summary.invalid_candle_count == 1
    assert summary.invalid_candle_fraction == pytest.approx(0.01)
    assert summary.wilson_lower == pytest.approx(lower)
    assert summary.wilson_upper == pytest.approx(upper)
    assert summary.violation_categories == {"HIGH_BELOW_CLOSE": 1}
    assert summary.material_defect is False
    assert summary.overwhelmingly_valid is False


def test_roundtrip_summary_rejects_malformed_violation_code_type() -> None:
    with pytest.raises(TypeError, match="violation codes"):
        summarize_roundtrip_rows(({"valid": False, "violation_codes": "BAD"},))


def test_reconstruction_errors_use_observed_close_denominator() -> None:
    result = normalized_reconstruction_errors(
        observed={"open": 100.0, "high": 105.0, "low": 95.0, "close": 100.0},
        reconstructed={"open": 101.0, "high": 103.0, "low": 96.0, "close": 98.0},
    )

    assert result == pytest.approx(
        {
            "open": 0.01,
            "high": 0.02,
            "low": 0.01,
            "close": 0.02,
            "range": 0.03,
        }
    )


def test_causal_volatility_regimes_keep_initial_rows_unavailable() -> None:
    closes = tuple(100.0 + index + (index % 3) for index in range(30))
    regimes = causal_volatility_regimes(closes, window=20)

    assert len(regimes) == len(closes)
    assert all(item.status == "not_computable" for item in regimes[:20])
    assert {item.regime for item in regimes[20:]} <= {"low", "middle", "high"}
    assert all(item.value is not None for item in regimes[20:])
