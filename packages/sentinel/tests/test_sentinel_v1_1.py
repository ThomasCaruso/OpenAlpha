from datetime import date
from pathlib import Path

from openalpha_sentinel.market_data import MarketDataRequest
from openalpha_sentinel.sentinel_v1_1 import (
    MASS_BUDGETS,
    SUPPORT_END_EXCLUSIVE,
    SUPPORT_START,
    SUPPORT_SYMBOLS,
    SUPPORT_WINDOW_LENGTH,
    SUPPORT_WINDOWS_PER_SYMBOL,
    locked_v1_1_origins,
    verify_experiment_hash,
)


def test_v1_1_scope_is_fixed_and_pre_holdout() -> None:
    assert SUPPORT_SYMBOLS == (
        "SPY",
        "QQQ",
        "IWM",
        "DIA",
        "TLT",
        "HYG",
        "GLD",
        "EFA",
        "EEM",
        "XLF",
    )
    assert SUPPORT_START == date(2017, 1, 1)
    assert SUPPORT_END_EXCLUSIVE == date(2024, 6, 29)
    assert SUPPORT_WINDOW_LENGTH == 512
    assert SUPPORT_WINDOWS_PER_SYMBOL == 3
    assert MASS_BUDGETS == ((8, 8), (16, 16), (32, 32))

    origins = locked_v1_1_origins()
    assert len(origins) == 12
    assert {item.asset for item in origins} == {"SPY", "QQQ"}
    assert max(
        session for item in origins for session in item.forecast_sessions
    ) < date(2025, 7, 1)


def test_v1_1_experiment_hash_matches_exact_yaml_bytes() -> None:
    assert verify_experiment_hash(
        Path("research/sentinel-v1_1/experiment.yaml"),
        Path("research/sentinel-v1_1/experiment.sha256"),
    )


def test_market_data_boundary_accepts_the_locked_support_corpus() -> None:
    for symbol in SUPPORT_SYMBOLS:
        request = MarketDataRequest(
            purpose="forecast_context",
            symbol=symbol,
            start_inclusive=date(2017, 1, 1),
            end_exclusive=date(2024, 6, 29),
            cutoff=date(2024, 6, 28),
            minimum_sessions=1536,
        )
        assert request.symbol == symbol
