from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Any

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest
from openalpha_sentinel.market_data import (
    MarketDataError,
    MarketDataRequest,
    YFinanceHistoricalBarsProvider,
)


def _sessions(start: date, end_inclusive: date) -> pd.DatetimeIndex:
    return xcals.get_calendar("XNYS").sessions_in_range(start.isoformat(), end_inclusive.isoformat())


def _synthetic_frame(start: date, end_inclusive: date) -> pd.DataFrame:
    sessions = _sessions(start, end_inclusive)
    close = np.arange(len(sessions), dtype=float) / 10.0 + 100.0
    return pd.DataFrame(
        {
            "Open": close - 0.2,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
            "Adj Close": close + 50.0,
            "Volume": np.full(len(sessions), 1_000_000.0),
            "Dividends": np.zeros(len(sessions)),
            "Stock Splits": np.zeros(len(sessions)),
        },
        index=sessions,
    )


def _context_request() -> MarketDataRequest:
    return MarketDataRequest(
        purpose="forecast_context",
        symbol="SPY",
        start_inclusive=date(2022, 6, 1),
        end_exclusive=date(2024, 7, 6),
        cutoff=date(2024, 7, 5),
        minimum_sessions=512,
    )


def _provider(frame_factory: Callable[..., pd.DataFrame], calls: list[dict[str, Any]]):
    def download(**kwargs: Any) -> pd.DataFrame:
        calls.append(kwargs)
        return frame_factory()

    return YFinanceHistoricalBarsProvider(
        downloader=download,
        client_version="1.5.2",
        clock=lambda: datetime(2026, 7, 30, 18, 0, tzinfo=UTC),
        synthetic=True,
    )


def test_context_download_uses_every_explicit_argument_and_excludes_adj_close() -> None:
    calls: list[dict[str, Any]] = []
    frame = _synthetic_frame(date(2022, 6, 1), date(2024, 7, 5))
    provider = _provider(lambda: frame, calls)

    snapshot = provider.fetch(_context_request())

    assert calls == [
        {
            "tickers": "SPY",
            "start": "2022-06-01",
            "end": "2024-07-06",
            "interval": "1d",
            "auto_adjust": False,
            "back_adjust": False,
            "repair": False,
            "actions": True,
            "progress": False,
            "threads": False,
            "timeout": 30.0,
            "prepost": False,
            "rounding": False,
            "keepna": False,
            "multi_level_index": False,
        }
    ]
    assert snapshot.provider == "yahoo_finance"
    assert snapshot.client == "yfinance"
    assert snapshot.client_version == "1.5.2"
    assert snapshot.adjustment == "raw"
    assert snapshot.first_session == date(2022, 6, 1)
    assert snapshot.last_session == date(2024, 7, 5)
    assert snapshot.row_count == len(_sessions(date(2022, 6, 1), date(2024, 7, 5)))
    assert snapshot.normalized_schema == ("session", "open", "high", "low", "close", "volume")
    assert snapshot.observations[0].close != float(frame.iloc[0]["Adj Close"])
    assert snapshot.retrieved_at == datetime(2026, 7, 30, 18, 0, tzinfo=UTC)
    assert snapshot.synthetic is True
    assert snapshot.pagination == "client_managed_not_exposed"


def test_corporate_actions_are_warnings_not_model_values() -> None:
    calls: list[dict[str, Any]] = []
    frame = _synthetic_frame(date(2022, 6, 1), date(2024, 7, 5))
    frame.loc[pd.Timestamp("2024-06-28"), "Dividends"] = 1.25
    snapshot = _provider(lambda: frame, calls).fetch(_context_request())

    assert snapshot.corporate_action_warnings[0].kind == "DIVIDEND"
    assert snapshot.corporate_action_warnings[0].session == date(2024, 6, 28)
    assert all(not hasattr(row, "dividends") for row in snapshot.observations)


def test_rejects_missing_xnys_session() -> None:
    calls: list[dict[str, Any]] = []
    frame = _synthetic_frame(date(2022, 6, 1), date(2024, 7, 5)).drop(
        pd.Timestamp("2024-07-03")
    )

    with pytest.raises(MarketDataError, match="missing XNYS"):
        _provider(lambda: frame, calls).fetch(_context_request())


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda frame: frame.assign(Close=np.inf), "finite"),
        (lambda frame: frame.assign(High=frame["Low"] - 1.0), "high"),
        (lambda frame: frame.assign(Volume=-1.0), "volume"),
    ],
)
def test_rejects_invalid_ohlcv(
    mutate: Callable[[pd.DataFrame], pd.DataFrame],
    message: str,
) -> None:
    calls: list[dict[str, Any]] = []
    frame = mutate(_synthetic_frame(date(2022, 6, 1), date(2024, 7, 5)))

    with pytest.raises(MarketDataError, match=message):
        _provider(lambda: frame, calls).fetch(_context_request())


def test_rejects_rows_after_cutoff_and_wrong_client_version() -> None:
    calls: list[dict[str, Any]] = []
    frame = _synthetic_frame(date(2022, 6, 1), date(2024, 7, 8))

    with pytest.raises(MarketDataError, match="cutoff"):
        _provider(lambda: frame, calls).fetch(_context_request())

    provider = YFinanceHistoricalBarsProvider(
        downloader=lambda **_: frame,
        client_version="1.5.1",
        clock=lambda: datetime.now(UTC),
    )
    with pytest.raises(MarketDataError, match="version"):
        provider.fetch(_context_request())


def test_outcome_request_is_the_exact_separate_five_session_window() -> None:
    calls: list[dict[str, Any]] = []
    frame = _synthetic_frame(date(2024, 7, 8), date(2024, 7, 12))
    request = MarketDataRequest(
        purpose="outcome",
        symbol="SPY",
        start_inclusive=date(2024, 7, 6),
        end_exclusive=date(2024, 7, 13),
        cutoff=date(2024, 7, 12),
        minimum_sessions=5,
    )

    snapshot = _provider(lambda: frame, calls).fetch(request)

    assert tuple(row.session for row in snapshot.observations) == (
        date(2024, 7, 8),
        date(2024, 7, 9),
        date(2024, 7, 10),
        date(2024, 7, 11),
        date(2024, 7, 12),
    )
