from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime
from itertools import pairwise
from typing import Annotated, Any, Literal, Self

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import yfinance as yf
from openalpha_research.artifacts import Sha256
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
from pydantic_core import ValidationError

from .contracts import OHLCVObservation

YFINANCE_VERSION = "1.5.2"
ProviderName = Literal["yahoo_finance"]
Purpose = Literal["forecast_context", "outcome"]
WarningKind = Literal["DIVIDEND", "STOCK_SPLIT"]
Nonempty = Annotated[str, StringConstraints(min_length=1, max_length=512)]
Downloader = Callable[..., pd.DataFrame | None]


class MarketDataError(RuntimeError):
    """A typed failure at the development market-data boundary."""


class FrozenModel(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
    )

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        if update:
            raise TypeError("model_copy updates bypass validation")
        return super().model_copy(update=None, deep=deep)


class MarketDataRequest(FrozenModel):
    purpose: Purpose
    symbol: Literal["SPY", "QQQ"]
    start_inclusive: date
    end_exclusive: date
    cutoff: date
    minimum_sessions: int = Field(ge=1, le=10_000)

    @model_validator(mode="after")
    def require_window(self) -> MarketDataRequest:
        if not self.start_inclusive < self.end_exclusive:
            raise ValueError("start must be before exclusive end")
        if not self.start_inclusive <= self.cutoff < self.end_exclusive:
            raise ValueError("cutoff must be inside the request window")
        if self.purpose == "forecast_context" and self.minimum_sessions < 512:
            raise ValueError("forecast context must require at least 512 sessions")
        if self.purpose == "outcome" and self.minimum_sessions != 5:
            raise ValueError("outcome request must require exactly five sessions")
        return self

    def download_parameters(self) -> dict[str, object]:
        return {
            "tickers": self.symbol,
            "start": self.start_inclusive.isoformat(),
            "end": self.end_exclusive.isoformat(),
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


class CorporateActionWarning(FrozenModel):
    kind: WarningKind
    session: date
    value: float


class MarketDataSnapshot(FrozenModel):
    provider: ProviderName
    client: Literal["yfinance"]
    client_version: Literal["1.5.2"]
    access_class: Literal["unofficial_public_interface"]
    intended_use: Literal["local_research_and_education"]
    redistribution: Literal["prohibited_by_project_policy"]
    adjustment: Literal["raw"]
    request_parameters: tuple[tuple[str, str], ...]
    retrieved_at: datetime
    normalized_schema: tuple[str, str, str, str, str, str]
    row_count: int = Field(ge=1)
    first_session: date
    last_session: date
    normalized_input_sha256: Sha256
    observations: tuple[OHLCVObservation, ...]
    quality_summary: tuple[Nonempty, ...]
    corporate_action_warnings: tuple[CorporateActionWarning, ...]
    timezone_normalization: Literal["provider_session_label_to_utc_midnight"]
    pagination: Literal["client_managed_not_exposed"]
    synthetic: bool


class YFinanceHistoricalBarsProvider:
    def __init__(
        self,
        *,
        downloader: Downloader,
        client_version: str,
        clock: Callable[[], datetime],
        synthetic: bool = False,
    ) -> None:
        self._downloader = downloader
        self._client_version = client_version
        self._clock = clock
        self._synthetic = synthetic

    def fetch(self, request: MarketDataRequest) -> MarketDataSnapshot:
        if self._client_version != YFINANCE_VERSION:
            raise MarketDataError(
                f"yfinance version mismatch: expected {YFINANCE_VERSION}, "
                f"observed {self._client_version}"
            )
        retrieved_at = self._clock()
        if retrieved_at.tzinfo is None or retrieved_at.utcoffset() is None:
            raise MarketDataError("retrieval clock must return timezone-aware timestamp")
        parameters = request.download_parameters()
        try:
            frame = self._downloader(**parameters)
        except Exception as error:
            raise MarketDataError(f"yfinance request failed: {type(error).__name__}: {error}") from error
        observations, warnings = self._normalize(frame, request)
        canonical_rows = [
            {
                "close": row.close,
                "high": row.high,
                "low": row.low,
                "open": row.open,
                "session": row.session.isoformat(),
                "volume": row.volume,
            }
            for row in observations
        ]
        normalized_bytes = json.dumps(
            canonical_rows,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return MarketDataSnapshot(
            provider="yahoo_finance",
            client="yfinance",
            client_version="1.5.2",
            access_class="unofficial_public_interface",
            intended_use="local_research_and_education",
            redistribution="prohibited_by_project_policy",
            adjustment="raw",
            request_parameters=tuple(
                sorted((name, _parameter_text(value)) for name, value in parameters.items())
            ),
            retrieved_at=retrieved_at,
            normalized_schema=("session", "open", "high", "low", "close", "volume"),
            row_count=len(observations),
            first_session=observations[0].session,
            last_session=observations[-1].session,
            normalized_input_sha256=hashlib.sha256(normalized_bytes).hexdigest(),
            observations=observations,
            quality_summary=(
                "STRICTLY_INCREASING_SESSIONS",
                "NO_DUPLICATE_SESSIONS",
                "COMPLETE_XNYS_WINDOW",
                "FINITE_VALID_OHLCV",
                "EXACT_CUTOFF",
            ),
            corporate_action_warnings=warnings,
            timezone_normalization="provider_session_label_to_utc_midnight",
            pagination="client_managed_not_exposed",
            synthetic=self._synthetic,
        )

    @staticmethod
    def _normalize(
        frame: object,
        request: MarketDataRequest,
    ) -> tuple[tuple[OHLCVObservation, ...], tuple[CorporateActionWarning, ...]]:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise MarketDataError("yfinance returned no rows")
        if isinstance(frame.columns, pd.MultiIndex):
            raise MarketDataError("unexpected multi-level yfinance columns")
        required = ("Open", "High", "Low", "Close", "Volume")
        missing_columns = tuple(column for column in required if column not in frame.columns)
        if missing_columns:
            raise MarketDataError(f"missing OHLCV columns: {', '.join(missing_columns)}")
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise MarketDataError("yfinance index must be DatetimeIndex")
        if frame.index.has_duplicates:
            raise MarketDataError("duplicate sessions")
        session_dates = tuple(timestamp.date() for timestamp in frame.index)
        if any(current >= following for current, following in pairwise(session_dates)):
            raise MarketDataError("sessions are not strictly increasing")
        if any(session > request.cutoff for session in session_dates):
            raise MarketDataError("row after declared cutoff")
        if session_dates[-1] != request.cutoff:
            raise MarketDataError("final session does not equal declared cutoff")

        calendar = xcals.get_calendar("XNYS")
        expected = tuple(
            timestamp.date()
            for timestamp in calendar.sessions_in_range(
                request.start_inclusive.isoformat(),
                request.cutoff.isoformat(),
            )
        )
        if session_dates != expected:
            missing = tuple(session for session in expected if session not in set(session_dates))
            extra = tuple(session for session in session_dates if session not in set(expected))
            raise MarketDataError(
                "missing XNYS sessions or unexpected rows: "
                f"missing={[item.isoformat() for item in missing]}, "
                f"extra={[item.isoformat() for item in extra]}"
            )
        if len(session_dates) < request.minimum_sessions:
            raise MarketDataError(
                f"insufficient sessions: expected at least {request.minimum_sessions}, "
                f"observed {len(session_dates)}"
            )

        values = frame.loc[:, list(required)].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise MarketDataError("OHLCV values must be finite")
        if np.any(values[:, 4] < 0):
            raise MarketDataError("volume must be nonnegative")
        if np.any(values[:, 1] < np.maximum.reduce((values[:, 0], values[:, 2], values[:, 3]))):
            raise MarketDataError("high must be at least open, close, and low")
        if np.any(values[:, 2] > np.minimum.reduce((values[:, 0], values[:, 1], values[:, 3]))):
            raise MarketDataError("low must be at most open, close, and high")
        observations: list[OHLCVObservation] = []
        for index, session in enumerate(session_dates):
            try:
                observations.append(
                    OHLCVObservation(
                        session=session,
                        timestamp=datetime.combine(session, datetime.min.time(), tzinfo=UTC),
                        open=float(values[index, 0]),
                        high=float(values[index, 1]),
                        low=float(values[index, 2]),
                        close=float(values[index, 3]),
                        volume=float(values[index, 4]),
                    )
                )
            except ValidationError as error:
                raise MarketDataError(f"invalid OHLCV row for {session}: {error}") from error

        warnings: list[CorporateActionWarning] = []
        action_columns: tuple[tuple[str, WarningKind], ...] = (
            ("Dividends", "DIVIDEND"),
            ("Stock Splits", "STOCK_SPLIT"),
        )
        for column, kind in action_columns:
            if column not in frame.columns:
                continue
            action_values = frame[column].to_numpy(dtype=float)
            if not np.isfinite(action_values).all():
                raise MarketDataError(f"{column} values must be finite")
            for session, value in zip(session_dates, action_values):
                if float(value) != 0.0:
                    warnings.append(
                        CorporateActionWarning(kind=kind, session=session, value=float(value))
                    )
        return tuple(observations), tuple(warnings)


def default_yfinance_provider() -> YFinanceHistoricalBarsProvider:
    return YFinanceHistoricalBarsProvider(
        downloader=yf.download,
        client_version=yf.__version__,
        clock=lambda: datetime.now(UTC),
    )


def _parameter_text(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)
