"""Locked Phase 2 provider contract, real client, and deterministic fake.

The real client is implemented here but performs no request at import time. It
resolves ``yfinance`` lazily so the base test suite never needs a provider
client, and it never writes credentials or raw responses to logs.

The fake provider generates structurally valid, timestamp-correct OHLCV series
for tests. Its artifacts are stamped ``provider_mode: fake`` so they can never be
confused with real evidence.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from datetime import date
from enum import StrEnum
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from ..calendars import CalendarName, sessions_in_half_open_range
from ..errors import BridgeFailure, BridgeTransformError, FailureCategory
from .optional import require_module

__all__ = [
    "LOCKED_TRAINING_SYMBOLS",
    "LOCKED_UNSEEN_SYMBOLS",
    "Candle",
    "DeterministicFakeProvider",
    "MarketSeries",
    "Phase2Provider",
    "ProviderMode",
    "RetrievalRequest",
    "YahooDailyProvider",
    "validate_series",
]

LOCKED_TRAINING_SYMBOLS: tuple[str, ...] = (
    "SPY", "QQQ", "XLF", "XLK", "XLE", "XLI", "XLV",
    "XLP", "XLY", "XLU", "TLT", "HYG", "EFA", "VNQ",
)  # fmt: skip
LOCKED_UNSEEN_SYMBOLS: tuple[str, ...] = ("IWM", "DIA", "GLD", "EEM", "SLV", "USO")

_MAX_PRICE = 1.0e12
_MIN_PRICE = 1.0e-12
_MAX_VOLUME = 1.0e15


class ProviderMode(StrEnum):
    REAL = "real"
    FAKE = "fake"


def _fail(code: str, message: str, *, field: str | None = None) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(
            category=FailureCategory.INVALID_INPUT,
            code=code,
            field=field,
            message=message,
        )
    )


class Candle(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    session: date
    open: float = Field(gt=0.0)
    high: float = Field(gt=0.0)
    low: float = Field(gt=0.0)
    close: float = Field(gt=0.0)
    volume: float = Field(ge=0.0)
    amount: float = Field(ge=0.0)


class RetrievalRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    symbol: str = Field(min_length=1)
    interval: Literal["1d"] = "1d"
    start: date
    end: date
    calendar: CalendarName = CalendarName.XNYS
    maximum_candles: int = Field(gt=0)


class MarketSeries(BaseModel):
    """A retrieved, normalized series plus its provenance."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    schema_version: Literal["openalpha.bridge.phase2.series.v1"] = (
        "openalpha.bridge.phase2.series.v1"
    )
    symbol: str
    interval: str
    provider: str
    provider_mode: ProviderMode
    client_version: str
    retrieval_timestamp: str | None
    candles: tuple[Candle, ...]

    @property
    def normalized_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(f"{self.symbol}|{self.interval}|{self.provider_mode.value}".encode())
        for candle in self.candles:
            digest.update(
                (
                    f"|{candle.session.isoformat()}"
                    f":{candle.open!r}:{candle.high!r}:{candle.low!r}"
                    f":{candle.close!r}:{candle.volume!r}:{candle.amount!r}"
                ).encode()
            )
        return digest.hexdigest()

    @property
    def sessions(self) -> tuple[date, ...]:
        return tuple(candle.session for candle in self.candles)


def validate_series(series: MarketSeries, *, expected_sessions: Sequence[date] | None = None) -> None:
    """Fail closed on schema, ordering, structural, or calendar violations."""
    if not series.candles:
        raise _fail("EMPTY_SERIES", f"{series.symbol} returned no candles")

    previous: date | None = None
    for index, candle in enumerate(series.candles):
        if previous is not None and candle.session <= previous:
            raise _fail(
                "NON_MONOTONIC_SESSIONS",
                f"{series.symbol} session {candle.session} does not follow {previous}",
            )
        previous = candle.session

        values = (candle.open, candle.high, candle.low, candle.close)
        if not all(math.isfinite(value) for value in values):
            raise _fail("NON_FINITE_PRICE", f"{series.symbol} non-finite price at index {index}")
        if not all(_MIN_PRICE <= value <= _MAX_PRICE for value in values):
            raise _fail("PRICE_OUT_OF_DOMAIN", f"{series.symbol} price out of bounds at {index}")
        if candle.volume > _MAX_VOLUME:
            raise _fail("VOLUME_OUT_OF_DOMAIN", f"{series.symbol} volume too large at {index}")
        if candle.high < max(candle.open, candle.close, candle.low):
            raise _fail("INVALID_HIGH", f"{series.symbol} high violates bounds at index {index}")
        if candle.low > min(candle.open, candle.close, candle.high):
            raise _fail("INVALID_LOW", f"{series.symbol} low violates bounds at index {index}")

    if expected_sessions is not None and series.sessions != tuple(expected_sessions):
        raise _fail(
            "CALENDAR_MISMATCH",
            f"{series.symbol} sessions do not match the declared calendar",
        )


class Phase2Provider(Protocol):
    """The retrieval interface the pipeline depends on."""

    @property
    def name(self) -> str: ...

    @property
    def mode(self) -> ProviderMode: ...

    @property
    def client_version(self) -> str: ...

    def fetch(self, request: RetrievalRequest) -> MarketSeries: ...


class YahooDailyProvider:
    """The locked Yahoo Finance client. Resolves ``yfinance`` lazily.

    Never invoked during local pipeline preparation.
    """

    def __init__(
        self,
        *,
        stage: str,
        max_retries: int = 3,
        timeout_seconds: int = 30,
    ) -> None:
        self._stage = stage
        self._max_retries = max_retries
        self._timeout_seconds = timeout_seconds

    @property
    def name(self) -> str:
        return "yahoo_finance"

    @property
    def mode(self) -> ProviderMode:
        return ProviderMode.REAL

    @property
    def client_version(self) -> str:
        module = require_module("yfinance", stage=self._stage)
        return str(getattr(module, "__version__", "unknown"))

    def fetch(self, request: RetrievalRequest) -> MarketSeries:
        from datetime import UTC, datetime

        yfinance = require_module("yfinance", stage=self._stage)
        ticker = yfinance.Ticker(request.symbol)

        last_error: Exception | None = None
        frame = None
        for _ in range(self._max_retries):
            try:
                frame = ticker.history(
                    start=request.start.isoformat(),
                    end=request.end.isoformat(),
                    interval=request.interval,
                    auto_adjust=False,
                    back_adjust=False,
                    repair=False,
                    actions=True,
                    timeout=self._timeout_seconds,
                )
                break
            except Exception as error:  # noqa: BLE001 - provider errors are opaque
                last_error = error
        if frame is None:
            # The provider exception text is not propagated; it can carry the
            # request URL and query parameters.
            raise _fail(
                "PROVIDER_REQUEST_FAILED",
                (
                    f"{self.name} did not return data for {request.symbol} after "
                    f"{self._max_retries} attempts ({type(last_error).__name__})"
                ),
            )

        candles: list[Candle] = []
        for stamp, row in frame.iterrows():
            volume = float(row["Volume"])
            ohlc_mean = (
                float(row["Open"]) + float(row["High"]) + float(row["Low"]) + float(row["Close"])
            ) / 4.0
            candles.append(
                Candle(
                    session=stamp.date(),
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=volume,
                    amount=volume * ohlc_mean,
                )
            )
        if len(candles) > request.maximum_candles:
            raise _fail(
                "RETRIEVAL_CAP_EXCEEDED",
                (
                    f"{request.symbol} returned {len(candles)} candles, "
                    f"above the cap of {request.maximum_candles}"
                ),
            )

        return MarketSeries(
            symbol=request.symbol,
            interval=request.interval,
            provider=self.name,
            provider_mode=ProviderMode.REAL,
            client_version=self.client_version,
            retrieval_timestamp=datetime.now(UTC).isoformat(),
            candles=tuple(candles),
        )


class DeterministicFakeProvider:
    """Reproducible synthetic OHLCV for pipeline tests.

    Output is a pure function of (symbol, session), so runs are byte-identical
    across machines. Series are stamped ``provider_mode: fake``.
    """

    def __init__(self, *, seed: int = 1729) -> None:
        self._seed = seed

    @property
    def name(self) -> str:
        return "deterministic_fake"

    @property
    def mode(self) -> ProviderMode:
        return ProviderMode.FAKE

    @property
    def client_version(self) -> str:
        return "fake-1"

    def _unit(self, symbol: str, session: date, salt: str) -> float:
        raw = f"{self._seed}|{symbol}|{session.isoformat()}|{salt}".encode()
        return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") / float(1 << 64)

    def fetch(self, request: RetrievalRequest) -> MarketSeries:
        sessions = sessions_in_half_open_range(
            request.start, request.end, calendar=request.calendar
        )[: request.maximum_candles]

        candles: list[Candle] = []
        level = 100.0 + 50.0 * self._unit(request.symbol, request.start, "level")
        for session in sessions:
            drift = (self._unit(request.symbol, session, "drift") - 0.5) * 0.02
            level = max(1.0, level * (1.0 + drift))
            open_price = level
            close_price = max(1.0, level * (1.0 + (self._unit(request.symbol, session, "c") - 0.5) * 0.02))
            spread = 1.0 + 0.01 * self._unit(request.symbol, session, "s")
            high = max(open_price, close_price) * spread
            low = min(open_price, close_price) / spread
            volume = 1.0e6 * (1.0 + self._unit(request.symbol, session, "v"))
            ohlc_mean = (open_price + high + low + close_price) / 4.0
            candles.append(
                Candle(
                    session=session,
                    open=open_price,
                    high=high,
                    low=low,
                    close=close_price,
                    volume=volume,
                    amount=volume * ohlc_mean,
                )
            )
            level = close_price

        return MarketSeries(
            symbol=request.symbol,
            interval=request.interval,
            provider=self.name,
            provider_mode=ProviderMode.FAKE,
            client_version=self.client_version,
            retrieval_timestamp=None,
            candles=tuple(candles),
        )
