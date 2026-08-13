from __future__ import annotations

import hashlib
import importlib
import math
from collections.abc import Sequence
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from .calendars import CalendarName, sessions_in_half_open_range
from .failures import FailureCategory, ResearchFailure, ResearchFailureError

__all__ = [
    "Candle",
    "DeterministicFakeProvider",
    "MarketDataProvider",
    "MarketSeries",
    "ProviderMode",
    "RetrievalRequest",
    "YahooDailyProvider",
    "validate_series",
]

_MAX_PRICE = 1.0e12
_MIN_PRICE = 1.0e-12
_MAX_VOLUME = 1.0e15
_HASH_COMPONENT_PATTERN = r"^[^|\r\n]+$"


class ProviderMode(StrEnum):
    REAL = "real"
    FAKE = "fake"


def _fail(code: str, message: str, *, field: str | None = None) -> ResearchFailureError:
    return ResearchFailureError(
        ResearchFailure(
            category=FailureCategory.INVALID_INPUT,
            code=code,
            field=field,
            message=message,
        )
    )


def _optional_module(name: str, *, extra: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise ResearchFailureError(
            ResearchFailure(
                category=FailureCategory.INVALID_CONFIGURATION,
                code="OPTIONAL_DEPENDENCY_MISSING",
                field=name,
                message=f"install openalpha-research-core[{extra}] to use {name}",
            )
        ) from error


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

    symbol: str = Field(min_length=1, pattern=_HASH_COMPONENT_PATTERN)
    interval: Literal["1d"] = "1d"
    start: date
    end: date
    calendar: CalendarName = CalendarName.XNYS
    maximum_candles: int = Field(gt=0)


class MarketSeries(BaseModel):
    """A normalized market series together with retrieval provenance."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    schema_version: Literal["openalpha.research.market-series.v1"] = (
        "openalpha.research.market-series.v1"
    )
    symbol: str = Field(min_length=1, pattern=_HASH_COMPONENT_PATTERN)
    interval: str = Field(min_length=1, pattern=_HASH_COMPONENT_PATTERN)
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


def validate_series(
    series: MarketSeries,
    *,
    expected_sessions: Sequence[date] | None = None,
) -> None:
    """Fail closed on empty, unordered, invalid, or calendar-drifted series."""
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

        prices = (candle.open, candle.high, candle.low, candle.close)
        if not all(math.isfinite(value) for value in prices):
            raise _fail("NON_FINITE_PRICE", f"{series.symbol} non-finite price at index {index}")
        if not all(_MIN_PRICE <= value <= _MAX_PRICE for value in prices):
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


class MarketDataProvider(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def mode(self) -> ProviderMode: ...

    @property
    def client_version(self) -> str: ...

    def fetch(self, request: RetrievalRequest) -> MarketSeries: ...


class YahooDailyProvider:
    """Daily Yahoo Finance client with lazy optional-dependency resolution."""

    def __init__(self, *, max_retries: int = 3, timeout_seconds: int = 30) -> None:
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
        module = _optional_module("yfinance", extra="yahoo")
        return str(getattr(module, "__version__", "unknown"))

    def fetch(self, request: RetrievalRequest) -> MarketSeries:
        yfinance = _optional_module("yfinance", extra="yahoo")
        ticker = yfinance.Ticker(request.symbol)
        last_error: Exception | None = None
        frame: Any | None = None
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
            except Exception as error:  # noqa: BLE001
                last_error = error
        if frame is None:
            raise _fail(
                "PROVIDER_REQUEST_FAILED",
                (
                    f"{self.name} returned no data after {self._max_retries} attempts "
                    f"({type(last_error).__name__})"
                ),
            )

        candles: list[Candle] = []
        for stamp, row in frame.iterrows():
            volume = float(row["Volume"])
            prices = tuple(float(row[name]) for name in ("Open", "High", "Low", "Close"))
            candles.append(
                Candle(
                    session=stamp.date(),
                    open=prices[0],
                    high=prices[1],
                    low=prices[2],
                    close=prices[3],
                    volume=volume,
                    amount=volume * sum(prices) / 4.0,
                )
            )
        if len(candles) > request.maximum_candles:
            raise _fail(
                "RETRIEVAL_CAP_EXCEEDED",
                (
                    f"{request.symbol} returned {len(candles)} candles, above the cap "
                    f"of {request.maximum_candles}"
                ),
            )
        return MarketSeries(
            symbol=request.symbol,
            interval=request.interval,
            provider=self.name,
            provider_mode=self.mode,
            client_version=self.client_version,
            retrieval_timestamp=datetime.now(UTC).isoformat(),
            candles=tuple(candles),
        )


class DeterministicFakeProvider:
    """Pure deterministic OHLCV generator for tests and dry runs."""

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
            request.start,
            request.end,
            calendar=request.calendar,
        )[: request.maximum_candles]
        candles: list[Candle] = []
        level = 100.0 + 50.0 * self._unit(request.symbol, request.start, "level")
        for session in sessions:
            drift = (self._unit(request.symbol, session, "drift") - 0.5) * 0.02
            level = max(1.0, level * (1.0 + drift))
            open_price = level
            close_price = max(
                1.0,
                level * (1.0 + (self._unit(request.symbol, session, "c") - 0.5) * 0.02),
            )
            spread = 1.0 + 0.01 * self._unit(request.symbol, session, "s")
            high = max(open_price, close_price) * spread
            low = min(open_price, close_price) / spread
            volume = 1.0e6 * (1.0 + self._unit(request.symbol, session, "v"))
            candles.append(
                Candle(
                    session=session,
                    open=open_price,
                    high=high,
                    low=low,
                    close=close_price,
                    volume=volume,
                    amount=volume * (open_price + high + low + close_price) / 4.0,
                )
            )
            level = close_price
        return MarketSeries(
            symbol=request.symbol,
            interval=request.interval,
            provider=self.name,
            provider_mode=self.mode,
            client_version=self.client_version,
            retrieval_timestamp=None,
            candles=tuple(candles),
        )
