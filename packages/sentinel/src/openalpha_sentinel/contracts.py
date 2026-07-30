from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date, datetime
from itertools import pairwise
from typing import Annotated, Any, Literal, Protocol, Self

from openalpha_research.artifacts import Sha256
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:/@-]{0,255}$")]
Revision = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
FailureCode = Annotated[str, StringConstraints(pattern=r"^[A-Z][A-Z0-9_]{1,63}$")]

MODEL_REPOSITORY = "NeoQuasar/Kronos-mini"
MODEL_REVISION = "f4e68697d9d5aed55cef5c96aabc3376bcad9f81"
TOKENIZER_REPOSITORY = "NeoQuasar/Kronos-Tokenizer-2k"
TOKENIZER_REVISION = "26966d0035065a0cae0ebad7af8ece35bc1fb51c"
SOURCE_REVISION = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"


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


class OHLCVObservation(FrozenModel):
    session: date
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    @field_validator("timestamp")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        return value

    @model_validator(mode="after")
    def require_valid_bar(self) -> OHLCVObservation:
        values = (self.open, self.high, self.low, self.close, self.volume)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("OHLCV values must be finite")
        if self.timestamp.date() != self.session:
            raise ValueError("timestamp date must match session")
        return self


class ForecastRequest(FrozenModel):
    model_repository: Literal["NeoQuasar/Kronos-mini"]
    model_revision: Literal["f4e68697d9d5aed55cef5c96aabc3376bcad9f81"]
    tokenizer_repository: Literal["NeoQuasar/Kronos-Tokenizer-2k"]
    tokenizer_revision: Literal["26966d0035065a0cae0ebad7af8ece35bc1fb51c"]
    source_revision: Literal["67b630e67f6a18c9e9be918d9b4337c960db1e9a"]
    symbol: Literal["SPY", "QQQ"]
    cutoff: date
    forecast_sessions: tuple[date, ...] = Field(min_length=5, max_length=5)
    observations: tuple[OHLCVObservation, ...] = Field(min_length=128)
    context_length: int
    forecast_horizon: int
    sampling_seed: int
    temperature: float
    top_p: float
    sample_count: int
    data_snapshot_sha256: Sha256
    experiment_sha256: Sha256
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def require_created_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def require_causal_order(self) -> ForecastRequest:
        if self.context_length not in {128, 256, 512}:
            raise ValueError("undeclared context_length")
        if self.forecast_horizon != 5:
            raise ValueError("forecast_horizon must be five")
        if self.sampling_seed not in {1729, 2027, 7919}:
            raise ValueError("undeclared sampling_seed")
        if self.temperature != 1.0 or self.top_p != 0.9:
            raise ValueError("sampling configuration does not match the experiment")
        if self.sample_count != 1:
            raise ValueError("sample_count must equal one")
        sessions = tuple(row.session for row in self.observations)
        if len(sessions) != len(set(sessions)):
            raise ValueError("observation sessions must be unique")
        if any(current >= following for current, following in pairwise(sessions)):
            raise ValueError("observation sessions must be strictly increasing")
        if len(self.observations) < self.context_length:
            raise ValueError("observations do not satisfy context_length")
        if any(row.volume < 0 for row in self.observations):
            raise ValueError("input observation volume must be nonnegative")
        if any(
            row.high < max(row.open, row.close, row.low)
            for row in self.observations
        ):
            raise ValueError("input high must be at least open, close, and low")
        if any(
            row.low > min(row.open, row.close, row.high)
            for row in self.observations
        ):
            raise ValueError("input low must be at most open, close, and high")
        if sessions[-1] != self.cutoff:
            raise ValueError("final observation must equal cutoff")
        if any(session > self.cutoff for session in sessions):
            raise ValueError("observation after cutoff")
        if any(
            current >= following
            for current, following in pairwise(self.forecast_sessions)
        ):
            raise ValueError("forecast sessions must be strictly increasing")
        if self.forecast_sessions[0] <= self.cutoff:
            raise ValueError("forecast sessions must follow cutoff")
        return self


class ForecastPath(FrozenModel):
    path_id: Identifier
    observations: tuple[OHLCVObservation, ...] = Field(min_length=5, max_length=5)
    canonical_sha256: Sha256

    @model_validator(mode="after")
    def require_ordered_path(self) -> ForecastPath:
        sessions = tuple(row.session for row in self.observations)
        if any(current >= following for current, following in pairwise(sessions)):
            raise ValueError("forecast path sessions must be strictly increasing")
        return self


class ForecastFailure(FrozenModel):
    code: FailureCode
    message: str = Field(min_length=1, max_length=2_000)


class ForecastResponse(FrozenModel):
    provider_id: Identifier
    checkpoint_id: Identifier
    request_id: Identifier
    generated_paths: tuple[ForecastPath, ...] = Field(max_length=1)
    inference_duration_ms: float = Field(ge=0.0)
    failure: ForecastFailure | None

    @model_validator(mode="after")
    def require_success_xor_failure(self) -> ForecastResponse:
        if self.failure is None and len(self.generated_paths) != 1:
            raise ValueError("successful response requires exactly one generated path")
        if self.failure is not None and self.generated_paths:
            raise ValueError("failure response cannot contain generated paths")
        return self


class ForecastProvider(Protocol):
    def forecast(self, request: ForecastRequest) -> ForecastResponse: ...
