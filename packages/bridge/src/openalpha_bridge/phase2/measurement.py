"""Measured, not estimated: provider calls, cache bytes, and GPU statistics.

Every value here is observed. Nothing is a literal, nothing is inferred, and no
monetary cost is computed: cost can only be derived after a real run from the
billed duration and the then-applicable price.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field

from ..errors import BridgeFailure, BridgeTransformError, FailureCategory
from .provider import MarketSeries, Phase2Provider, ProviderMode, RetrievalRequest

__all__ = [
    "CacheMeasurement",
    "GpuMeasurement",
    "ObservedRetrieval",
    "ObservingProvider",
    "TimingScopes",
    "directory_bytes",
    "gpu_snapshot",
    "reset_gpu_statistics",
]


def _fail(code: str, message: str) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            message=message,
        )
    )


def directory_bytes(root: Path | str) -> int:
    """Total bytes of every regular file under ``root``. Filesystem truth."""
    path = Path(root)
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


class ObservedRetrieval(BaseModel):
    """One provider call, exactly as it was issued."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    symbol: str
    interval: str
    start: str
    end_exclusive: str
    maximum_candles: int
    returned_candles: int = Field(ge=0)


class ObservingProvider:
    """Wraps a provider and records every call it actually made.

    The canary reports this count. A second call, or a request whose bounds
    differ from the amended window, fails closed rather than being averaged away.
    """

    def __init__(self, inner: Phase2Provider) -> None:
        self._inner = inner
        self.calls: list[ObservedRetrieval] = []

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def mode(self) -> ProviderMode:
        return self._inner.mode

    @property
    def client_version(self) -> str:
        return self._inner.client_version

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def fetch(self, request: RetrievalRequest) -> MarketSeries:
        series = self._inner.fetch(request)
        self.calls.append(
            ObservedRetrieval(
                symbol=request.symbol,
                interval=request.interval,
                start=request.start.isoformat(),
                end_exclusive=request.end.isoformat(),
                maximum_candles=request.maximum_candles,
                returned_candles=len(series.candles),
            )
        )
        return series

    def assert_exactly(
        self,
        *,
        symbol: str,
        start: str,
        end_exclusive: str,
        maximum_candles: int,
    ) -> ObservedRetrieval:
        """Exactly one call, with exactly the amended bounds."""
        if self.call_count != 1:
            raise _fail(
                "CANARY_UNEXPECTED_PROVIDER_CALL_COUNT",
                f"the canary must issue exactly one provider call, observed {self.call_count}",
            )
        call = self.calls[0]
        drift = [
            name
            for name, got, want in (
                ("symbol", call.symbol, symbol),
                ("start", call.start, start),
                ("end_exclusive", call.end_exclusive, end_exclusive),
                ("maximum_candles", call.maximum_candles, maximum_candles),
            )
            if got != want
        ]
        if drift:
            raise _fail(
                "CANARY_UNEXPECTED_PROVIDER_REQUEST",
                f"the provider call did not match the amended window: {', '.join(drift)}",
            )
        return call


class CacheMeasurement(BaseModel):
    """Filesystem byte counts around asset resolution and shard creation."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    asset_cache_bytes_before: int = Field(ge=0)
    asset_cache_bytes_after: int = Field(ge=0)
    newly_downloaded_asset_bytes: int = Field(ge=0)
    total_asset_cache_bytes: int = Field(ge=0)
    feature_cache_bytes_before: int = Field(ge=0)
    feature_cache_bytes_after: int = Field(ge=0)
    new_shard_bytes: int = Field(ge=0)
    total_canary_feature_cache_bytes: int = Field(ge=0)


class GpuMeasurement(BaseModel):
    """Peak memory, or an explicit absence when no accelerator is present."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    cuda_available: bool
    device_name: str | None = None
    peak_allocated_bytes: int | None = None
    peak_reserved_bytes: int | None = None


class TimingScopes(BaseModel):
    """Separately measured phases. Wall time is not the same as GPU-path time."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    asset_resolution_seconds: float
    primary_extraction_seconds: float
    deterministic_replay_seconds: float
    causality_extraction_seconds: float
    cache_verification_seconds: float
    official_numerical_path_seconds: float
    total_worker_wall_seconds: float
    #: Deliberately absent. Cost is only knowable after a real run, from the
    #: billed duration and the price in force at that time. An estimate here
    #: would read as a measurement, so none is offered.
    estimated_monetary_cost: None = None


def _torch() -> Any | None:
    try:
        import torch  # pyright: ignore[reportMissingImports]
    except ImportError:
        return None
    return torch


def synchronize() -> None:
    """Block until queued CUDA work completes, so a timer measures real work."""
    torch = _torch()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_gpu_statistics() -> None:
    torch = _torch()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def gpu_snapshot() -> GpuMeasurement:
    torch = _torch()
    if torch is None or not torch.cuda.is_available():
        return GpuMeasurement(cuda_available=False)
    torch.cuda.synchronize()
    return GpuMeasurement(
        cuda_available=True,
        device_name=str(torch.cuda.get_device_name(0)),
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
    )


class scoped_timer:
    """Times a scope, synchronizing CUDA at both ends when it is available."""

    def __init__(self) -> None:
        self.seconds: float = 0.0
        self._started: float = 0.0

    def __enter__(self) -> Self:
        synchronize()
        self._started = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> Literal[False]:
        synchronize()
        self.seconds = round(time.perf_counter() - self._started, 6)
        return False
