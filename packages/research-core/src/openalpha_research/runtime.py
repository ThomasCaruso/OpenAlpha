from __future__ import annotations

import importlib
import time
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict

__all__ = [
    "GpuMeasurement",
    "directory_bytes",
    "gpu_snapshot",
    "reset_gpu_statistics",
    "scoped_timer",
    "synchronize",
]


def directory_bytes(root: Path | str) -> int:
    """Return the observed size of all regular files below ``root``."""
    path = Path(root)
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


class GpuMeasurement(BaseModel):
    """Peak memory or an explicit absence when no accelerator is available."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    cuda_available: bool
    device_name: str | None = None
    peak_allocated_bytes: int | None = None
    peak_reserved_bytes: int | None = None


def _torch() -> Any | None:
    try:
        return importlib.import_module("torch")
    except ImportError:
        return None


def synchronize() -> None:
    """Wait for queued CUDA work when an accelerator runtime is available."""
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
    """Measure a scope while synchronizing optional accelerator work."""

    def __init__(self) -> None:
        self.seconds = 0.0
        self._started = 0.0

    def __enter__(self) -> Self:
        synchronize()
        self._started = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> Literal[False]:
        synchronize()
        self.seconds = round(time.perf_counter() - self._started, 6)
        return False
