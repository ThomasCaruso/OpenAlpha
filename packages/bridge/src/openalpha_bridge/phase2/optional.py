"""Optional-dependency boundary for the Phase 2 execution pipeline.

Importing :mod:`openalpha_bridge` must never pull in Torch, a provider client, or
any Kronos asset. Heavy dependencies are resolved lazily, here, and only by the
execution pipeline. A missing dependency produces a typed, actionable failure
rather than a bare ``ImportError``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from importlib import import_module
from types import ModuleType

from ..errors import BridgeFailure, BridgeTransformError, FailureCategory

__all__ = [
    "BridgeExtra",
    "OptionalDependency",
    "missing_optional_dependency",
    "require_module",
]


class BridgeExtra(StrEnum):
    """Installable extras that gate the heavy execution dependencies."""

    BRIDGE_CORE = "bridge-core"
    BRIDGE_KRONOS = "bridge-kronos"
    BRIDGE_TRAINING = "bridge-training"
    BRIDGE_GPU = "bridge-gpu"
    BRIDGE_CLOUD = "bridge-cloud"
    BRIDGE_MODAL = "bridge-modal"


@dataclass(frozen=True, slots=True)
class OptionalDependency:
    """A lazily imported module and the extra that provides it."""

    module: str
    extra: BridgeExtra

    @property
    def install_command(self) -> str:
        return f"uv sync --extra {self.extra.value}"


#: The complete set of dependencies the pipeline may import lazily.
_KNOWN: dict[str, OptionalDependency] = {
    "torch": OptionalDependency("torch", BridgeExtra.BRIDGE_TRAINING),
    "yfinance": OptionalDependency("yfinance", BridgeExtra.BRIDGE_CORE),
    "huggingface_hub": OptionalDependency("huggingface_hub", BridgeExtra.BRIDGE_KRONOS),
    "safetensors": OptionalDependency("safetensors", BridgeExtra.BRIDGE_KRONOS),
    "boto3": OptionalDependency("boto3", BridgeExtra.BRIDGE_CLOUD),
    "fastapi": OptionalDependency("fastapi", BridgeExtra.BRIDGE_CLOUD),
    "modal": OptionalDependency("modal", BridgeExtra.BRIDGE_MODAL),
}


def missing_optional_dependency(
    module: str,
    *,
    stage: str,
    extra: BridgeExtra | None = None,
) -> BridgeTransformError:
    """Build the typed MISSING_OPTIONAL_DEPENDENCY failure."""
    known = _KNOWN.get(module)
    resolved_extra = extra or (known.extra if known else BridgeExtra.BRIDGE_CORE)
    command = f"uv sync --extra {resolved_extra.value}"
    return BridgeTransformError(
        BridgeFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code="MISSING_OPTIONAL_DEPENDENCY",
            field=module,
            message=(
                f"stage={stage} requires the optional dependency '{module}', "
                f"provided by extra '{resolved_extra.value}'. Install it with: {command}"
            ),
        )
    )


def require_module(module: str, *, stage: str, extra: BridgeExtra | None = None) -> ModuleType:
    """Import ``module`` lazily or fail with MISSING_OPTIONAL_DEPENDENCY."""
    try:
        return import_module(module)
    except ImportError as error:
        raise missing_optional_dependency(module, stage=stage, extra=extra) from error
