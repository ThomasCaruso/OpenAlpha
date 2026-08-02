"""Frozen inference diagnostic for constrained Kronos decoding.

DEVELOPMENT DIAGNOSTIC - NOT HOLDOUT OR TRADING EVIDENCE.

Specified by research/bridge-v0/phase2-frozen-inference-diagnostic.yaml, which
is separately hashed and amends nothing. No training, no fine-tuning, no
parameter update, no optimizer, no Bridge head, no Stage B or Stage C, no
checkpointing, no gate evaluation, and no held-out partition.

Imports are lazy so that importing this package pulls in neither Torch nor any
official asset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .conclusion import DiagnosticConclusion
    from .spec import (
        CLAIM_BOUNDARY,
        DIAGNOSTIC_SPECIFICATION_NAME,
        DIAGNOSTIC_SPECIFICATION_SHA256,
    )

__all__ = [
    "CLAIM_BOUNDARY",
    "DIAGNOSTIC_SPECIFICATION_NAME",
    "DIAGNOSTIC_SPECIFICATION_SHA256",
    "DiagnosticConclusion",
]


def __getattr__(name: str) -> object:
    if name in {
        "CLAIM_BOUNDARY",
        "DIAGNOSTIC_SPECIFICATION_NAME",
        "DIAGNOSTIC_SPECIFICATION_SHA256",
    }:
        from . import spec

        return getattr(spec, name)
    if name == "DiagnosticConclusion":
        from .conclusion import DiagnosticConclusion

        return DiagnosticConclusion
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
