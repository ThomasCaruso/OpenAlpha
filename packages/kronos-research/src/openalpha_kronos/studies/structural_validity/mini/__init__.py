"""Frozen inference diagnostic for constrained Kronos decoding.

DEVELOPMENT DIAGNOSTIC - NOT HOLDOUT OR TRADING EVIDENCE.

Operative specification: research/bridge-v0/phase2-frozen-inference-diagnostic-v2.yaml,
which supersedes v1 before execution. v1 is preserved byte-identical. Neither
amends the sealed experiment or its amendments.

No training, no fine-tuning, no parameter update, no optimizer, no Bridge head,
no Stage B or Stage C, no checkpointing, no gate evaluation, no held-out
partition.

Imports are lazy so that importing this package pulls in neither Torch nor any
official asset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..common.conclusion import DiagnosticConclusion
    from .spec import (
        CLAIM_BOUNDARY,
        V1_SPECIFICATION_SHA256,
        V2_SPECIFICATION_NAME,
        V2_SPECIFICATION_SHA256,
    )

__all__ = [
    "CLAIM_BOUNDARY",
    "V1_SPECIFICATION_SHA256",
    "V2_SPECIFICATION_NAME",
    "V2_SPECIFICATION_SHA256",
    "DiagnosticConclusion",
]

_SPEC_NAMES = {
    "CLAIM_BOUNDARY",
    "V1_SPECIFICATION_SHA256",
    "V2_SPECIFICATION_NAME",
    "V2_SPECIFICATION_SHA256",
}


def __getattr__(name: str) -> object:
    if name in _SPEC_NAMES:
        from . import spec

        return getattr(spec, name)
    if name == "DiagnosticConclusion":
        from ..common.conclusion import DiagnosticConclusion

        return DiagnosticConclusion
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
