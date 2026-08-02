"""The context-only normalization state, fitted once and carried explicitly.

``KronosPredictor.predict`` computes mean and population standard deviation
over the frame it is handed, standardizes with a 1e-5 epsilon, and clips
symmetrically at 5. It then inverts with ``x * (std + eps) + mean``.

v1 left this implicit inside an adapter, which allows two failure modes that
are invisible from the outside: a state silently refitted on data that includes
the target, and a codec that remembers whatever it normalized last. Both are
removed here by making the state an immutable value that every method must be
handed and that every result records the hash of.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..errors import BridgeFailure, BridgeTransformError, FailureCategory
from .official_input import OFFICIAL_COLUMNS, OfficialRow

__all__ = [
    "CLIP_POLICY",
    "CLIP_VALUE",
    "EPSILON",
    "NormalizationState",
    "fit_context_state",
]

#: predict() adds this to the standard deviation, forward and inverse alike.
EPSILON: Final[float] = 1e-5

#: KronosPredictor default clip.
CLIP_VALUE: Final[float] = 5.0

CLIP_POLICY: Final[str] = "symmetric_clip_after_standardization"


def _fail(code: str, message: str) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            message=message,
        )
    )


class NormalizationState(BaseModel):
    """Immutable. Fitted exactly once, from the context rows only."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    schema_version: Literal["openalpha.bridge.diagnostic.normalization.v1"] = (
        "openalpha.bridge.diagnostic.normalization.v1"
    )
    columns: tuple[str, ...]
    mean: tuple[float, ...]
    standard_deviation: tuple[float, ...]
    epsilon: float = EPSILON
    clip_policy: str = CLIP_POLICY
    clip_value: float = CLIP_VALUE
    #: How many candles it was fitted from. 448, never 512.
    fitted_candle_count: int = Field(gt=0)
    #: Population standard deviation, matching numpy's default ddof of 0.
    standard_deviation_ddof: Literal[0] = 0
    state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def normalize(self, rows: tuple[OfficialRow, ...]) -> tuple[tuple[float, ...], ...]:
        """Standardize then clip, exactly as predict() does."""
        out: list[tuple[float, ...]] = []
        for row in rows:
            values = row.channels()
            standardized = tuple(
                _clip(
                    (value - self.mean[index]) / (self.standard_deviation[index] + self.epsilon),
                    self.clip_value,
                )
                for index, value in enumerate(values)
            )
            out.append(standardized)
        return tuple(out)

    def invert(self, normalized: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
        """``x * (std + eps) + mean``. Affine, and applied per column."""
        out: list[tuple[float, ...]] = []
        for values in normalized:
            if len(values) != len(self.columns):
                raise _fail(
                    "NORMALIZATION_CHANNEL_COUNT_MISMATCH",
                    f"expected {len(self.columns)} channels, got {len(values)}",
                )
            out.append(
                tuple(
                    value * (self.standard_deviation[index] + self.epsilon) + self.mean[index]
                    for index, value in enumerate(values)
                )
            )
        return tuple(out)

    def assert_matches(self, other: NormalizationState) -> None:
        """Two methods must have used the same state, not merely similar ones."""
        if self.state_sha256 != other.state_sha256:
            raise _fail(
                "NORMALIZATION_STATE_MISMATCH",
                (
                    f"normalization state {other.state_sha256} does not match the "
                    f"fitted state {self.state_sha256}; every method must use the one "
                    "state fitted from the context"
                ),
            )


def _clip(value: float, limit: float) -> float:
    if math.isnan(value):
        return value
    return max(-limit, min(limit, value))


def _state_digest(
    *,
    columns: tuple[str, ...],
    mean: tuple[float, ...],
    standard_deviation: tuple[float, ...],
    epsilon: float,
    clip_policy: str,
    clip_value: float,
    fitted_candle_count: int,
) -> str:
    """Canonical JSON over every field that changes the transform."""
    payload = {
        "columns": list(columns),
        "mean": [repr(value) for value in mean],
        "standard_deviation": [repr(value) for value in standard_deviation],
        "epsilon": repr(epsilon),
        "clip_policy": clip_policy,
        "clip_value": repr(clip_value),
        "fitted_candle_count": fitted_candle_count,
        "standard_deviation_ddof": 0,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fit_context_state(context: tuple[OfficialRow, ...]) -> NormalizationState:
    """Fit from the context rows only.

    Refitting on the full sequence would let the target's mean and variance
    reach the reconstruction, which is leakage even though no parameter moves.
    """
    if not context:
        raise _fail("NORMALIZATION_NO_CONTEXT", "cannot fit a state from zero candles")

    count = len(context)
    columns = OFFICIAL_COLUMNS
    means: list[float] = []
    deviations: list[float] = []
    for index in range(len(columns)):
        values = [row.channels()[index] for row in context]
        if not all(math.isfinite(value) for value in values):
            raise _fail(
                "NORMALIZATION_NON_FINITE_INPUT",
                f"column {columns[index]} contains a non-finite value in the context",
            )
        mean = sum(values) / count
        # Population variance, ddof = 0, matching numpy's default.
        variance = sum((value - mean) ** 2 for value in values) / count
        means.append(mean)
        deviations.append(math.sqrt(variance))

    digest = _state_digest(
        columns=columns,
        mean=tuple(means),
        standard_deviation=tuple(deviations),
        epsilon=EPSILON,
        clip_policy=CLIP_POLICY,
        clip_value=CLIP_VALUE,
        fitted_candle_count=count,
    )
    return NormalizationState(
        columns=columns,
        mean=tuple(means),
        standard_deviation=tuple(deviations),
        fitted_candle_count=count,
        state_sha256=digest,
    )
