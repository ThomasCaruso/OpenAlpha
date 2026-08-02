"""The context-only normalization state, in the official float32 arithmetic.

``KronosPredictor.predict`` does, in this order:

    x = values.astype(np.float32)
    x_mean = np.mean(x, axis=0)
    x_std  = np.std(x, axis=0)
    x = (x - x_mean) / (x_std + 1e-5)
    x = np.clip(x, -5, 5)

Every one of those runs in float32. Computing the statistics in Python double
precision instead gives a number that is close but not identical, and the
result is fed into a quantizer: a value a few ULPs from a bin boundary can
land in a different bin, which changes a discrete token id and therefore
changes the decoded candle. So this reproduces the arithmetic exactly rather
than approximately.

The state itself stays immutable and explicit. It is fitted once from the
context rows, and every method that uses it records its hash. The hash is over
the raw float32 bytes of the statistics, never over a decimal repr: two
distinct float32 values can print the same way, and a digest that cannot tell
them apart is not an identity.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Final, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..errors import BridgeFailure, BridgeTransformError, FailureCategory
from .official_input import OFFICIAL_COLUMNS, OfficialRow

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.typing import NDArray

__all__ = [
    "CLIP_POLICY",
    "CLIP_VALUE",
    "EPSILON",
    "NormalizationState",
    "context_matrix",
    "fit_context_state",
]

#: predict() adds this to the standard deviation, forward and inverse alike.
#: Kept as a Python float so the expression matches the source literally; under
#: NEP 50 a Python scalar does not upcast a float32 array.
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


def context_matrix(rows: tuple[OfficialRow, ...]) -> NDArray[np.float32]:
    """The six-channel matrix, float32, in the official column order.

    ``predict`` builds this with ``.values.astype(np.float32)``; the cast
    happens before any statistic is taken, so it happens here too.
    """
    return np.array([row.channels() for row in rows], dtype=np.float32)


def _as_float32(values: tuple[float, ...]) -> NDArray[np.float32]:
    """Rebuild an exact float32 vector from stored values.

    float32 to float64 is lossless, so the round trip through the model's
    ``float`` fields recovers the original bits exactly.
    """
    return np.asarray(values, dtype=np.float32)


class NormalizationState(BaseModel):
    """Immutable. Fitted exactly once, from the context rows only."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    schema_version: Literal["openalpha.bridge.diagnostic.normalization.v2"] = (
        "openalpha.bridge.diagnostic.normalization.v2"
    )
    columns: tuple[str, ...]
    #: float32 statistics, widened to Python float for storage. The widening is
    #: exact, so ``_as_float32`` recovers the original bits.
    mean: tuple[float, ...]
    standard_deviation: tuple[float, ...]
    #: The same statistics as raw little-endian float32 bytes, hex encoded.
    #: This is the numerical identity; the decimal fields above are for reading.
    mean_float32_hex: str
    standard_deviation_float32_hex: str
    dtype: Literal["float32"] = "float32"
    epsilon: float = EPSILON
    clip_policy: str = CLIP_POLICY
    clip_value: float = CLIP_VALUE
    #: How many candles it was fitted from. 448, never 512.
    fitted_candle_count: int = Field(gt=0)
    #: Population standard deviation, matching numpy's default ddof of 0.
    standard_deviation_ddof: Literal[0] = 0
    state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def mean_array(self) -> NDArray[np.float32]:
        return _as_float32(self.mean)

    def std_array(self) -> NDArray[np.float32]:
        return _as_float32(self.standard_deviation)

    def normalize_array(self, matrix: NDArray[np.float32]) -> NDArray[np.float32]:
        """``clip((x - mean) / (std + eps), -clip, +clip)``, all in float32."""
        if matrix.dtype != np.float32:
            raise _fail(
                "NORMALIZATION_DTYPE_MISMATCH",
                f"the input matrix is {matrix.dtype}, expected float32",
            )
        if matrix.shape[1] != len(self.columns):
            raise _fail(
                "NORMALIZATION_CHANNEL_COUNT_MISMATCH",
                f"expected {len(self.columns)} channels, got {matrix.shape[1]}",
            )
        standardized = (matrix - self.mean_array()) / (self.std_array() + self.epsilon)
        clipped = np.clip(standardized, -self.clip_value, self.clip_value)
        # Every operand is float32 and no Python scalar upcasts under NEP 50,
        # so this is a no-op that states the invariant rather than changing it.
        return np.asarray(clipped, dtype=np.float32)

    def invert_array(self, matrix: NDArray[np.float32]) -> NDArray[np.float32]:
        """``x * (std + eps) + mean``, in float32."""
        if matrix.dtype != np.float32:
            raise _fail(
                "NORMALIZATION_DTYPE_MISMATCH",
                f"the input matrix is {matrix.dtype}, expected float32",
            )
        if matrix.shape[1] != len(self.columns):
            raise _fail(
                "NORMALIZATION_CHANNEL_COUNT_MISMATCH",
                f"expected {len(self.columns)} channels, got {matrix.shape[1]}",
            )
        restored = matrix * (self.std_array() + self.epsilon) + self.mean_array()
        return np.asarray(restored, dtype=np.float32)

    def normalize(self, rows: tuple[OfficialRow, ...]) -> tuple[tuple[float, ...], ...]:
        """Convenience over ``normalize_array``. Widening to float is exact."""
        return _to_rows(self.normalize_array(context_matrix(rows)))

    def invert(self, normalized: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
        """Convenience over ``invert_array``.

        The input arrives as Python floats because it came back from a tensor.
        Casting to float32 is the same narrowing the official path performs
        when its own float32 tensor is read.
        """
        matrix = np.asarray(normalized, dtype=np.float32)
        if matrix.ndim != 2:
            raise _fail(
                "NORMALIZATION_CHANNEL_COUNT_MISMATCH",
                f"expected a two dimensional array, got {matrix.ndim} dimensions",
            )
        return _to_rows(self.invert_array(matrix))

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


def _to_rows(matrix: NDArray[np.float32]) -> tuple[tuple[float, ...], ...]:
    """float32 to Python float, only at the boundary where rows are built."""
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _state_digest(
    *,
    columns: tuple[str, ...],
    mean: NDArray[np.float32],
    standard_deviation: NDArray[np.float32],
    epsilon: float,
    clip_policy: str,
    clip_value: float,
    fitted_candle_count: int,
) -> str:
    """sha256 over the raw float32 bytes plus the transform metadata.

    Hashing ``repr`` of the statistics would be wrong: distinct float32 values
    can share a decimal representation, so a digest built from text can call
    two different transforms the same. The bytes cannot.
    """
    digest = hashlib.sha256()
    metadata = {
        "schema": "openalpha.bridge.diagnostic.normalization.v2",
        "columns": list(columns),
        "dtype": "float32",
        "epsilon": float(np.float32(epsilon)),
        "clip_policy": clip_policy,
        "clip_value": float(np.float32(clip_value)),
        "fitted_candle_count": fitted_candle_count,
        "standard_deviation_ddof": 0,
    }
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(b"mean:")
    digest.update(mean.astype("<f4").tobytes())
    digest.update(b"std:")
    digest.update(standard_deviation.astype("<f4").tobytes())
    return digest.hexdigest()


def fit_context_state(context: tuple[OfficialRow, ...]) -> NormalizationState:
    """Fit from the context rows only, in float32.

    Refitting on the full sequence would let the target's mean and variance
    reach the reconstruction, which is leakage even though no parameter moves.
    """
    if not context:
        raise _fail("NORMALIZATION_NO_CONTEXT", "cannot fit a state from zero candles")

    matrix = context_matrix(context)
    if not np.isfinite(matrix).all():
        column = OFFICIAL_COLUMNS[int(np.argmin(np.isfinite(matrix).all(axis=0)))]
        raise _fail(
            "NORMALIZATION_NON_FINITE_INPUT",
            f"column {column} contains a non-finite value in the context",
        )

    # Exactly the pinned expressions. np.mean and np.std over a float32 array
    # accumulate in float32 and return float32, and np.std defaults to ddof 0.
    mean = np.mean(matrix, axis=0)
    standard_deviation = np.std(matrix, axis=0)

    digest = _state_digest(
        columns=OFFICIAL_COLUMNS,
        mean=mean,
        standard_deviation=standard_deviation,
        epsilon=EPSILON,
        clip_policy=CLIP_POLICY,
        clip_value=CLIP_VALUE,
        fitted_candle_count=len(context),
    )
    return NormalizationState(
        columns=OFFICIAL_COLUMNS,
        mean=tuple(float(value) for value in mean),
        standard_deviation=tuple(float(value) for value in standard_deviation),
        mean_float32_hex=mean.astype("<f4").tobytes().hex(),
        standard_deviation_float32_hex=standard_deviation.astype("<f4").tobytes().hex(),
        fitted_candle_count=len(context),
        state_sha256=digest,
    )
