"""The moving-block percentile bootstrap, as a study-neutral core.

Two studies need the same estimator over different panel sizes, and neither may
weaken the other's guards to get it. This module owns the arithmetic and nothing
else: it takes pre-summed cluster totals and knows nothing about assets,
ordinals, feature sets, or which study called it. Each study keeps its own
typed cluster model, its own cluster-count guard and its own error codes, and
delegates only the numbers.

**This code reproduces published research.** The zero-shot benchmark's terminal
artifact records intervals computed by the original in-study implementation, and
a regression test recomputes them from that artifact's own recorded clusters.
Any change to the draw order, the truncation of the final block, the percentile
indices, or the seeding will break that test -- which is the point. Treat the
sequence below as fixed.

The design it implements, and why:

* One draw selects a BLOCK START, never a single cluster and never a row. That
  preserves temporal dependence between overlapping windows.
* A cluster is indivisible. Its members enter together, which preserves
  cross-sectional dependence within one period.
* There is no wraparound. The last cluster never adjoins the first, because
  those periods are not adjacent in time.
* The final block is truncated so exactly ``cluster_count`` positions are
  retained, and all ``blocks`` draws are still made, so the generator advances
  identically regardless of truncation.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from .errors import BridgeFailure, BridgeTransformError, FailureCategory

__all__ = [
    "MovingBlockCore",
    "block_start_count",
    "blocks_per_resample",
    "moving_block_percentile_interval",
]

_MINIMUM_CONFIDENCE: Final[float] = 0.0
_MAXIMUM_CONFIDENCE: Final[float] = 1.0


def _fail(code: str, message: str, *, field: str | None = None) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            field=field,
            message=message,
        )
    )


def block_start_count(cluster_count: int, block_length: int) -> int:
    """Valid contiguous starts with no wraparound: ``n - L + 1``."""
    return cluster_count - block_length + 1


def blocks_per_resample(cluster_count: int, block_length: int) -> int:
    """``ceil(n / L)`` blocks, enough to cover the panel before truncation."""
    return -(-cluster_count // block_length)


class MovingBlockCore(BaseModel):
    """The numeric result. Callers wrap this in their own reported type."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    cluster_count: int = Field(ge=1)
    observation_count: int = Field(ge=1)
    block_length: int = Field(ge=1)
    available_block_starts: int = Field(ge=1)
    blocks_drawn_per_resample: int = Field(ge=1)
    resamples: int = Field(ge=1)
    confidence_level: float
    seed: int
    point_estimate: float
    lower: float
    upper: float

    @property
    def excludes_zero(self) -> bool:
        return (self.lower > 0.0 and self.upper > 0.0) or (
            self.lower < 0.0 and self.upper < 0.0
        )

    @property
    def excludes_zero_favorably(self) -> bool:
        return self.lower > 0.0


def moving_block_percentile_interval(
    cluster_totals: Sequence[float],
    *,
    observations: int,
    block_length: int,
    seed: int,
    resamples: int,
    confidence_level: float,
) -> MovingBlockCore:
    """Percentile interval of the mean, resampling consecutive cluster blocks.

    ``cluster_totals[j]`` is the SUM of cluster ``j``'s member values, in
    chronological order. ``observations`` is the total number of underlying
    members across all clusters, and is the denominator of every resampled mean;
    passing it explicitly keeps this function ignorant of panel structure.

    Structural validation -- how many clusters there must be, what they contain,
    which identities they carry -- belongs to the calling study, not here. This
    function checks only what it needs to compute a number.
    """
    count = len(cluster_totals)
    if count < 1:
        raise _fail("RESAMPLING_NO_CLUSTERS", "at least one cluster is required")
    if block_length < 1:
        raise _fail(
            "RESAMPLING_BLOCK_LENGTH_INVALID",
            f"block length must be positive, got {block_length}",
            field="block_length",
        )
    if block_length > count:
        raise _fail(
            "RESAMPLING_BLOCK_LENGTH_INVALID",
            (
                f"block length {block_length} exceeds the {count} available clusters; "
                "no contiguous block of that length exists without wraparound"
            ),
            field="block_length",
        )
    if resamples < 1:
        raise _fail(
            "RESAMPLING_RESAMPLES_INVALID",
            f"the resample count must be positive, got {resamples}",
            field="resamples",
        )
    if not _MINIMUM_CONFIDENCE < confidence_level < _MAXIMUM_CONFIDENCE:
        raise _fail(
            "RESAMPLING_CONFIDENCE_INVALID",
            f"the confidence level must lie strictly between 0 and 1, got {confidence_level}",
            field="confidence_level",
        )
    if observations < 1:
        raise _fail(
            "RESAMPLING_OBSERVATIONS_INVALID",
            f"the observation count must be positive, got {observations}",
            field="observations",
        )
    for index, total in enumerate(cluster_totals):
        if not math.isfinite(total):
            raise _fail(
                "RESAMPLING_NON_FINITE_TOTAL",
                f"cluster {index} has a non-finite total",
            )

    starts = block_start_count(count, block_length)
    blocks = blocks_per_resample(count, block_length)
    # The final block contributes only as many clusters as remain, so exactly
    # `count` positions are retained.
    retained_in_last = count - block_length * (blocks - 1)

    # Precomputed so the inner loop is a lookup. This changes no arithmetic:
    # the same clusters are summed, just once each instead of per resample.
    block_totals = [sum(cluster_totals[s : s + block_length]) for s in range(starts)]
    partial_totals = [sum(cluster_totals[s : s + retained_in_last]) for s in range(starts)]

    generator = random.Random(seed)
    means: list[float] = []
    for _ in range(resamples):
        total = 0.0
        for block in range(blocks):
            # Every block draws, including the truncated final one, so the
            # generator advances identically whether or not truncation applies.
            drawn = generator.randrange(starts)
            total += block_totals[drawn] if block < blocks - 1 else partial_totals[drawn]
        means.append(total / observations)
    means.sort()

    tail = (1.0 - confidence_level) / 2.0
    return MovingBlockCore(
        cluster_count=count,
        observation_count=observations,
        block_length=block_length,
        available_block_starts=starts,
        blocks_drawn_per_resample=blocks,
        resamples=resamples,
        confidence_level=confidence_level,
        seed=seed,
        point_estimate=sum(cluster_totals) / observations,
        lower=means[math.floor(tail * (resamples - 1))],
        upper=means[math.ceil((1.0 - tail) * (resamples - 1))],
    )
