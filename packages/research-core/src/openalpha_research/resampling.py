from __future__ import annotations

import math
import random
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from .failures import FailureCategory, ResearchFailure, ResearchFailureError

__all__ = [
    "MovingBlockCore",
    "block_start_count",
    "blocks_per_resample",
    "moving_block_percentile_interval",
]


def _fail(code: str, message: str, *, field: str | None = None) -> ResearchFailureError:
    return ResearchFailureError(
        ResearchFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            field=field,
            message=message,
        )
    )


def block_start_count(cluster_count: int, block_length: int) -> int:
    """Return the number of valid contiguous starts without wraparound."""
    return cluster_count - block_length + 1


def blocks_per_resample(cluster_count: int, block_length: int) -> int:
    """Return enough blocks to cover all positions before truncation."""
    return -(-cluster_count // block_length)


class MovingBlockCore(BaseModel):
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
    """Compute a percentile interval by resampling contiguous cluster blocks."""
    count = len(cluster_totals)
    if count < 1:
        raise _fail("RESAMPLING_NO_CLUSTERS", "at least one cluster is required")
    if block_length < 1 or block_length > count:
        raise _fail(
            "RESAMPLING_BLOCK_LENGTH_INVALID",
            f"block length must lie between 1 and {count}, got {block_length}",
            field="block_length",
        )
    if resamples < 1:
        raise _fail(
            "RESAMPLING_RESAMPLES_INVALID",
            f"the resample count must be positive, got {resamples}",
            field="resamples",
        )
    if not 0.0 < confidence_level < 1.0:
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
    retained_in_last = count - block_length * (blocks - 1)
    block_totals = [sum(cluster_totals[start : start + block_length]) for start in range(starts)]
    partial_totals = [
        sum(cluster_totals[start : start + retained_in_last]) for start in range(starts)
    ]

    generator = random.Random(seed)
    means: list[float] = []
    for _ in range(resamples):
        total = 0.0
        for block in range(blocks):
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
