from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from functools import lru_cache
from itertools import pairwise
from typing import Any, Literal

from pydantic import Field, model_validator

from .contracts import (
    MODEL_REPOSITORY,
    MODEL_REVISION,
    SOURCE_REVISION,
    TOKENIZER_REPOSITORY,
    TOKENIZER_REVISION,
    FrozenModel,
    OHLCVObservation,
)
from .development_manifest import require_outcome_sessions

EXPERIMENT_SHA256 = "55e9d2eb9a9b64feac394c40e295fa16679edc7b8e3cfa803676774e6d86d0a3"
SEEDS = (1729, 2027, 7919)
METHODS = (
    "RAW_AUTOREGRESSIVE",
    "STEPWISE_PROJECT_REENCODE",
    "VALID_CANDIDATE_RESAMPLING",
)
LOCKED_CUTOFFS = (
    date(2024, 7, 5),
    date(2024, 9, 13),
    date(2024, 11, 22),
    date(2025, 1, 31),
    date(2025, 4, 11),
    date(2025, 6, 20),
)


class Phase3BOrigin(FrozenModel):
    origin_id: str = Field(pattern=r"^sentinel-v1-(SPY|QQQ)-\d{4}-\d{2}-\d{2}-h5$")
    asset: Literal["SPY", "QQQ"]
    cutoff: date
    forecast_sessions: tuple[date, ...] = Field(min_length=5, max_length=5)

    @model_validator(mode="after")
    def require_lock(self) -> Phase3BOrigin:
        expected = f"sentinel-v1-{self.asset}-{self.cutoff.isoformat()}-h5"
        if self.origin_id != expected:
            raise ValueError("origin ID does not match the locked identity")
        if self.cutoff not in LOCKED_CUTOFFS:
            raise ValueError("origin cutoff is not in the Phase 3B lock")
        if self.forecast_sessions[-1] >= date(2025, 7, 1):
            raise ValueError("Phase 3B outcome may not enter the holdout period")
        return self


@lru_cache(maxsize=1)
def locked_phase3b_origins() -> tuple[Phase3BOrigin, ...]:
    return tuple(
        Phase3BOrigin(
            origin_id=f"sentinel-v1-{asset}-{cutoff.isoformat()}-h5",
            asset=asset,
            cutoff=cutoff,
            forecast_sessions=require_outcome_sessions(cutoff),
        )
        for cutoff in LOCKED_CUTOFFS
        for asset in ("SPY", "QQQ")
    )


def build_constrained_worker_requests(
    *,
    origin: Phase3BOrigin,
    observations: Sequence[OHLCVObservation],
) -> tuple[dict[str, Any], ...]:
    rows = tuple(observations)
    if len(rows) < 512:
        raise ValueError("Phase 3B requires at least 512 causal observations")
    context = rows[-512:]
    if context[-1].session != origin.cutoff:
        raise ValueError("final causal observation must equal the origin cutoff")
    if any(left.session >= right.session for left, right in pairwise(context)):
        raise ValueError("causal observations must be strictly increasing")
    serialized = [
        {
            "timestamp": item.timestamp.isoformat(),
            "open": item.open,
            "high": item.high,
            "low": item.low,
            "close": item.close,
            "volume": item.volume,
        }
        for item in context
    ]
    return tuple(
        {
            "origin_id": origin.origin_id,
            "symbol": origin.asset,
            "cutoff": origin.cutoff.isoformat(),
            "model_repository": MODEL_REPOSITORY,
            "model_revision": MODEL_REVISION,
            "tokenizer_repository": TOKENIZER_REPOSITORY,
            "tokenizer_revision": TOKENIZER_REVISION,
            "source_revision": SOURCE_REVISION,
            "context_length": 512,
            "sampling_seed": seed,
            "temperature": 1.0,
            "top_p": 0.9,
            "top_k": 0,
            "sample_count": 1,
            "forecast_horizon": 5,
            "methods": list(METHODS),
            "candidate_budget_initial": 16,
            "candidate_budget_expanded": 64,
            "observations": serialized,
            "forecast_sessions": [
                session.isoformat() for session in origin.forecast_sessions
            ],
            "official_parity_probe": seed == 1729,
        }
        for seed in SEEDS
    )
