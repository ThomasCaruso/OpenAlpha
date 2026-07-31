from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Literal

from pydantic import Field, model_validator

from .contracts import FrozenModel
from .structural_validity import (
    AuditCandle,
    PathValidity,
    ProjectionAdjustment,
    constraint_projection_v0,
    validate_forecast_path,
)

MethodId = Literal[
    "RAW_AUTOREGRESSIVE",
    "TERMINAL_PROJECTION",
    "STEPWISE_PROJECT_REENCODE",
    "VALID_CANDIDATE_RESAMPLING",
]


class CandidateTokenPair(FrozenModel):
    coarse_token: int = Field(ge=0, le=1023)
    fine_token: int = Field(ge=0, le=1023)
    joint_log_probability: float
    rank: int = Field(ge=1)
    decoded_candle: AuditCandle
    valid: bool

    @model_validator(mode="after")
    def require_finite_probability(self) -> CandidateTokenPair:
        if not math.isfinite(self.joint_log_probability):
            raise ValueError("joint log probability must be finite")
        return self


class CandidateSelection(FrozenModel):
    selected: CandidateTokenPair
    candidates_considered: int = Field(ge=1)
    valid_candidate_count: int = Field(ge=1)
    rejection_count: int = Field(ge=0)
    auxiliary_fraction: float = Field(ge=0.0, lt=1.0)


class TerminalProjectionResult(FrozenModel):
    method: Literal["TERMINAL_PROJECTION"]
    path_id: str
    raw_candles: tuple[AuditCandle, ...]
    projected_candles: tuple[AuditCandle, ...]
    adjustments: tuple[ProjectionAdjustment, ...]
    raw_validity: PathValidity
    projected_validity: PathValidity
    raw_log_return: float
    projected_log_return: float


def terminal_projection(
    *,
    path_id: str,
    candles: Sequence[AuditCandle],
    expected_sessions: Sequence,
    cutoff_close: float,
    cutoff_volume: float,
) -> TerminalProjectionResult:
    raw = tuple(candles)
    raw_validity = validate_forecast_path(
        path_id=path_id,
        candles=raw,
        expected_sessions=expected_sessions,
        cutoff_close=cutoff_close,
        cutoff_volume=cutoff_volume,
    )
    projection = constraint_projection_v0(
        path_id=path_id,
        candles=raw,
        cutoff_close=cutoff_close,
    )
    projected_validity = validate_forecast_path(
        path_id=f"{path_id}-terminal-projection",
        candles=projection.projected_candles,
        expected_sessions=expected_sessions,
        cutoff_close=cutoff_close,
        cutoff_volume=cutoff_volume,
    )
    if not projected_validity.valid:
        raise ValueError("terminal projection did not produce a structurally valid path")
    return TerminalProjectionResult(
        method="TERMINAL_PROJECTION",
        path_id=path_id,
        raw_candles=raw,
        projected_candles=projection.projected_candles,
        adjustments=projection.adjustments,
        raw_validity=raw_validity,
        projected_validity=projected_validity,
        raw_log_return=projection.original_log_return,
        projected_log_return=projection.projected_log_return,
    )


def deterministic_auxiliary_fraction(*, origin_id: str, seed: int, step: int) -> float:
    if not origin_id:
        raise ValueError("origin_id must be nonempty")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if step < 1:
        raise ValueError("step must be positive")
    payload = f"sentinel-v1-candidate-selection|{origin_id}|{seed}|{step}".encode()
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return integer / 2**64


def choose_valid_candidate(
    candidates: Sequence[CandidateTokenPair],
    *,
    auxiliary_fraction: float,
) -> CandidateSelection:
    if not 0.0 <= auxiliary_fraction < 1.0:
        raise ValueError("auxiliary fraction must be in [0, 1)")
    materialized = tuple(candidates)
    if not materialized:
        raise ValueError("candidate list must be nonempty")
    valid = tuple(item for item in materialized if item.valid)
    if not valid:
        raise ValueError("no valid candidate within the declared budget")
    maximum = max(item.joint_log_probability for item in valid)
    weights = tuple(math.exp(item.joint_log_probability - maximum) for item in valid)
    total = sum(weights)
    threshold = auxiliary_fraction * total
    cumulative = 0.0
    selected = valid[-1]
    for candidate, weight in zip(valid, weights):
        cumulative += weight
        if threshold < cumulative:
            selected = candidate
            break
    return CandidateSelection(
        selected=selected,
        candidates_considered=len(materialized),
        valid_candidate_count=len(valid),
        rejection_count=len(materialized) - len(valid),
        auxiliary_fraction=auxiliary_fraction,
    )
