"""Methods A, B, C and D.

Each returns a typed result and records what the specification says it records.
None of them corrects, retries, substitutes or drops anything: an invalid path
stays invalid in the record, because the point of the diagnostic is to find out
how often that happens and where it comes from.
"""

from __future__ import annotations

import math
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .backends import ForecastModel, GeneratedPath, TokenizerCodec, TokenPair
from .metrics import ForecastError, forecast_error, mean_of
from .spec import (
    OFFICIAL_INFERENCE_SETTINGS,
    ROLLOUT_SEEDS,
    InferenceSettings,
)
from .validity import (
    DecodedCandle,
    PathValidity,
    ProjectionOutcome,
    path_validity,
    project_path,
)

__all__ = [
    "MethodAResult",
    "MethodBResult",
    "MethodCResult",
    "MethodDResult",
    "RolloutRecord",
    "run_method_a",
    "run_method_b",
    "run_method_c",
    "run_method_d",
]


def _tokens_as_pairs(tokens: tuple[TokenPair, ...]) -> dict[str, tuple[int, ...]]:
    return {
        "coarse": tuple(t.coarse for t in tokens),
        "fine": tuple(t.fine for t in tokens),
    }


class MethodAResult(BaseModel):
    """Tokenizer round trip on the known real sequence. No generation."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    method: Literal["A_TOKENIZER_ROUND_TRIP"] = "A_TOKENIZER_ROUND_TRIP"
    coarse_token_ids: tuple[int, ...]
    fine_token_ids: tuple[int, ...]
    reconstruction: tuple[DecodedCandle, ...]
    validity: PathValidity
    #: Reconstruction error over the whole encoded sequence.
    reconstruction_error: ForecastError
    seconds: float


def run_method_a(
    *, codec: TokenizerCodec, candles: tuple[DecodedCandle, ...], anchor_close: float
) -> MethodAResult:
    """Encode the real sequence and decode it straight back.

    Determines whether the tokenizer decoder itself creates invalid candles from
    in-distribution encoder tokens. If it does, everything downstream is
    confounded and no statement about generation can be made.
    """
    started = time.perf_counter()
    tokens = codec.encode(candles)
    reconstruction = codec.decode(tokens)
    streams = _tokens_as_pairs(tokens)
    return MethodAResult(
        coarse_token_ids=streams["coarse"],
        fine_token_ids=streams["fine"],
        reconstruction=reconstruction,
        validity=path_validity(reconstruction),
        reconstruction_error=forecast_error(
            reconstruction, candles, anchor_close=anchor_close
        ),
        seconds=round(time.perf_counter() - started, 6),
    )


class MethodBResult(BaseModel):
    """One official frozen forecast, decoded without correction."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    method: Literal["B_OFFICIAL_FROZEN_FORECAST"] = "B_OFFICIAL_FROZEN_FORECAST"
    seed: int
    coarse_token_ids: tuple[int, ...]
    fine_token_ids: tuple[int, ...]
    step_log_probabilities: tuple[float, ...]
    total_log_probability: float
    settings: InferenceSettings
    raw_decoded: tuple[DecodedCandle, ...]
    validity: PathValidity
    forecast_error: ForecastError
    seconds: float


def run_method_b(
    *,
    model: ForecastModel,
    codec: TokenizerCodec,
    context: tuple[DecodedCandle, ...],
    target: tuple[DecodedCandle, ...],
    anchor_close: float,
    settings: InferenceSettings = OFFICIAL_INFERENCE_SETTINGS,
    seed: int = ROLLOUT_SEEDS[0],
) -> MethodBResult:
    """Generate future tokens from the context and decode them raw.

    Determines whether invalidity enters primarily through generated
    future-token combinations rather than through the decoder.
    """
    started = time.perf_counter()
    generated = model.generate(
        context,
        steps=settings.prediction_length,
        seed=seed,
        temperature=settings.temperature,
        top_k=settings.top_k,
        top_p=settings.top_p,
    )
    decoded = codec.decode(generated.tokens)
    streams = _tokens_as_pairs(generated.tokens)
    return MethodBResult(
        seed=seed,
        coarse_token_ids=streams["coarse"],
        fine_token_ids=streams["fine"],
        step_log_probabilities=generated.step_log_probabilities,
        total_log_probability=generated.total_log_probability,
        settings=settings,
        raw_decoded=decoded,
        validity=path_validity(decoded),
        forecast_error=forecast_error(decoded, target, anchor_close=anchor_close),
        seconds=round(time.perf_counter() - started, 6),
    )


class MethodCResult(BaseModel):
    """Terminal projection of Method B's raw output. Tokens untouched."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    method: Literal["C_DETERMINISTIC_TERMINAL_PROJECTION"] = (
        "C_DETERMINISTIC_TERMINAL_PROJECTION"
    )
    projection: ProjectionOutcome
    validity_before: PathValidity
    validity_after: PathValidity
    #: True when projection turned an invalid path into a valid one. False when
    #: the path was already valid, or when projection did not repair it.
    restores_validity: bool
    forecast_error_before: ForecastError
    forecast_error_after: ForecastError
    #: Positive means projection reduced the primary error.
    primary_error_improvement: float | None
    seconds: float


def run_method_c(
    *, forecast: MethodBResult, target: tuple[DecodedCandle, ...], anchor_close: float
) -> MethodCResult:
    """Repair geometry only, and see whether prediction quality moves."""
    started = time.perf_counter()
    outcome = project_path(forecast.raw_decoded)
    after = path_validity(outcome.projected)
    error_after = forecast_error(outcome.projected, target, anchor_close=anchor_close)

    before_primary = forecast.forecast_error.close_return_mae
    after_primary = error_after.close_return_mae
    improvement = (
        before_primary - after_primary
        if before_primary is not None and after_primary is not None
        else None
    )
    return MethodCResult(
        projection=outcome,
        validity_before=forecast.validity,
        validity_after=after,
        restores_validity=forecast.validity.path_is_invalid and not after.path_is_invalid,
        forecast_error_before=forecast.forecast_error,
        forecast_error_after=error_after,
        primary_error_improvement=improvement,
        seconds=round(time.perf_counter() - started, 6),
    )


class RolloutRecord(BaseModel):
    """One stochastic trajectory, preserved whether it is valid or not."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    index: int
    seed: int
    coarse_token_ids: tuple[int, ...]
    fine_token_ids: tuple[int, ...]
    total_log_probability: float
    raw_decoded: tuple[DecodedCandle, ...]
    valid: bool
    invalid_candle_count: int
    forecast_error: ForecastError


class MethodDResult(BaseModel):
    """Valid-rollout filtering across the fixed seed set."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    method: Literal["D_VALID_ROLLOUT_FILTERING"] = "D_VALID_ROLLOUT_FILTERING"
    settings: InferenceSettings
    seeds: tuple[int, ...]
    rollouts: tuple[RolloutRecord, ...]

    rollout_count: int = Field(ge=0)
    valid_rollout_count: int = Field(ge=0)
    valid_rollout_fraction: float

    #: Mean primary error within each group. None when a group is empty, which
    #: is a statement about the group rather than a zero.
    valid_group_mean_primary_error: float | None
    invalid_group_mean_primary_error: float | None

    #: Averaged over valid paths only. None when there are none, and in that
    #: case nothing is substituted for it.
    valid_only_ensemble: tuple[DecodedCandle, ...] | None
    valid_only_ensemble_error: ForecastError | None
    #: What the official procedure would report, averaging every rollout.
    all_rollout_ensemble: tuple[DecodedCandle, ...] | None
    all_rollout_ensemble_error: ForecastError | None

    distinct_token_paths: int = Field(ge=0)
    repeated_path_count: int = Field(ge=0)
    most_repeated_path_occurrences: int = Field(ge=0)
    mean_pairwise_close_dispersion: float | None

    total_seconds: float
    mean_seconds_per_rollout: float


def _ensemble(paths: list[tuple[DecodedCandle, ...]]) -> tuple[DecodedCandle, ...] | None:
    """Elementwise mean across paths. None when there is nothing to average."""
    if not paths:
        return None
    steps = len(paths[0])
    if any(len(path) != steps for path in paths):
        return None
    averaged: list[DecodedCandle] = []
    for step in range(steps):
        rows = [path[step] for path in paths]
        count = len(rows)
        averaged.append(
            DecodedCandle(
                open=sum(r.open for r in rows) / count,
                high=sum(r.high for r in rows) / count,
                low=sum(r.low for r in rows) / count,
                close=sum(r.close for r in rows) / count,
                volume=sum(r.volume for r in rows) / count,
            )
        )
    return tuple(averaged)


def run_method_d(
    *,
    model: ForecastModel,
    codec: TokenizerCodec,
    context: tuple[DecodedCandle, ...],
    target: tuple[DecodedCandle, ...],
    anchor_close: float,
    settings: InferenceSettings = OFFICIAL_INFERENCE_SETTINGS,
    seeds: tuple[int, ...] = ROLLOUT_SEEDS,
) -> MethodDResult:
    """Generate the fixed seed set and compare valid-only against all-rollout.

    When at least one valid path exists the valid-only ensemble averages those
    paths and only those. When none exists, the ensemble is None and the caller
    emits a typed NO_VALID_ROLLOUTS result: nothing is projected in, and no path
    is substituted.
    """
    started = time.perf_counter()
    records: list[RolloutRecord] = []

    for index, seed in enumerate(seeds):
        generated: GeneratedPath = model.generate(
            context,
            steps=settings.prediction_length,
            seed=seed,
            temperature=settings.temperature,
            top_k=settings.top_k,
            top_p=settings.top_p,
        )
        decoded = codec.decode(generated.tokens)
        validity = path_validity(decoded)
        streams = _tokens_as_pairs(generated.tokens)
        records.append(
            RolloutRecord(
                index=index,
                seed=seed,
                coarse_token_ids=streams["coarse"],
                fine_token_ids=streams["fine"],
                total_log_probability=generated.total_log_probability,
                raw_decoded=decoded,
                valid=not validity.path_is_invalid,
                invalid_candle_count=validity.invalid_candle_count,
                forecast_error=forecast_error(decoded, target, anchor_close=anchor_close),
            )
        )

    total_seconds = time.perf_counter() - started

    valid = [r for r in records if r.valid]
    invalid = [r for r in records if not r.valid]

    def primaries(group: list[RolloutRecord]) -> list[float]:
        return [
            r.forecast_error.close_return_mae
            for r in group
            if r.forecast_error.defined and r.forecast_error.close_return_mae is not None
        ]

    valid_paths = [r.raw_decoded for r in valid]
    all_paths = [r.raw_decoded for r in records]

    valid_ensemble = _ensemble(valid_paths)
    all_ensemble = _ensemble(all_paths)

    # Diversity: identical token sequences mean the sampler is collapsing.
    signatures = [(r.coarse_token_ids, r.fine_token_ids) for r in records]
    counts: dict[tuple, int] = {}
    for signature in signatures:
        counts[signature] = counts.get(signature, 0) + 1
    repeated = sum(count for count in counts.values() if count > 1)

    dispersion: float | None = None
    if len(all_paths) > 1 and all(len(p) == len(all_paths[0]) for p in all_paths):
        steps = len(all_paths[0])
        if steps:
            per_step: list[float] = []
            for step in range(steps):
                closes = [p[step].close for p in all_paths]
                if all(math.isfinite(c) for c in closes):
                    mean_close = sum(closes) / len(closes)
                    per_step.append(
                        sum(abs(c - mean_close) for c in closes) / len(closes)
                    )
            dispersion = mean_of(per_step)

    return MethodDResult(
        settings=settings,
        seeds=tuple(seeds),
        rollouts=tuple(records),
        rollout_count=len(records),
        valid_rollout_count=len(valid),
        valid_rollout_fraction=(len(valid) / len(records)) if records else 0.0,
        valid_group_mean_primary_error=mean_of(primaries(valid)),
        invalid_group_mean_primary_error=mean_of(primaries(invalid)),
        valid_only_ensemble=valid_ensemble,
        valid_only_ensemble_error=(
            forecast_error(valid_ensemble, target, anchor_close=anchor_close)
            if valid_ensemble is not None
            else None
        ),
        all_rollout_ensemble=all_ensemble,
        all_rollout_ensemble_error=(
            forecast_error(all_ensemble, target, anchor_close=anchor_close)
            if all_ensemble is not None
            else None
        ),
        distinct_token_paths=len(counts),
        repeated_path_count=repeated,
        most_repeated_path_occurrences=max(counts.values()) if counts else 0,
        mean_pairwise_close_dispersion=dispersion,
        total_seconds=round(total_seconds, 6),
        mean_seconds_per_rollout=round(total_seconds / len(records), 6) if records else 0.0,
    )
