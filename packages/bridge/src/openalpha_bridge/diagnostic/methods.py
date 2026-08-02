"""Methods A, B, C and D over the official six-channel contract.

Every method is handed the one context-fitted normalization state and records
its hash, so a state refitted somewhere along the way is detectable rather than
assumed absent. None of them corrects, retries, substitutes or drops anything.

Methods B and D take their forecast rows from ``GeneratedPath.raw_decoded_suffix``,
which the backend produced by decoding the concatenated context and generated
token window the way the pinned source does. They never call ``codec.decode``
on the generated tokens alone: that would restart the tokenizer decoder at
position zero without the preceding context and is a different computation.
Method A still uses the codec, because its round trip genuinely is a decode of
one complete 512-token window.
"""

from __future__ import annotations

import math
import time
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..errors import BridgeFailure, BridgeTransformError, FailureCategory
from .backends import ForecastModel, GeneratedPath, StepSampling, TokenizerCodec, TokenPair
from .metrics import (
    ForecastError,
    ReconstructionError,
    forecast_error,
    mean_of,
    reconstruction_error,
)
from .normalization import NormalizationState
from .official_input import OfficialRow, OfficialSeries, TimeStamp
from .spec import OFFICIAL_INFERENCE_SETTINGS, ROLLOUT_SEEDS, InferenceSettings
from .validity import PathValidity, ProjectionOutcome, path_validity, project_path

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


def _fail(code: str, message: str) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            message=message,
        )
    )


def _streams(tokens: tuple[TokenPair, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return tuple(t.coarse for t in tokens), tuple(t.fine for t in tokens)


class MethodAResult(BaseModel):
    """Tokenizer round trip on all 512 known candles. No generation."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    method: Literal["A_TOKENIZER_ROUND_TRIP"] = "A_TOKENIZER_ROUND_TRIP"
    normalization_state_sha256: str
    coarse_token_ids: tuple[int, ...]
    fine_token_ids: tuple[int, ...]
    reconstruction: tuple[OfficialRow, ...]

    #: Over all 512 rows and all six channels, with returns taken from
    #: transitions 1..511 internal to the sequence.
    full_sequence_error: ReconstructionError
    #: Over the final 64 rows only, so it is comparable with the forecast
    #: methods without pretending it is a forecast.
    target_suffix_error: ReconstructionError

    validity_all: PathValidity
    validity_target_suffix: PathValidity
    #: Any invalid round-trip candle at all. Independent of materiality.
    structural_invalidity_observed: bool
    seconds: float


def run_method_a(
    *, codec: TokenizerCodec, series: OfficialSeries, state: NormalizationState
) -> MethodAResult:
    """Encode and decode all 512 known candles under the context-only state."""
    started = time.perf_counter()
    rows = series.rows
    tokens = codec.encode(rows, state=state)
    reconstruction = codec.decode(tokens, state=state, sessions=series.sessions)
    coarse, fine = _streams(tokens)

    suffix = len(series.target)
    validity_all = path_validity(reconstruction)
    return MethodAResult(
        normalization_state_sha256=state.state_sha256,
        coarse_token_ids=coarse,
        fine_token_ids=fine,
        reconstruction=reconstruction,
        full_sequence_error=reconstruction_error(reconstruction, rows),
        target_suffix_error=reconstruction_error(reconstruction[-suffix:], rows[-suffix:]),
        validity_all=validity_all,
        validity_target_suffix=path_validity(reconstruction[-suffix:]),
        structural_invalidity_observed=validity_all.invalid_candle_count > 0,
        seconds=round(time.perf_counter() - started, 6),
    )


class MethodBResult(BaseModel):
    """One official frozen forecast, decoded without correction."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    method: Literal["B_OFFICIAL_FROZEN_FORECAST"] = "B_OFFICIAL_FROZEN_FORECAST"
    normalization_state_sha256: str
    seed: int
    coarse_token_ids: tuple[int, ...]
    fine_token_ids: tuple[int, ...]
    sampling: tuple[StepSampling, ...]
    total_path_sampling_log_probability: float
    settings: InferenceSettings
    raw_decoded: tuple[OfficialRow, ...]
    validity: PathValidity
    forecast_error: ForecastError
    seconds: float


def _generate(
    model: ForecastModel,
    series: OfficialSeries,
    state: NormalizationState,
    settings: InferenceSettings,
    seed: int,
) -> GeneratedPath:
    return model.generate(
        series.context,
        context_stamps=series.context_stamps(),
        target_stamps=series.target_stamps(),
        target_sessions=_target_sessions(series),
        state=state,
        steps=settings.prediction_length,
        seed=seed,
        temperature=settings.temperature,
        top_k=settings.top_k,
        top_p=settings.top_p,
    )


def _target_sessions(series: OfficialSeries) -> tuple[date, ...]:
    return tuple(row.session for row in series.target)


def run_method_b(
    *,
    model: ForecastModel,
    codec: TokenizerCodec,
    series: OfficialSeries,
    state: NormalizationState,
    settings: InferenceSettings = OFFICIAL_INFERENCE_SETTINGS,
    seed: int = ROLLOUT_SEEDS[0],
) -> MethodBResult:
    """Generate future tokens from the 448 context candles and decode raw."""
    started = time.perf_counter()
    generated = _generate(model, series, state, settings, seed)
    # The official decode, performed by the backend over the concatenated
    # context and generated tokens. Re-decoding generated.tokens here would
    # restart the tokenizer decoder at position zero with no preceding
    # context, which is a different computation.
    decoded = generated.raw_decoded_suffix
    if len(decoded) != settings.prediction_length:
        raise _fail(
            "DIAGNOSTIC_DECODED_SUFFIX_LENGTH_MISMATCH",
            f"the backend returned {len(decoded)} decoded rows, expected "
            f"{settings.prediction_length}",
        )
    coarse, fine = _streams(generated.tokens)
    anchor = series.context[-1].close
    return MethodBResult(
        normalization_state_sha256=state.state_sha256,
        seed=seed,
        coarse_token_ids=coarse,
        fine_token_ids=fine,
        sampling=generated.sampling,
        total_path_sampling_log_probability=generated.total_path_sampling_log_probability,
        settings=settings,
        raw_decoded=decoded,
        validity=path_validity(decoded),
        forecast_error=forecast_error(decoded, series.target, anchor_close=anchor),
        seconds=round(time.perf_counter() - started, 6),
    )


class MethodCResult(BaseModel):
    """Terminal projection of Method B's raw output. Tokens untouched."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    method: Literal["C_DETERMINISTIC_TERMINAL_PROJECTION"] = "C_DETERMINISTIC_TERMINAL_PROJECTION"
    projection: ProjectionOutcome
    validity_before: PathValidity
    validity_after: PathValidity
    restores_validity: bool
    forecast_error_before: ForecastError
    forecast_error_after: ForecastError
    primary_error_improvement: float | None
    seconds: float


def run_method_c(*, forecast: MethodBResult, series: OfficialSeries) -> MethodCResult:
    """Repair geometry only, and see whether prediction quality moves."""
    started = time.perf_counter()
    anchor = series.context[-1].close
    outcome = project_path(forecast.raw_decoded)
    after = path_validity(outcome.projected)
    error_after = forecast_error(outcome.projected, series.target, anchor_close=anchor)

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
    total_path_sampling_log_probability: float
    raw_decoded: tuple[OfficialRow, ...]
    valid: bool
    invalid_candle_count: int
    forecast_error: ForecastError


class MethodDResult(BaseModel):
    """Valid-rollout filtering across the fixed seed set, at one origin."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    method: Literal["D_VALID_ROLLOUT_FILTERING"] = "D_VALID_ROLLOUT_FILTERING"
    normalization_state_sha256: str
    settings: InferenceSettings
    seeds: tuple[int, ...]
    rollouts: tuple[RolloutRecord, ...]
    #: One forecast origin. Recorded so no reader mistakes this for a study.
    forecast_origins: Literal[1] = 1

    rollout_count: int = Field(ge=0)
    valid_rollout_count: int = Field(ge=0)
    valid_rollout_fraction: float

    valid_group_mean_primary_error: float | None
    invalid_group_mean_primary_error: float | None

    valid_only_ensemble: tuple[OfficialRow, ...] | None
    valid_only_ensemble_error: ForecastError | None
    #: Arithmetic mean over all 64 separately seeded rollouts. This is NOT the
    #: official sample_count=64 ensemble: that repeats the context along the
    #: batch dimension and draws from one RNG stream. No parity is claimed.
    manual_seeded_ensemble: tuple[OfficialRow, ...] | None
    manual_seeded_ensemble_error: ForecastError | None
    manual_ensemble_is_official: Literal[False] = False

    distinct_token_paths: int = Field(ge=0)
    repeated_path_count: int = Field(ge=0)
    most_repeated_path_occurrences: int = Field(ge=0)
    mean_pairwise_close_dispersion: float | None

    total_seconds: float
    mean_seconds_per_rollout: float


def _ensemble(
    paths: list[tuple[OfficialRow, ...]], sessions: tuple[date, ...]
) -> tuple[OfficialRow, ...] | None:
    """Elementwise mean across paths, in the inverse-normalized domain."""
    if not paths:
        return None
    steps = len(paths[0])
    if any(len(path) != steps for path in paths) or steps != len(sessions):
        return None
    averaged: list[OfficialRow] = []
    for step in range(steps):
        rows = [path[step] for path in paths]
        count = len(rows)
        averaged.append(
            OfficialRow(
                session=sessions[step],
                open=sum(r.open for r in rows) / count,
                high=sum(r.high for r in rows) / count,
                low=sum(r.low for r in rows) / count,
                close=sum(r.close for r in rows) / count,
                volume=sum(r.volume for r in rows) / count,
                amount=sum(r.amount for r in rows) / count,
            )
        )
    return tuple(averaged)


def run_method_d(
    *,
    model: ForecastModel,
    codec: TokenizerCodec,
    series: OfficialSeries,
    state: NormalizationState,
    settings: InferenceSettings = OFFICIAL_INFERENCE_SETTINGS,
    seeds: tuple[int, ...] = ROLLOUT_SEEDS,
) -> MethodDResult:
    """Generate the fixed seed set and compare valid-only against all-rollout.

    When no valid path exists the valid-only ensemble is None and the caller
    emits NO_VALID_ROLLOUTS_OBSERVED. Nothing is projected in and no path is
    substituted.
    """
    started = time.perf_counter()
    sessions = _target_sessions(series)
    anchor = series.context[-1].close
    records: list[RolloutRecord] = []

    for index, seed in enumerate(seeds):
        generated = _generate(model, series, state, settings, seed)
        # The official decode, from the backend. Never a generated-only decode.
        decoded = generated.raw_decoded_suffix
        if len(decoded) != settings.prediction_length:
            raise _fail(
                "DIAGNOSTIC_DECODED_SUFFIX_LENGTH_MISMATCH",
                f"rollout {index} returned {len(decoded)} decoded rows, expected "
                f"{settings.prediction_length}",
            )
        validity = path_validity(decoded)
        coarse, fine = _streams(generated.tokens)
        records.append(
            RolloutRecord(
                index=index,
                seed=seed,
                coarse_token_ids=coarse,
                fine_token_ids=fine,
                total_path_sampling_log_probability=(generated.total_path_sampling_log_probability),
                raw_decoded=decoded,
                valid=not validity.path_is_invalid,
                invalid_candle_count=validity.invalid_candle_count,
                forecast_error=forecast_error(decoded, series.target, anchor_close=anchor),
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

    valid_ensemble = _ensemble([r.raw_decoded for r in valid], sessions)
    manual_ensemble = _ensemble([r.raw_decoded for r in records], sessions)

    signatures = [(r.coarse_token_ids, r.fine_token_ids) for r in records]
    counts: dict[tuple, int] = {}
    for signature in signatures:
        counts[signature] = counts.get(signature, 0) + 1
    repeated = sum(count for count in counts.values() if count > 1)

    dispersion: float | None = None
    all_paths = [r.raw_decoded for r in records]
    if len(all_paths) > 1 and all(len(p) == len(all_paths[0]) for p in all_paths):
        per_step: list[float] = []
        for step in range(len(all_paths[0])):
            closes = [p[step].close for p in all_paths]
            if all(math.isfinite(c) for c in closes):
                mean_close = sum(closes) / len(closes)
                per_step.append(sum(abs(c - mean_close) for c in closes) / len(closes))
        dispersion = mean_of(per_step)

    return MethodDResult(
        normalization_state_sha256=state.state_sha256,
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
            forecast_error(valid_ensemble, series.target, anchor_close=anchor)
            if valid_ensemble is not None
            else None
        ),
        manual_seeded_ensemble=manual_ensemble,
        manual_seeded_ensemble_error=(
            forecast_error(manual_ensemble, series.target, anchor_close=anchor)
            if manual_ensemble is not None
            else None
        ),
        distinct_token_paths=len(counts),
        repeated_path_count=repeated,
        most_repeated_path_occurrences=max(counts.values()) if counts else 0,
        mean_pairwise_close_dispersion=dispersion,
        total_seconds=round(total_seconds, 6),
        mean_seconds_per_rollout=round(total_seconds / len(records), 6) if records else 0.0,
    )


__all__ += ["TimeStamp"]
