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
import random
import time
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..errors import BridgeFailure, BridgeTransformError, FailureCategory
from .backends import ForecastModel, GeneratedPath, StepSampling, TokenizerCodec, TokenPair
from .metrics import (
    ForecastError,
    PersistenceComparison,
    ReconstructionError,
    compare_to_persistence,
    forecast_error,
    mean_of,
    reconstruction_error,
    spearman,
)
from .normalization import ClippingReport, NormalizationState, clipping_report
from .official_input import OfficialRow, OfficialSeries, TimeStamp
from .spec import (
    CONTROL_REPETITIONS,
    CONTROL_SEED,
    OFFICIAL_INFERENCE_SETTINGS,
    ROLLOUT_SEEDS,
    InferenceSettings,
)
from .validity import PathValidity, ProjectionOutcome, path_validity, project_path

__all__ = [
    "MethodAResult",
    "MethodBResult",
    "MethodCResult",
    "MethodDResult",
    "RolloutRecord",
    "SizeMatchedControls",
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

    #: How much of the input the clip touched before the tokenizer saw it.
    clipping_all: ClippingReport
    clipping_target_suffix: ClippingReport
    #: Invalidity split by whether the input row was clipped. A row whose input
    #: was distorted by the clip cannot be attributed to the decoder.
    invalid_rows_with_clipped_input: int
    invalid_rows_with_unclipped_input: int
    clipped_input_rows: int
    unclipped_input_rows: int
    invalid_fraction_given_clipped_input: float | None
    invalid_fraction_given_unclipped_input: float | None
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

    clip_all = clipping_report(state, rows)
    clipped_rows = set(clip_all.clipped_row_indices)
    invalid_rows = {c.index for c in validity_all.per_candle if not c.valid}
    clipped_count = len(clipped_rows)
    unclipped_count = len(rows) - clipped_count
    invalid_clipped = len(invalid_rows & clipped_rows)
    invalid_unclipped = len(invalid_rows - clipped_rows)

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
        clipping_all=clip_all,
        clipping_target_suffix=clipping_report(state, rows[-suffix:]),
        invalid_rows_with_clipped_input=invalid_clipped,
        invalid_rows_with_unclipped_input=invalid_unclipped,
        clipped_input_rows=clipped_count,
        unclipped_input_rows=unclipped_count,
        invalid_fraction_given_clipped_input=(
            invalid_clipped / clipped_count if clipped_count else None
        ),
        invalid_fraction_given_unclipped_input=(
            invalid_unclipped / unclipped_count if unclipped_count else None
        ),
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
    persistence: PersistenceComparison
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
        persistence=compare_to_persistence(
            decoded, series.target, anchor_close=anchor, label="method_b_raw"
        ),
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
    persistence_after: PersistenceComparison
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
        persistence_after=compare_to_persistence(
            outcome.projected,
            series.target,
            anchor_close=anchor,
            label="method_c_projected",
        ),
        seconds=round(time.perf_counter() - started, 6),
    )


class SizeMatchedControls(BaseModel):
    """Ensembles of exactly k paths drawn from all rollouts, k = valid count.

    The valid-only ensemble averages k paths while the manual ensemble averages
    all of them, and averaging fewer paths cancels less noise. Comparing the
    two therefore mixes selection-by-validity with ensemble size, and the size
    term alone can dominate: with validity assigned at random and k around a
    fifth of the rollouts, the smaller ensemble looks roughly twice as bad.

    These controls hold k fixed and vary only which paths are chosen, so the
    remaining difference is attributable to validity.
    """

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    k: int = Field(ge=0)
    seed: int
    repetitions: int = Field(ge=0)
    #: Controls actually scored. Fewer than `repetitions` when an error was
    #: undefined, and zero when k is 0.
    scored_repetitions: int = Field(ge=0)
    #: True when k equals the rollout count, so every draw is the same set and
    #: the control is identical to the manual ensemble by construction.
    degenerate: bool
    sampling: Literal["without_replacement_within_each_repetition"] = (
        "without_replacement_within_each_repetition"
    )

    mean_primary_error: float | None = None
    median_primary_error: float | None = None
    standard_deviation_primary_error: float | None = None
    minimum_primary_error: float | None = None
    maximum_primary_error: float | None = None
    #: Fraction of controls whose error is greater than the valid-only error.
    #: 1.0 means valid-only beat every control.
    valid_only_percentile_rank: float | None = None
    relative_improvement_against_mean: float | None = None
    relative_improvement_against_median: float | None = None
    undefined_reason: str | None = None


def _size_matched_controls(
    *,
    paths: list[tuple[OfficialRow, ...]],
    k: int,
    sessions: tuple[date, ...],
    target: tuple[OfficialRow, ...],
    anchor: float,
    valid_only_error: float | None,
    seed: int = CONTROL_SEED,
    repetitions: int = CONTROL_REPETITIONS,
) -> SizeMatchedControls:
    """Draw k paths without replacement, repeatedly, and score each draw."""
    total = len(paths)
    if k <= 0:
        return SizeMatchedControls(
            k=k,
            seed=seed,
            repetitions=repetitions,
            scored_repetitions=0,
            degenerate=False,
            undefined_reason="no valid rollouts, so there is no size to match",
        )
    if k > total:
        return SizeMatchedControls(
            k=k,
            seed=seed,
            repetitions=repetitions,
            scored_repetitions=0,
            degenerate=False,
            undefined_reason=f"k {k} exceeds the {total} available rollouts",
        )

    degenerate = k == total
    # Every draw of k = total is the same set, so one repetition says all there
    # is to say and the rest would be identical work.
    effective = 1 if degenerate else repetitions

    generator = random.Random(seed)
    errors: list[float] = []
    indices = list(range(total))
    for _ in range(effective):
        chosen = generator.sample(indices, k)
        ensemble = _ensemble([paths[i] for i in chosen], sessions)
        if ensemble is None:
            continue
        error = forecast_error(ensemble, target, anchor_close=anchor)
        if error.defined and error.close_return_mae is not None:
            errors.append(error.close_return_mae)

    if not errors:
        return SizeMatchedControls(
            k=k,
            seed=seed,
            repetitions=repetitions,
            scored_repetitions=0,
            degenerate=degenerate,
            undefined_reason="every control ensemble produced an undefined error",
        )

    ordered = sorted(errors)
    count = len(ordered)
    mean_error = sum(ordered) / count
    median_error = (
        ordered[count // 2] if count % 2 else (ordered[count // 2 - 1] + ordered[count // 2]) / 2.0
    )
    variance = sum((value - mean_error) ** 2 for value in ordered) / count
    rank = (
        sum(1 for value in ordered if value > valid_only_error) / count
        if valid_only_error is not None
        else None
    )
    return SizeMatchedControls(
        k=k,
        seed=seed,
        repetitions=repetitions,
        scored_repetitions=count,
        degenerate=degenerate,
        mean_primary_error=mean_error,
        median_primary_error=median_error,
        standard_deviation_primary_error=variance**0.5,
        minimum_primary_error=ordered[0],
        maximum_primary_error=ordered[-1],
        valid_only_percentile_rank=rank,
        relative_improvement_against_mean=(
            1.0 - valid_only_error / mean_error
            if valid_only_error is not None and mean_error
            else None
        ),
        relative_improvement_against_median=(
            1.0 - valid_only_error / median_error
            if valid_only_error is not None and median_error
            else None
        ),
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
    #: Secondary, unconfounded evidence: per-path errors, not ensembles.
    group_absolute_difference: float | None
    group_relative_difference: float | None
    #: Descriptive rank association between a rollout's invalid candle count
    #: and its forecast error, at this origin only.
    invalidity_error_spearman: float | None

    #: The de-confounded comparison the filtering rule uses.
    size_matched_controls: SizeMatchedControls

    valid_only_ensemble: tuple[OfficialRow, ...] | None
    valid_only_ensemble_error: ForecastError | None
    #: Arithmetic mean over all 64 separately seeded rollouts. This is NOT the
    #: official sample_count=64 ensemble: that repeats the context along the
    #: batch dimension and draws from one RNG stream. No parity is claimed.
    manual_seeded_ensemble: tuple[OfficialRow, ...] | None
    manual_seeded_ensemble_error: ForecastError | None
    manual_ensemble_is_official: Literal[False] = False

    #: Persistence comparisons for the two ensembles.
    valid_only_persistence: PersistenceComparison | None
    manual_ensemble_persistence: PersistenceComparison | None

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

    valid_only_error = (
        forecast_error(valid_ensemble, series.target, anchor_close=anchor)
        if valid_ensemble is not None
        else None
    )
    controls = _size_matched_controls(
        paths=[r.raw_decoded for r in records],
        k=len(valid),
        sessions=sessions,
        target=series.target,
        anchor=anchor,
        valid_only_error=(
            valid_only_error.close_return_mae
            if valid_only_error is not None and valid_only_error.defined
            else None
        ),
    )

    valid_mean = mean_of(primaries(valid))
    invalid_mean = mean_of(primaries(invalid))
    absolute_difference = (
        invalid_mean - valid_mean if valid_mean is not None and invalid_mean is not None else None
    )
    relative_difference = (
        absolute_difference / invalid_mean
        if absolute_difference is not None and invalid_mean
        else None
    )
    scorable = [
        r
        for r in records
        if r.forecast_error.defined and r.forecast_error.close_return_mae is not None
    ]
    association = spearman(
        [float(r.invalid_candle_count) for r in scorable],
        [float(r.forecast_error.close_return_mae) for r in scorable],  # type: ignore[arg-type]
    )

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
        valid_group_mean_primary_error=valid_mean,
        invalid_group_mean_primary_error=invalid_mean,
        group_absolute_difference=absolute_difference,
        group_relative_difference=relative_difference,
        invalidity_error_spearman=association,
        size_matched_controls=controls,
        valid_only_ensemble=valid_ensemble,
        valid_only_ensemble_error=valid_only_error,
        manual_seeded_ensemble=manual_ensemble,
        manual_seeded_ensemble_error=(
            forecast_error(manual_ensemble, series.target, anchor_close=anchor)
            if manual_ensemble is not None
            else None
        ),
        valid_only_persistence=compare_to_persistence(
            valid_ensemble, series.target, anchor_close=anchor, label="valid_only_ensemble"
        ),
        manual_ensemble_persistence=compare_to_persistence(
            manual_ensemble, series.target, anchor_close=anchor, label="manual_seeded_ensemble"
        ),
        distinct_token_paths=len(counts),
        repeated_path_count=repeated,
        most_repeated_path_occurrences=max(counts.values()) if counts else 0,
        mean_pairwise_close_dispersion=dispersion,
        total_seconds=round(total_seconds, 6),
        mean_seconds_per_rollout=round(total_seconds / len(records), 6) if records else 0.0,
    )


__all__ += ["TimeStamp"]
