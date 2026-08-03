"""The zero-shot benchmark measurement.

One provider request per asset, twenty-five preregistered origins per asset, two
preregistered sampling configurations, eight preregistered seeds. Normalization
is refitted from context rows at every origin, so nothing a later window
contains can reach an earlier one.

Structural validity is recorded and never used. No path is repaired, filtered,
reordered or selected on it, and the decision layer is not given it.
"""

from __future__ import annotations

import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..diagnostic.backends import ForecastModel, ResolvedDiagnosticAssets, TokenizerCodec
from ..diagnostic.metrics import spearman
from ..diagnostic.normalization import (
    ClippingReport,
    NormalizationState,
    clipping_report,
    fit_context_state,
)
from ..diagnostic.official_input import (
    OFFICIAL_COLUMNS,
    ColumnPresence,
    OfficialRow,
    OfficialSeries,
)
from ..diagnostic.safe_logging import StageTracker
from ..diagnostic.validity import path_validity
from ..errors import BridgeFailure, BridgeTransformError, FailureCategory
from ..phase2.measurement import GpuMeasurement, gpu_snapshot, reset_gpu_statistics
from ..phase2.provider import Phase2Provider, RetrievalRequest, validate_series
from .aggregation import (
    BootstrapInterval,
    DistributionSummary,
    ExtendedForecastMetrics,
    StepSummary,
    absolute_return_errors,
    extended_metrics,
    paired_bootstrap,
    relative_skill,
    step_summaries,
    summarize,
)
from .baselines import PRIMARY_BASELINE_ID, build_baselines
from .decision import (
    AssetSupport,
    ConfigurationEvidence,
    ZeroShotDecision,
    decide,
)
from .invocation import ZeroShotInvocation
from .origins import ORIGIN_INDEX_OFFSETS, OriginSelection, resolve_origins, verify_origin_policy
from .spec import (
    ASSET_PANEL,
    BOOTSTRAP_CONFIDENCE_LEVEL,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CALENDAR,
    CONTEXT_CANDLES,
    ENSEMBLE_SEEDS,
    FREQUENCY,
    HORIZON_CANDLES,
    PRIMARY_METRIC,
    REQUIRED_SESSIONS,
    RETRIEVAL_END_EXCLUSIVE,
    RETRIEVAL_MAXIMUM_CANDLES,
    RETRIEVAL_START_INCLUSIVE,
    SAMPLING_CONFIGURATIONS,
    SECONDARY_METRICS,
    WINDOW_CANDLES,
    ZERO_SHOT_CLAIM_BOUNDARY,
    ZERO_SHOT_EXPERIMENT_ID,
    ZERO_SHOT_SPECIFICATION_NAME,
    ZERO_SHOT_SPECIFICATION_SHA256,
    ZERO_SHOT_SUCCESS_SCHEMA_VERSION,
    SamplingConfiguration,
    prove_zero_shot_budget,
    verify_pinned_pair,
    verify_zero_shot_specification,
)

__all__ = [
    "AssetRetrieval",
    "OriginConfigurationResult",
    "OriginResult",
    "PathRecord",
    "ZeroShotBenchmarkArtifact",
    "run_zero_shot_benchmark",
]

_EXPECTED_MODEL_REPOSITORY: Final[str] = "NeoQuasar/Kronos-base"
_EXPECTED_TOKENIZER_REPOSITORY: Final[str] = "NeoQuasar/Kronos-Tokenizer-base"
_EXPECTED_MODEL_REVISION: Final[str] = "2b554741eca47781b64468546e77fef3e85130e6"
_EXPECTED_TOKENIZER_REVISION: Final[str] = "0e0117387f39004a9016484a186a908917e22426"
_MINIMUM_SPEARMAN_POINTS: Final[int] = 3


def _fail(code: str, message: str, *, field: str | None = None) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            field=field,
            message=message,
        )
    )


class _CountingProvider:
    """Wraps the provider so the request count is measured, not asserted."""

    __slots__ = ("_inner", "requests")

    def __init__(self, inner: Phase2Provider) -> None:
        self._inner = inner
        self.requests: list[RetrievalRequest] = []

    def fetch(self, request: RetrievalRequest) -> Any:
        self.requests.append(request)
        return self._inner.fetch(request)


class PathRecord(BaseModel):
    """One generated path. Tokens are kept; decoded rows are not.

    Storing 1600 decoded twelve-row paths would make the terminal artifact large
    without adding identity: the token pair stream plus the normalization state
    determines the decode exactly, and the aggregate metrics are kept alongside.
    """

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    seed: int
    coarse_token_ids: tuple[int, ...]
    fine_token_ids: tuple[int, ...]
    total_path_sampling_log_probability: float
    close_return_mae: float | None = None
    #: Descriptive only. Never used to filter, order, repair or select.
    invalid_candle_count: int = Field(ge=0)
    invalid_candle_fraction: float


class StructuralObservation(BaseModel):
    """Descriptive structural fields. Excluded from every decision rule."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    paths: int = Field(ge=0)
    invalid_paths: int = Field(ge=0)
    mean_invalid_candle_fraction: float | None = None
    ensemble_invalid_candle_count: int = Field(ge=0)
    ensemble_invalid_candle_fraction: float
    invalidity_error_spearman: float | None = None
    used_in_any_decision_rule: Literal[False] = False
    paths_filtered_by_validity: Literal[0] = 0
    paths_repaired: Literal[0] = 0


class BaselineRecord(BaseModel):
    """One baseline scored at one asset-origin."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    baseline_id: str
    is_primary: bool
    accessed_target_data: Literal[False] = False
    metrics: ExtendedForecastMetrics
    relative_skill_against_persistence: float | None = None


class OriginConfigurationResult(BaseModel):
    """One asset-origin under one sampling configuration."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    configuration_label: str
    temperature: float
    paths: tuple[PathRecord, ...]
    #: The preregistered primary candidate.
    ensemble_metrics: ExtendedForecastMetrics
    #: The preregistered secondary candidate: ensemble member zero on its own.
    single_path_metrics: ExtendedForecastMetrics
    ensemble_close_forecast: tuple[float, ...]
    ensemble_relative_skill: float | None = None
    single_path_relative_skill: float | None = None
    #: persistence minus candidate on the primary metric. Positive is favorable.
    ensemble_paired_difference: float | None = None
    beats_persistence: bool
    ensemble_step_absolute_return_errors: tuple[float, ...]
    structural: StructuralObservation
    seconds: float


class OriginResult(BaseModel):
    """Everything measured at one asset-origin."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    asset: str
    ordinal: int = Field(ge=0)
    origin_index: int = Field(ge=0)
    context_first_session: date
    context_last_session: date
    target_first_session: date
    target_last_session: date
    context_candles: int
    horizon_candles: int
    anchor_close: float
    target_closes: tuple[float, ...]
    normalization_state_sha256: str
    normalization_fitted_candle_count: int
    context_clipping: ClippingReport
    persistence_metrics: ExtendedForecastMetrics
    persistence_step_absolute_return_errors: tuple[float, ...]
    baselines: tuple[BaselineRecord, ...]
    configurations: tuple[OriginConfigurationResult, ...]


class AssetRetrieval(BaseModel):
    """One asset's single retrieval, and what was used from it."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    asset: str
    provider_requests: int = Field(ge=0)
    sessions_retrieved: int = Field(ge=0)
    sessions_used: int = Field(ge=0)
    first_session_used: date
    last_session_used: date
    series_sha256: str
    column_presence: ColumnPresence


class AssetAggregate(BaseModel):
    """One asset under one configuration."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    asset: str
    origins_scored: int = Field(ge=0)
    relative_skill: DistributionSummary
    candidate_primary_error: DistributionSummary
    persistence_primary_error: DistributionSummary
    fraction_beating_persistence: float | None = None
    supports_configuration: bool = False


class ConfigurationAggregate(BaseModel):
    """Pooled and per-asset aggregates for one temperature configuration."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    configuration_label: str
    temperature: float
    origins_scored: int = Field(ge=0)
    candidate_primary_error: DistributionSummary
    persistence_primary_error: DistributionSummary
    relative_skill: DistributionSummary
    paired_difference: DistributionSummary
    fraction_beating_persistence: float | None = None
    single_path_relative_skill: DistributionSummary
    secondary_metrics: dict[str, DistributionSummary]
    assets: tuple[AssetAggregate, ...]
    steps: tuple[StepSummary, ...]
    bootstrap: BootstrapInterval


class ZeroShotBenchmarkArtifact(BaseModel):
    """The complete zero-shot benchmark result."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    schema_version: Literal["openalpha.bridge.zero_shot.kronos_zero_shot_benchmark.v1"] = (
        ZERO_SHOT_SUCCESS_SCHEMA_VERSION
    )
    claim_boundary: Literal["DEVELOPMENT BENCHMARK - NOT HOLDOUT OR TRADING EVIDENCE"] = (
        ZERO_SHOT_CLAIM_BOUNDARY
    )
    evidence_class: Literal["development_compatibility_canary"] = (
        "development_compatibility_canary"
    )
    experiment_id: Literal["openalpha-kronos-zero-shot-benchmark-v1"] = ZERO_SHOT_EXPERIMENT_ID
    model_family: Literal["kronos-base"] = "kronos-base"
    study_type: Literal["zero_shot_forecast_benchmark"] = "zero_shot_forecast_benchmark"
    is_structural_repair_experiment: Literal[False] = False
    specification_name: str = ZERO_SHOT_SPECIFICATION_NAME
    specification_sha256: str = ZERO_SHOT_SPECIFICATION_SHA256

    run_id: str
    source_commit: str
    deployed_commit: str

    assets: tuple[str, ...]
    frequency: str
    calendar: str
    retrieval_start_inclusive: str
    retrieval_end_exclusive: str
    required_sessions: int
    context_candles: int
    horizon_candles: int
    context_budget: dict[str, int | bool]
    origin_index_offsets: tuple[int, ...]
    origins_per_asset: int
    total_asset_origins: int
    sampling_configurations: tuple[SamplingConfiguration, ...]
    ensemble_seeds: tuple[int, ...]
    primary_candidate: Literal["ensemble_mean"] = "ensemble_mean"
    secondary_candidate: Literal["deterministic_single_path"] = "deterministic_single_path"
    primary_metric: str = PRIMARY_METRIC
    secondary_metrics: tuple[str, ...] = SECONDARY_METRICS

    retrievals: tuple[AssetRetrieval, ...]
    provider_request_count: int = Field(ge=0)
    origins: tuple[OriginResult, ...]
    aggregates: tuple[ConfigurationAggregate, ...]
    decision: ZeroShotDecision
    outcome_summary: str

    assets_resolved: ResolvedDiagnosticAssets
    total_generations: int = Field(ge=0)
    parameter_sha256_before: str
    parameter_sha256_after: str
    parameters_unmodified: bool
    gpu_measurement: GpuMeasurement
    total_seconds: float
    completed_at: datetime

    training_performed: Literal[False] = False
    optimizer_constructed: Literal[False] = False
    held_out_partition_opened: Literal[False] = False
    forecasts_repaired: Literal[False] = False
    paths_filtered_by_structural_validity: Literal[False] = False
    structural_validity_used_in_decision: Literal[False] = False
    market_backtest_performed: Literal[False] = False
    authorizes_training: Literal[False] = False
    authorizes_fine_tuning: Literal[False] = False
    authorizes_representation_probe: Literal[False] = False
    authorizes_stage_b: Literal[False] = False
    authorizes_stage_c: Literal[False] = False
    authorizes_test_opening: Literal[False] = False
    authorizes_production_inference: Literal[False] = False
    authorizes_trading_claims: Literal[False] = False
    scientific_result_available: Literal[False] = False


def _require_pinned_pair(assets: ResolvedDiagnosticAssets) -> None:
    """The loaded pair must be the preregistered one. No substitution."""
    if assets.model_repository != _EXPECTED_MODEL_REPOSITORY:
        raise _fail(
            "ZERO_SHOT_WRONG_MODEL_REPOSITORY",
            f"loaded model {assets.model_repository}, expected {_EXPECTED_MODEL_REPOSITORY}",
            field="model_repository",
        )
    if assets.tokenizer_repository != _EXPECTED_TOKENIZER_REPOSITORY:
        raise _fail(
            "ZERO_SHOT_WRONG_TOKENIZER_REPOSITORY",
            (
                f"loaded tokenizer {assets.tokenizer_repository}, expected "
                f"{_EXPECTED_TOKENIZER_REPOSITORY}"
            ),
            field="tokenizer_repository",
        )
    if assets.model_revision != _EXPECTED_MODEL_REVISION:
        raise _fail(
            "ZERO_SHOT_WRONG_MODEL_REVISION",
            f"loaded model revision {assets.model_revision}",
            field="model_revision",
        )
    if assets.tokenizer_revision != _EXPECTED_TOKENIZER_REVISION:
        raise _fail(
            "ZERO_SHOT_WRONG_TOKENIZER_REVISION",
            f"loaded tokenizer revision {assets.tokenizer_revision}",
            field="tokenizer_revision",
        )
    if assets.trainable_parameter_count != 0:
        raise _fail(
            "ZERO_SHOT_PARAMETERS_NOT_FROZEN",
            f"{assets.trainable_parameter_count} parameters are trainable, expected zero",
            field="trainable_parameter_count",
        )


def _require_configurations(configurations: tuple[SamplingConfiguration, ...]) -> None:
    """Exactly the two preregistered temperatures, in the declared order."""
    declared = tuple((c.label, c.temperature) for c in SAMPLING_CONFIGURATIONS)
    observed = tuple((c.label, c.temperature) for c in configurations)
    if observed != declared:
        raise _fail(
            "ZERO_SHOT_CONFIGURATIONS_NOT_PREREGISTERED",
            (
                f"configurations {observed} are not the preregistered {declared}; a third "
                "temperature may not be added and the two may not be changed"
            ),
            field="sampling_configurations",
        )


def _require_seeds(seeds: tuple[int, ...]) -> None:
    if seeds != ENSEMBLE_SEEDS:
        raise _fail(
            "ZERO_SHOT_SEEDS_NOT_PREREGISTERED",
            f"seeds {seeds} are not the preregistered {ENSEMBLE_SEEDS}",
            field="ensemble_seeds",
        )


def _rows_from(series: Any) -> tuple[OfficialRow, ...]:
    return tuple(
        OfficialRow(
            session=candle.session,
            open=candle.open,
            high=candle.high,
            low=candle.low,
            close=candle.close,
            volume=candle.volume,
            amount=candle.amount,
        )
        for candle in series.candles
    )


def _window_series(asset: str, rows: tuple[OfficialRow, ...]) -> OfficialSeries:
    official = OfficialSeries(
        symbol=asset,
        frequency=FREQUENCY,
        calendar=CALENDAR,
        columns=OFFICIAL_COLUMNS,
        column_presence=ColumnPresence(volume=True, amount=True),
        rows=rows,
        context_target_boundary=CONTEXT_CANDLES,
    )
    official.assert_contract(expected_rows=WINDOW_CANDLES, expected_boundary=CONTEXT_CANDLES)
    return official


def _mean_path(
    paths: list[tuple[OfficialRow, ...]], sessions: tuple[date, ...]
) -> tuple[OfficialRow, ...]:
    """Per-channel arithmetic mean across paths. No weighting, no trimming."""
    count = len(paths)
    rows: list[OfficialRow] = []
    for step, session in enumerate(sessions):
        channels = [path[step].channels() for path in paths]
        averaged = [sum(values) / count for values in zip(*channels, strict=True)]
        rows.append(
            OfficialRow(
                session=session,
                open=averaged[0],
                high=averaged[1],
                low=averaged[2],
                close=averaged[3],
                volume=averaged[4],
                amount=averaged[5],
            )
        )
    return tuple(rows)


def _generate(
    model: ForecastModel,
    series: OfficialSeries,
    state: NormalizationState,
    configuration: SamplingConfiguration,
    seed: int,
) -> Any:
    return model.generate(
        series.context,
        context_stamps=series.context_stamps(),
        target_stamps=series.target_stamps(),
        target_sessions=tuple(row.session for row in series.target),
        state=state,
        steps=configuration.prediction_length,
        seed=seed,
        temperature=configuration.temperature,
        top_k=configuration.top_k,
        top_p=configuration.top_p,
    )


def _run_origin_configuration(
    *,
    model: ForecastModel,
    series: OfficialSeries,
    state: NormalizationState,
    configuration: SamplingConfiguration,
    seeds: tuple[int, ...],
    anchor: float,
    persistence_primary: float | None,
) -> OriginConfigurationResult:
    started = time.perf_counter()
    target = series.target
    target_sessions = tuple(row.session for row in target)

    decoded_paths: list[tuple[OfficialRow, ...]] = []
    records: list[PathRecord] = []
    for seed in seeds:
        generated = _generate(model, series, state, configuration, seed)
        decoded = generated.raw_decoded_suffix
        if len(decoded) != configuration.prediction_length:
            raise _fail(
                "ZERO_SHOT_DECODED_SUFFIX_LENGTH_MISMATCH",
                (
                    f"the backend returned {len(decoded)} decoded rows, expected "
                    f"{configuration.prediction_length}"
                ),
            )
        decoded_paths.append(decoded)
        validity = path_validity(decoded)
        metrics = extended_metrics(decoded, target, anchor_close=anchor)
        records.append(
            PathRecord(
                seed=seed,
                coarse_token_ids=tuple(pair.coarse for pair in generated.tokens),
                fine_token_ids=tuple(pair.fine for pair in generated.tokens),
                total_path_sampling_log_probability=(
                    generated.total_path_sampling_log_probability
                ),
                close_return_mae=metrics.close_return_mae,
                invalid_candle_count=validity.invalid_candle_count,
                invalid_candle_fraction=validity.invalid_candle_fraction,
            )
        )

    # The primary candidate. Every path contributes; none is dropped, reordered
    # or preferred, and structural validity plays no part in the average.
    ensemble = _mean_path(decoded_paths, target_sessions)
    ensemble_metrics = extended_metrics(ensemble, target, anchor_close=anchor)
    single_metrics = extended_metrics(decoded_paths[0], target, anchor_close=anchor)
    ensemble_validity = path_validity(ensemble)

    fractions = [record.invalid_candle_fraction for record in records]
    errors = [
        record.close_return_mae for record in records if record.close_return_mae is not None
    ]
    correlation = (
        spearman(fractions, errors)
        if len(errors) == len(fractions) and len(errors) >= _MINIMUM_SPEARMAN_POINTS
        else None
    )

    step_errors = absolute_return_errors(ensemble, target, anchor_close=anchor) or []
    candidate_primary = ensemble_metrics.close_return_mae
    paired = (
        persistence_primary - candidate_primary
        if persistence_primary is not None and candidate_primary is not None
        else None
    )

    return OriginConfigurationResult(
        configuration_label=configuration.label,
        temperature=configuration.temperature,
        paths=tuple(records),
        ensemble_metrics=ensemble_metrics,
        single_path_metrics=single_metrics,
        ensemble_close_forecast=tuple(row.close for row in ensemble),
        ensemble_relative_skill=relative_skill(candidate_primary, persistence_primary),
        single_path_relative_skill=relative_skill(
            single_metrics.close_return_mae, persistence_primary
        ),
        ensemble_paired_difference=paired,
        beats_persistence=bool(paired is not None and paired > 0.0),
        ensemble_step_absolute_return_errors=tuple(step_errors),
        structural=StructuralObservation(
            paths=len(records),
            invalid_paths=sum(1 for record in records if record.invalid_candle_count > 0),
            mean_invalid_candle_fraction=(sum(fractions) / len(fractions)) if fractions else None,
            ensemble_invalid_candle_count=ensemble_validity.invalid_candle_count,
            ensemble_invalid_candle_fraction=ensemble_validity.invalid_candle_fraction,
            invalidity_error_spearman=correlation,
        ),
        seconds=round(time.perf_counter() - started, 6),
    )


def _run_origin(
    *,
    model: ForecastModel,
    asset: str,
    rows: tuple[OfficialRow, ...],
    selection: OriginSelection,
    configurations: tuple[SamplingConfiguration, ...],
    seeds: tuple[int, ...],
) -> OriginResult:
    window = rows[selection.context_start_index : selection.target_end_exclusive]
    series = _window_series(asset, window)
    context = series.context
    target = series.target
    target_sessions = tuple(row.session for row in target)

    # Refitted here, from context rows only. Nothing from the target, and
    # nothing from any other origin, can reach this state.
    state = fit_context_state(context)
    if state.fitted_candle_count != CONTEXT_CANDLES:
        raise _fail(
            "ZERO_SHOT_NORMALIZATION_FITTED_ON_WRONG_ROWS",
            (
                f"the normalization state was fitted from {state.fitted_candle_count} rows, "
                f"expected exactly {CONTEXT_CANDLES} context rows"
            ),
        )
    anchor = context[-1].close

    baseline_paths = build_baselines(context, target_sessions=target_sessions)
    mapping = baseline_paths.as_mapping()
    persistence_metrics = extended_metrics(
        mapping[PRIMARY_BASELINE_ID], target, anchor_close=anchor
    )
    persistence_primary = persistence_metrics.close_return_mae

    baselines: list[BaselineRecord] = []
    for baseline_id, path in mapping.items():
        metrics = extended_metrics(path, target, anchor_close=anchor)
        baselines.append(
            BaselineRecord(
                baseline_id=baseline_id,
                is_primary=baseline_id == PRIMARY_BASELINE_ID,
                metrics=metrics,
                relative_skill_against_persistence=relative_skill(
                    metrics.close_return_mae, persistence_primary
                ),
            )
        )

    results = tuple(
        _run_origin_configuration(
            model=model,
            series=series,
            state=state,
            configuration=configuration,
            seeds=seeds,
            anchor=anchor,
            persistence_primary=persistence_primary,
        )
        for configuration in configurations
    )

    persistence_steps = (
        absolute_return_errors(mapping[PRIMARY_BASELINE_ID], target, anchor_close=anchor) or []
    )
    return OriginResult(
        asset=asset,
        ordinal=selection.ordinal,
        origin_index=selection.origin_index,
        context_first_session=selection.context_first_session,
        context_last_session=selection.context_last_session,
        target_first_session=selection.target_first_session,
        target_last_session=selection.target_last_session,
        context_candles=CONTEXT_CANDLES,
        horizon_candles=HORIZON_CANDLES,
        anchor_close=anchor,
        target_closes=tuple(row.close for row in target),
        normalization_state_sha256=state.state_sha256,
        normalization_fitted_candle_count=state.fitted_candle_count,
        context_clipping=clipping_report(state, context),
        persistence_metrics=persistence_metrics,
        persistence_step_absolute_return_errors=tuple(persistence_steps),
        baselines=tuple(baselines),
        configurations=results,
    )


def _aggregate(
    *,
    configuration: SamplingConfiguration,
    origins: tuple[OriginResult, ...],
) -> ConfigurationAggregate:
    skills: list[float] = []
    candidate_errors: list[float] = []
    persistence_errors: list[float] = []
    paired: list[float] = []
    single_skills: list[float] = []
    wins = 0
    scored = 0
    secondary: dict[str, list[float]] = {name: [] for name in SECONDARY_METRICS}
    per_asset: dict[str, list[tuple[float | None, float | None, float | None, bool]]] = {
        asset: [] for asset in ASSET_PANEL
    }
    candidate_steps: list[list[float]] = []
    persistence_steps: list[list[float]] = []

    for origin in origins:
        result = next(
            (r for r in origin.configurations if r.configuration_label == configuration.label),
            None,
        )
        if result is None:
            continue
        scored += 1
        if result.ensemble_relative_skill is not None:
            skills.append(result.ensemble_relative_skill)
        if result.ensemble_metrics.close_return_mae is not None:
            candidate_errors.append(result.ensemble_metrics.close_return_mae)
        if origin.persistence_metrics.close_return_mae is not None:
            persistence_errors.append(origin.persistence_metrics.close_return_mae)
        if result.ensemble_paired_difference is not None:
            paired.append(result.ensemble_paired_difference)
        if result.single_path_relative_skill is not None:
            single_skills.append(result.single_path_relative_skill)
        if result.beats_persistence:
            wins += 1
        metrics = result.ensemble_metrics
        if metrics.close_mae is not None:
            secondary["close_mae"].append(metrics.close_mae)
        if metrics.close_return_rmse is not None:
            secondary["close_return_rmse"].append(metrics.close_return_rmse)
        if metrics.median_absolute_return_error is not None:
            secondary["median_absolute_return_error"].append(
                metrics.median_absolute_return_error
            )
        if metrics.directional is not None and metrics.directional.accuracy is not None:
            secondary["directional_accuracy"].append(metrics.directional.accuracy)
        per_asset.setdefault(origin.asset, []).append(
            (
                result.ensemble_relative_skill,
                metrics.close_return_mae,
                origin.persistence_metrics.close_return_mae,
                result.beats_persistence,
            )
        )
        candidate_steps.append(list(result.ensemble_step_absolute_return_errors))
        persistence_steps.append(list(origin.persistence_step_absolute_return_errors))

    assets: list[AssetAggregate] = []
    for asset in ASSET_PANEL:
        entries = per_asset.get(asset, [])
        asset_skills = [entry[0] for entry in entries if entry[0] is not None]
        asset_candidate = [entry[1] for entry in entries if entry[1] is not None]
        asset_persistence = [entry[2] for entry in entries if entry[2] is not None]
        asset_wins = sum(1 for entry in entries if entry[3])
        fraction = (asset_wins / len(entries)) if entries else None
        skill_summary = summarize(asset_skills)
        supports = bool(
            skill_summary.median is not None
            and skill_summary.median > 0.0
            and fraction is not None
            and fraction >= 0.60
        )
        assets.append(
            AssetAggregate(
                asset=asset,
                origins_scored=len(entries),
                relative_skill=skill_summary,
                candidate_primary_error=summarize(asset_candidate),
                persistence_primary_error=summarize(asset_persistence),
                fraction_beating_persistence=fraction,
                supports_configuration=supports,
            )
        )

    return ConfigurationAggregate(
        configuration_label=configuration.label,
        temperature=configuration.temperature,
        origins_scored=scored,
        candidate_primary_error=summarize(candidate_errors),
        persistence_primary_error=summarize(persistence_errors),
        relative_skill=summarize(skills),
        paired_difference=summarize(paired),
        fraction_beating_persistence=(wins / scored) if scored else None,
        single_path_relative_skill=summarize(single_skills),
        secondary_metrics={name: summarize(values) for name, values in secondary.items()},
        assets=tuple(assets),
        steps=step_summaries(
            candidate_step_errors=candidate_steps,
            persistence_step_errors=persistence_steps,
            horizon=HORIZON_CANDLES,
        ),
        bootstrap=paired_bootstrap(
            paired,
            seed=BOOTSTRAP_SEED,
            resamples=BOOTSTRAP_RESAMPLES,
            confidence_level=BOOTSTRAP_CONFIDENCE_LEVEL,
        ),
    )


def _limitations(
    *, retrievals: tuple[AssetRetrieval, ...], origins: tuple[OriginResult, ...]
) -> tuple[str, ...]:
    """Predeclared Z5 triggers, and only those."""
    found: list[str] = []
    if len(retrievals) != len(ASSET_PANEL):
        found.append("a required asset could not be retrieved under identical semantics")
    for retrieval in retrievals:
        if retrieval.sessions_used != REQUIRED_SESSIONS:
            found.append(f"{retrieval.asset} supplied fewer than the required sessions")
        if retrieval.provider_requests != 1:
            found.append(f"{retrieval.asset} did not receive exactly one provider request")
    expected_origins = len(ASSET_PANEL) * len(ORIGIN_INDEX_OFFSETS)
    if len(origins) != expected_origins:
        found.append("an asset-origin could not form a complete context and target window")
    for origin in origins:
        if not origin.persistence_metrics.defined:
            found.append(f"the primary metric was undefined at {origin.asset} origin {origin.ordinal}")
        for result in origin.configurations:
            if not result.ensemble_metrics.defined:
                found.append(
                    f"a generation or scoring step did not complete at {origin.asset} "
                    f"origin {origin.ordinal}"
                )
    return tuple(dict.fromkeys(found))


def _evidence(aggregate: ConfigurationAggregate) -> ConfigurationEvidence:
    return ConfigurationEvidence(
        label=aggregate.configuration_label,
        temperature=aggregate.temperature,
        origins_scored=aggregate.origins_scored,
        pooled_median_relative_skill=aggregate.relative_skill.median,
        pooled_fraction_beating_persistence=aggregate.fraction_beating_persistence,
        assets=tuple(
            AssetSupport(
                asset=asset.asset,
                origins_scored=asset.origins_scored,
                median_relative_skill=asset.relative_skill.median,
                fraction_beating_persistence=asset.fraction_beating_persistence,
                supports=asset.supports_configuration,
            )
            for asset in aggregate.assets
        ),
        bootstrap=aggregate.bootstrap,
    )


def run_zero_shot_benchmark(
    *,
    provider: Phase2Provider,
    codec: TokenizerCodec,
    model: ForecastModel,
    assets: ResolvedDiagnosticAssets,
    invocation: ZeroShotInvocation,
    research_root: Path | str,
    parameter_digest: Any,
    configurations: tuple[SamplingConfiguration, ...] = SAMPLING_CONFIGURATIONS,
    seeds: tuple[int, ...] = ENSEMBLE_SEEDS,
    now: datetime | None = None,
    stage: StageTracker | None = None,
) -> ZeroShotBenchmarkArtifact:
    """Run the whole benchmark once and return its complete result."""
    tracker = stage or StageTracker()
    started = time.perf_counter()
    reset_gpu_statistics()

    tracker.enter("verify_specification")
    verify_zero_shot_specification(research_root)
    verify_pinned_pair()
    _require_pinned_pair(assets)
    _require_configurations(configurations)
    _require_seeds(seeds)
    verify_origin_policy()
    budget = prove_zero_shot_budget()

    # The codec is not used directly here: every encode and decode happens
    # inside the official generation path, which is the only faithful order.
    # It is still required so a caller cannot supply half a runtime.
    if codec is None:  # pragma: no cover - defensive
        raise _fail("ZERO_SHOT_MISSING_CODEC", "a tokenizer codec is required")

    parameter_before = str(parameter_digest())

    counting = _CountingProvider(provider)
    retrievals: list[AssetRetrieval] = []
    rows_by_asset: dict[str, tuple[OfficialRow, ...]] = {}

    tracker.enter("retrieve_series")
    for asset in ASSET_PANEL:
        before = len(counting.requests)
        series = counting.fetch(
            RetrievalRequest(
                symbol=asset,
                start=date.fromisoformat(RETRIEVAL_START_INCLUSIVE),
                end=date.fromisoformat(RETRIEVAL_END_EXCLUSIVE),
                maximum_candles=RETRIEVAL_MAXIMUM_CANDLES,
            )
        )
        validate_series(series)
        retrieved = _rows_from(series)
        if len(retrieved) < REQUIRED_SESSIONS:
            raise _fail(
                "ZERO_SHOT_INSUFFICIENT_SESSIONS",
                (
                    f"{asset} supplied {len(retrieved)} sessions in the declared window, "
                    f"which is fewer than the {REQUIRED_SESSIONS} the origin policy needs; "
                    "the benchmark fails closed rather than reaching into earlier data"
                ),
                field=asset,
            )
        # Exactly the first REQUIRED_SESSIONS, so provider surplus cannot change
        # which sessions are evaluated.
        used = retrieved[:REQUIRED_SESSIONS]
        rows_by_asset[asset] = used
        retrievals.append(
            AssetRetrieval(
                asset=asset,
                provider_requests=len(counting.requests) - before,
                sessions_retrieved=len(retrieved),
                sessions_used=len(used),
                first_session_used=used[0].session,
                last_session_used=used[-1].session,
                series_sha256=series.normalized_sha256,
                column_presence=ColumnPresence(volume=True, amount=True),
            )
        )

    if len(counting.requests) != len(ASSET_PANEL):
        raise _fail(
            "ZERO_SHOT_UNEXPECTED_PROVIDER_CALL_COUNT",
            (
                f"the provider was called {len(counting.requests)} times, expected exactly "
                f"{len(ASSET_PANEL)}, one per asset"
            ),
        )

    tracker.enter("resolve_origins")
    origins: list[OriginResult] = []
    for asset in ASSET_PANEL:
        rows = rows_by_asset[asset]
        selections = resolve_origins(asset=asset, sessions=tuple(row.session for row in rows))
        for selection in selections:
            tracker.enter("generate_forecasts")
            origins.append(
                _run_origin(
                    model=model,
                    asset=asset,
                    rows=rows,
                    selection=selection,
                    configurations=configurations,
                    seeds=seeds,
                )
            )

    tracker.enter("verify_parameters")
    parameter_after = str(parameter_digest())
    if parameter_before != parameter_after:
        raise _fail(
            "ZERO_SHOT_PARAMETERS_MODIFIED",
            "the parameter digest changed during the benchmark; the model was not frozen",
        )

    tracker.enter("aggregate")
    aggregates = tuple(
        _aggregate(configuration=configuration, origins=tuple(origins))
        for configuration in configurations
    )

    tracker.enter("compute_decision")
    decision = decide(
        configurations=tuple(_evidence(aggregate) for aggregate in aggregates),
        limitations=_limitations(retrievals=tuple(retrievals), origins=tuple(origins)),
    )

    total_generations = len(origins) * len(configurations) * len(seeds)
    return ZeroShotBenchmarkArtifact(
        run_id=invocation.run_id,
        source_commit=invocation.source_commit,
        deployed_commit=invocation.deployed_commit,
        assets=ASSET_PANEL,
        frequency=FREQUENCY,
        calendar=CALENDAR,
        retrieval_start_inclusive=RETRIEVAL_START_INCLUSIVE,
        retrieval_end_exclusive=RETRIEVAL_END_EXCLUSIVE,
        required_sessions=REQUIRED_SESSIONS,
        context_candles=CONTEXT_CANDLES,
        horizon_candles=HORIZON_CANDLES,
        context_budget=budget,
        origin_index_offsets=ORIGIN_INDEX_OFFSETS,
        origins_per_asset=len(ORIGIN_INDEX_OFFSETS),
        total_asset_origins=len(origins),
        sampling_configurations=configurations,
        ensemble_seeds=seeds,
        retrievals=tuple(retrievals),
        provider_request_count=len(counting.requests),
        origins=tuple(origins),
        aggregates=aggregates,
        decision=decision,
        outcome_summary=(
            f"{decision.outcome.value}; {decision.zero_shot_generation_direction.value}"
        ),
        assets_resolved=assets,
        total_generations=total_generations,
        parameter_sha256_before=parameter_before,
        parameter_sha256_after=parameter_after,
        parameters_unmodified=parameter_before == parameter_after,
        gpu_measurement=gpu_snapshot(),
        total_seconds=round(time.perf_counter() - started, 6),
        completed_at=now or datetime.now(UTC),
    )
