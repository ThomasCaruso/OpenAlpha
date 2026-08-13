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

from openalpha_research.failures import FailureCategory, ResearchFailure, ResearchFailureError
from openalpha_research.providers import MarketDataProvider, RetrievalRequest, validate_series
from openalpha_research.runtime import GpuMeasurement, gpu_snapshot, reset_gpu_statistics
from pydantic import BaseModel, ConfigDict, Field

from openalpha_kronos.evaluation.metrics import spearman
from openalpha_kronos.evaluation.validity import path_validity
from openalpha_kronos.model.contracts import ForecastModel, ResolvedDiagnosticAssets, TokenizerCodec
from openalpha_kronos.model.input import (
    OFFICIAL_COLUMNS,
    ColumnPresence,
    OfficialRow,
    OfficialSeries,
)
from openalpha_kronos.model.normalization import (
    ClippingReport,
    NormalizationState,
    clipping_report,
    fit_context_state,
)
from openalpha_kronos.studies.structural_validity.mini.safe_logging import StageTracker

from .aggregation import (
    BOOTSTRAP_ASSETS_PER_CLUSTER,
    BOOTSTRAP_AVAILABLE_BLOCK_STARTS,
    BOOTSTRAP_BASE_RESAMPLING_UNIT,
    BOOTSTRAP_BLOCK_LENGTH,
    BOOTSTRAP_BLOCKS_PER_RESAMPLE,
    BOOTSTRAP_CLUSTER_COUNT,
    BOOTSTRAP_METHOD,
    BOOTSTRAP_OBSERVATION_COUNT,
    BOOTSTRAP_TEMPORAL_RESAMPLING_UNIT,
    DistributionSummary,
    ExtendedForecastMetrics,
    MovingBlockBootstrapInterval,
    OriginCluster,
    StepSummary,
    absolute_return_errors,
    extended_metrics,
    paired_origin_moving_block_bootstrap,
    relative_skill,
    step_summaries,
    summarize,
    undefined_moving_block_bootstrap,
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
    "CrossAssetAlignment",
    "OriginConfigurationResult",
    "OriginResult",
    "PathRecord",
    "ZeroShotBenchmarkArtifact",
    "run_zero_shot_benchmark",
    "verify_cross_asset_alignment",
]

_EXPECTED_MODEL_REPOSITORY: Final[str] = "NeoQuasar/Kronos-base"
_EXPECTED_TOKENIZER_REPOSITORY: Final[str] = "NeoQuasar/Kronos-Tokenizer-base"
_EXPECTED_MODEL_REVISION: Final[str] = "2b554741eca47781b64468546e77fef3e85130e6"
_EXPECTED_TOKENIZER_REVISION: Final[str] = "0e0117387f39004a9016484a186a908917e22426"
_MINIMUM_SPEARMAN_POINTS: Final[int] = 3


def _fail(code: str, message: str, *, field: str | None = None) -> ResearchFailureError:
    return ResearchFailureError(
        ResearchFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            field=field,
            message=message,
        )
    )


class _CountingProvider:
    """Wraps the provider so the request count is measured, not asserted."""

    __slots__ = ("_inner", "requests")

    def __init__(self, inner: MarketDataProvider) -> None:
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


class CrossAssetAlignment(BaseModel):
    """Proof that one ordinal is one shared chronological window.

    A cluster only means something if the four ETFs at an ordinal really are the
    same market window. The four are expected to share the XNYS calendar, but
    expectation is not evidence, so this is proved from the retrieved sessions
    before anything is clustered.
    """

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    ordinals_checked: int = Field(ge=0)
    assets_checked: tuple[str, ...]
    target_dates_aligned: Literal[True] = True
    context_dates_aligned: Literal[True] = True
    proved_from_retrieved_sessions: Literal[True] = True


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
    #: The clusters the interval was actually computed from, recorded so the
    #: interval can be recomputed from the artifact alone.
    origin_clusters: tuple[OriginCluster, ...]
    #: The only interval Z1 reads. There is no flat asset-origin alternative and
    #: no independent origin-cluster alternative.
    moving_block_bootstrap: MovingBlockBootstrapInterval


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

    bootstrap_method: str
    bootstrap_base_resampling_unit: str
    bootstrap_temporal_resampling_unit: str
    bootstrap_cluster_count: int
    bootstrap_assets_per_cluster: int
    bootstrap_observation_count: int
    bootstrap_block_length: int
    bootstrap_available_block_starts: int
    bootstrap_blocks_per_resample: int
    bootstrap_resamples: int
    bootstrap_confidence_level: float
    bootstrap_seed: int

    retrievals: tuple[AssetRetrieval, ...]
    provider_request_count: int = Field(ge=0)
    cross_asset_alignment: CrossAssetAlignment
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


def verify_cross_asset_alignment(
    origins: tuple[OriginResult, ...],
) -> CrossAssetAlignment:
    """Prove every ordinal is the same chronological window for all four ETFs.

    Clustering assumes an ordinal names one shared market window. If SPY's
    ordinal 7 covered different sessions from DIA's ordinal 7, a cluster would
    be pooling unrelated windows and the interval built from it would be
    meaningless.

    On any mismatch this fails closed, naming the ordinal and the disagreeing
    assets. It does not shrink the panel, shift an origin, or fall back to
    resampling individual rows -- each of those would silently convert a data
    problem into a narrower interval.
    """
    by_ordinal: dict[int, dict[str, tuple[date, date, date, date]]] = {}
    for origin in origins:
        by_ordinal.setdefault(origin.ordinal, {})[origin.asset] = (
            origin.target_first_session,
            origin.target_last_session,
            origin.context_first_session,
            origin.context_last_session,
        )

    for ordinal in sorted(by_ordinal):
        per_asset = by_ordinal[ordinal]
        missing = [asset for asset in ASSET_PANEL if asset not in per_asset]
        if missing:
            raise _fail(
                "ZERO_SHOT_CROSS_ASSET_ORIGIN_MISALIGNED",
                (
                    f"ordinal {ordinal} is missing {sorted(missing)}; a cluster must carry "
                    f"every asset of the preregistered panel {list(ASSET_PANEL)}"
                ),
                field=f"ordinal_{ordinal}",
            )

        reference_asset = ASSET_PANEL[0]
        reference = per_asset[reference_asset]
        for asset in ASSET_PANEL[1:]:
            observed = per_asset[asset]
            if observed[:2] != reference[:2]:
                raise _fail(
                    "ZERO_SHOT_CROSS_ASSET_ORIGIN_MISALIGNED",
                    (
                        f"ordinal {ordinal} target window disagrees: {reference_asset} covers "
                        f"{reference[0]}..{reference[1]} but {asset} covers "
                        f"{observed[0]}..{observed[1]}; the four assets must resolve to one "
                        "shared chronological window before they can be clustered"
                    ),
                    field=f"ordinal_{ordinal}",
                )
            if observed[2:] != reference[2:]:
                raise _fail(
                    "ZERO_SHOT_CROSS_ASSET_ORIGIN_MISALIGNED",
                    (
                        f"ordinal {ordinal} context window disagrees: {reference_asset} covers "
                        f"{reference[2]}..{reference[3]} but {asset} covers "
                        f"{observed[2]}..{observed[3]}; context alignment is required so a "
                        "cluster represents one market window end to end"
                    ),
                    field=f"ordinal_{ordinal}",
                )

    return CrossAssetAlignment(
        ordinals_checked=len(by_ordinal),
        assets_checked=ASSET_PANEL,
    )


def _build_clusters(
    *, configuration_label: str, origins: tuple[OriginResult, ...]
) -> tuple[tuple[OriginCluster, ...] | None, str | None]:
    """Assemble one cluster per ordinal, or explain why it is impossible.

    Returns ``(None, reason)`` when any paired difference is undefined. The
    bootstrap is never run on a partial panel: a missing observation must become
    a declared limitation, never a quietly smaller and narrower sample.
    """
    by_ordinal: dict[int, dict[str, float | None]] = {}
    for origin in origins:
        result = next(
            (r for r in origin.configurations if r.configuration_label == configuration_label),
            None,
        )
        if result is None:
            return None, (
                f"configuration {configuration_label} is missing at {origin.asset} "
                f"ordinal {origin.ordinal}"
            )
        by_ordinal.setdefault(origin.ordinal, {})[origin.asset] = (
            result.ensemble_paired_difference
        )

    clusters: list[OriginCluster] = []
    for ordinal in sorted(by_ordinal):
        per_asset = by_ordinal[ordinal]
        differences: list[float] = []
        for asset in ASSET_PANEL:
            value = per_asset.get(asset)
            if value is None:
                return None, (
                    f"the paired difference is undefined at ordinal {ordinal} for {asset}, "
                    "so the cluster bootstrap cannot be computed on the full panel"
                )
            differences.append(value)
        clusters.append(
            OriginCluster(
                ordinal=ordinal,
                assets=ASSET_PANEL,
                paired_differences=tuple(differences),
            )
        )
    return tuple(clusters), None


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

    # The decision-bearing interval. Built from whole origin clusters so the
    # four ETFs of one chronological window stay inseparable; `paired` above is
    # kept only for its descriptive mean/median/spread and never resampled.
    clusters, reason = _build_clusters(
        configuration_label=configuration.label, origins=origins
    )
    if clusters is None:
        interval = undefined_moving_block_bootstrap(
            seed=BOOTSTRAP_SEED,
            resamples=BOOTSTRAP_RESAMPLES,
            confidence_level=BOOTSTRAP_CONFIDENCE_LEVEL,
            reason=reason or "the origin clusters could not be assembled",
        )
        clusters = ()
    else:
        interval = paired_origin_moving_block_bootstrap(
            clusters,
            block_length=BOOTSTRAP_BLOCK_LENGTH,
            seed=BOOTSTRAP_SEED,
            resamples=BOOTSTRAP_RESAMPLES,
            confidence_level=BOOTSTRAP_CONFIDENCE_LEVEL,
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
        origin_clusters=clusters,
        moving_block_bootstrap=interval,
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
            if result.ensemble_paired_difference is None:
                # An undefined difference must surface as a limitation. Dropping
                # it would shrink the cluster panel and narrow the interval,
                # which is precisely the failure this design removes.
                found.append(
                    f"the paired difference is undefined at {origin.asset} "
                    f"origin {origin.ordinal}, so the cluster bootstrap is incomplete"
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
        # Only the clustered interval is handed to the decision layer.
        bootstrap=aggregate.moving_block_bootstrap,
    )


def run_zero_shot_benchmark(
    *,
    provider: MarketDataProvider,
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

    # Proved before anything is aggregated or clustered. A cluster is only a
    # shared market window if the four assets really resolve to the same dates.
    tracker.enter("verify_cross_asset_alignment")
    alignment = verify_cross_asset_alignment(tuple(origins))

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
        bootstrap_method=BOOTSTRAP_METHOD,
        bootstrap_base_resampling_unit=BOOTSTRAP_BASE_RESAMPLING_UNIT,
        bootstrap_temporal_resampling_unit=BOOTSTRAP_TEMPORAL_RESAMPLING_UNIT,
        bootstrap_cluster_count=BOOTSTRAP_CLUSTER_COUNT,
        bootstrap_assets_per_cluster=BOOTSTRAP_ASSETS_PER_CLUSTER,
        bootstrap_observation_count=BOOTSTRAP_OBSERVATION_COUNT,
        bootstrap_block_length=BOOTSTRAP_BLOCK_LENGTH,
        bootstrap_available_block_starts=BOOTSTRAP_AVAILABLE_BLOCK_STARTS,
        bootstrap_blocks_per_resample=BOOTSTRAP_BLOCKS_PER_RESAMPLE,
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
        bootstrap_confidence_level=BOOTSTRAP_CONFIDENCE_LEVEL,
        bootstrap_seed=BOOTSTRAP_SEED,
        retrievals=tuple(retrievals),
        provider_request_count=len(counting.requests),
        cross_asset_alignment=alignment,
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
