"""Run the Kronos-base replication and produce one immutable base artifact.

The measurement is deliberately the mini study's: normalization, methods A to
D, the metrics and the decision rules are imported, not reimplemented. If the
arithmetic differed, a difference between the two studies could not be
attributed to the model family, which is the only thing this study is asking
about.

What is not shared is identity. The artifact below has its own schema version,
its own specification hash, its own model-family field, and its own namespace.
No object written here can be read as a mini result, and no mini result can be
read as a base one.

There is no code path from here into training, an optimizer, a gradient, Stage
B, Stage C, checkpointing, or any held-out partition.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..diagnostic.backends import (
    SAMPLING_PROBABILITY_DEFINITION,
    ForecastModel,
    ResolvedDiagnosticAssets,
    TokenizerCodec,
)
from ..diagnostic.conclusion import (
    ConclusionOutcome,
    DiagnosticConclusion,
    NextExperiment,
    decide,
)
from ..diagnostic.methods import (
    MethodAResult,
    MethodBResult,
    MethodCResult,
    MethodDResult,
    run_method_a,
    run_method_b,
    run_method_c,
    run_method_d,
)
from ..diagnostic.normalization import NormalizationState, fit_context_state
from ..diagnostic.official_input import ColumnPresence
from ..diagnostic.runner import _compare_b_to_rollout_zero, build_official_series
from ..diagnostic.safe_logging import StageTracker
from ..diagnostic.spec import (
    CONTEXT_CANDLES,
    OFFICIAL_INFERENCE_SETTINGS,
    PRIMARY_METRIC,
    ROLLOUT_SEEDS,
    TARGET_CANDLES,
    TOTAL_CANDLES,
    WINDOW,
    InferenceSettings,
)
from ..errors import BridgeFailure, BridgeTransformError, FailureCategory
from ..phase2.measurement import GpuMeasurement, gpu_snapshot, reset_gpu_statistics
from ..phase2.provider import Phase2Provider, RetrievalRequest, validate_series
from ..phase2.states import EvidenceClass
from .invocation import BaseStudyInvocation
from .spec import (
    BASE_CLAIM_BOUNDARY,
    BASE_EXPERIMENT_ID,
    BASE_SPECIFICATION_NAME,
    BASE_SUCCESS_SCHEMA_VERSION,
    KRONOS_BASE_SPEC,
    KRONOS_BASE_TOKENIZER_SPEC,
    MAXIMUM_CONTEXT,
    BaseForecastModelSpec,
    prove_context_budget,
    verify_base_specification,
)

__all__ = [
    "ContextBudgetProof",
    "KronosBaseDiagnosticArtifact",
    "PriorMiniStudyReference",
    "run_kronos_base_diagnostic",
]


def _fail(code: str, message: str) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(category=FailureCategory.INVALID_CONFIGURATION, code=code, message=message)
    )


class ContextBudgetProof(BaseModel):
    """Recorded, not assumed: the window fits and nothing is dropped."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    context_candles: int
    target_candles: int
    maximum_context: int
    sum: int
    fills_budget_exactly: bool
    truncation_occurs: Literal[False] = False


class PriorMiniStudyReference(BaseModel):
    """What this replicates, named exactly, claimed to be nothing more.

    Present so a reader of a base artifact can find the mini study without
    having to guess, and so the artifact itself states that the two are not
    interchangeable rather than leaving that to prose elsewhere.
    """

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    study: Literal["bridge-v0 frozen inference diagnostic (Kronos-mini)"] = (
        "bridge-v0 frozen inference diagnostic (Kronos-mini)"
    )
    operative_specification: str = "phase2-frozen-inference-diagnostic-v4.yaml"
    operative_specification_sha256: str = (
        "bd407722adfc3ebf92eb187828d42c2c9cfa57b2fdc0406121f27d39a5c44977"
    )
    completed_run_id: str = "canary_0a92fde788bd685c"
    completed_artifact_sha256: str = (
        "8e3d8a333c11c1939c5d182db73012fc3e0fc01da42368eb23f71661aadab18c"
    )
    primary_conclusion: str = "ROUNDTRIP_MATERIAL_INVALIDITY"
    model_repository: str = "NeoQuasar/Kronos-mini"
    tokenizer_repository: str = "NeoQuasar/Kronos-Tokenizer-2k"
    results_are_interchangeable: Literal[False] = False


class KronosBaseDiagnosticArtifact(BaseModel):
    """The single immutable base artifact. Authorizes nothing, by type."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    schema_version: Literal["openalpha.bridge.base_study.kronos_base_diagnostic.v1"] = (
        BASE_SUCCESS_SCHEMA_VERSION
    )
    claim_boundary: Literal["DEVELOPMENT DIAGNOSTIC - NOT HOLDOUT OR TRADING EVIDENCE"] = (
        BASE_CLAIM_BOUNDARY
    )
    evidence_class: Literal[EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY] = (
        EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY
    )

    experiment_id: Literal["openalpha-kronos-base-replication-v1"] = BASE_EXPERIMENT_ID
    specification_name: str
    specification_sha256: str
    model_family: Literal["kronos-base"] = "kronos-base"
    prior_study: PriorMiniStudyReference = PriorMiniStudyReference()

    conclusion: DiagnosticConclusion
    matched_findings: tuple[DiagnosticConclusion, ...]
    recommended_next_experiment: NextExperiment
    decision: ConclusionOutcome

    run_id: str
    source_commit: str
    deployed_commit: str

    symbol: str
    frequency: str
    calendar: str
    retrieval_start_inclusive: str
    retrieval_end_exclusive: str
    retrieved_sessions: int
    ordered_columns: tuple[str, ...]
    column_presence: ColumnPresence
    context_candles: int
    target_candles: int
    context_target_boundary: int
    first_session: str
    last_session: str
    candle_data_sha256: str
    provider_request_count: int = Field(ge=0)

    context_budget: ContextBudgetProof
    normalization_state: NormalizationState
    assets: ResolvedDiagnosticAssets
    forecast_model: BaseForecastModelSpec
    tokenizer_repository: str
    tokenizer_revision: str
    inference_settings: InferenceSettings
    rollout_seeds: tuple[int, ...]
    primary_metric: str
    sampling_probability_definition: str

    method_a: MethodAResult
    method_b: MethodBResult
    method_c: MethodCResult
    method_d: MethodDResult

    parameter_sha256_before: str
    parameter_sha256_after: str
    parameters_unmodified: bool

    gpu_measurement: GpuMeasurement
    total_seconds: float
    completed_at: datetime

    training_performed: Literal[False] = False
    optimizer_constructed: Literal[False] = False
    authorizes_training: Literal[False] = False
    authorizes_stage_b: Literal[False] = False
    authorizes_stage_c: Literal[False] = False
    authorizes_test_opening: Literal[False] = False
    authorizes_production_inference: Literal[False] = False
    authorizes_trading_claims: Literal[False] = False
    scientific_result_available: Literal[False] = False
    held_out_partition_opened: Literal[False] = False


def _require_base_pair(assets: ResolvedDiagnosticAssets) -> None:
    """The loaded pair must be the base pair, and paired as released.

    A crossed pair -- base weights with the 2k tokenizer, or mini weights with
    the base tokenizer -- would produce numbers that look like a result and
    measure nothing, because the forecaster was trained against a specific
    token space. This is checked against the resolved identities rather than
    against what the caller intended to load.
    """
    if assets.model_repository != KRONOS_BASE_SPEC.repository:
        raise _fail(
            "BASE_STUDY_WRONG_MODEL_REPOSITORY",
            (
                f"the loaded model is {assets.model_repository}, expected "
                f"{KRONOS_BASE_SPEC.repository}; the base study never runs another family"
            ),
        )
    if assets.tokenizer_repository != KRONOS_BASE_TOKENIZER_SPEC.repository:
        raise _fail(
            "BASE_STUDY_WRONG_TOKENIZER_REPOSITORY",
            (
                f"the loaded tokenizer is {assets.tokenizer_repository}, expected "
                f"{KRONOS_BASE_TOKENIZER_SPEC.repository}; a crossed model and "
                "tokenizer pair measures neither model"
            ),
        )
    if assets.model_revision != KRONOS_BASE_SPEC.revision:
        raise _fail(
            "BASE_STUDY_WRONG_MODEL_REVISION",
            f"the loaded model revision is {assets.model_revision}, expected "
            f"{KRONOS_BASE_SPEC.revision}",
        )
    if assets.tokenizer_revision != KRONOS_BASE_TOKENIZER_SPEC.revision:
        raise _fail(
            "BASE_STUDY_WRONG_TOKENIZER_REVISION",
            f"the loaded tokenizer revision is {assets.tokenizer_revision}, expected "
            f"{KRONOS_BASE_TOKENIZER_SPEC.revision}",
        )
    if KRONOS_BASE_SPEC.paired_tokenizer_repository != KRONOS_BASE_TOKENIZER_SPEC.repository:
        raise _fail(
            "BASE_STUDY_PAIRING_CONTRADICTION",
            (
                f"{KRONOS_BASE_SPEC.name} is released paired with "
                f"{KRONOS_BASE_SPEC.paired_tokenizer_repository}, but the pinned "
                f"tokenizer is {KRONOS_BASE_TOKENIZER_SPEC.repository}"
            ),
        )


class _CountingProvider:
    """Wraps the provider and records every retrieval the study issued."""

    def __init__(self, inner: Phase2Provider) -> None:
        self._inner = inner
        self.requests: list[RetrievalRequest] = []

    def fetch(self, request: RetrievalRequest):
        self.requests.append(request)
        return self._inner.fetch(request)


def run_kronos_base_diagnostic(
    *,
    provider: Phase2Provider,
    codec: TokenizerCodec,
    model: ForecastModel,
    assets: ResolvedDiagnosticAssets,
    invocation: BaseStudyInvocation,
    research_root: Path | str,
    parameter_digest: Any,
    settings: InferenceSettings = OFFICIAL_INFERENCE_SETTINGS,
    seeds: tuple[int, ...] = ROLLOUT_SEEDS,
    now: datetime | None = None,
    stage: StageTracker | None = None,
) -> KronosBaseDiagnosticArtifact:
    """Execute the base replication and compute the preregistered conclusion."""
    started = time.perf_counter()
    stamped = now or datetime.now(UTC)
    reset_gpu_statistics()
    track = (stage or StageTracker()).enter

    specifications = verify_base_specification(research_root)
    _require_base_pair(assets)
    budget = ContextBudgetProof(
        **prove_context_budget(  # type: ignore[arg-type]
            context_candles=CONTEXT_CANDLES,
            target_candles=TARGET_CANDLES,
            maximum_context=MAXIMUM_CONTEXT,
        )
    )
    if settings.max_context != MAXIMUM_CONTEXT:
        raise _fail(
            "BASE_STUDY_UNEXPECTED_MAX_CONTEXT",
            f"the settings declare max_context {settings.max_context}, expected {MAXIMUM_CONTEXT}",
        )
    if not budget.fills_budget_exactly:
        raise _fail(
            "BASE_STUDY_WINDOW_DOES_NOT_FILL_BUDGET",
            (
                f"{budget.context_candles} + {budget.target_candles} = {budget.sum}, "
                f"which is not the maximum context {budget.maximum_context}; the "
                "direct replication requires the exact window"
            ),
        )

    parameter_before = str(parameter_digest())

    track("retrieve_series")
    observing = _CountingProvider(provider)
    series = observing.fetch(
        RetrievalRequest(
            symbol=WINDOW.symbol,
            start=datetime.fromisoformat(WINDOW.start_inclusive).date(),
            end=datetime.fromisoformat(WINDOW.end_exclusive).date(),
            maximum_candles=TOTAL_CANDLES,
        )
    )
    validate_series(series)
    if len(observing.requests) != 1:
        raise _fail(
            "BASE_STUDY_UNEXPECTED_PROVIDER_CALL_COUNT",
            f"exactly one retrieval is permitted, observed {len(observing.requests)}",
        )
    if len(series.candles) != TOTAL_CANDLES:
        raise _fail(
            "BASE_STUDY_UNEXPECTED_SESSION_COUNT",
            f"expected {TOTAL_CANDLES} sessions, retrieved {len(series.candles)}",
        )

    official = build_official_series(series)

    track("fit_normalization")
    state = fit_context_state(official.context)
    if state.fitted_candle_count != CONTEXT_CANDLES:
        raise _fail(
            "NORMALIZATION_FITTED_ON_WRONG_ROWS",
            (
                f"the normalization state was fitted from {state.fitted_candle_count} "
                f"candles, expected exactly {CONTEXT_CANDLES} context candles"
            ),
        )

    track("method_a")
    method_a = run_method_a(codec=codec, series=official, state=state)
    track("method_b")
    method_b = run_method_b(
        model=model, codec=codec, series=official, state=state, settings=settings, seed=seeds[0]
    )
    track("method_c")
    method_c = run_method_c(forecast=method_b, series=official)
    track("method_d")
    method_d = run_method_d(
        model=model, codec=codec, series=official, state=state, settings=settings, seeds=seeds
    )

    for name, observed in (
        ("A", method_a.normalization_state_sha256),
        ("B", method_b.normalization_state_sha256),
        ("D", method_d.normalization_state_sha256),
    ):
        if observed != state.state_sha256:
            raise _fail(
                "NORMALIZATION_STATE_MISMATCH",
                f"method {name} used normalization state {observed}, expected {state.state_sha256}",
            )

    track("verify_parameters")
    parameter_after = str(parameter_digest())
    if parameter_after != parameter_before:
        raise _fail(
            "BASE_STUDY_PARAMETERS_MODIFIED",
            (
                "the frozen parameters changed during the base replication; training "
                "and fine-tuning are prohibited by the specification"
            ),
        )
    if assets.trainable_parameter_count != 0:
        raise _fail(
            "BASE_STUDY_PARAMETERS_NOT_FROZEN",
            (
                f"{assets.trainable_parameter_count} parameters are trainable; every "
                "official parameter must be frozen"
            ),
        )

    track("compute_decision")
    reproducibility = _compare_b_to_rollout_zero(method_b, method_d)
    decision = decide(
        method_a=method_a,
        method_b=method_b,
        method_c=method_c,
        method_d=method_d,
        reproducibility=reproducibility,
    )

    return KronosBaseDiagnosticArtifact(
        specification_name=BASE_SPECIFICATION_NAME,
        specification_sha256=specifications[BASE_SPECIFICATION_NAME],
        conclusion=decision.primary_conclusion,
        matched_findings=decision.matched_findings,
        recommended_next_experiment=decision.recommended_next_experiment,
        decision=decision,
        run_id=invocation.run_id,
        source_commit=invocation.source_commit,
        deployed_commit=invocation.deployed_commit,
        symbol=official.symbol,
        frequency=official.frequency,
        calendar=official.calendar,
        retrieval_start_inclusive=WINDOW.start_inclusive,
        retrieval_end_exclusive=WINDOW.end_exclusive,
        retrieved_sessions=len(official.rows),
        ordered_columns=official.columns,
        column_presence=official.column_presence,
        context_candles=len(official.context),
        target_candles=len(official.target),
        context_target_boundary=official.context_target_boundary,
        first_session=official.sessions[0].isoformat(),
        last_session=official.sessions[-1].isoformat(),
        candle_data_sha256=series.normalized_sha256,
        provider_request_count=len(observing.requests),
        context_budget=budget,
        normalization_state=state,
        assets=assets,
        forecast_model=KRONOS_BASE_SPEC,
        tokenizer_repository=KRONOS_BASE_TOKENIZER_SPEC.repository,
        tokenizer_revision=KRONOS_BASE_TOKENIZER_SPEC.revision,
        inference_settings=settings,
        rollout_seeds=tuple(seeds),
        primary_metric=PRIMARY_METRIC,
        sampling_probability_definition=SAMPLING_PROBABILITY_DEFINITION,
        method_a=method_a,
        method_b=method_b,
        method_c=method_c,
        method_d=method_d,
        parameter_sha256_before=parameter_before,
        parameter_sha256_after=parameter_after,
        parameters_unmodified=True,
        gpu_measurement=gpu_snapshot(),
        total_seconds=round(time.perf_counter() - started, 6),
        completed_at=stamped,
    )
