"""Run the four methods and produce one immutable decision artifact.

There is no code path from here into training, an optimizer, a gradient, the
Bridge head, Stage B, Stage C, checkpointing, gate evaluation, or opening any
held-out partition. Exactly one provider retrieval is issued and it is
measured, not asserted.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..errors import BridgeFailure, BridgeTransformError, FailureCategory
from ..phase2.invocation import WorkerInvocation
from ..phase2.measurement import GpuMeasurement, gpu_snapshot, reset_gpu_statistics
from ..phase2.provider import Phase2Provider, RetrievalRequest, validate_series
from .backends import (
    SAMPLING_PROBABILITY_DEFINITION,
    ForecastModel,
    ResolvedDiagnosticAssets,
    TokenizerCodec,
)
from .conclusion import ConclusionOutcome, DiagnosticConclusion, decide
from .methods import (
    MethodAResult,
    MethodBResult,
    MethodCResult,
    MethodDResult,
    run_method_a,
    run_method_b,
    run_method_c,
    run_method_d,
)
from .normalization import NormalizationState, fit_context_state
from .official_input import OFFICIAL_COLUMNS, ColumnPresence, OfficialRow, OfficialSeries
from .spec import (
    CONTEXT_CANDLES,
    KRONOS_MINI_SPEC,
    OFFICIAL_INFERENCE_SETTINGS,
    PRIMARY_METRIC,
    ROLLOUT_SEEDS,
    TOTAL_CANDLES,
    V1_SPECIFICATION_NAME,
    V2_SPECIFICATION_NAME,
    WINDOW,
    ForecastModelSpec,
    InferenceSettings,
    verify_diagnostic_specifications,
)

__all__ = [
    "DIAGNOSTIC_SCHEMA_VERSION",
    "DiagnosticArtifact",
    "build_official_series",
    "run_frozen_inference_diagnostic",
]

DIAGNOSTIC_SCHEMA_VERSION: Literal["openalpha.bridge.diagnostic.frozen_inference.v2"] = (
    "openalpha.bridge.diagnostic.frozen_inference.v2"
)


def _fail(code: str, message: str) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            message=message,
        )
    )


class DiagnosticArtifact(BaseModel):
    """The single immutable artifact. Authorizes nothing, by type."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    schema_version: Literal["openalpha.bridge.diagnostic.frozen_inference.v2"] = (
        DIAGNOSTIC_SCHEMA_VERSION
    )
    claim_boundary: Literal["DEVELOPMENT DIAGNOSTIC - NOT HOLDOUT OR TRADING EVIDENCE"] = (
        "DEVELOPMENT DIAGNOSTIC - NOT HOLDOUT OR TRADING EVIDENCE"
    )
    conclusion: DiagnosticConclusion
    decision: ConclusionOutcome

    run_id: str
    source_commit: str
    deployed_commit: str
    specification_v1_name: str
    specification_v1_sha256: str
    specification_v2_name: str
    specification_v2_sha256: str
    operative_specification: str

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

    normalization_state: NormalizationState
    assets: ResolvedDiagnosticAssets
    forecast_model: ForecastModelSpec
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


class _CountingProvider:
    """Wraps the provider and records every retrieval the diagnostic issued."""

    def __init__(self, inner: Phase2Provider) -> None:
        self._inner = inner
        self.requests: list[RetrievalRequest] = []

    def fetch(self, request: RetrievalRequest):
        self.requests.append(request)
        return self._inner.fetch(request)


def build_official_series(series: Any) -> OfficialSeries:
    """Carry the retrieved series into the official six-channel identity.

    The provider supplies amount, so nothing is synthesised here. If it ever
    stopped doing so, the presence mask would say so rather than the official
    fallback quietly manufacturing a column that then reads as retrieved.
    """
    rows = tuple(
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
    official = OfficialSeries(
        symbol=series.symbol,
        frequency=WINDOW.frequency,
        calendar=WINDOW.calendar,
        columns=OFFICIAL_COLUMNS,
        column_presence=ColumnPresence(volume=True, amount=True),
        rows=rows,
        context_target_boundary=CONTEXT_CANDLES,
    )
    official.assert_contract(expected_rows=TOTAL_CANDLES, expected_boundary=CONTEXT_CANDLES)
    return official


def run_frozen_inference_diagnostic(
    *,
    provider: Phase2Provider,
    codec: TokenizerCodec,
    model: ForecastModel,
    assets: ResolvedDiagnosticAssets,
    invocation: WorkerInvocation,
    research_root: Path | str,
    parameter_digest: Any,
    settings: InferenceSettings = OFFICIAL_INFERENCE_SETTINGS,
    seeds: tuple[int, ...] = ROLLOUT_SEEDS,
    now: datetime | None = None,
) -> DiagnosticArtifact:
    """Execute methods A, B, C and D and compute the preregistered conclusion.

    ``parameter_digest`` is a zero-argument callable returning a hash of every
    loaded parameter. It is called before and after, and a difference is a
    typed failure: the specification prohibits parameter updates, so the
    prohibition is checked rather than trusted.
    """
    started = time.perf_counter()
    stamped = now or datetime.now(UTC)
    reset_gpu_statistics()

    specifications = verify_diagnostic_specifications(research_root)
    parameter_before = str(parameter_digest())

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
            "DIAGNOSTIC_UNEXPECTED_PROVIDER_CALL_COUNT",
            f"exactly one retrieval is permitted, observed {len(observing.requests)}",
        )
    if len(series.candles) != TOTAL_CANDLES:
        raise _fail(
            "DIAGNOSTIC_UNEXPECTED_SESSION_COUNT",
            f"expected {TOTAL_CANDLES} sessions, retrieved {len(series.candles)}",
        )

    official = build_official_series(series)

    # Fitted exactly once, from the 448 context rows only. Every method is
    # handed this object; none may fit its own.
    state = fit_context_state(official.context)
    if state.fitted_candle_count != CONTEXT_CANDLES:
        raise _fail(
            "NORMALIZATION_FITTED_ON_WRONG_ROWS",
            (
                f"the normalization state was fitted from {state.fitted_candle_count} "
                f"candles, expected exactly {CONTEXT_CANDLES} context candles"
            ),
        )

    method_a = run_method_a(codec=codec, series=official, state=state)
    method_b = run_method_b(
        model=model,
        codec=codec,
        series=official,
        state=state,
        settings=settings,
        seed=seeds[0],
    )
    method_c = run_method_c(forecast=method_b, series=official)
    method_d = run_method_d(
        model=model,
        codec=codec,
        series=official,
        state=state,
        settings=settings,
        seeds=seeds,
    )

    # Every method must have used the one fitted state.
    for name, observed in (
        ("A", method_a.normalization_state_sha256),
        ("B", method_b.normalization_state_sha256),
        ("D", method_d.normalization_state_sha256),
    ):
        if observed != state.state_sha256:
            raise _fail(
                "NORMALIZATION_STATE_MISMATCH",
                (
                    f"method {name} used normalization state {observed}, expected "
                    f"{state.state_sha256}"
                ),
            )

    parameter_after = str(parameter_digest())
    if parameter_after != parameter_before:
        raise _fail(
            "DIAGNOSTIC_PARAMETERS_MODIFIED",
            (
                "the frozen parameters changed during the diagnostic; training and "
                "fine-tuning are prohibited by the specification"
            ),
        )
    if assets.trainable_parameter_count != 0:
        raise _fail(
            "DIAGNOSTIC_PARAMETERS_NOT_FROZEN",
            (
                f"{assets.trainable_parameter_count} parameters are trainable; every "
                "official parameter must be frozen"
            ),
        )

    decision = decide(method_a=method_a, method_b=method_b, method_c=method_c, method_d=method_d)

    return DiagnosticArtifact(
        conclusion=decision.conclusion,
        decision=decision,
        run_id=invocation.run_id,
        source_commit=invocation.source_commit,
        deployed_commit=invocation.deployed_commit,
        specification_v1_name=V1_SPECIFICATION_NAME,
        specification_v1_sha256=specifications[V1_SPECIFICATION_NAME],
        specification_v2_name=V2_SPECIFICATION_NAME,
        specification_v2_sha256=specifications[V2_SPECIFICATION_NAME],
        operative_specification=V2_SPECIFICATION_NAME,
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
        normalization_state=state,
        assets=assets,
        forecast_model=KRONOS_MINI_SPEC,
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
