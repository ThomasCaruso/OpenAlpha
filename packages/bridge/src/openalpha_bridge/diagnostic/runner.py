"""Run the four methods and produce one immutable decision artifact.

There is no code path from here into training, an optimizer, a gradient, the
Bridge head, Stage B, Stage C, checkpointing, gate evaluation, or opening any
held-out partition. Exactly one provider retrieval is issued and it is measured,
not asserted.
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
from .backends import ForecastModel, ResolvedDiagnosticAssets, TokenizerCodec
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
from .spec import (
    CONTEXT_CANDLES,
    DIAGNOSTIC_SPECIFICATION_NAME,
    KRONOS_MINI_SPEC,
    OFFICIAL_INFERENCE_SETTINGS,
    PRIMARY_METRIC,
    ROLLOUT_SEEDS,
    TARGET_CANDLES,
    TOTAL_CANDLES,
    WINDOW,
    ForecastModelSpec,
    InferenceSettings,
    verify_diagnostic_specification,
)
from .validity import DecodedCandle

__all__ = [
    "DIAGNOSTIC_SCHEMA_VERSION",
    "DiagnosticArtifact",
    "run_frozen_inference_diagnostic",
]

DIAGNOSTIC_SCHEMA_VERSION: Literal["openalpha.bridge.diagnostic.frozen_inference.v1"] = (
    "openalpha.bridge.diagnostic.frozen_inference.v1"
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

    schema_version: Literal["openalpha.bridge.diagnostic.frozen_inference.v1"] = (
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
    specification_name: str
    specification_sha256: str

    symbol: str
    interval: str
    retrieval_start_inclusive: str
    retrieval_end_exclusive: str
    retrieved_candles: int
    context_candles: int
    target_candles: int
    candle_data_sha256: str
    #: Measured at the diagnostic-to-provider boundary, not asserted.
    provider_request_count: int = Field(ge=0)

    assets: ResolvedDiagnosticAssets
    forecast_model: ForecastModelSpec
    inference_settings: InferenceSettings
    rollout_seeds: tuple[int, ...]
    primary_metric: str

    method_a: MethodAResult
    method_b: MethodBResult
    method_c: MethodCResult
    method_d: MethodDResult

    #: Hashed before and after every method. Equality is the evidence that no
    #: parameter was modified; nothing here asserts it without measuring.
    parameter_sha256_before: str
    parameter_sha256_after: str
    parameters_unmodified: bool

    gpu_measurement: GpuMeasurement
    total_seconds: float
    completed_at: datetime

    # Fixed by type. No conclusion can flip any of these.
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

    # The specification's bytes, verified before anything is executed against it.
    specification_sha256 = verify_diagnostic_specification(research_root)

    parameter_before = str(parameter_digest())

    # Exactly one retrieval, of exactly the amended window.
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
            "DIAGNOSTIC_UNEXPECTED_CANDLE_COUNT",
            f"expected {TOTAL_CANDLES} candles, retrieved {len(series.candles)}",
        )

    full = tuple(
        DecodedCandle(
            open=c.open, high=c.high, low=c.low, close=c.close, volume=c.volume
        )
        for c in series.candles
    )
    context = full[:CONTEXT_CANDLES]
    target = full[CONTEXT_CANDLES:]
    if len(context) != CONTEXT_CANDLES or len(target) != TARGET_CANDLES:
        raise _fail(
            "DIAGNOSTIC_WINDOW_SPLIT_MISMATCH",
            f"split produced {len(context)} context and {len(target)} target candles",
        )
    anchor_close = context[-1].close

    # A: does the decoder itself produce invalid candles from real tokens?
    method_a = run_method_a(codec=codec, candles=full, anchor_close=full[0].close)
    # B: does invalidity enter through generated future tokens?
    method_b = run_method_b(
        model=model,
        codec=codec,
        context=context,
        target=target,
        anchor_close=anchor_close,
        settings=settings,
        seed=seeds[0],
    )
    # C: does repairing geometry alone help prediction quality?
    method_c = run_method_c(forecast=method_b, target=target, anchor_close=anchor_close)
    # D: does conditioning on validity improve forecasting without training?
    method_d = run_method_d(
        model=model,
        codec=codec,
        context=context,
        target=target,
        anchor_close=anchor_close,
        settings=settings,
        seeds=seeds,
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

    decision = decide(
        method_a=method_a, method_b=method_b, method_c=method_c, method_d=method_d
    )

    return DiagnosticArtifact(
        conclusion=decision.conclusion,
        decision=decision,
        run_id=invocation.run_id,
        source_commit=invocation.source_commit,
        deployed_commit=invocation.deployed_commit,
        specification_name=DIAGNOSTIC_SPECIFICATION_NAME,
        specification_sha256=specification_sha256,
        symbol=WINDOW.symbol,
        interval=WINDOW.interval,
        retrieval_start_inclusive=WINDOW.start_inclusive,
        retrieval_end_exclusive=WINDOW.end_exclusive,
        retrieved_candles=len(series.candles),
        context_candles=len(context),
        target_candles=len(target),
        candle_data_sha256=series.normalized_sha256,
        provider_request_count=len(observing.requests),
        assets=assets,
        forecast_model=KRONOS_MINI_SPEC,
        inference_settings=settings,
        rollout_seeds=tuple(seeds),
        primary_metric=PRIMARY_METRIC,
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
