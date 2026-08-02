"""A compatibility probe. Not a diagnostic, and not empirical evidence.

It answers one question: does the deployed image actually execute the official
frozen path end to end? It retrieves no market data, computes no forecast
metric, reaches no scientific conclusion, and authorizes nothing.

Its result is deliberately a different type, with a different outcome code and
a different storage name, from anything the frozen diagnostic produces. It can
never be written under the diagnostic's key.
"""

from __future__ import annotations

import time
from datetime import UTC, date, datetime, timedelta
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..errors import BridgeFailure, BridgeTransformError, FailureCategory
from ..phase2.measurement import GpuMeasurement, gpu_snapshot, reset_gpu_statistics
from .backends import ResolvedDiagnosticAssets
from .normalization import NormalizationState, context_matrix, fit_context_state
from .official_input import OFFICIAL_COLUMNS, OfficialRow, official_stamp
from .spec import KRONOS_MINI_SPEC

__all__ = [
    "PROBE_ARTIFACT_NAME",
    "PROBE_OUTCOME_PASSED",
    "PROBE_SCHEMA_VERSION",
    "RuntimeProbeResult",
    "run_frozen_inference_runtime_probe",
    "synthetic_probe_context",
]

PROBE_SCHEMA_VERSION: Final[str] = "openalpha.bridge.diagnostic.runtime_probe.v1"
PROBE_OUTCOME_PASSED: Final[str] = "FROZEN_INFERENCE_RUNTIME_PROBE_PASSED"

#: Its own object name. Never the diagnostic's.
PROBE_ARTIFACT_NAME: Final[str] = "frozen_inference_runtime_probe.json"

#: Small enough to be cheap, long enough to exercise a real encode.
PROBE_CONTEXT_LENGTH: Final[int] = 64
PROBE_STEPS: Final[int] = 1


def _fail(code: str, message: str) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            message=message,
        )
    )


def synthetic_probe_context(length: int = PROBE_CONTEXT_LENGTH) -> tuple[OfficialRow, ...]:
    """A deterministic synthetic six-channel context.

    Fabricated on purpose. The probe must never retrieve market data, and
    synthetic input is enough to answer whether the path executes.
    """
    start = date(2020, 1, 1)
    rows: list[OfficialRow] = []
    level = 100.0
    for index in range(length):
        level = level * (1.0 + 0.0005 * ((index % 7) - 3) / 3.0)
        close = level * 1.0002
        volume = 1.0e6 + index * 10.0
        rows.append(
            OfficialRow(
                session=start + timedelta(days=index),
                open=level,
                high=max(level, close) * 1.003,
                low=min(level, close) * 0.997,
                close=close,
                volume=volume,
                amount=volume * close,
            )
        )
    return tuple(rows)


class RuntimeEnvironment(BaseModel):
    """What the image turned out to be."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    torch_version: str
    torch_cuda_version: str | None
    cuda_available: bool
    device_name: str | None
    device_capability: str | None
    numpy_version: str


class RuntimeProbeResult(BaseModel):
    """A compatibility observation. Authorizes nothing, by type."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    schema_version: Literal["openalpha.bridge.diagnostic.runtime_probe.v1"] = (
        "openalpha.bridge.diagnostic.runtime_probe.v1"
    )
    #: Deliberately not any outcome the diagnostic can produce.
    outcome: Literal["FROZEN_INFERENCE_RUNTIME_PROBE_PASSED"] = PROBE_OUTCOME_PASSED
    claim_boundary: Literal[
        "RUNTIME COMPATIBILITY PROBE - NOT A DIAGNOSTIC AND NOT EMPIRICAL EVIDENCE"
    ] = "RUNTIME COMPATIBILITY PROBE - NOT A DIAGNOSTIC AND NOT EMPIRICAL EVIDENCE"

    run_id: str
    deployed_commit: str
    environment: RuntimeEnvironment
    assets: ResolvedDiagnosticAssets

    synthetic_context_rows: int = Field(gt=0)
    ordered_columns: tuple[str, ...]
    normalization_state: NormalizationState
    encoded_token_count: int = Field(gt=0)
    coarse_token_range: tuple[int, int]
    fine_token_range: tuple[int, int]
    generated_steps: int = Field(gt=0)
    decoded_window_length: int = Field(gt=0)
    decoded_suffix_rows: int = Field(gt=0)
    decoded_channels: int = Field(gt=0)

    parameter_sha256_before: str
    parameter_sha256_after: str
    parameters_unmodified: bool
    total_parameter_count: int = Field(ge=0)
    trainable_parameter_count: int = Field(ge=0)

    gpu_measurement: GpuMeasurement
    elapsed_seconds: float
    completed_at: datetime

    market_data_retrieved: Literal[False] = False
    diagnostic_conclusion_written: Literal[False] = False
    scientific_result_available: Literal[False] = False
    authorizes_training: Literal[False] = False
    authorizes_stage_b: Literal[False] = False
    authorizes_stage_c: Literal[False] = False
    authorizes_test_opening: Literal[False] = False
    authorizes_the_frozen_diagnostic: Literal[False] = False
    authorizes_production_inference: Literal[False] = False
    authorizes_trading_claims: Literal[False] = False


def run_frozen_inference_runtime_probe(
    *,
    codec: Any,
    model: Any,
    assets: ResolvedDiagnosticAssets,
    parameter_digest: Any,
    environment: RuntimeEnvironment,
    run_id: str,
    deployed_commit: str,
    context_length: int = PROBE_CONTEXT_LENGTH,
    steps: int = PROBE_STEPS,
    now: datetime | None = None,
) -> RuntimeProbeResult:
    """Push one synthetic context through the whole official path.

    Normalization, encode, one autoregressive coarse and fine step, the full
    context-plus-generated decode, and the final suffix slice. Everything is
    checked: token bounds, tensor shapes, six channels out, and a parameter
    hash that has not moved.
    """
    if not environment.cuda_available:
        raise _fail(
            "RUNTIME_PROBE_NO_CUDA",
            "CUDA is not available, so the deployed GPU path cannot be exercised",
        )
    if assets.trainable_parameter_count != 0:
        raise _fail(
            "RUNTIME_PROBE_PARAMETERS_NOT_FROZEN",
            f"{assets.trainable_parameter_count} parameters are trainable",
        )

    started = time.perf_counter()
    reset_gpu_statistics()
    before = str(parameter_digest())

    context = synthetic_probe_context(context_length)
    state = fit_context_state(context)
    matrix = context_matrix(context)
    if matrix.shape != (context_length, len(OFFICIAL_COLUMNS)):
        raise _fail(
            "RUNTIME_PROBE_SHAPE_MISMATCH",
            f"the synthetic matrix is {matrix.shape}, expected "
            f"({context_length}, {len(OFFICIAL_COLUMNS)})",
        )

    tokens = codec.encode(context, state=state)
    if len(tokens) != context_length:
        raise _fail(
            "RUNTIME_PROBE_TOKEN_LENGTH_MISMATCH",
            f"encode produced {len(tokens)} tokens for {context_length} rows",
        )
    coarse = [t.coarse for t in tokens]
    fine = [t.fine for t in tokens]
    for name, values, vocabulary in (
        ("coarse", coarse, KRONOS_MINI_SPEC.coarse_vocabulary),
        ("fine", fine, KRONOS_MINI_SPEC.fine_vocabulary),
    ):
        if min(values) < 0 or max(values) >= vocabulary:
            raise _fail(
                "RUNTIME_PROBE_TOKEN_OUT_OF_VOCABULARY",
                f"{name} ids span [{min(values)}, {max(values)}], outside [0, {vocabulary - 1}]",
            )

    target_sessions = tuple(
        context[-1].session + timedelta(days=offset + 1) for offset in range(steps)
    )
    generated = model.generate(
        context,
        context_stamps=tuple(official_stamp(row.session) for row in context),
        target_stamps=tuple(official_stamp(session) for session in target_sessions),
        target_sessions=target_sessions,
        state=state,
        steps=steps,
        seed=0,
        temperature=1.0,
        top_k=0,
        top_p=0.9,
    )
    if len(generated.tokens) != steps:
        raise _fail(
            "RUNTIME_PROBE_TOKEN_LENGTH_MISMATCH",
            f"generate produced {len(generated.tokens)} steps, expected {steps}",
        )
    suffix = generated.raw_decoded_suffix
    if len(suffix) != steps:
        raise _fail(
            "RUNTIME_PROBE_DECODE_SHAPE_MISMATCH",
            f"the decoded suffix is {len(suffix)} rows, expected {steps}",
        )
    for row in suffix:
        if len(row.channels()) != len(OFFICIAL_COLUMNS):
            raise _fail(
                "RUNTIME_PROBE_DECODE_SHAPE_MISMATCH",
                "a decoded row does not carry all six channels",
            )

    after = str(parameter_digest())
    if after != before:
        raise _fail(
            "RUNTIME_PROBE_PARAMETERS_MODIFIED",
            "the frozen parameters changed while the probe was running",
        )

    return RuntimeProbeResult(
        run_id=run_id,
        deployed_commit=deployed_commit,
        environment=environment,
        assets=assets,
        synthetic_context_rows=len(context),
        ordered_columns=OFFICIAL_COLUMNS,
        normalization_state=state,
        encoded_token_count=len(tokens),
        coarse_token_range=(min(coarse), max(coarse)),
        fine_token_range=(min(fine), max(fine)),
        generated_steps=len(generated.tokens),
        decoded_window_length=context_length + steps,
        decoded_suffix_rows=len(suffix),
        decoded_channels=len(OFFICIAL_COLUMNS),
        parameter_sha256_before=before,
        parameter_sha256_after=after,
        parameters_unmodified=True,
        total_parameter_count=assets.total_parameter_count,
        trainable_parameter_count=assets.trainable_parameter_count,
        gpu_measurement=gpu_snapshot(),
        elapsed_seconds=round(time.perf_counter() - started, 6),
        completed_at=now or datetime.now(UTC),
    )
