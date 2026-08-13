"""Does the deployed image execute the official path for this benchmark?

A compatibility probe, not a benchmark and not empirical evidence. It retrieves
no market data, computes no forecast metric, evaluates no decision rule, writes
no conclusion, and authorizes nothing -- including the benchmark itself.

The measurement is the one the base study already validated against real Modal,
because the question is identical and a second implementation could disagree
with the first for reasons that have nothing to do with the model. What differs
is the outer identity: its own schema, its own outcome code, and its own
authorization field, so a benchmark probe result can never be quoted as a
structural-diagnostic probe result.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from openalpha_research.failures import FailureCategory, ResearchFailure, ResearchFailureError
from pydantic import BaseModel, ConfigDict, Field

from openalpha_kronos.model.contracts import ResolvedDiagnosticAssets
from openalpha_kronos.studies.structural_validity.base.runtime_probe import run_base_runtime_probe
from openalpha_kronos.studies.structural_validity.mini.runtime_probe import (
    RuntimeEnvironment,
    RuntimeProbeResult,
)

from .spec import (
    ASSET_PANEL,
    CONTEXT_CANDLES,
    HORIZON_CANDLES,
    MAXIMUM_CONTEXT,
    ZERO_SHOT_EXPERIMENT_ID,
    ZERO_SHOT_MODEL_SPEC,
    ZERO_SHOT_PROBE_SCHEMA_VERSION,
    ZERO_SHOT_TOKENIZER_SPEC,
    verify_pinned_pair,
)

__all__ = [
    "ZERO_SHOT_PROBE_OUTCOME_PASSED",
    "ZeroShotRuntimeProbeResult",
    "run_zero_shot_runtime_probe",
]

ZERO_SHOT_PROBE_OUTCOME_PASSED: Literal["KRONOS_ZERO_SHOT_RUNTIME_PROBE_PASSED"] = (
    "KRONOS_ZERO_SHOT_RUNTIME_PROBE_PASSED"
)


def _fail(code: str, message: str) -> ResearchFailureError:
    return ResearchFailureError(
        ResearchFailure(category=FailureCategory.INVALID_CONFIGURATION, code=code, message=message)
    )


class ZeroShotRuntimeProbeResult(BaseModel):
    """A benchmark-family compatibility observation. Authorizes nothing."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    schema_version: Literal["openalpha.bridge.zero_shot.runtime_probe.v1"] = (
        ZERO_SHOT_PROBE_SCHEMA_VERSION
    )
    #: Deliberately not any outcome the mini probe, the base probe or either
    #: diagnostic can produce.
    outcome: Literal["KRONOS_ZERO_SHOT_RUNTIME_PROBE_PASSED"] = ZERO_SHOT_PROBE_OUTCOME_PASSED
    claim_boundary: Literal[
        "RUNTIME COMPATIBILITY PROBE - NOT A BENCHMARK AND NOT EMPIRICAL EVIDENCE"
    ] = "RUNTIME COMPATIBILITY PROBE - NOT A BENCHMARK AND NOT EMPIRICAL EVIDENCE"

    experiment_id: Literal["openalpha-kronos-zero-shot-benchmark-v1"] = ZERO_SHOT_EXPERIMENT_ID
    study_type: Literal["zero_shot_forecast_benchmark"] = "zero_shot_forecast_benchmark"
    model_family: Literal["kronos-base"] = "kronos-base"
    model_repository: str
    model_revision: str
    tokenizer_repository: str
    tokenizer_revision: str
    maximum_context: int = Field(gt=0)
    benchmark_context_candles: int = CONTEXT_CANDLES
    benchmark_horizon_candles: int = HORIZON_CANDLES
    benchmark_assets: tuple[str, ...] = ASSET_PANEL

    measurement: RuntimeProbeResult

    market_data_retrieved: Literal[False] = False
    decision_rules_evaluated: Literal[False] = False
    authorizes_the_zero_shot_benchmark: Literal[False] = False
    authorizes_the_representation_probe: Literal[False] = False
    authorizes_training: Literal[False] = False
    authorizes_trading_claims: Literal[False] = False


def _require_benchmark_pair(assets: ResolvedDiagnosticAssets) -> None:
    """The probe must have loaded the pair this benchmark preregistered."""
    verify_pinned_pair()
    if assets.model_repository != ZERO_SHOT_MODEL_SPEC.repository:
        raise _fail(
            "ZERO_SHOT_PROBE_WRONG_MODEL_REPOSITORY",
            f"the loaded model is {assets.model_repository}, expected "
            f"{ZERO_SHOT_MODEL_SPEC.repository}",
        )
    if assets.model_revision != ZERO_SHOT_MODEL_SPEC.revision:
        raise _fail(
            "ZERO_SHOT_PROBE_WRONG_MODEL_REVISION",
            f"the loaded model revision is {assets.model_revision}, expected "
            f"{ZERO_SHOT_MODEL_SPEC.revision}",
        )
    if assets.tokenizer_repository != ZERO_SHOT_TOKENIZER_SPEC.repository:
        raise _fail(
            "ZERO_SHOT_PROBE_WRONG_TOKENIZER_REPOSITORY",
            (
                f"the loaded tokenizer is {assets.tokenizer_repository}, expected "
                f"{ZERO_SHOT_TOKENIZER_SPEC.repository}; a crossed pair measures neither model"
            ),
        )
    if assets.tokenizer_revision != ZERO_SHOT_TOKENIZER_SPEC.revision:
        raise _fail(
            "ZERO_SHOT_PROBE_WRONG_TOKENIZER_REVISION",
            f"the loaded tokenizer revision is {assets.tokenizer_revision}, expected "
            f"{ZERO_SHOT_TOKENIZER_SPEC.revision}",
        )


def run_zero_shot_runtime_probe(
    *,
    codec: Any,
    model: Any,
    assets: ResolvedDiagnosticAssets,
    parameter_digest: Any,
    environment: RuntimeEnvironment,
    run_id: str,
    deployed_commit: str,
    now: datetime | None = None,
) -> ZeroShotRuntimeProbeResult:
    """Push one synthetic context through the official path, under this identity.

    Reuses the base study's already validated probe body for the measurement and
    wraps it in this benchmark's own identity. The inner result keeps its own
    authorization fields, all false; this outer result adds its own, also all
    false, so neither can be read as permission for the other's study.
    """
    _require_benchmark_pair(assets)
    inner = run_base_runtime_probe(
        codec=codec,
        model=model,
        assets=assets,
        parameter_digest=parameter_digest,
        environment=environment,
        run_id=run_id,
        deployed_commit=deployed_commit,
        now=now,
    )
    return ZeroShotRuntimeProbeResult(
        model_repository=assets.model_repository,
        model_revision=assets.model_revision,
        tokenizer_repository=assets.tokenizer_repository,
        tokenizer_revision=assets.tokenizer_revision,
        maximum_context=MAXIMUM_CONTEXT,
        measurement=inner.measurement,
    )
