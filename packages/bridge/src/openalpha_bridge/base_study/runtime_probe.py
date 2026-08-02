"""Does the deployed image actually execute the official base path?

A compatibility probe, not a replication and not empirical evidence. It
retrieves no market data, computes no forecast metric, writes no scientific
conclusion, and authorizes nothing -- including the base diagnostic itself.

The measurement is the mini probe's, because the question is identical and a
second implementation could disagree with the first for reasons that have
nothing to do with the model. What differs is the identity: its own schema, its
own outcome code, and an explicit check that what got loaded is the base pair
and not something that merely also has weights.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..diagnostic.backends import ResolvedDiagnosticAssets
from ..diagnostic.runtime_probe import (
    RuntimeEnvironment,
    RuntimeProbeResult,
    run_frozen_inference_runtime_probe,
)
from ..errors import BridgeFailure, BridgeTransformError, FailureCategory
from .spec import (
    BASE_PROBE_SCHEMA_VERSION,
    KRONOS_BASE_SPEC,
    KRONOS_BASE_TOKENIZER_SPEC,
    MAXIMUM_CONTEXT,
    BaseForecastModelSpec,
)

__all__ = [
    "BASE_PROBE_OUTCOME_PASSED",
    "BaseRuntimeProbeResult",
    "run_base_runtime_probe",
]

BASE_PROBE_OUTCOME_PASSED: Literal["KRONOS_BASE_RUNTIME_PROBE_PASSED"] = (
    "KRONOS_BASE_RUNTIME_PROBE_PASSED"
)


def _fail(code: str, message: str) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(category=FailureCategory.INVALID_CONFIGURATION, code=code, message=message)
    )


class BaseRuntimeProbeResult(BaseModel):
    """A base-family compatibility observation. Authorizes nothing, by type."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    schema_version: Literal["openalpha.bridge.base_study.runtime_probe.v1"] = (
        BASE_PROBE_SCHEMA_VERSION
    )
    #: Deliberately not any outcome the mini probe or either diagnostic can
    #: produce, so a base probe result can never be quoted as a mini one.
    outcome: Literal["KRONOS_BASE_RUNTIME_PROBE_PASSED"] = BASE_PROBE_OUTCOME_PASSED
    claim_boundary: Literal[
        "RUNTIME COMPATIBILITY PROBE - NOT A DIAGNOSTIC AND NOT EMPIRICAL EVIDENCE"
    ] = "RUNTIME COMPATIBILITY PROBE - NOT A DIAGNOSTIC AND NOT EMPIRICAL EVIDENCE"

    model_family: Literal["kronos-base"] = "kronos-base"
    forecast_model: BaseForecastModelSpec
    model_repository: str
    model_revision: str
    tokenizer_repository: str
    tokenizer_revision: str
    maximum_context: int = Field(gt=0)

    measurement: RuntimeProbeResult

    authorizes_the_base_diagnostic: Literal[False] = False


def _require_base_pair(assets: ResolvedDiagnosticAssets) -> None:
    if assets.model_repository != KRONOS_BASE_SPEC.repository:
        raise _fail(
            "BASE_PROBE_WRONG_MODEL_REPOSITORY",
            f"the loaded model is {assets.model_repository}, expected "
            f"{KRONOS_BASE_SPEC.repository}",
        )
    if assets.model_revision != KRONOS_BASE_SPEC.revision:
        raise _fail(
            "BASE_PROBE_WRONG_MODEL_REVISION",
            f"the loaded model revision is {assets.model_revision}, expected "
            f"{KRONOS_BASE_SPEC.revision}",
        )
    if assets.tokenizer_repository != KRONOS_BASE_TOKENIZER_SPEC.repository:
        raise _fail(
            "BASE_PROBE_WRONG_TOKENIZER_REPOSITORY",
            (
                f"the loaded tokenizer is {assets.tokenizer_repository}, expected "
                f"{KRONOS_BASE_TOKENIZER_SPEC.repository}; a crossed pair measures "
                "neither model"
            ),
        )
    if assets.tokenizer_revision != KRONOS_BASE_TOKENIZER_SPEC.revision:
        raise _fail(
            "BASE_PROBE_WRONG_TOKENIZER_REVISION",
            f"the loaded tokenizer revision is {assets.tokenizer_revision}, expected "
            f"{KRONOS_BASE_TOKENIZER_SPEC.revision}",
        )


def run_base_runtime_probe(
    *,
    codec: Any,
    model: Any,
    assets: ResolvedDiagnosticAssets,
    parameter_digest: Any,
    environment: RuntimeEnvironment,
    run_id: str,
    deployed_commit: str,
    now: datetime | None = None,
) -> BaseRuntimeProbeResult:
    """Push one synthetic context through the whole official base path."""
    _require_base_pair(assets)
    measurement = run_frozen_inference_runtime_probe(
        codec=codec,
        model=model,
        assets=assets,
        parameter_digest=parameter_digest,
        environment=environment,
        run_id=run_id,
        deployed_commit=deployed_commit,
        now=now,
    )
    return BaseRuntimeProbeResult(
        forecast_model=KRONOS_BASE_SPEC,
        model_repository=assets.model_repository,
        model_revision=assets.model_revision,
        tokenizer_repository=assets.tokenizer_repository,
        tokenizer_revision=assets.tokenizer_revision,
        maximum_context=MAXIMUM_CONTEXT,
        measurement=measurement,
    )
