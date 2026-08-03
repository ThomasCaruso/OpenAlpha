"""The zero-shot benchmark worker body, independent of any compute backend.

Same contract the two completed studies earned: validate every identifier, check
for an existing terminal artifact before anything is loaded or fetched, enter the
official runtime once and keep it entered for the whole computation, and cover
the entire publication with sanitised logging.

There is no import of and no call into either completed study's worker, runner
or artifact namespace, CloudRunner, training, an optimizer, checkpoint code,
Stage B, Stage C, test opening, held-out evaluation, structural projection or
validity filtering. Execution stops after the artifact is written.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from ..cloud.objectstore import ObjectStore
from ..diagnostic.safe_logging import StageTracker, log_operational_failure
from ..errors import BridgeTransformError
from ..phase2.states import EvidenceClass
from .artifact import ZeroShotTerminalArtifact, publish_zero_shot_artifact
from .invocation import ZeroShotInvocation
from .runner import run_zero_shot_benchmark
from .spec import ZERO_SHOT_EXPERIMENT_ID

__all__ = ["ZeroShotWorkerResult", "run_zero_shot_benchmark_worker"]


class ZeroShotWorkerResult(BaseModel):
    """What the worker returns. The evidence is the stored artifact."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    artifact_key: str
    content_sha256: str
    outcome: str
    decision_outcome: str | None
    experiment_id: Literal["openalpha-kronos-zero-shot-benchmark-v1"] = ZERO_SHOT_EXPERIMENT_ID
    model_family: Literal["kronos-base"] = "kronos-base"
    study_type: Literal["zero_shot_forecast_benchmark"] = "zero_shot_forecast_benchmark"
    evidence_class: Literal[EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY] = (
        EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY
    )
    already_existed: bool
    deployed_commit: str
    authorizes_training: Literal[False] = False
    authorizes_fine_tuning: Literal[False] = False
    authorizes_representation_probe: Literal[False] = False
    authorizes_stage_b: Literal[False] = False
    authorizes_stage_c: Literal[False] = False
    authorizes_test_opening: Literal[False] = False
    authorizes_production_inference: Literal[False] = False
    authorizes_trading_claims: Literal[False] = False
    scientific_result_available: Literal[False] = False
    payload: dict[str, Any]


def run_zero_shot_benchmark_worker(
    *,
    store: ObjectStore,
    resolve_runtime: Any,
    provider_factory: Any,
    research_root: Path | str,
    source_commit: str,
    deployed_commit: str,
    run_id: str,
    now: datetime | None = None,
    logger: logging.Logger | None = None,
) -> ZeroShotWorkerResult:
    """Run the benchmark once and store exactly one terminal artifact."""
    stage = StageTracker("validate_invocation")
    invocation = ZeroShotInvocation.validate_all(
        run_id=run_id, source_commit=source_commit, deployed_commit=deployed_commit
    )

    def execute():
        stage.enter("enter_official_runtime")
        with resolve_runtime() as runtime:
            stage.enter("construct_provider")
            provider = provider_factory()
            return run_zero_shot_benchmark(
                provider=provider,
                codec=runtime.codec,
                model=runtime.model,
                assets=runtime.assets,
                invocation=invocation,
                research_root=Path(research_root),
                parameter_digest=runtime.parameter_digest,
                now=now,
                stage=stage,
            )

    def execute_and_log():
        try:
            return execute()
        except BridgeTransformError:
            raise
        except Exception as error:
            log_operational_failure(
                error,
                run_id=invocation.run_id,
                deployed_commit=invocation.deployed_commit,
                stage=stage.stage,
                logger=logger,
            )
            raise

    try:
        artifact: ZeroShotTerminalArtifact = publish_zero_shot_artifact(
            store, invocation=invocation, execute=execute_and_log, now=now, stage=stage
        )
    except BridgeTransformError:
        raise
    except Exception as error:
        log_operational_failure(
            error,
            run_id=invocation.run_id,
            deployed_commit=invocation.deployed_commit,
            stage=stage.stage,
            logger=logger,
        )
        raise

    return ZeroShotWorkerResult(
        artifact_key=artifact.key,
        content_sha256=artifact.content_sha256,
        outcome=artifact.outcome,
        decision_outcome=artifact.decision_outcome,
        already_existed=artifact.already_existed,
        deployed_commit=invocation.deployed_commit,
        payload=artifact.payload,
    )
