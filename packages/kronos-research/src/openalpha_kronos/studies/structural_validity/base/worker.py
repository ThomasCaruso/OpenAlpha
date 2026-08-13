"""The base replication worker body, independent of any compute backend.

Structurally the same contract the mini worker earned: validate every
identifier, check for an existing terminal artifact before anything is loaded,
enter the official runtime once and keep it entered for the whole computation,
and cover the entire publication with sanitised logging.

There is no import of, and no call into, the mini diagnostic worker, the mini
artifact namespace, CloudRunner, training, an optimizer, checkpoint code, Stage
B, Stage C, test opening, or any held-out evaluation. Execution stops after the
artifact is written.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from openalpha_research.failures import ResearchFailureError
from openalpha_research.objectstore import ObjectStore
from pydantic import BaseModel, ConfigDict

from openalpha_kronos.studies.provenance import EvidenceClass
from openalpha_kronos.studies.structural_validity.mini.safe_logging import (
    StageTracker,
    log_operational_failure,
)

from .artifact import BaseStudyTerminalArtifact, publish_base_artifact
from .invocation import BaseStudyInvocation
from .runner import run_kronos_base_diagnostic
from .spec import BASE_EXPERIMENT_ID

__all__ = ["BaseStudyWorkerResult", "run_base_study_worker"]


class BaseStudyWorkerResult(BaseModel):
    """What the worker returns. The evidence is the stored artifact."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    artifact_key: str
    content_sha256: str
    outcome: str
    conclusion: str | None
    experiment_id: Literal["openalpha-kronos-base-replication-v1"] = BASE_EXPERIMENT_ID
    model_family: Literal["kronos-base"] = "kronos-base"
    evidence_class: Literal[EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY] = (
        EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY
    )
    already_existed: bool
    deployed_commit: str
    authorizes_training: Literal[False] = False
    authorizes_stage_b: Literal[False] = False
    authorizes_stage_c: Literal[False] = False
    authorizes_test_opening: Literal[False] = False
    authorizes_production_inference: Literal[False] = False
    authorizes_trading_claims: Literal[False] = False
    scientific_result_available: Literal[False] = False
    payload: dict[str, Any]


def run_base_study_worker(
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
) -> BaseStudyWorkerResult:
    """Run the base replication once and store exactly one terminal artifact."""
    stage = StageTracker("validate_invocation")
    invocation = BaseStudyInvocation.validate_all(
        run_id=run_id, source_commit=source_commit, deployed_commit=deployed_commit
    )

    def execute():
        stage.enter("enter_official_runtime")
        with resolve_runtime() as runtime:
            stage.enter("construct_provider")
            provider = provider_factory()
            return run_kronos_base_diagnostic(
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
        except ResearchFailureError:
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
        artifact: BaseStudyTerminalArtifact = publish_base_artifact(
            store, invocation=invocation, execute=execute_and_log, now=now, stage=stage
        )
    except ResearchFailureError:
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

    return BaseStudyWorkerResult(
        artifact_key=artifact.key,
        content_sha256=artifact.content_sha256,
        outcome=artifact.outcome,
        conclusion=artifact.conclusion,
        already_existed=artifact.already_existed,
        deployed_commit=invocation.deployed_commit,
        payload=artifact.payload,
    )
