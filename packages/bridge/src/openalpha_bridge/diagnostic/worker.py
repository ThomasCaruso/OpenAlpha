"""The frozen diagnostic worker body, independent of any compute backend.

The logic lives here rather than inside the Modal function so it can be
executed in a test with doubles. The Modal function is a shell.

There is no import of, and no call into, CloudRunner, training, an optimizer,
checkpoint code, Stage B, Stage C, test opening, or any held-out evaluation.
Execution stops after the artifact is written.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from ..cloud.objectstore import ObjectStore
from ..errors import BridgeTransformError
from ..phase2.invocation import WorkerInvocation
from ..phase2.states import EvidenceClass
from .artifact import DiagnosticTerminalArtifact, publish_diagnostic_artifact
from .runner import run_frozen_inference_diagnostic
from .safe_logging import StageTracker, log_operational_failure

__all__ = ["DiagnosticWorkerResult", "run_diagnostic_worker"]


class DiagnosticWorkerResult(BaseModel):
    """What the worker returns. The evidence is the stored artifact."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    artifact_key: str
    content_sha256: str
    outcome: str
    conclusion: str | None
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


def run_diagnostic_worker(
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
) -> DiagnosticWorkerResult:
    """Run the diagnostic once and store exactly one terminal artifact.

    Order is enforced by structure:

    1. Every identifier is validated. Nothing below runs on an unchecked
       string, so no path, cache or object key is built from one.
    2. An existing terminal artifact is retrieved and verified. If one exists,
       ``resolve_runtime`` is never entered and ``provider_factory`` is never
       called, so no weight is loaded and no retrieval is issued.
    3. Only then is the official runtime entered, and it stays entered for the
       whole computation. The tokenizer and the model do their inference while
       the context that imported them is still open, which is the lifetime the
       runtime probe exercises too.

    4. The whole publication is covered by safe logging, not only the
       computation. Publication can fail on its own -- an unreachable store, an
       undecodable existing artifact, a payload that will not serialise -- and
       in that state no durable artifact can be written at all. The sanitised
       log is then the only evidence there is, so it has to exist.

    ``resolve_runtime`` is a zero-argument callable returning a context manager
    that yields an object with ``codec``, ``model``, ``assets`` and
    ``parameter_digest``.
    """
    stage = StageTracker("validate_invocation")
    invocation = WorkerInvocation.validate_all(
        run_id=run_id, source_commit=source_commit, deployed_commit=deployed_commit
    )

    def execute():
        # publish_diagnostic_artifact calls this only when no terminal artifact
        # exists, so entering the runtime here is what keeps a duplicate
        # invocation from loading anything.
        stage.enter("enter_official_runtime")
        with resolve_runtime() as runtime:
            stage.enter("construct_provider")
            provider = provider_factory()
            return run_frozen_inference_diagnostic(
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
            # Typed failures already say what went wrong, in a message this
            # code wrote. They need no sanitising and no extra logging.
            raise
        except Exception as error:
            # The artifact will carry only the class and a fixed message, so
            # the safe location detail has to go somewhere. It goes here.
            log_operational_failure(
                error,
                run_id=invocation.run_id,
                deployed_commit=invocation.deployed_commit,
                stage=stage.stage,
                logger=logger,
            )
            raise

    try:
        artifact: DiagnosticTerminalArtifact = publish_diagnostic_artifact(
            store, invocation=invocation, execute=execute_and_log, now=now, stage=stage
        )
    except BridgeTransformError:
        # A typed failure that escapes publication is a rejection this code
        # wrote, not an unexpected fault. It needs no sanitising.
        raise
    except Exception as error:
        # Publication itself failed: the object store was unreachable, an
        # existing artifact could not be decoded, a payload would not
        # serialise. Nothing durable can be written in that state, so the
        # sanitised log is the only evidence that will survive. An execution
        # failure that publication already converted into a terminal artifact
        # never reaches here, so it is not logged twice.
        log_operational_failure(
            error,
            run_id=invocation.run_id,
            deployed_commit=invocation.deployed_commit,
            stage=stage.stage,
            logger=logger,
        )
        raise

    return DiagnosticWorkerResult(
        artifact_key=artifact.key,
        content_sha256=artifact.content_sha256,
        outcome=artifact.outcome,
        conclusion=artifact.conclusion,
        already_existed=artifact.already_existed,
        deployed_commit=invocation.deployed_commit,
        payload=artifact.payload,
    )
