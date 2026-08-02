"""The frozen diagnostic worker body, independent of any compute backend.

The logic lives here rather than inside the Modal function so it can be
executed in a test with doubles. The Modal function is a shell.

There is no import of, and no call into, CloudRunner, training, an optimizer,
checkpoint code, Stage B, Stage C, test opening, or any held-out evaluation.
Execution stops after the artifact is written.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from ..cloud.objectstore import ObjectStore
from ..phase2.invocation import WorkerInvocation
from ..phase2.states import EvidenceClass
from .artifact import DiagnosticTerminalArtifact, publish_diagnostic_artifact
from .runner import run_frozen_inference_diagnostic

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
    resolve: Any,
    provider_factory: Any,
    research_root: Path | str,
    source_commit: str,
    deployed_commit: str,
    run_id: str,
    now: datetime | None = None,
) -> DiagnosticWorkerResult:
    """Run the diagnostic once and store exactly one terminal artifact.

    Order is enforced by structure:

    1. Every identifier is validated. Nothing below runs on an unchecked
       string, so no path, cache or object key is built from one.
    2. An existing terminal artifact is retrieved and verified. If one exists,
       neither the provider nor the model is touched: ``resolve`` and
       ``provider_factory`` are callables and are never invoked.
    3. Only then are the assets resolved and the provider constructed.

    ``resolve`` returns ``(codec, model, assets, parameter_digest)``. It is a
    callable rather than a value so that a duplicate invocation genuinely
    avoids loading weights rather than merely avoiding using them.
    """
    invocation = WorkerInvocation.validate_all(
        run_id=run_id, source_commit=source_commit, deployed_commit=deployed_commit
    )

    def execute():
        codec, model, assets, parameter_digest = resolve()
        return run_frozen_inference_diagnostic(
            provider=provider_factory(),
            codec=codec,
            model=model,
            assets=assets,
            invocation=invocation,
            research_root=Path(research_root),
            parameter_digest=parameter_digest,
            now=now,
        )

    artifact: DiagnosticTerminalArtifact = publish_diagnostic_artifact(
        store, invocation=invocation, execute=execute, now=now
    )
    return DiagnosticWorkerResult(
        artifact_key=artifact.key,
        content_sha256=artifact.content_sha256,
        outcome=artifact.outcome,
        conclusion=artifact.conclusion,
        already_existed=artifact.already_existed,
        deployed_commit=invocation.deployed_commit,
        payload=artifact.payload,
    )
