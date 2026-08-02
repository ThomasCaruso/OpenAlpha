"""The Stage A canary worker body, independent of any compute backend.

The logic lives here rather than inside the Modal function so it can be
executed in a test with fakes. A worker whose only tests are assertions about
its source text is a worker nobody has run.

The Modal function is a shell: it builds the real store, provider and backend,
reads the deployed commit out of the image, and calls ``run_canary_worker``.

There is no code path from here into Stage B, Stage C, training, checkpoint
creation or selection, metric or gate evaluation, or test-partition opening.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from ..cloud.objectstore import ObjectStore
from ..errors import BridgeFailure, BridgeTransformError, FailureCategory
from .cache import FeatureCache
from .canary import run_stage_a_canary
from .canary_artifact import TerminalArtifact, publish_terminal_artifact
from .kronos import KronosBackend
from .provider import Phase2Provider
from .states import EvidenceClass

__all__ = ["CanaryWorkerResult", "run_canary_worker"]

_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class CanaryWorkerResult(BaseModel):
    """What the worker returns to its caller.

    A convenience summary. The evidence is the immutable artifact in the object
    store, and this restates its identity rather than standing in for it.
    """

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    artifact_key: str
    content_sha256: str
    outcome: str
    evidence_class: Literal[EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY] = (
        EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY
    )
    already_existed: bool
    authorizes_stage_b: Literal[False] = False
    authorizes_real_run: Literal[False] = False
    scientific_result_available: Literal[False] = False
    deployed_commit: str
    report: dict[str, Any]


def run_canary_worker(
    *,
    store: ObjectStore,
    provider: Phase2Provider,
    backend: KronosBackend,
    feature_cache_root: Path | str,
    asset_cache_root: Path | str,
    research_root: Path | str,
    source_commit: str,
    deployed_commit: str,
    run_id: str,
    now: datetime | None = None,
) -> CanaryWorkerResult:
    """Execute the canary once and store exactly one terminal artifact."""
    if not _COMMIT_PATTERN.fullmatch(deployed_commit):
        raise BridgeTransformError(
            BridgeFailure(
                category=FailureCategory.INVALID_CONFIGURATION,
                code="CANARY_WORKER_INVALID_DEPLOYED_COMMIT",
                message=(
                    "deployed_commit must be exactly forty lowercase hex characters; "
                    "the worker must know what code it is running"
                ),
            )
        )

    def execute():
        return run_stage_a_canary(
            provider=provider,
            backend=backend,
            cache=FeatureCache(Path(feature_cache_root) / run_id),
            research_root=Path(research_root),
            source_commit=source_commit,
            deployed_commit=deployed_commit,
            run_id=run_id,
            # Measured from the filesystem before and after asset resolution.
            asset_cache_root=Path(asset_cache_root),
            now=now,
        )

    artifact: TerminalArtifact = publish_terminal_artifact(
        store,
        run_id=run_id,
        source_commit=source_commit,
        deployed_commit=deployed_commit,
        execute=execute,
        now=now,
    )
    return CanaryWorkerResult(
        artifact_key=artifact.key,
        content_sha256=artifact.content_sha256,
        outcome=artifact.outcome,
        already_existed=artifact.already_existed,
        deployed_commit=deployed_commit,
        report=artifact.payload,
    )
