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

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from ..cloud.objectstore import ObjectStore
from .cache import FeatureCache
from .canary import run_stage_a_canary
from .canary_artifact import TerminalArtifact, publish_terminal_artifact
from .invocation import WorkerInvocation
from .kronos import KronosBackend
from .provider import Phase2Provider
from .states import EvidenceClass

__all__ = ["CanaryWorkerResult", "run_canary_worker"]


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
    """Execute the canary once and store exactly one terminal artifact.

    Order matters and is enforced by structure rather than by convention:

    1. Every identifier is validated. Nothing below this line runs on an
       unchecked string.
    2. An existing terminal artifact is retrieved and verified. If one exists
       the work is not repeated, so no provider request is issued, no asset is
       resolved, and no cache directory is created.
    3. Only then is anything constructed from the identifiers.
    """
    # 1. Validated before a path, a cache, an object key, a provider call or an
    #    asset resolution can be derived from any of these strings.
    invocation = WorkerInvocation.validate_all(
        run_id=run_id, source_commit=source_commit, deployed_commit=deployed_commit
    )

    def execute():
        # Constructed inside the callable so that a duplicate invocation, which
        # never calls it, creates no cache directory and touches no provider.
        return run_stage_a_canary(
            provider=provider,
            backend=backend,
            cache=FeatureCache(Path(feature_cache_root) / invocation.run_id),
            research_root=Path(research_root),
            source_commit=invocation.source_commit,
            deployed_commit=invocation.deployed_commit,
            run_id=invocation.run_id,
            # Measured from the filesystem before and after asset resolution.
            asset_cache_root=Path(asset_cache_root),
            now=now,
        )

    # 2. Checks for an existing artifact first and returns it verified; only
    #    calls execute when this run id has no terminal result yet.
    artifact: TerminalArtifact = publish_terminal_artifact(
        store,
        invocation=invocation,
        execute=execute,
        now=now,
    )
    return CanaryWorkerResult(
        artifact_key=artifact.key,
        content_sha256=artifact.content_sha256,
        outcome=artifact.outcome,
        already_existed=artifact.already_existed,
        deployed_commit=invocation.deployed_commit,
        report=artifact.payload,
    )
