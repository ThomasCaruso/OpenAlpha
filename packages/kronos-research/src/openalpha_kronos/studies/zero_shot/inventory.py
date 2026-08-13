"""Read-only inventory of this benchmark's artifact namespace.

Lists what exists under the benchmark root and nothing else. It cannot see, and
must never be able to see, either completed study's namespace: the prefix is
fixed here rather than passed in, so no caller can point it at
``openalpha-compatibility/bridge-phase2`` or
``openalpha-compatibility/kronos-base-diagnostic``.

It reads. It never writes, never deletes, and never loads a weight.
"""

from __future__ import annotations

from datetime import datetime
from typing import Final, Literal

from openalpha_research.objectstore import ObjectStore
from pydantic import BaseModel, ConfigDict, Field

from .spec import (
    ZERO_SHOT_ARTIFACT_OBJECT_NAME,
    ZERO_SHOT_ARTIFACT_ROOT,
    ZERO_SHOT_RUN_ID_PATTERN,
)

__all__ = ["ZeroShotArtifactInventory", "ZeroShotArtifactRecord", "inventory_zero_shot_artifacts"]

_RUNS_PREFIX: Final[str] = f"{ZERO_SHOT_ARTIFACT_ROOT}/runs/"

#: Roots this inventory must never report on. Listed so the guarantee is a
#: value a test can read, not a claim in a docstring.
FOREIGN_ROOTS: Final[tuple[str, ...]] = (
    "openalpha-compatibility/bridge-phase2",
    "openalpha-compatibility/kronos-base-diagnostic",
)


class ZeroShotArtifactRecord(BaseModel):
    """One terminal object under the benchmark root."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    key: str
    run_id: str
    run_id_is_well_formed: bool
    is_terminal_object: bool


class ZeroShotArtifactInventory(BaseModel):
    """What exists under the benchmark root, at one moment."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    artifact_root: str = ZERO_SHOT_ARTIFACT_ROOT
    object_name: str = ZERO_SHOT_ARTIFACT_OBJECT_NAME
    records: tuple[ZeroShotArtifactRecord, ...]
    run_count: int = Field(ge=0)
    deployed_commit: str
    inspected_at: datetime
    read_only: Literal[True] = True
    deletion_supported: Literal[False] = False
    weights_loaded: Literal[False] = False
    foreign_roots_inspected: Literal[False] = False


def inventory_zero_shot_artifacts(
    store: ObjectStore, *, deployed_commit: str, inspected_at: datetime
) -> ZeroShotArtifactInventory:
    """List the benchmark's own artifacts. The prefix is not a parameter."""
    keys = store.list_keys(_RUNS_PREFIX)
    records: list[ZeroShotArtifactRecord] = []
    runs: set[str] = set()
    for key in sorted(keys):
        remainder = key[len(_RUNS_PREFIX) :]
        run_id, _, tail = remainder.partition("/")
        runs.add(run_id)
        records.append(
            ZeroShotArtifactRecord(
                key=key,
                run_id=run_id,
                run_id_is_well_formed=bool(ZERO_SHOT_RUN_ID_PATTERN.fullmatch(run_id)),
                is_terminal_object=tail == ZERO_SHOT_ARTIFACT_OBJECT_NAME,
            )
        )
    return ZeroShotArtifactInventory(
        records=tuple(records),
        run_count=len(runs),
        deployed_commit=deployed_commit,
        inspected_at=inspected_at,
    )
