"""Run identity for the base study, checked before it can name anything.

Deliberately not a widening of ``WorkerInvocation``. The mini invocation
accepts only ``canary_<hex>`` and this one accepts only ``base_<hex>``, so the
two identifier spaces are disjoint by construction: no string validates under
both, and no object key of one study can be produced from an identifier of the
other.
"""

from __future__ import annotations

import re
from typing import Final

from pydantic import BaseModel, ConfigDict

from ..errors import BridgeFailure, BridgeTransformError, FailureCategory
from ..phase2.invocation import RUN_ID_PATTERN as MINI_RUN_ID_PATTERN
from .spec import BASE_RUN_ID_PATTERN

__all__ = ["MINI_RUN_ID_PATTERN", "BaseStudyInvocation"]

COMMIT_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{40}$")


def _fail(code: str, message: str, *, field: str | None = None) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            field=field,
            message=message,
        )
    )


class BaseStudyInvocation(BaseModel):
    """Base-study run identity."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    run_id: str
    source_commit: str
    deployed_commit: str

    @classmethod
    def validate_all(
        cls, *, run_id: object, source_commit: object, deployed_commit: object
    ) -> BaseStudyInvocation:
        for name, value in (
            ("run_id", run_id),
            ("source_commit", source_commit),
            ("deployed_commit", deployed_commit),
        ):
            if not isinstance(value, str):
                raise _fail(
                    "INVOCATION_IDENTIFIER_NOT_A_STRING",
                    f"{name} must be a string, got {type(value).__name__}",
                    field=name,
                )

        assert isinstance(run_id, str)
        assert isinstance(source_commit, str)
        assert isinstance(deployed_commit, str)

        if MINI_RUN_ID_PATTERN.fullmatch(run_id):
            raise _fail(
                "MINI_RUN_ID_REFUSED_BY_BASE_STUDY",
                (
                    f"{run_id} is a Kronos-mini run identifier; the base study "
                    "must never write under, read from, or reuse a mini identity"
                ),
                field="run_id",
            )
        if not BASE_RUN_ID_PATTERN.fullmatch(run_id):
            raise _fail(
                "INVALID_BASE_RUN_ID",
                (
                    "run_id must match base_<8-32 lowercase hex>; separators, "
                    "traversal, absolute paths and whitespace are refused before "
                    "the value can name a directory or an object key"
                ),
                field="run_id",
            )
        if not COMMIT_PATTERN.fullmatch(source_commit):
            raise _fail(
                "INVALID_SOURCE_COMMIT",
                "source_commit must be exactly forty lowercase hexadecimal characters",
                field="source_commit",
            )
        if not COMMIT_PATTERN.fullmatch(deployed_commit):
            raise _fail(
                "INVALID_DEPLOYED_COMMIT",
                "deployed_commit must be exactly forty lowercase hexadecimal characters",
                field="deployed_commit",
            )
        if source_commit != deployed_commit:
            raise _fail(
                "SOURCE_COMMIT_MISMATCH",
                (
                    f"source_commit {source_commit} does not match the commit baked "
                    f"into the deployed image {deployed_commit}; the worker must run "
                    "the code that was deployed"
                ),
                field="source_commit",
            )
        return cls(run_id=run_id, source_commit=source_commit, deployed_commit=deployed_commit)
