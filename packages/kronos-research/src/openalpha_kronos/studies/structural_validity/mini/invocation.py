"""Validated identity of one Kronos-mini structural-validity invocation."""

from __future__ import annotations

import re
from typing import Final

from openalpha_research.failures import FailureCategory, ResearchFailure, ResearchFailureError
from pydantic import BaseModel, ConfigDict

from ...provenance import MINI_RUN_ID_PATTERN

__all__ = ["COMMIT_PATTERN", "MINI_RUN_ID_PATTERN", "WorkerInvocation"]

COMMIT_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{40}$")


def _fail(code: str, message: str, *, field: str | None = None) -> ResearchFailureError:
    return ResearchFailureError(
        ResearchFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            field=field,
            message=message,
        )
    )


class WorkerInvocation(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    run_id: str
    source_commit: str
    deployed_commit: str

    @classmethod
    def validate_all(
        cls, *, run_id: object, source_commit: object, deployed_commit: object
    ) -> WorkerInvocation:
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
        if not MINI_RUN_ID_PATTERN.fullmatch(run_id):
            raise _fail(
                "INVALID_RUN_ID",
                "run_id must match canary_<8-32 lowercase hex>",
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
                    f"into the deployed image {deployed_commit}"
                ),
                field="source_commit",
            )
        return cls(run_id=run_id, source_commit=source_commit, deployed_commit=deployed_commit)
