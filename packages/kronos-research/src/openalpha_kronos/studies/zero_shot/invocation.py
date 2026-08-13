"""Run identity for the zero-shot benchmark, checked before it names anything.

A third identifier space. ``canary_``, ``base_`` and ``zsb_`` are pairwise
disjoint by construction: no string validates under more than one, so no object
key of any study can be produced from another study's identifier. Both foreign
prefixes are refused explicitly and by name, before this study's own pattern is
consulted, so the failure says which study the identifier belongs to.
"""

from __future__ import annotations

import re
from typing import Final

from openalpha_research.failures import FailureCategory, ResearchFailure, ResearchFailureError
from pydantic import BaseModel, ConfigDict

from openalpha_kronos.studies.provenance import MINI_RUN_ID_PATTERN
from openalpha_kronos.studies.structural_validity.base.spec import BASE_RUN_ID_PATTERN

from .spec import ZERO_SHOT_RUN_ID_PATTERN

__all__ = ["BASE_RUN_ID_PATTERN", "MINI_RUN_ID_PATTERN", "ZeroShotInvocation"]

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


class ZeroShotInvocation(BaseModel):
    """Zero-shot benchmark run identity."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    run_id: str
    source_commit: str
    deployed_commit: str

    @classmethod
    def validate_all(
        cls, *, run_id: object, source_commit: object, deployed_commit: object
    ) -> ZeroShotInvocation:
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
                "MINI_RUN_ID_REFUSED_BY_ZERO_SHOT_BENCHMARK",
                (
                    f"{run_id} is a Kronos-mini diagnostic run identifier; the zero-shot "
                    "benchmark must never write under, read from, or reuse it"
                ),
                field="run_id",
            )
        if BASE_RUN_ID_PATTERN.fullmatch(run_id):
            raise _fail(
                "BASE_RUN_ID_REFUSED_BY_ZERO_SHOT_BENCHMARK",
                (
                    f"{run_id} is a Kronos-base structural diagnostic run identifier; the "
                    "zero-shot benchmark must never write under, read from, or reuse it"
                ),
                field="run_id",
            )
        if not ZERO_SHOT_RUN_ID_PATTERN.fullmatch(run_id):
            raise _fail(
                "INVALID_ZERO_SHOT_RUN_ID",
                (
                    "run_id must match zsb_<8-32 lowercase hex>; separators, traversal, "
                    "absolute paths and whitespace are refused before the value can name "
                    "a directory or an object key"
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
                    f"source_commit {source_commit} does not match the commit baked into "
                    f"the deployed image {deployed_commit}; the worker must run the code "
                    "that was deployed"
                ),
                field="source_commit",
            )
        return cls(run_id=run_id, source_commit=source_commit, deployed_commit=deployed_commit)
