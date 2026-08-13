"""Run identity for the representation probe.

A fourth identifier space. ``canary_``, ``base_``, ``zsb_`` and ``frp_`` are
pairwise disjoint by construction, so no object key of any study can be produced
from another study's identifier. All three foreign prefixes are refused
explicitly and by name, before this study's own pattern is consulted, so the
failure says which study the identifier belongs to.
"""

from __future__ import annotations

import re
from typing import Final

from openalpha_kronos.studies.structural_validity.base.spec import BASE_RUN_ID_PATTERN
from openalpha_kronos.studies.zero_shot.spec import ZERO_SHOT_RUN_ID_PATTERN
from pydantic import BaseModel, ConfigDict

from ..errors import BridgeFailure, BridgeTransformError, FailureCategory
from ..phase2.invocation import RUN_ID_PATTERN as MINI_RUN_ID_PATTERN
from .spec import PROBE_RUN_ID_PATTERN

__all__ = [
    "BASE_RUN_ID_PATTERN",
    "MINI_RUN_ID_PATTERN",
    "ZERO_SHOT_RUN_ID_PATTERN",
    "ProbeInvocation",
]

COMMIT_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{40}$")

_FOREIGN: Final[tuple[tuple[re.Pattern[str], str, str], ...]] = (
    (MINI_RUN_ID_PATTERN, "MINI_RUN_ID_REFUSED_BY_REPRESENTATION_PROBE", "Kronos-mini diagnostic"),
    (
        BASE_RUN_ID_PATTERN,
        "BASE_RUN_ID_REFUSED_BY_REPRESENTATION_PROBE",
        "Kronos-base structural diagnostic",
    ),
    (
        ZERO_SHOT_RUN_ID_PATTERN,
        "ZERO_SHOT_RUN_ID_REFUSED_BY_REPRESENTATION_PROBE",
        "Kronos zero-shot benchmark",
    ),
)


def _fail(code: str, message: str, *, field: str | None = None) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            field=field,
            message=message,
        )
    )


class ProbeInvocation(BaseModel):
    """Representation-probe run identity."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    run_id: str
    source_commit: str
    deployed_commit: str

    @classmethod
    def validate_all(
        cls, *, run_id: object, source_commit: object, deployed_commit: object
    ) -> ProbeInvocation:
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

        for pattern, code, study in _FOREIGN:
            if pattern.fullmatch(run_id):
                raise _fail(
                    code,
                    (
                        f"{run_id} is a {study} run identifier; the representation probe "
                        "must never write under, read from, or reuse it"
                    ),
                    field="run_id",
                )
        if not PROBE_RUN_ID_PATTERN.fullmatch(run_id):
            raise _fail(
                "INVALID_PROBE_RUN_ID",
                (
                    "run_id must match frp_<8-32 lowercase hex>; separators, traversal, "
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
