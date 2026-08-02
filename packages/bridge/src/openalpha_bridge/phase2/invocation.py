"""The validated identity of one worker invocation.

Validation belongs at the outermost boundary, before anything derived from
these strings exists. A run id that has not been checked must never reach a
filesystem path, a cache directory, an object-store key, a provider, or asset
resolution: by the time a malformed value is caught downstream it has already
been used to name something.

Constructing this object is the check. There is no way to hold an unvalidated
invocation identity.
"""

from __future__ import annotations

import re
from typing import Final

from pydantic import BaseModel, ConfigDict

from ..errors import BridgeFailure, BridgeTransformError, FailureCategory

__all__ = ["COMMIT_PATTERN", "RUN_ID_PATTERN", "WorkerInvocation"]

#: Exactly forty lowercase hex characters. Not a prefix, not a tag, not HEAD.
COMMIT_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{40}$")

#: ``canary_`` followed by 8 to 32 lowercase hex characters. No separator that
#: could traverse a directory, no whitespace, no absolute path.
RUN_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^canary_[0-9a-f]{8,32}$")


def _fail(code: str, message: str, *, field: str | None = None) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            field=field,
            message=message,
        )
    )


class WorkerInvocation(BaseModel):
    """Run identity, checked before it can name anything."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    run_id: str
    #: What the caller asked to run.
    source_commit: str
    #: What the image was built from.
    deployed_commit: str

    @classmethod
    def validate_all(
        cls, *, run_id: object, source_commit: object, deployed_commit: object
    ) -> WorkerInvocation:
        """Check every identifier, or refuse to produce an invocation.

        Types are checked explicitly rather than left to pydantic, so a caller
        passing a Path or an int gets a typed failure instead of a validation
        error that reads like an internal fault.
        """
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

        if not RUN_ID_PATTERN.fullmatch(run_id):
            raise _fail(
                "INVALID_RUN_ID",
                (
                    "run_id must match canary_<8-32 lowercase hex>; separators, "
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
