"""Operational failure detail that is safe to emit.

The durable artifact deliberately carries no exception text, because an
artifact is readable by anyone with bucket access and exception messages carry
provider URLs, credentials, response fragments and local paths. That left the
failure undiagnosable: the artifact pointed at the worker logs and the worker
logged nothing.

This emits the missing half to the logs, and only the half that is safe: the
exception class, the stage it died in, and a stack of ``basename:function:line``
locations. Never ``str(exception)``, never its arguments, never locals, never a
full path.
"""

from __future__ import annotations

import logging
import traceback
from types import TracebackType
from typing import Final

__all__ = [
    "LOGGER_NAME",
    "STAGES",
    "SafeFailureRecord",
    "StageTracker",
    "log_operational_failure",
    "sanitized_frames",
]

LOGGER_NAME: Final[str] = "openalpha.diagnostic"

#: Every stage the worker and runner can be in. Ordered as execution reaches
#: them, so a recorded stage also says how far the run got.
STAGES: Final[tuple[str, ...]] = (
    "validate_invocation",
    "verify_existing_artifact",
    "enter_official_runtime",
    "construct_provider",
    "retrieve_series",
    "fit_normalization",
    "method_a",
    "method_b",
    "method_c",
    "method_d",
    "verify_parameters",
    "compute_decision",
    "serialize_artifact",
    "write_artifact",
)


class StageTracker:
    """Where execution currently is. Mutable by design, read on failure."""

    __slots__ = ("_stage",)

    def __init__(self, stage: str = "validate_invocation") -> None:
        self._stage = stage

    @property
    def stage(self) -> str:
        return self._stage

    def enter(self, stage: str) -> None:
        # An unknown stage is recorded rather than rejected: losing the failure
        # because its label was unexpected would be worse than an odd label.
        self._stage = stage

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"StageTracker({self._stage!r})"


class SafeFailureRecord:
    """What is safe to say about an unexpected exception."""

    __slots__ = ("deployed_commit", "exception_class", "frames", "run_id", "stage")

    def __init__(
        self,
        *,
        run_id: str,
        deployed_commit: str,
        stage: str,
        exception_class: str,
        frames: tuple[str, ...],
    ) -> None:
        self.run_id = run_id
        self.deployed_commit = deployed_commit
        self.stage = stage
        self.exception_class = exception_class
        self.frames = frames

    def as_message(self) -> str:
        joined = " <- ".join(self.frames) if self.frames else "no frames"
        return (
            "diagnostic operational failure"
            f" run_id={self.run_id}"
            f" deployed_commit={self.deployed_commit}"
            f" stage={self.stage}"
            f" exception_class={self.exception_class}"
            f" frames=[{joined}]"
        )


def _basename(filename: str) -> str:
    """The file's own name, never the directory that held it."""
    cleaned = filename.replace("\\", "/")
    return cleaned.rsplit("/", 1)[-1] or cleaned


def sanitized_frames(traceback_object: TracebackType | None, *, limit: int = 20) -> tuple[str, ...]:
    """``basename:function:line`` per frame, innermost last.

    Nothing else is taken. ``traceback.extract_tb`` also exposes the source
    line, which can contain a literal credential or URL, so the text is
    discarded and only the location survives.
    """
    if traceback_object is None:
        return ()
    frames = traceback.extract_tb(traceback_object)[-limit:]
    return tuple(f"{_basename(frame.filename)}:{frame.name}:{frame.lineno}" for frame in frames)


def log_operational_failure(
    error: BaseException,
    *,
    run_id: str,
    deployed_commit: str,
    stage: str,
    logger: logging.Logger | None = None,
) -> SafeFailureRecord:
    """Emit the safe half of an unexpected failure and return what was said.

    ``logger.exception`` is deliberately not used: it formats the exception,
    which reintroduces exactly the text this exists to keep out.
    """
    record = SafeFailureRecord(
        run_id=run_id,
        deployed_commit=deployed_commit,
        stage=stage,
        exception_class=type(error).__name__,
        frames=sanitized_frames(error.__traceback__),
    )
    (logger or logging.getLogger(LOGGER_NAME)).error(record.as_message())
    return record
