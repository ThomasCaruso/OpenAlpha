from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass
from types import TracebackType

__all__ = [
    "LOGGER_NAME",
    "SafeFailureRecord",
    "StageTracker",
    "log_operational_failure",
    "sanitized_frames",
]

LOGGER_NAME = "openalpha.research"


class StageTracker:
    """Mutable record of a caller-defined execution stage."""

    __slots__ = ("_stage",)

    def __init__(self, stage: str = "initial") -> None:
        self._stage = stage

    @property
    def stage(self) -> str:
        return self._stage

    def enter(self, stage: str) -> None:
        self._stage = stage


@dataclass(frozen=True, slots=True)
class SafeFailureRecord:
    """Exception metadata safe for operational logging."""

    run_id: str
    deployed_commit: str
    stage: str
    exception_class: str
    frames: tuple[str, ...]

    def as_message(self) -> str:
        joined = " <- ".join(self.frames) if self.frames else "no frames"
        return (
            "operational failure"
            f" run_id={self.run_id}"
            f" deployed_commit={self.deployed_commit}"
            f" stage={self.stage}"
            f" exception_class={self.exception_class}"
            f" frames=[{joined}]"
        )


def _basename(filename: str) -> str:
    cleaned = filename.replace(chr(92), "/")
    return cleaned.rsplit("/", 1)[-1] or cleaned


def sanitized_frames(
    traceback_object: TracebackType | None,
    *,
    limit: int = 20,
) -> tuple[str, ...]:
    """Return only ``basename:function:line`` locations, innermost last."""
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
    """Log exception class and sanitized locations without exception text."""
    record = SafeFailureRecord(
        run_id=run_id,
        deployed_commit=deployed_commit,
        stage=stage,
        exception_class=type(error).__name__,
        frames=sanitized_frames(error.__traceback__),
    )
    (logger or logging.getLogger(LOGGER_NAME)).error(record.as_message())
    return record
