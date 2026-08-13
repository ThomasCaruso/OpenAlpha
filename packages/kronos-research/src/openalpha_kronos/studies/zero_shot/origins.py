"""Deterministic origin selection, fixed before execution.

Origins are integer index offsets into the ascending session sequence, not
calendar dates. That is the only way to preregister one selection policy that
resolves identically for four symbols whose provider histories may begin on
different sessions: an index rule cannot silently pick a different window for
one asset, and it cannot be nudged after seeing a result.

The stride equals the horizon, so an asset's 25 target windows are exactly
disjoint and contiguous. No target row is scored twice.
"""

from __future__ import annotations

from datetime import date
from typing import Final

from openalpha_research.failures import FailureCategory, ResearchFailure, ResearchFailureError
from pydantic import BaseModel, ConfigDict, Field

from .spec import (
    CONTEXT_CANDLES,
    HORIZON_CANDLES,
    ORIGIN_FIRST_INDEX,
    ORIGIN_STRIDE,
    ORIGINS_PER_ASSET,
    REQUIRED_SESSIONS,
)

__all__ = [
    "ORIGIN_INDEX_OFFSETS",
    "OriginSelection",
    "resolve_origins",
    "verify_origin_policy",
]

#: The preregistered offsets, computed once from the declared rule. Every asset
#: receives this same tuple; there is no per-asset branch anywhere in this file.
ORIGIN_INDEX_OFFSETS: Final[tuple[int, ...]] = tuple(
    ORIGIN_FIRST_INDEX + ORIGIN_STRIDE * index for index in range(ORIGINS_PER_ASSET)
)


def _fail(code: str, message: str, *, field: str | None = None) -> ResearchFailureError:
    return ResearchFailureError(
        ResearchFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            field=field,
            message=message,
        )
    )


class OriginSelection(BaseModel):
    """One asset-origin: where its context and target begin and end."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    asset: str
    #: Position in the preregistered offset tuple, 0..24.
    ordinal: int = Field(ge=0)
    #: Index of the first target row in the session sequence.
    origin_index: int = Field(ge=0)
    context_start_index: int = Field(ge=0)
    context_end_exclusive: int = Field(gt=0)
    target_start_index: int = Field(ge=0)
    target_end_exclusive: int = Field(gt=0)
    #: Resolved at execution from the retrieved sessions, recorded for audit.
    context_first_session: date
    context_last_session: date
    target_first_session: date
    target_last_session: date

    @property
    def context_length(self) -> int:
        return self.context_end_exclusive - self.context_start_index

    @property
    def target_length(self) -> int:
        return self.target_end_exclusive - self.target_start_index


def verify_origin_policy() -> dict[str, int]:
    """Check the declared arithmetic is self-consistent before it is used."""
    if len(ORIGIN_INDEX_OFFSETS) != ORIGINS_PER_ASSET:
        raise _fail(
            "ZERO_SHOT_ORIGIN_POLICY_INVALID",
            f"expected {ORIGINS_PER_ASSET} offsets, derived {len(ORIGIN_INDEX_OFFSETS)}",
        )
    if ORIGIN_INDEX_OFFSETS[0] < CONTEXT_CANDLES:
        raise _fail(
            "ZERO_SHOT_ORIGIN_POLICY_INVALID",
            (
                f"the first origin index {ORIGIN_INDEX_OFFSETS[0]} leaves fewer than "
                f"{CONTEXT_CANDLES} context rows before it"
            ),
        )
    if ORIGIN_STRIDE < HORIZON_CANDLES:
        raise _fail(
            "ZERO_SHOT_ORIGIN_POLICY_INVALID",
            (
                f"stride {ORIGIN_STRIDE} is shorter than the horizon {HORIZON_CANDLES}, "
                "so target windows would overlap and a target row would be scored twice"
            ),
        )
    needed = ORIGIN_INDEX_OFFSETS[-1] + HORIZON_CANDLES
    if needed != REQUIRED_SESSIONS:
        raise _fail(
            "ZERO_SHOT_ORIGIN_POLICY_INVALID",
            f"the last target row ends at {needed}, but {REQUIRED_SESSIONS} sessions are declared",
        )
    return {
        "origins_per_asset": ORIGINS_PER_ASSET,
        "first_origin_index": ORIGIN_INDEX_OFFSETS[0],
        "last_origin_index": ORIGIN_INDEX_OFFSETS[-1],
        "stride": ORIGIN_STRIDE,
        "required_sessions": REQUIRED_SESSIONS,
    }


def resolve_origins(*, asset: str, sessions: tuple[date, ...]) -> tuple[OriginSelection, ...]:
    """Resolve the preregistered offsets against one asset's sessions.

    ``sessions`` must already be the trimmed, ascending, deduplicated prefix of
    exactly ``REQUIRED_SESSIONS`` sessions. Trimming happens at retrieval so
    that this function has no opportunity to choose anything.
    """
    verify_origin_policy()
    if len(sessions) != REQUIRED_SESSIONS:
        raise _fail(
            "ZERO_SHOT_SESSION_COUNT_MISMATCH",
            (
                f"{asset} supplied {len(sessions)} sessions, expected exactly "
                f"{REQUIRED_SESSIONS}; origins are index based and cannot be resolved "
                "against a different length"
            ),
            field=asset,
        )
    if list(sessions) != sorted(sessions):
        raise _fail(
            "ZERO_SHOT_SESSIONS_UNSORTED", f"{asset} sessions are not ascending", field=asset
        )
    if len(set(sessions)) != len(sessions):
        raise _fail(
            "ZERO_SHOT_DUPLICATE_SESSION", f"{asset} has a repeated session", field=asset
        )

    selections: list[OriginSelection] = []
    for ordinal, origin_index in enumerate(ORIGIN_INDEX_OFFSETS):
        context_start = origin_index - CONTEXT_CANDLES
        target_end = origin_index + HORIZON_CANDLES
        if context_start < 0 or target_end > len(sessions):
            raise _fail(
                "ZERO_SHOT_INCOMPLETE_WINDOW",
                (
                    f"{asset} origin {ordinal} at index {origin_index} does not have a "
                    f"complete {CONTEXT_CANDLES} + {HORIZON_CANDLES} window"
                ),
                field=asset,
            )
        selections.append(
            OriginSelection(
                asset=asset,
                ordinal=ordinal,
                origin_index=origin_index,
                context_start_index=context_start,
                context_end_exclusive=origin_index,
                target_start_index=origin_index,
                target_end_exclusive=target_end,
                context_first_session=sessions[context_start],
                context_last_session=sessions[origin_index - 1],
                target_first_session=sessions[origin_index],
                target_last_session=sessions[target_end - 1],
            )
        )

    windows = [(s.target_start_index, s.target_end_exclusive) for s in selections]
    covered: set[int] = set()
    for start, end in windows:
        span = set(range(start, end))
        if span & covered:
            raise _fail(
                "ZERO_SHOT_TARGET_WINDOWS_OVERLAP",
                f"{asset} target windows overlap, so a target row would be scored twice",
                field=asset,
            )
        covered |= span
    return tuple(selections)
