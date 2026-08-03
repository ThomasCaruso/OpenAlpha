"""The four preregistered baselines. None of them may see a target row.

Every function here takes the context rows and the target *sessions* -- the
dates the forecast is for -- and never the target rows. That is a structural
guarantee rather than a promise: a baseline cannot use information it was not
handed, and the signature is what the leakage tests assert against.

The target sessions are needed only to stamp the produced rows, so a baseline
path can be scored by the same six-channel machinery as a model path.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Final

from pydantic import BaseModel, ConfigDict

from ..diagnostic.official_input import OfficialRow
from ..errors import BridgeFailure, BridgeTransformError, FailureCategory

__all__ = [
    "BASELINE_IDS",
    "PRIMARY_BASELINE_ID",
    "BaselinePaths",
    "build_baselines",
    "context_drift",
    "context_mean_return",
    "last_close_level",
    "zero_return_persistence",
]

PRIMARY_BASELINE_ID: Final[str] = "zero_return_persistence"

BASELINE_IDS: Final[tuple[str, ...]] = (
    "zero_return_persistence",
    "last_close_level",
    "context_mean_return",
    "context_drift",
)


def _fail(code: str, message: str) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            message=message,
        )
    )


def _flat_path(level: float, target_sessions: tuple[date, ...]) -> tuple[OfficialRow, ...]:
    """A constant-level path. Volume and amount are held at zero.

    Only the close drives every metric this benchmark scores. The other price
    channels are set to the same level so the row stays a well formed candle;
    volume and amount are zero rather than invented, because a baseline that
    manufactured a plausible volume would be asserting something it does not know.
    """
    return tuple(
        OfficialRow(
            session=session,
            open=level,
            high=level,
            low=level,
            close=level,
            volume=0.0,
            amount=0.0,
        )
        for session in target_sessions
    )


def _levelled_path(levels: list[float], target_sessions: tuple[date, ...]) -> tuple[OfficialRow, ...]:
    return tuple(
        OfficialRow(
            session=session,
            open=level,
            high=level,
            low=level,
            close=level,
            volume=0.0,
            amount=0.0,
        )
        for level, session in zip(levels, target_sessions, strict=True)
    )


def _anchor(context: tuple[OfficialRow, ...]) -> float:
    if not context:
        raise _fail("ZERO_SHOT_BASELINE_NO_CONTEXT", "a baseline needs at least one context row")
    anchor = context[-1].close
    if not math.isfinite(anchor) or anchor <= 0.0:
        raise _fail(
            "ZERO_SHOT_BASELINE_ANCHOR_UNUSABLE",
            "the final context close is not a usable positive anchor",
        )
    return anchor


def zero_return_persistence(
    context: tuple[OfficialRow, ...], *, target_sessions: tuple[date, ...]
) -> tuple[OfficialRow, ...]:
    """The primary baseline: every predicted log return is exactly zero."""
    return _flat_path(_anchor(context), target_sessions)


def last_close_level(
    context: tuple[OfficialRow, ...], *, target_sessions: tuple[date, ...]
) -> tuple[OfficialRow, ...]:
    """The final observed context close carried forward as a level forecast.

    Identical by construction to ``zero_return_persistence``: holding the level
    constant is the same statement as predicting zero return. It is named
    separately because the two are conventionally reported in different metric
    spaces, and the identity is asserted by test rather than left implicit.
    """
    return _flat_path(_anchor(context), target_sessions)


def context_mean_return(
    context: tuple[OfficialRow, ...], *, target_sessions: tuple[date, ...]
) -> tuple[OfficialRow, ...]:
    """Compound the mean internal context log return forward from the anchor."""
    anchor = _anchor(context)
    returns: list[float] = []
    for index in range(1, len(context)):
        previous = context[index - 1].close
        current = context[index].close
        if not math.isfinite(previous) or not math.isfinite(current):
            raise _fail(
                "ZERO_SHOT_BASELINE_CONTEXT_NON_FINITE",
                "the context contains a non-finite close, so its mean return is undefined",
            )
        if previous <= 0.0 or current <= 0.0:
            raise _fail(
                "ZERO_SHOT_BASELINE_CONTEXT_NON_POSITIVE",
                "a context close is not positive, so its log return is undefined",
            )
        returns.append(math.log(current / previous))
    if not returns:
        raise _fail(
            "ZERO_SHOT_BASELINE_NO_TRANSITIONS",
            "the context has fewer than two rows, so it has no internal return",
        )
    mean_return = sum(returns) / len(returns)

    levels: list[float] = []
    level = anchor
    for _ in target_sessions:
        level *= math.exp(mean_return)
        levels.append(level)
    return _levelled_path(levels, target_sessions)


def context_drift(
    context: tuple[OfficialRow, ...], *, target_sessions: tuple[date, ...]
) -> tuple[OfficialRow, ...]:
    """Linear level extrapolation across the context.

    Per-step increment is ``(last - first) / (n - 1)``, added forward from the
    anchor. Distinct from ``context_mean_return``, which is geometric in return
    space; on a trending context the two separate.
    """
    anchor = _anchor(context)
    if len(context) < 2:
        raise _fail(
            "ZERO_SHOT_BASELINE_NO_TRANSITIONS",
            "the context has fewer than two rows, so it has no drift",
        )
    first = context[0].close
    if not math.isfinite(first):
        raise _fail(
            "ZERO_SHOT_BASELINE_CONTEXT_NON_FINITE",
            "the first context close is not finite, so drift is undefined",
        )
    increment = (anchor - first) / (len(context) - 1)

    levels: list[float] = []
    level = anchor
    for _ in target_sessions:
        level += increment
        levels.append(level)
    return _levelled_path(levels, target_sessions)


class BaselinePaths(BaseModel):
    """All four baseline paths for one asset-origin."""

    model_config = ConfigDict(allow_inf_nan=True, extra="forbid", frozen=True, strict=True)

    zero_return_persistence: tuple[OfficialRow, ...]
    last_close_level: tuple[OfficialRow, ...]
    context_mean_return: tuple[OfficialRow, ...]
    context_drift: tuple[OfficialRow, ...]

    def as_mapping(self) -> dict[str, tuple[OfficialRow, ...]]:
        return {
            "zero_return_persistence": self.zero_return_persistence,
            "last_close_level": self.last_close_level,
            "context_mean_return": self.context_mean_return,
            "context_drift": self.context_drift,
        }


def build_baselines(
    context: tuple[OfficialRow, ...], *, target_sessions: tuple[date, ...]
) -> BaselinePaths:
    """Every baseline, built from the context and the target dates only."""
    return BaselinePaths(
        zero_return_persistence=zero_return_persistence(context, target_sessions=target_sessions),
        last_close_level=last_close_level(context, target_sessions=target_sessions),
        context_mean_return=context_mean_return(context, target_sessions=target_sessions),
        context_drift=context_drift(context, target_sessions=target_sessions),
    )
