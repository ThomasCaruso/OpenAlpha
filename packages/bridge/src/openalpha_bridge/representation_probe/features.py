"""Window construction, targets, and the design matrix.

Origins are integer index offsets, spaced by the horizon so target windows are
exactly disjoint. Each origin yields one sample: one feature row and one target.

The split between what a feature builder may see and what a target may see is
enforced by signature. Every builder in :mod:`controls` and the extractor in
:mod:`extraction` receives context rows only; targets are computed here, in a
separate function, from rows the builders never receive.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from ..diagnostic.official_input import OfficialRow
from ..errors import BridgeFailure, BridgeTransformError, FailureCategory
from .spec import ASSET_PANEL, CONTEXT_CANDLES, HORIZON_CANDLES, STRIDE

__all__ = [
    "OriginWindow",
    "Sample",
    "cumulative_log_return",
    "origin_offsets",
    "resolve_windows",
    "verify_disjoint_targets",
]

_MINIMUM_CONTEXT_FOR_FIRST_ORIGIN: Final[int] = CONTEXT_CANDLES


def _fail(code: str, message: str, *, field: str | None = None) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            field=field,
            message=message,
        )
    )


def origin_offsets(session_count: int, *, limit: int | None = None) -> tuple[int, ...]:
    """Index offsets of every complete non-overlapping origin in a sequence.

    An origin index is the index of the FIRST TARGET ROW. Offsets start at
    ``CONTEXT_CANDLES`` because an origin needs a full context behind it, and
    step by ``STRIDE`` so targets never overlap.
    """
    if session_count < CONTEXT_CANDLES + HORIZON_CANDLES:
        return ()
    offsets: list[int] = []
    index = _MINIMUM_CONTEXT_FOR_FIRST_ORIGIN
    while index + HORIZON_CANDLES <= session_count:
        offsets.append(index)
        index += STRIDE
    if limit is not None:
        offsets = offsets[:limit]
    return tuple(offsets)


class OriginWindow(BaseModel):
    """One asset-origin: its context rows, its target rows, and its dates."""

    model_config = ConfigDict(allow_inf_nan=True, extra="forbid", frozen=True, strict=True)

    asset: str
    ordinal: int = Field(ge=0)
    origin_index: int = Field(ge=0)
    context: tuple[OfficialRow, ...]
    target: tuple[OfficialRow, ...]

    @property
    def context_first_session(self) -> date:
        return self.context[0].session

    @property
    def context_last_session(self) -> date:
        return self.context[-1].session

    @property
    def target_first_session(self) -> date:
        return self.target[0].session

    @property
    def target_last_session(self) -> date:
        return self.target[-1].session

    @property
    def anchor_close(self) -> float:
        """The final observed context close. The model genuinely had this."""
        return self.context[-1].close


class Sample(BaseModel):
    """One row of the design matrix, with its target and its provenance."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    asset: str
    ordinal: int = Field(ge=0)
    origin_index: int = Field(ge=0)
    context_first_session: date
    context_last_session: date
    target_first_session: date
    target_last_session: date
    anchor_close: float
    #: Cumulative log return over the horizon. The regression target.
    target_return: float
    #: Its sign. The classification target. Zero maps to 0.
    target_direction: int


def resolve_windows(
    *, asset: str, rows: tuple[OfficialRow, ...], limit: int | None = None
) -> tuple[OriginWindow, ...]:
    """Slice a session sequence into complete, non-overlapping origin windows."""
    offsets = origin_offsets(len(rows), limit=limit)
    windows: list[OriginWindow] = []
    for ordinal, origin_index in enumerate(offsets):
        context = rows[origin_index - CONTEXT_CANDLES : origin_index]
        target = rows[origin_index : origin_index + HORIZON_CANDLES]
        if len(context) != CONTEXT_CANDLES or len(target) != HORIZON_CANDLES:
            raise _fail(
                "PROBE_INCOMPLETE_WINDOW",
                (
                    f"{asset} ordinal {ordinal} has {len(context)} context and "
                    f"{len(target)} target rows, expected {CONTEXT_CANDLES} and "
                    f"{HORIZON_CANDLES}"
                ),
                field=asset,
            )
        windows.append(
            OriginWindow(
                asset=asset,
                ordinal=ordinal,
                origin_index=origin_index,
                context=context,
                target=target,
            )
        )
    verify_disjoint_targets(asset=asset, windows=tuple(windows))
    return tuple(windows)


def verify_disjoint_targets(*, asset: str, windows: tuple[OriginWindow, ...]) -> None:
    """No target row may be scored twice."""
    covered: set[int] = set()
    for window in windows:
        span = set(range(window.origin_index, window.origin_index + HORIZON_CANDLES))
        if span & covered:
            raise _fail(
                "PROBE_TARGET_WINDOWS_OVERLAP",
                (
                    f"{asset} target windows overlap at ordinal {window.ordinal}; "
                    "stride must be at least the horizon"
                ),
                field=asset,
            )
        covered |= span


def cumulative_log_return(window: OriginWindow) -> float:
    """log(final target close / final context close), over the whole horizon.

    Computed here and never inside a feature builder, so a feature set cannot
    reach a target value through the code that constructs it.
    """
    anchor = window.anchor_close
    final = window.target[-1].close
    if not math.isfinite(anchor) or anchor <= 0.0:
        raise _fail(
            "PROBE_TARGET_ANCHOR_UNUSABLE",
            f"{window.asset} ordinal {window.ordinal}: anchor close is not usable",
        )
    if not math.isfinite(final) or final <= 0.0:
        raise _fail(
            "PROBE_TARGET_CLOSE_UNUSABLE",
            f"{window.asset} ordinal {window.ordinal}: final target close is not usable",
        )
    return math.log(final / anchor)


def build_sample(window: OriginWindow) -> Sample:
    """Attach the targets to one origin's provenance."""
    value = cumulative_log_return(window)
    return Sample(
        asset=window.asset,
        ordinal=window.ordinal,
        origin_index=window.origin_index,
        context_first_session=window.context_first_session,
        context_last_session=window.context_last_session,
        target_first_session=window.target_first_session,
        target_last_session=window.target_last_session,
        anchor_close=window.anchor_close,
        target_return=value,
        target_direction=(1 if value > 0.0 else (-1 if value < 0.0 else 0)),
    )


def verify_cross_asset_alignment(
    windows_by_asset: dict[str, tuple[OriginWindow, ...]],
) -> dict[str, int]:
    """Every ordinal must be one shared chronological window for all assets.

    Clustering in the bootstrap assumes an ordinal names one market window. If
    SPY's ordinal 7 covered different sessions from DIA's, a cluster would pool
    unrelated windows.
    """
    missing = [asset for asset in ASSET_PANEL if asset not in windows_by_asset]
    if missing:
        raise _fail(
            "PROBE_CROSS_ASSET_ORIGIN_MISALIGNED",
            f"missing assets {sorted(missing)}; the panel must be complete",
        )
    counts = {asset: len(windows_by_asset[asset]) for asset in ASSET_PANEL}
    if len(set(counts.values())) != 1:
        raise _fail(
            "PROBE_CROSS_ASSET_ORIGIN_MISALIGNED",
            f"assets yielded different origin counts: {counts}",
        )

    reference_asset = ASSET_PANEL[0]
    for ordinal, reference in enumerate(windows_by_asset[reference_asset]):
        for asset in ASSET_PANEL[1:]:
            other = windows_by_asset[asset][ordinal]
            if (
                other.target_first_session != reference.target_first_session
                or other.target_last_session != reference.target_last_session
            ):
                raise _fail(
                    "PROBE_CROSS_ASSET_ORIGIN_MISALIGNED",
                    (
                        f"ordinal {ordinal} target window disagrees: {reference_asset} "
                        f"covers {reference.target_first_session}.."
                        f"{reference.target_last_session} but {asset} covers "
                        f"{other.target_first_session}..{other.target_last_session}"
                    ),
                    field=f"ordinal_{ordinal}",
                )
            if (
                other.context_first_session != reference.context_first_session
                or other.context_last_session != reference.context_last_session
            ):
                raise _fail(
                    "PROBE_CROSS_ASSET_ORIGIN_MISALIGNED",
                    (
                        f"ordinal {ordinal} context window disagrees: {reference_asset} "
                        f"covers {reference.context_first_session}.."
                        f"{reference.context_last_session} but {asset} covers "
                        f"{other.context_first_session}..{other.context_last_session}"
                    ),
                    field=f"ordinal_{ordinal}",
                )
    return {"ordinals_checked": counts[reference_asset], "assets_checked": len(ASSET_PANEL)}


__all__ += ["build_sample", "verify_cross_asset_alignment"]
