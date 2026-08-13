from __future__ import annotations

import importlib
from datetime import date, timedelta
from enum import StrEnum
from functools import lru_cache
from typing import Any

from .failures import FailureCategory, ResearchFailure, ResearchFailureError

__all__ = [
    "CalendarName",
    "sessions_in_half_open_range",
    "xnys_holidays",
]

_XNYS_START = date(2000, 1, 1)
_XNYS_END = date(2030, 12, 31)
_XNYS_END_EXCLUSIVE = _XNYS_END + timedelta(days=1)


class CalendarName(StrEnum):
    XNYS = "XNYS"
    CONTINUOUS = "CONTINUOUS"


def _fail(code: str, message: str) -> ResearchFailureError:
    return ResearchFailureError(
        ResearchFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            message=message,
        )
    )


@lru_cache(maxsize=1)
def _xnys_calendar() -> Any:
    """Load the pinned official calendar only when XNYS behavior is requested."""
    try:
        calendars = importlib.import_module("exchange_calendars")
    except ImportError as error:
        raise _fail(
            "CALENDAR_DEPENDENCY_MISSING",
            "the pinned exchange calendar dependency is unavailable",
        ) from error
    return calendars.get_calendar("XNYS", start=_XNYS_START, end=_XNYS_END)


def _validate_xnys_range(start: date, end: date) -> None:
    if start < _XNYS_START or start > _XNYS_END or end > _XNYS_END_EXCLUSIVE:
        raise _fail(
            "XNYS_RANGE_UNSUPPORTED",
            (
                "XNYS dates must lie within the supported range "
                f"{_XNYS_START.isoformat()} through {_XNYS_END.isoformat()}"
            ),
        )


@lru_cache(maxsize=31)
def xnys_holidays(year: int) -> frozenset[date]:
    """Return weekday XNYS non-sessions from the pinned official calendar."""
    start = date(year, 1, 1)
    end = date(year, 12, 31)
    _validate_xnys_range(start, end)
    sessions = {
        timestamp.date()
        for timestamp in _xnys_calendar().sessions_in_range(start, end)
    }
    span = (end - start).days + 1
    return frozenset(
        day
        for offset in range(span)
        if (day := start + timedelta(days=offset)).weekday() < 5 and day not in sessions
    )


def sessions_in_half_open_range(
    start: date,
    end: date,
    *,
    calendar: CalendarName = CalendarName.XNYS,
) -> tuple[date, ...]:
    """Return sessions in ``[start, end)`` for an exchange or continuous calendar."""
    if end < start:
        raise _fail(
            "INVERTED_DATE_RANGE",
            f"end {end.isoformat()} precedes start {start.isoformat()}",
        )
    span = (end - start).days
    if calendar is CalendarName.CONTINUOUS:
        return tuple(start + timedelta(days=offset) for offset in range(span))

    _validate_xnys_range(start, end)
    if start == end:
        return ()
    inclusive_end = end - timedelta(days=1)
    return tuple(
        timestamp.date()
        for timestamp in _xnys_calendar().sessions_in_range(start, inclusive_end)
    )
