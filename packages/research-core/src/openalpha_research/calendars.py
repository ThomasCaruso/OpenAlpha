from __future__ import annotations

from datetime import date, timedelta
from enum import StrEnum
from functools import lru_cache

from .failures import FailureCategory, ResearchFailure, ResearchFailureError

__all__ = [
    "XNYS_AD_HOC_CLOSURES",
    "CalendarName",
    "sessions_in_half_open_range",
    "xnys_holidays",
]


class CalendarName(StrEnum):
    XNYS = "XNYS"
    CONTINUOUS = "CONTINUOUS"


XNYS_AD_HOC_CLOSURES: frozenset[date] = frozenset(
    {
        date(2012, 10, 29),
        date(2012, 10, 30),
        date(2018, 12, 5),
        date(2025, 1, 9),
    }
)

_JUNETEENTH_FIRST_OBSERVED_YEAR = 2022


def _easter_sunday(year: int) -> date:
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    lunar = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * lunar) // 451
    month, day = divmod(h + lunar - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, ordinal: int) -> date:
    first = date(year, month, 1)
    first += timedelta(days=(weekday - first.weekday()) % 7)
    return first + timedelta(weeks=ordinal - 1)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    end = date(year, 12, 31) if month == 12 else date(year, month + 1, 1) - timedelta(days=1)
    return end - timedelta(days=(end.weekday() - weekday) % 7)


def _weekend_observed(day: date) -> date:
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


@lru_cache(maxsize=256)
def xnys_holidays(year: int) -> frozenset[date]:
    """Return observed full-day XNYS closures for a calendar year."""
    observed: set[date] = set()
    new_year = date(year, 1, 1)
    observed.add(new_year + timedelta(days=1) if new_year.weekday() == 6 else new_year)
    observed.add(_nth_weekday(year, 1, 0, 3))
    observed.add(_nth_weekday(year, 2, 0, 3))
    observed.add(_easter_sunday(year) - timedelta(days=2))
    observed.add(_last_weekday(year, 5, 0))
    if year >= _JUNETEENTH_FIRST_OBSERVED_YEAR:
        observed.add(_weekend_observed(date(year, 6, 19)))
    observed.add(_weekend_observed(date(year, 7, 4)))
    observed.add(_nth_weekday(year, 9, 0, 1))
    observed.add(_nth_weekday(year, 11, 3, 4))
    observed.add(_weekend_observed(date(year, 12, 25)))
    in_year = {day for day in observed if day.year == year}
    return frozenset(in_year | {day for day in XNYS_AD_HOC_CLOSURES if day.year == year})


def sessions_in_half_open_range(
    start: date,
    end: date,
    *,
    calendar: CalendarName = CalendarName.XNYS,
) -> tuple[date, ...]:
    """Return sessions in ``[start, end)`` for an exchange or continuous calendar."""
    if end < start:
        raise ResearchFailureError(
            ResearchFailure(
                category=FailureCategory.INVALID_CONFIGURATION,
                code="INVERTED_DATE_RANGE",
                message=f"end {end.isoformat()} precedes start {start.isoformat()}",
            )
        )
    span = (end - start).days
    if calendar is CalendarName.CONTINUOUS:
        return tuple(start + timedelta(days=offset) for offset in range(span))

    holidays: set[date] = set()
    for year in range(start.year, end.year + 1):
        holidays.update(xnys_holidays(year))
    return tuple(
        start + timedelta(days=offset)
        for offset in range(span)
        if (start + timedelta(days=offset)).weekday() < 5
        and (start + timedelta(days=offset)) not in holidays
    )
