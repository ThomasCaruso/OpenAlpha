"""Session calendars for Bridge partition and window construction.

The Phase 2 primary provider is Yahoo Finance at ``1d`` over exchange-traded US
ETFs, so the applicable calendar is XNYS. A 24/7 calendar is provided for the
Phase 4 Binance slices only; it is not used by Phase 2.
"""

from __future__ import annotations

from datetime import date, timedelta
from enum import StrEnum
from functools import lru_cache

from .errors import BridgeFailure, BridgeTransformError, FailureCategory

__all__ = [
    "XNYS_AD_HOC_CLOSURES",
    "CalendarName",
    "sessions_in_half_open_range",
    "xnys_holidays",
]


class CalendarName(StrEnum):
    XNYS = "XNYS"
    CRYPTO_24_7 = "CRYPTO_24_7"


# Full-day XNYS closures that no recurring rule generates.
XNYS_AD_HOC_CLOSURES: frozenset[date] = frozenset(
    {
        date(2012, 10, 29),  # Hurricane Sandy
        date(2012, 10, 30),  # Hurricane Sandy
        date(2018, 12, 5),  # G.H.W. Bush national day of mourning
        date(2025, 1, 9),  # Carter national day of mourning
    }
)

_JUNETEENTH_FIRST_OBSERVED_YEAR = 2022


def _easter_sunday(year: int) -> date:
    """Anonymous Gregorian computus."""
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
    if month == 12:
        end = date(year, 12, 31)
    else:
        end = date(year, month + 1, 1) - timedelta(days=1)
    return end - timedelta(days=(end.weekday() - weekday) % 7)


def _weekend_observed(day: date) -> date:
    """Saturday holidays move to the preceding Friday, Sunday to the next Monday."""
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


@lru_cache(maxsize=256)
def xnys_holidays(year: int) -> frozenset[date]:
    """Observed XNYS full-day closures for a calendar year."""
    observed: set[date] = set()

    # New Year's Day rolls forward from Sunday but is not observed on the
    # preceding Friday when it falls on a Saturday.
    new_year = date(year, 1, 1)
    observed.add(new_year + timedelta(days=1) if new_year.weekday() == 6 else new_year)

    observed.add(_nth_weekday(year, 1, 0, 3))  # Martin Luther King Jr. Day
    observed.add(_nth_weekday(year, 2, 0, 3))  # Washington's Birthday
    observed.add(_easter_sunday(year) - timedelta(days=2))  # Good Friday
    observed.add(_last_weekday(year, 5, 0))  # Memorial Day
    if year >= _JUNETEENTH_FIRST_OBSERVED_YEAR:
        observed.add(_weekend_observed(date(year, 6, 19)))
    observed.add(_weekend_observed(date(year, 7, 4)))  # Independence Day
    observed.add(_nth_weekday(year, 9, 0, 1))  # Labor Day
    observed.add(_nth_weekday(year, 11, 3, 4))  # Thanksgiving
    observed.add(_weekend_observed(date(year, 12, 25)))  # Christmas

    in_year = {day for day in observed if day.year == year}
    return frozenset(in_year | {day for day in XNYS_AD_HOC_CLOSURES if day.year == year})


def sessions_in_half_open_range(
    start: date,
    end: date,
    *,
    calendar: CalendarName = CalendarName.XNYS,
) -> tuple[date, ...]:
    """Sessions in ``[start, end)`` for the named calendar.

    XNYS excludes weekends and observed holidays. CRYPTO_24_7 includes every
    calendar day, Saturdays and Sundays among them, and is reserved for the
    Phase 4 Binance slices.
    """
    if end < start:
        raise BridgeTransformError(
            BridgeFailure(
                category=FailureCategory.INVALID_CONFIGURATION,
                code="INVERTED_DATE_RANGE",
                message=f"end {end.isoformat()} precedes start {start.isoformat()}",
            )
        )

    if calendar is CalendarName.CRYPTO_24_7:
        span = (end - start).days
        return tuple(start + timedelta(days=offset) for offset in range(span))

    holidays: set[date] = set()
    for year in range(start.year, end.year + 1):
        holidays |= xnys_holidays(year)

    result: list[date] = []
    cursor = start
    while cursor < end:
        if cursor.weekday() < 5 and cursor not in holidays:
            result.append(cursor)
        cursor += timedelta(days=1)
    return tuple(result)
