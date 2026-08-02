"""The official six-channel input, with timestamps preserved.

v1 carried five channels and no session. That cannot execute the official path
faithfully: ``KronosPredictor.predict`` takes
``['open','high','low','close','volume','amount']`` in that exact order, and
``calc_time_stamps`` derives five stamp features from the timestamp of every
context and target row. Dropping either produces a tensor the model was not
trained against.

Values are unconstrained on purpose. A container that refuses to hold a
non-finite or non-positive price cannot be used to measure how often a decoder
emits one.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..errors import BridgeFailure, BridgeTransformError, FailureCategory

__all__ = [
    "OFFICIAL_COLUMNS",
    "OFFICIAL_STAMP_COLUMNS",
    "ColumnPresence",
    "OfficialRow",
    "OfficialSeries",
    "TimeStamp",
    "official_stamp",
]

#: Exactly the order predict() assembles: price_cols + [vol_col, amt_vol].
OFFICIAL_COLUMNS: Final[tuple[str, ...]] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
)

#: calc_time_stamps builds these five, in this order.
OFFICIAL_STAMP_COLUMNS: Final[tuple[str, ...]] = (
    "minute",
    "hour",
    "weekday",
    "day",
    "month",
)


def _fail(code: str, message: str) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            message=message,
        )
    )


class OfficialRow(BaseModel):
    """One session: its timestamp and all six official channels."""

    model_config = ConfigDict(allow_inf_nan=True, extra="forbid", frozen=True, strict=True)

    session: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float

    def channels(self) -> tuple[float, ...]:
        """The six values in the official column order."""
        return (self.open, self.high, self.low, self.close, self.volume, self.amount)


class TimeStamp(BaseModel):
    """The five stamp features the official path derives from a timestamp."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    minute: int
    hour: int
    weekday: int
    day: int
    month: int

    def values(self) -> tuple[int, ...]:
        return (self.minute, self.hour, self.weekday, self.day, self.month)


def official_stamp(session: date) -> TimeStamp:
    """Reproduce calc_time_stamps for one session.

    Daily sessions leave minute and hour at zero. They are still emitted,
    because the official path expects five stamp columns and presenting four
    would change the tensor shape the model sees.
    """
    if isinstance(session, datetime):
        minute, hour = session.minute, session.hour
    else:
        # A plain date has no clock. calc_time_stamps reads .dt.minute and
        # .dt.hour off a pandas timestamp, which for a midnight daily session
        # is zero; matching that here keeps the stamp tensor identical.
        minute, hour = 0, 0
    return TimeStamp(
        minute=minute,
        hour=hour,
        weekday=session.weekday(),
        day=session.day,
        month=session.month,
    )


class ColumnPresence(BaseModel):
    """Which columns were retrieved, and which the official fallback made up.

    predict() fills a missing volume with 0.0 and a missing amount with
    ``volume * mean(open, high, low, close)``. A synthesised value must never
    be mistaken for a retrieved one, so the distinction is carried in the
    artifact rather than lost at the boundary.
    """

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    open: bool = True
    high: bool = True
    low: bool = True
    close: bool = True
    volume: bool
    amount: bool

    def as_mask(self) -> dict[str, bool]:
        return {name: getattr(self, name) for name in OFFICIAL_COLUMNS}

    @property
    def all_retrieved(self) -> bool:
        return all(self.as_mask().values())


class OfficialSeries(BaseModel):
    """The complete input identity: rows, order, presence, calendar, boundary."""

    model_config = ConfigDict(allow_inf_nan=True, extra="forbid", frozen=True, strict=True)

    schema_version: Literal["openalpha.bridge.diagnostic.official_series.v1"] = (
        "openalpha.bridge.diagnostic.official_series.v1"
    )
    symbol: str
    frequency: str
    calendar: str
    columns: tuple[str, ...]
    column_presence: ColumnPresence
    rows: tuple[OfficialRow, ...]
    #: Index of the first target row. Rows before it are context.
    context_target_boundary: int = Field(ge=0)

    @property
    def context(self) -> tuple[OfficialRow, ...]:
        return self.rows[: self.context_target_boundary]

    @property
    def target(self) -> tuple[OfficialRow, ...]:
        return self.rows[self.context_target_boundary :]

    @property
    def sessions(self) -> tuple[date, ...]:
        return tuple(row.session for row in self.rows)

    def stamps(self) -> tuple[TimeStamp, ...]:
        return tuple(official_stamp(row.session) for row in self.rows)

    def context_stamps(self) -> tuple[TimeStamp, ...]:
        return tuple(official_stamp(row.session) for row in self.context)

    def target_stamps(self) -> tuple[TimeStamp, ...]:
        return tuple(official_stamp(row.session) for row in self.target)

    def assert_contract(self, *, expected_rows: int, expected_boundary: int) -> None:
        """Every structural expectation, checked rather than assumed."""
        if self.columns != OFFICIAL_COLUMNS:
            raise _fail(
                "OFFICIAL_COLUMN_ORDER_MISMATCH",
                (
                    f"columns must be {list(OFFICIAL_COLUMNS)} in that order, got "
                    f"{list(self.columns)}"
                ),
            )
        if len(self.columns) != 6:
            raise _fail(
                "OFFICIAL_CHANNEL_COUNT_MISMATCH",
                f"the official path is six channels, got {len(self.columns)}",
            )
        if len(self.rows) != expected_rows:
            raise _fail(
                "OFFICIAL_ROW_COUNT_MISMATCH",
                f"expected {expected_rows} sessions, got {len(self.rows)}",
            )
        if self.context_target_boundary != expected_boundary:
            raise _fail(
                "OFFICIAL_BOUNDARY_MISMATCH",
                (
                    f"context/target boundary must be {expected_boundary}, got "
                    f"{self.context_target_boundary}"
                ),
            )
        sessions = self.sessions
        if len(set(sessions)) != len(sessions):
            raise _fail("OFFICIAL_DUPLICATE_SESSION", "a session appears more than once")
        if list(sessions) != sorted(sessions):
            raise _fail("OFFICIAL_SESSIONS_UNSORTED", "sessions are not in ascending order")
