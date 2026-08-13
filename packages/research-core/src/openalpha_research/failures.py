from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["FailureCategory", "ResearchFailure", "ResearchFailureError"]


class FailureCategory(StrEnum):
    INVALID_INPUT = "INVALID_INPUT"
    OUT_OF_DOMAIN = "OUT_OF_DOMAIN"
    UNSUPPORTED_NUMERICAL = "UNSUPPORTED_NUMERICAL"
    INVALID_SHAPE = "INVALID_SHAPE"
    INVALID_CONFIGURATION = "INVALID_CONFIGURATION"


class ResearchFailure(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
    )

    schema_version: Literal["openalpha.research.failure.v1"] = (
        "openalpha.research.failure.v1"
    )
    category: FailureCategory
    code: str = Field(min_length=2, max_length=96, pattern=r"^[A-Z][A-Z0-9_]+$")
    sequence_index: int | None = Field(default=None, ge=0)
    candle_index: int | None = Field(default=None, ge=0)
    field: str | None = None
    observed_value: float | str | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    message: str = Field(min_length=1)


class ResearchFailureError(ValueError):
    """Typed fail-closed result for invalid research inputs or configuration."""

    def __init__(self, *failures: ResearchFailure) -> None:
        if not failures:
            raise ValueError("at least one ResearchFailure is required")
        self.failures = tuple(failures)
        first = self.failures[0]
        location = ""
        if first.sequence_index is not None:
            location += f" sequence={first.sequence_index}"
        if first.candle_index is not None:
            location += f" candle={first.candle_index}"
        if first.field is not None:
            location += f" field={first.field}"
        super().__init__(f"{first.code}:{location} {first.message}".strip())
