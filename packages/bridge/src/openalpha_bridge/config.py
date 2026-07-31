from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class VolumeMode(StrEnum):
    VOLUME_REQUIRED = "VOLUME_REQUIRED"
    VOLUME_OPTIONAL = "VOLUME_OPTIONAL"
    PRICE_ONLY = "PRICE_ONLY"


class BridgeRepresentationConfig(BaseModel):
    """Immutable, versioned numerical contract for Bridge financial decoding."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
    )

    representation_version: Literal["openalpha.bridge.financial.v1"] = (
        "openalpha.bridge.financial.v1"
    )
    gap_return_limit: float = Field(default=math.log(4.0), gt=0.0)
    body_return_limit: float = Field(default=math.log(4.0), gt=0.0)
    upper_wick_transform: Literal["bounded_stable_softplus"] = "bounded_stable_softplus"
    upper_wick_limit: float = Field(default=math.log(4.0), gt=0.0)
    lower_wick_transform: Literal["bounded_stable_softplus"] = "bounded_stable_softplus"
    lower_wick_limit: float = Field(default=math.log(4.0), gt=0.0)
    volume_mode: VolumeMode = VolumeMode.VOLUME_OPTIONAL
    volume_transform: Literal["bounded_stable_softplus_log1p"] = "bounded_stable_softplus_log1p"
    volume_limit: float = Field(default=math.log1p(1.0e15), gt=0.0)
    numerical_epsilon: float = Field(default=1.0e-14, gt=0.0)
    output_dtype: Literal["float32", "float64"] = "float32"
    reference_dtype: Literal["float64"] = "float64"
    overflow_policy: Literal["raise"] = "raise"
    out_of_domain_policy: Literal["raise"] = "raise"
    supported_feature_order: tuple[str, ...] = ("open", "high", "low", "close", "volume")
    minimum_price: float = Field(default=1.0e-12, gt=0.0)
    maximum_price: float = Field(default=1.0e12, gt=0.0)
    maximum_volume: float = Field(default=1.0e15, ge=0.0)
    log_price_guard: tuple[float, float] = (-300.0, 300.0)
    maximum_decode_steps: int = Field(default=64, gt=0, le=4096)

    @model_validator(mode="after")
    def validate_contract(self) -> BridgeRepresentationConfig:
        if self.minimum_price >= self.maximum_price:
            raise ValueError("minimum_price must be below maximum_price")
        if len(self.log_price_guard) != 2 or self.log_price_guard[0] >= self.log_price_guard[1]:
            raise ValueError("log_price_guard must be an increasing pair")
        guard_low, guard_high = self.log_price_guard
        if guard_low < -700.0 or guard_high > 700.0:
            raise ValueError(
                "log_price_guard must remain inside the safe float64 exponential range"
            )
        if math.log(self.minimum_price) <= guard_low or math.log(self.maximum_price) >= guard_high:
            raise ValueError(
                "configured source price bounds must lie strictly inside log_price_guard"
            )

        maximum_reconstructed_log = (
            math.log(self.maximum_price)
            + self.gap_return_limit
            + self.body_return_limit
            + self.upper_wick_limit
        )
        minimum_reconstructed_log = (
            math.log(self.minimum_price)
            - self.gap_return_limit
            - self.body_return_limit
            - self.lower_wick_limit
        )
        if maximum_reconstructed_log >= guard_high or minimum_reconstructed_log <= guard_low:
            raise ValueError(
                "channel limits can leave log_price_guard in one supported decode step"
            )
        if self.volume_limit > math.log1p(self.maximum_volume):
            raise ValueError("volume_limit cannot exceed log1p(maximum_volume)")
        if self.numerical_epsilon >= min(
            self.gap_return_limit,
            self.body_return_limit,
            self.upper_wick_limit,
            self.lower_wick_limit,
            self.volume_limit,
        ):
            raise ValueError("numerical_epsilon must be smaller than every channel limit")

        price_order = ("open", "high", "low", "close")
        volume_order = (*price_order, "volume")
        if self.volume_mode is VolumeMode.PRICE_ONLY:
            if self.supported_feature_order != price_order:
                raise ValueError("PRICE_ONLY requires exactly the four OHLC feature columns")
        elif self.supported_feature_order != volume_order:
            raise ValueError("volume-enabled configuration requires ordered OHLCV feature columns")
        return self

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        if update:
            raise TypeError("model_copy updates bypass validation")
        return super().model_copy(update=None, deep=deep)

    def canonical_bytes(self) -> bytes:
        payload = {
            "config": self.model_dump(mode="json", exclude_none=False),
            "schema_version": "openalpha.bridge.config.v1",
        }
        return json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    @property
    def canonical_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()
