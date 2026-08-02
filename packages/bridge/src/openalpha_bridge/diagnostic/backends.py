"""What the diagnostic needs from the official assets, and nothing more.

Two narrow protocols. Both are read-only by construction: neither exposes a way
to update a parameter, and no implementation in this package constructs an
optimizer, computes a gradient, or writes a checkpoint.

The deterministic fakes exist so every branch of the diagnostic, including the
ones that require a broken tokenizer or a model that never emits a valid path,
can be executed in a test without touching a network or an official asset.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from .validity import DecodedCandle

__all__ = [
    "ForecastModel",
    "GeneratedPath",
    "ResolvedDiagnosticAssets",
    "TokenPair",
    "TokenizerCodec",
]


class TokenPair(BaseModel):
    """One step of the two-stream Kronos token representation."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    coarse: int = Field(ge=0)
    fine: int = Field(ge=0)


class GeneratedPath(BaseModel):
    """One rollout, exactly as the frozen model produced it."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    seed: int
    tokens: tuple[TokenPair, ...]
    #: Log probability of each selected token pair, per step. Preserved so a
    #: rollout's likelihood can be compared against its validity.
    step_log_probabilities: tuple[float, ...]
    total_log_probability: float


class ResolvedDiagnosticAssets(BaseModel):
    """Observed identity of the assets that were actually loaded."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    tokenizer_repository: str
    tokenizer_revision: str
    tokenizer_config_sha256: str
    tokenizer_weights_sha256: str
    model_repository: str
    model_revision: str
    model_config_sha256: str
    model_weights_sha256: str
    #: Every loaded parameter, hashed. Compared before and after the diagnostic
    #: so "no parameter was modified" is measured rather than asserted.
    parameter_sha256: str
    trainable_parameter_count: int = Field(ge=0)
    total_parameter_count: int = Field(ge=0)


@runtime_checkable
class TokenizerCodec(Protocol):
    """The official tokenizer, used only to encode and decode."""

    def encode(self, candles: tuple[DecodedCandle, ...]) -> tuple[TokenPair, ...]:
        """Real candles to token pairs."""
        ...

    def decode(self, tokens: tuple[TokenPair, ...]) -> tuple[DecodedCandle, ...]:
        """Token pairs to candles, with no correction of any kind."""
        ...


@runtime_checkable
class ForecastModel(Protocol):
    """The pinned frozen forecasting model, used only to generate."""

    def generate(
        self,
        context: tuple[DecodedCandle, ...],
        *,
        steps: int,
        seed: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> GeneratedPath:
        """Produce ``steps`` future token pairs from ``context``.

        Implementations must not mutate any parameter. The diagnostic hashes the
        parameters before and after and fails if they differ.
        """
        ...
