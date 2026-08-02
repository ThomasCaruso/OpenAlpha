"""What the diagnostic needs from the official assets, and nothing more.

Two narrow protocols. Both are read-only by construction: neither exposes a way
to update a parameter, and no implementation in this package constructs an
optimizer, computes a gradient, or writes a checkpoint.

Both take an explicit ``NormalizationState``. Neither may retain a mutable
"last normalization" of its own, because a remembered state is a state nobody
can audit.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from .normalization import NormalizationState
from .official_input import OfficialRow, TimeStamp

__all__ = [
    "SAMPLING_PROBABILITY_DEFINITION",
    "ForecastModel",
    "GeneratedPath",
    "ResolvedDiagnosticAssets",
    "StepSampling",
    "TokenPair",
    "TokenizerCodec",
]

#: What every stored probability in this package means, in one place.
#:
#: Derived from sample_from_logits in the pinned source, which divides logits
#: by the temperature, applies top-k or top-p filtering, then softmaxes. The
#: softmax is over the filtered vector, whose removed entries are -inf and so
#: receive zero probability; the survivors therefore sum to one.
SAMPLING_PROBABILITY_DEFINITION: Literal[
    "log_softmax_of_temperature_scaled_then_top_k_top_p_filtered_logits_renormalized"
] = "log_softmax_of_temperature_scaled_then_top_k_top_p_filtered_logits_renormalized"


class TokenPair(BaseModel):
    """One step of the two-stream Kronos token representation."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    coarse: int = Field(ge=0)
    fine: int = Field(ge=0)


class StepSampling(BaseModel):
    """The sampling probabilities for one generated step.

    These are sampling probabilities, not model likelihoods. The distribution
    they come from has been temperature-scaled and nucleus-filtered, so mass
    the model assigned to removed tokens has been redistributed over the kept
    set. They are not comparable across different settings, and they are never
    labelled likelihood.
    """

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    definition: Literal[
        "log_softmax_of_temperature_scaled_then_top_k_top_p_filtered_logits_renormalized"
    ] = SAMPLING_PROBABILITY_DEFINITION
    measured_after_temperature: Literal[True] = True
    measured_after_top_k_top_p_filtering: Literal[True] = True
    filtered_distribution_renormalized: Literal[True] = True

    #: log p(selected coarse token) under the filtered s1 distribution.
    coarse_log_probability: float
    #: log p(selected fine token | selected coarse token) under the filtered s2
    #: distribution, which decode_s2 conditions on the sampled coarse token.
    fine_conditional_log_probability: float

    @property
    def pair_log_probability(self) -> float:
        return self.coarse_log_probability + self.fine_conditional_log_probability


class GeneratedPath(BaseModel):
    """One rollout, exactly as the frozen model produced it."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    seed: int
    tokens: tuple[TokenPair, ...]
    sampling: tuple[StepSampling, ...]
    #: Sum of pair log probabilities over the generated steps. A path sampling
    #: log probability, not a model likelihood of the path.
    total_path_sampling_log_probability: float


class ResolvedDiagnosticAssets(BaseModel):
    """Observed identity of the assets and source that were actually loaded."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    tokenizer_repository: str
    tokenizer_revision: str
    tokenizer_config_sha256: str
    tokenizer_weights_sha256: str
    model_repository: str
    model_revision: str
    model_config_sha256: str
    model_weights_sha256: str
    #: Both digests for each source file: as committed upstream, and normalized
    #: to CRLF as the sealed experiment records them.
    source_revision: str
    source_as_committed_sha256: dict[str, str]
    source_crlf_normalized_sha256: dict[str, str]
    #: Every loaded parameter, hashed. Compared before and after so "no
    #: parameter was modified" is measured rather than asserted.
    parameter_sha256: str
    trainable_parameter_count: int = Field(ge=0)
    total_parameter_count: int = Field(ge=0)


@runtime_checkable
class TokenizerCodec(Protocol):
    """The official tokenizer, used only to encode and decode."""

    def encode(
        self, rows: tuple[OfficialRow, ...], *, state: NormalizationState
    ) -> tuple[TokenPair, ...]:
        """Normalize with the supplied state, then encode. No retained state."""
        ...

    def decode(
        self, tokens: tuple[TokenPair, ...], *, state: NormalizationState, sessions: tuple
    ) -> tuple[OfficialRow, ...]:
        """Decode, then inverse-normalize with the supplied state.

        ``sessions`` supplies the timestamps for the decoded rows, which the
        token stream does not carry. No correction of any kind is applied.
        """
        ...


@runtime_checkable
class ForecastModel(Protocol):
    """The pinned frozen forecasting model, used only to generate."""

    def generate(
        self,
        context: tuple[OfficialRow, ...],
        *,
        context_stamps: tuple[TimeStamp, ...],
        target_stamps: tuple[TimeStamp, ...],
        state: NormalizationState,
        steps: int,
        seed: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> GeneratedPath:
        """Produce ``steps`` future token pairs from ``context``.

        Implementations must not mutate any parameter. The diagnostic hashes
        the parameters before and after and fails if they differ.
        """
        ...
