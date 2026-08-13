# pyright: reportMissingImports=false
# Torch is deliberately absent from the base development environment: the
# ordinary test suite must run without it, and this module imports it lazily
# inside the one function that needs a GPU.
"""Frozen hidden-state extraction through the official public accessor.

``Kronos.decode_s1`` returns ``(s1_logits, x)`` where ``x`` is the post-norm
output of the full transformer stack, shape ``[batch, seq_len, d_model]``. The
official ``auto_regressive_inference`` calls exactly this and passes ``x`` into
``decode_s2``, so reading it exercises the released code path rather than a
private detour. Nothing upstream is modified, forked, hooked or patched.

The input tensors are built with the same helpers the generation path uses, so
the representation is taken from byte-identical inputs to the ones the
completed zero-shot benchmark fed the model. Reimplementing that normalisation
here would risk a silent divergence and make the two studies incomparable.

Nothing in this module decodes a candle, samples a token, or touches a target
row. It reads one vector per origin.
"""

from __future__ import annotations

from typing import Any, Final, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from ..diagnostic.normalization import NormalizationState
from ..diagnostic.official_backend import _rows_to_tensor, _stamps_to_tensor
from ..diagnostic.official_input import OfficialRow, TimeStamp
from ..errors import BridgeFailure, BridgeTransformError, FailureCategory
from .spec import CONTEXT_CANDLES, REPRESENTATION_DIMENSION

__all__ = [
    "ExtractedRepresentation",
    "HiddenStateExtractor",
    "OfficialHiddenStateExtractor",
]

#: The official clip, applied to the standardized tensor exactly as predict()
#: does before the tokenizer sees it.
_CLIP: Final[float] = 5.0


def _fail(code: str, message: str, *, field: str | None = None) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            field=field,
            message=message,
        )
    )


class ExtractedRepresentation(BaseModel):
    """One origin's hidden state, plus what was observed while taking it."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    #: The final-position vector. One row of the design matrix.
    vector: tuple[float, ...]
    dimension: int = Field(ge=1)
    sequence_length: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    position_used: int = -1
    #: Recorded so a reader can confirm the encode saw the whole context.
    encoded_token_count: int = Field(ge=0)
    gradients_computed: bool = False
    target_rows_seen: bool = False


@runtime_checkable
class HiddenStateExtractor(Protocol):
    """What the probe needs from the official assets, and nothing more.

    Deliberately narrower than ``ForecastModel``: there is no ``generate`` here,
    so this study structurally cannot sample a path.
    """

    def hidden_state(
        self,
        context: tuple[OfficialRow, ...],
        *,
        context_stamps: tuple[TimeStamp, ...],
        state: NormalizationState,
    ) -> ExtractedRepresentation:
        """Return the final-position hidden state for one context window."""
        ...


class OfficialHiddenStateExtractor:
    """Reads ``decode_s1``'s context tensor from the pinned frozen model.

    Constructed from the raw loaded modules rather than from a forecasting
    wrapper, so no generation code is reachable from this object at all.
    """

    __slots__ = ("_clip", "_device", "_model", "_tokenizer")

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        device: str = "cuda",
        clip: float = _CLIP,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._clip = clip

    def hidden_state(
        self,
        context: tuple[OfficialRow, ...],
        *,
        context_stamps: tuple[TimeStamp, ...],
        state: NormalizationState,
    ) -> ExtractedRepresentation:
        import torch

        if len(context) != CONTEXT_CANDLES:
            raise _fail(
                "PROBE_CONTEXT_LENGTH_MISMATCH",
                f"{len(context)} context rows, expected exactly {CONTEXT_CANDLES}",
            )
        if len(context_stamps) != len(context):
            raise _fail(
                "PROBE_STAMP_SHAPE_MISMATCH",
                f"{len(context_stamps)} stamps for {len(context)} context rows",
            )
        if state.fitted_candle_count != len(context):
            raise _fail(
                "PROBE_NORMALIZATION_FITTED_ON_WRONG_ROWS",
                (
                    f"the normalization state was fitted from {state.fitted_candle_count} "
                    f"rows, expected exactly the {len(context)} context rows"
                ),
            )

        # The same tensors the generation path builds. Identical normalization,
        # identical clip, identical stamp layout.
        x = _rows_to_tensor(context, state, self._device)
        x = torch.clip(x, -self._clip, self._clip)
        x_stamp = _stamps_to_tensor(context_stamps, self._device)

        with torch.inference_mode():
            encoded = self._tokenizer.encode(x, half=True)
            coarse, fine = encoded[0], encoded[1]
            if tuple(coarse.shape) != (1, len(context)):
                raise _fail(
                    "PROBE_ENCODE_SHAPE_MISMATCH",
                    f"coarse ids are {tuple(coarse.shape)}, expected (1, {len(context)})",
                )

            # The public accessor. Returns (s1_logits, context representation);
            # only the second element is read, and the logits are discarded.
            _, hidden = self._model.decode_s1(coarse, fine, x_stamp)

            if hidden.dim() != 3:
                raise _fail(
                    "PROBE_HIDDEN_STATE_RANK_MISMATCH",
                    f"decode_s1 returned rank {hidden.dim()}, expected 3 [B, T, D]",
                )
            batch, seq_len, dim = (int(v) for v in hidden.shape)
            if batch != 1:
                raise _fail(
                    "PROBE_HIDDEN_STATE_SHAPE_MISMATCH",
                    f"batch dimension is {batch}, expected 1",
                )
            if seq_len != len(context):
                raise _fail(
                    "PROBE_HIDDEN_STATE_SHAPE_MISMATCH",
                    f"sequence length is {seq_len}, expected {len(context)}",
                )
            if dim != REPRESENTATION_DIMENSION:
                raise _fail(
                    "PROBE_HIDDEN_STATE_DIMENSION_MISMATCH",
                    (
                        f"hidden dimension is {dim}, expected "
                        f"{REPRESENTATION_DIMENSION}; the pinned Kronos-base d_model"
                    ),
                )

            # x[:, -1, :]: the state the model would have predicted from.
            final = hidden[0, -1, :]
            if not bool(torch.isfinite(final).all()):
                raise _fail(
                    "PROBE_HIDDEN_STATE_NON_FINITE",
                    "the extracted hidden state contains a non-finite value",
                )
            vector = tuple(float(v) for v in final.detach().to("cpu").tolist())

        if len(vector) != REPRESENTATION_DIMENSION:
            raise _fail(
                "PROBE_HIDDEN_STATE_DIMENSION_MISMATCH",
                f"extracted {len(vector)} values, expected {REPRESENTATION_DIMENSION}",
            )
        return ExtractedRepresentation(
            vector=vector,
            dimension=len(vector),
            sequence_length=seq_len,
            batch_size=batch,
            encoded_token_count=int(coarse.shape[1]),
        )
