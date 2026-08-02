"""Runtime-observed manifest of the official Kronos components.

Every field is read from the tokenizer that was actually loaded. Nothing here is
a hardcoded string, so the manifest is evidence about the assets in front of us
rather than a restatement of what the documentation claims.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..errors import BridgeFailure, BridgeTransformError, FailureCategory

__all__ = [
    "ObservedEncode",
    "ObservedKronosComponents",
    "ObservedModule",
    "assert_observed_matches_locks",
    "describe_module",
]


def _fail(code: str, message: str, *, field: str | None = None) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            field=field,
            message=message,
        )
    )


def describe_module(obj: Any) -> tuple[str, str]:
    """The class name and fully qualified module of a live object."""
    cls = type(obj)
    return cls.__name__, getattr(cls, "__module__", "unknown")


class ObservedModule(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    attribute: str
    class_name: str
    module: str


class ObservedLinear(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    attribute: str
    class_name: str
    module: str
    in_features: int
    out_features: int
    has_bias: bool


class ObservedEncode(BaseModel):
    """What ``encode(half=True)`` actually returned."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    return_container: str
    tensor_count: int = Field(ge=0)
    batched_shapes: tuple[tuple[int, ...], ...]
    batched_dtypes: tuple[str, ...]
    coarse_flat_shape: tuple[int, ...]
    fine_flat_shape: tuple[int, ...]
    coarse_flat_dtype: str
    fine_flat_dtype: str


class ObservedKronosComponents(BaseModel):
    """Observed identity and shape of every component the Bridge path touches."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    schema_version: Literal["openalpha.bridge.phase2.observed_components.v1"] = (
        "openalpha.bridge.phase2.observed_components.v1"
    )
    tokenizer_class: str
    tokenizer_module: str
    indices_to_bits_attribute: str
    projection: ObservedLinear
    decoder_attribute: str
    decoder_container_class: str
    decoder_container_module: str
    decoder_block_count: int = Field(ge=0)
    decoder_blocks: tuple[ObservedModule, ...]
    head: ObservedLinear
    encode: ObservedEncode | None = None
    device: str
    training_mode: bool
    total_parameters: int = Field(ge=0)
    trainable_parameters: int = Field(ge=0)


def assert_observed_matches_locks(
    observed: ObservedKronosComponents,
    *,
    quantized_latent_dimension: int,
    decoder_hidden_dimension: int,
    tokenizer_input_dimension: int,
    expected_decoder_blocks: int,
    expected_sequence_length: int,
) -> None:
    """Every locked structural expectation, checked against what was observed."""
    projection = observed.projection
    if (projection.in_features, projection.out_features) != (
        quantized_latent_dimension,
        decoder_hidden_dimension,
    ):
        raise _fail(
            "OBSERVED_PROJECTION_MISMATCH",
            (
                f"post_quant_embed must be Linear({quantized_latent_dimension}, "
                f"{decoder_hidden_dimension}), observed Linear("
                f"{projection.in_features}, {projection.out_features})"
            ),
        )
    if observed.decoder_block_count != expected_decoder_blocks:
        raise _fail(
            "OBSERVED_DECODER_BLOCK_COUNT_MISMATCH",
            (
                f"expected {expected_decoder_blocks} decoder blocks, observed "
                f"{observed.decoder_block_count}"
            ),
        )
    if len(observed.decoder_blocks) != observed.decoder_block_count:
        raise _fail(
            "OBSERVED_DECODER_BLOCK_INVENTORY_MISMATCH",
            "the observed decoder block inventory does not match the reported count",
        )
    head = observed.head
    if (head.in_features, head.out_features) != (
        decoder_hidden_dimension,
        tokenizer_input_dimension,
    ):
        raise _fail(
            "OBSERVED_HEAD_MISMATCH",
            (
                f"head must be Linear({decoder_hidden_dimension}, "
                f"{tokenizer_input_dimension}), observed Linear("
                f"{head.in_features}, {head.out_features})"
            ),
        )
    if observed.training_mode:
        raise _fail(
            "OBSERVED_TOKENIZER_IN_TRAINING_MODE",
            "the official tokenizer must be in evaluation mode",
        )
    if observed.trainable_parameters != 0:
        raise _fail(
            "OBSERVED_TOKENIZER_NOT_FROZEN",
            (
                f"every official parameter must be frozen, observed "
                f"{observed.trainable_parameters} trainable"
            ),
        )
    if observed.total_parameters <= 0:
        raise _fail(
            "OBSERVED_TOKENIZER_HAS_NO_PARAMETERS",
            "the loaded tokenizer reports no parameters at all",
        )

    encode = observed.encode
    if encode is None:
        raise _fail(
            "OBSERVED_ENCODE_NOT_RECORDED",
            "encode was never observed, so its contract cannot be asserted",
        )
    if encode.tensor_count != 2:
        raise _fail(
            "OBSERVED_ENCODE_TENSOR_COUNT",
            f"encode(half=True) must return exactly two tensors, observed {encode.tensor_count}",
        )
    expected_batched = (1, expected_sequence_length)
    for index, shape in enumerate(encode.batched_shapes):
        if tuple(shape) != expected_batched:
            raise _fail(
                "OBSERVED_ENCODE_SHAPE_MISMATCH",
                (
                    f"encode tensor {index} must be batched {list(expected_batched)}, "
                    f"observed {list(shape)}"
                ),
            )
