"""Stage A feature extraction: candles to the locked 269-dimensional tensor.

The numerical path is exactly the preregistered one:

    512 candles (448 causal prefix + 64 scored suffix)
      -> six ordered channels (open, high, low, close, volume, amount)
      -> population mean/std computed from the PREFIX ONLY, then applied to the
         whole window, clipped to the official [-5, 5]
      -> official KronosTokenizer.encode(..., half=True)
      -> official coarse/fine identifiers, 0..1023
      -> official indices_to_bits(..., half=True): [T, 20] bipolar, 1/sqrt(20)
      -> official post_quant_embed: Linear(20, 256)
      -> the three official causal decoder blocks
      -> [T, 256] frozen hidden state
      -> concatenated with 13 causal scale features
      -> [T, 269] Bridge input

269 = 256 frozen decoder hidden dimensions + 13 causal scale features.

Nothing here trains, and nothing advances the state machine. Extraction fails
closed on any contract deviation.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import date

import numpy as np
from numpy.typing import NDArray

from ..errors import BridgeFailure, BridgeTransformError, FailureCategory
from ..windowing import (
    CONTEXT_PREFIX_LENGTH,
    EXAMPLE_LENGTH,
    SCORED_SUFFIX_LENGTH,
    SequenceSpec,
    score_mask,
    score_mask_sha256,
)
from .cache import CachedExample, CacheIdentity
from .kronos import (
    BRIDGE_INPUT_DIMENSION,
    DECODER_HIDDEN_DIMENSION,
    QUANTIZED_LATENT_DIMENSION,
    SCALE_FEATURE_COUNT,
    SCALE_FEATURE_ORDER,
    KronosBackend,
    assert_bridge_input,
    assert_token_ranges,
)
from .provider import Candle

__all__ = [
    "CLIP_RANGE",
    "NORMALIZATION_EPSILON",
    "OFFICIAL_CHANNEL_ORDER",
    "ExtractedSequence",
    "FeatureExtractor",
    "candle_matrix",
    "compute_scale_features",
    "forward_representation",
    "normalize_with_prefix_state",
]

#: experiment.yaml features.ordered_columns
OFFICIAL_CHANNEL_ORDER: tuple[str, ...] = ("open", "high", "low", "close", "volume", "amount")

#: experiment.yaml features.normalization_epsilon and features.clip
NORMALIZATION_EPSILON = 1.0e-5
CLIP_RANGE: tuple[float, float] = (-5.0, 5.0)

#: mathematical_contract log_price_guard and price bounds
_MINIMUM_PRICE = 1.0e-12
_MAXIMUM_PRICE = 1.0e12
_RETURN_LIMIT = math.log(4.0)
_ZERO_TOLERANCE = 1.0e-14


def _fail(
    code: str,
    message: str,
    *,
    category: FailureCategory = FailureCategory.INVALID_INPUT,
    candle_index: int | None = None,
    field: str | None = None,
) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(
            category=category,
            code=code,
            candle_index=candle_index,
            field=field,
            message=message,
        )
    )


def candle_matrix(candles: tuple[Candle, ...]) -> NDArray[np.float64]:
    """Ordered six-channel matrix in float64, the locked reference dtype."""
    return np.array(
        [
            (c.open, c.high, c.low, c.close, c.volume, c.amount)
            for c in candles
        ],
        dtype=np.float64,
    )


def normalize_with_prefix_state(
    matrix: NDArray[np.float64],
    *,
    prefix_length: int = CONTEXT_PREFIX_LENGTH,
) -> tuple[NDArray[np.float32], NDArray[np.float64], NDArray[np.float64]]:
    """Normalize the whole window using PREFIX-ONLY population statistics.

    The preregistration is explicit: "The six-feature population mean and
    standard deviation are computed from the prefix only. The prefix and real
    suffix are normalized with that frozen state." Using suffix statistics would
    leak the scored target into its own encoder input.
    """
    if matrix.shape[0] <= prefix_length:
        raise _fail(
            "SEQUENCE_TOO_SHORT",
            f"need more than {prefix_length} candles, got {matrix.shape[0]}",
        )
    prefix = matrix[:prefix_length]
    mean = prefix.mean(axis=0)
    # Population standard deviation (ddof=0), as preregistered.
    std = prefix.std(axis=0, ddof=0)
    normalized = (matrix - mean) / (std + NORMALIZATION_EPSILON)
    clipped = np.clip(normalized, CLIP_RANGE[0], CLIP_RANGE[1])
    if not np.isfinite(clipped).all():
        raise _fail(
            "NON_FINITE_NORMALIZED_INPUT",
            "normalization produced non-finite values",
            category=FailureCategory.UNSUPPORTED_NUMERICAL,
        )
    return clipped.astype(np.float32), mean, std


def compute_scale_features(
    matrix: NDArray[np.float64],
    *,
    prefix_length: int = CONTEXT_PREFIX_LENGTH,
) -> NDArray[np.float32]:
    """The 13 causal scale features, in the locked order.

    Every value is a function of the 448-candle prefix only, so no scored
    candle can influence them. They are broadcast across all timesteps by the
    caller, which is what makes them "causal" for every position in the window.

    The formulas follow the locked feature names in
    experiment.yaml bridge_architecture.tensors.scale_features.ordered_features:
    a `*_mean_relative_to_anchor` name is that channel's prefix mean divided by
    the anchor, a `log1p_*_std_over_anchor` name is log1p of that channel's
    prefix standard deviation divided by the anchor, and the volume and amount
    features are log1p of the raw prefix statistics.
    """
    prefix = matrix[:prefix_length]
    anchor = float(prefix[-1, 3])  # last observed close before the scored suffix
    if not math.isfinite(anchor) or not (_MINIMUM_PRICE <= anchor <= _MAXIMUM_PRICE):
        raise _fail(
            "INVALID_ANCHOR_CLOSE",
            f"prefix anchor close {anchor!r} is outside the supported domain",
        )

    mean = prefix.mean(axis=0)
    std = prefix.std(axis=0, ddof=0)

    values = np.array(
        [
            math.log(anchor),
            mean[0] / anchor,
            mean[1] / anchor,
            mean[2] / anchor,
            mean[3] / anchor,
            math.log1p(std[0] / anchor),
            math.log1p(std[1] / anchor),
            math.log1p(std[2] / anchor),
            math.log1p(std[3] / anchor),
            math.log1p(mean[4]),
            math.log1p(std[4]),
            math.log1p(mean[5]),
            math.log1p(std[5]),
        ],
        dtype=np.float64,
    )
    if values.shape[0] != SCALE_FEATURE_COUNT:
        raise _fail(
            "INVALID_SCALE_FEATURE_COUNT",
            f"expected {SCALE_FEATURE_COUNT} scale features, built {values.shape[0]}",
            category=FailureCategory.INVALID_SHAPE,
        )
    if not np.isfinite(values).all():
        offenders = [
            SCALE_FEATURE_ORDER[i] for i, v in enumerate(values) if not math.isfinite(v)
        ]
        raise _fail(
            "NON_FINITE_SCALE_FEATURE",
            f"non-finite scale feature(s): {offenders}",
            category=FailureCategory.UNSUPPORTED_NUMERICAL,
        )
    return values.astype(np.float32)


def forward_representation(
    suffix: tuple[Candle, ...], *, anchor_close: float
) -> NDArray[np.float32]:
    """The locked five constrained channels for the scored suffix.

    gap, body, upper, lower, log1p_volume, computed as differences of logarithms
    exactly as docs/BRIDGE_MATHEMATICAL_REPRESENTATION.md specifies. The anchor
    for the first candle is the last observed close before the suffix; later
    candles chain from the preceding real close.
    """
    if len(suffix) != SCORED_SUFFIX_LENGTH:
        raise _fail(
            "INVALID_SUFFIX_LENGTH",
            f"scored suffix must hold {SCORED_SUFFIX_LENGTH} candles, got {len(suffix)}",
            category=FailureCategory.INVALID_SHAPE,
        )

    rows = np.empty((SCORED_SUFFIX_LENGTH, 5), dtype=np.float64)
    previous_close = float(anchor_close)
    for index, candle in enumerate(suffix):
        values = (candle.open, candle.high, candle.low, candle.close)
        if not all(_MINIMUM_PRICE <= v <= _MAXIMUM_PRICE for v in values):
            raise _fail(
                "PRICE_OUT_OF_DOMAIN",
                f"candle {candle.session.isoformat()} price outside the supported domain",
                candle_index=index,
                category=FailureCategory.OUT_OF_DOMAIN,
            )
        log_open = math.log(candle.open)
        log_close = math.log(candle.close)
        gap = log_open - math.log(previous_close)
        body = log_close - log_open
        upper = math.log(candle.high) - max(log_open, log_close)
        lower = min(log_open, log_close) - math.log(candle.low)

        for name, value in (("gap", gap), ("body", body), ("upper", upper), ("lower", lower)):
            if abs(value) > _RETURN_LIMIT:
                raise _fail(
                    "RETURN_LIMIT_EXCEEDED",
                    f"{name} {value!r} exceeds the locked log(4) limit",
                    candle_index=index,
                    field=name,
                    category=FailureCategory.OUT_OF_DOMAIN,
                )
        # Roundoff-only negatives canonicalize to positive zero; a genuinely
        # negative wick is a structural failure, not a rounding artefact.
        upper = 0.0 if -_ZERO_TOLERANCE <= upper < 0.0 else upper
        lower = 0.0 if -_ZERO_TOLERANCE <= lower < 0.0 else lower
        if upper < 0.0 or lower < 0.0:
            raise _fail(
                "NEGATIVE_WICK",
                f"candle {candle.session.isoformat()} has a negative wick",
                candle_index=index,
            )

        rows[index] = (gap, body, upper, lower, math.log1p(candle.volume))
        previous_close = candle.close

    if not np.isfinite(rows).all():
        raise _fail(
            "NON_FINITE_TARGET",
            "forward representation produced non-finite values",
            category=FailureCategory.UNSUPPORTED_NUMERICAL,
        )
    return rows.astype(np.float32)


@dataclass(frozen=True, slots=True)
class ExtractedSequence:
    """One fully extracted 512-candle example."""

    sequence_id: str
    symbol: str
    interval: str
    partition: str
    target_start: date
    target_end: date
    coarse_ids: NDArray[np.int64]
    fine_ids: NDArray[np.int64]
    bipolar_latent: NDArray[np.float32]
    frozen_hidden: NDArray[np.float32]
    causal_features: NDArray[np.float32]
    bridge_input: NDArray[np.float32]
    constrained_targets: NDArray[np.float32]
    initial_previous_close: float
    source_data_sha256: str

    @property
    def score_mask_sha256(self) -> str:
        return score_mask_sha256()

    def canonical_sha256(self) -> str:
        """Content hash over every extracted tensor, for determinism checks."""
        digest = hashlib.sha256()
        digest.update(f"{self.sequence_id}|{self.symbol}|{self.interval}".encode())
        for array in (
            self.coarse_ids,
            self.fine_ids,
            self.bipolar_latent,
            self.frozen_hidden,
            self.causal_features,
            self.bridge_input,
            self.constrained_targets,
        ):
            digest.update(np.ascontiguousarray(array).tobytes())
        return digest.hexdigest()

    def to_cached_example(self, identity: CacheIdentity) -> CachedExample:
        from ..windowing import Partition

        example = CachedExample(
            identity=identity,
            sequence_id=self.sequence_id,
            symbol=self.symbol,
            interval=self.interval,
            partition=Partition(self.partition),
            coarse_ids=self.coarse_ids,
            fine_ids=self.fine_ids,
            bipolar_latent=self.bipolar_latent,
            frozen_hidden=self.frozen_hidden,
            causal_features=self.causal_features,
            constrained_targets=self.constrained_targets,
            initial_previous_close=self.initial_previous_close,
            volume_mask=np.ones(EXAMPLE_LENGTH, dtype=np.bool_),
            sequence_mask=np.ones(EXAMPLE_LENGTH, dtype=np.bool_),
            score_mask=np.asarray(score_mask(), dtype=np.bool_),
            source_data_sha256=self.source_data_sha256,
        )
        example.validate_shapes()
        return example


class FeatureExtractor:
    """Builds locked Bridge inputs from real candles and a frozen Kronos backend."""

    def __init__(self, backend: KronosBackend) -> None:
        self._backend = backend

    @property
    def backend(self) -> KronosBackend:
        return self._backend

    def extract(
        self,
        *,
        spec: SequenceSpec,
        candles: tuple[Candle, ...],
        source_data_sha256: str,
    ) -> ExtractedSequence:
        self._validate_window(spec=spec, candles=candles)

        matrix = candle_matrix(candles)
        normalized, _, _ = normalize_with_prefix_state(matrix)

        coarse_ids, fine_ids = self._backend.encode_tokens(normalized)
        assert_token_ranges(coarse_ids, fine_ids)
        self._assert_shape("coarse_ids", coarse_ids, (EXAMPLE_LENGTH,))
        self._assert_shape("fine_ids", fine_ids, (EXAMPLE_LENGTH,))

        latent = self._backend.bipolar_latent(coarse_ids, fine_ids)
        self._assert_shape("bipolar_latent", latent, (EXAMPLE_LENGTH, QUANTIZED_LATENT_DIMENSION))

        projected = self._backend.project_latent(latent)
        self._assert_shape("projected", projected, (EXAMPLE_LENGTH, DECODER_HIDDEN_DIMENSION))

        hidden = self._backend.decoder_trunk(projected)
        self._assert_shape("frozen_hidden", hidden, (EXAMPLE_LENGTH, DECODER_HIDDEN_DIMENSION))

        scale = compute_scale_features(matrix)
        causal_features = np.broadcast_to(scale, (EXAMPLE_LENGTH, SCALE_FEATURE_COUNT))
        causal_features = np.ascontiguousarray(causal_features, dtype=np.float32)

        bridge_input = np.concatenate([hidden, causal_features], axis=1).astype(np.float32)
        assert_bridge_input(bridge_input, sequence_length=EXAMPLE_LENGTH)
        if bridge_input.shape[1] != BRIDGE_INPUT_DIMENSION:  # pragma: no cover - asserted above
            raise _fail(
                "INVALID_BRIDGE_INPUT_SHAPE",
                f"expected {BRIDGE_INPUT_DIMENSION} features",
                category=FailureCategory.INVALID_SHAPE,
            )

        anchor = float(matrix[CONTEXT_PREFIX_LENGTH - 1, 3])
        targets = forward_representation(
            candles[CONTEXT_PREFIX_LENGTH:], anchor_close=anchor
        )

        return ExtractedSequence(
            sequence_id=spec.sequence_id,
            symbol=spec.symbol,
            interval=spec.interval,
            partition=spec.partition.value,
            target_start=spec.target_start,
            target_end=spec.target_end,
            coarse_ids=coarse_ids,
            fine_ids=fine_ids,
            bipolar_latent=latent,
            frozen_hidden=hidden,
            causal_features=causal_features,
            bridge_input=bridge_input,
            constrained_targets=targets,
            initial_previous_close=anchor,
            source_data_sha256=source_data_sha256,
        )

    # ---------------------------------------------------------------- guards

    def _validate_window(self, *, spec: SequenceSpec, candles: tuple[Candle, ...]) -> None:
        if len(candles) != EXAMPLE_LENGTH:
            raise _fail(
                "INVALID_EXAMPLE_LENGTH",
                f"expected {EXAMPLE_LENGTH} candles, got {len(candles)}",
                category=FailureCategory.INVALID_SHAPE,
            )
        sessions = [candle.session for candle in candles]
        for index in range(1, EXAMPLE_LENGTH):
            if sessions[index] <= sessions[index - 1]:
                raise _fail(
                    "NON_MONOTONIC_SESSIONS",
                    f"session {sessions[index]} does not follow {sessions[index - 1]}",
                    candle_index=index,
                )
        if sessions[0] != spec.prefix_start:
            raise _fail(
                "PREFIX_START_MISMATCH",
                f"window starts {sessions[0]}, specification says {spec.prefix_start}",
            )
        if sessions[CONTEXT_PREFIX_LENGTH - 1] != spec.prefix_end:
            raise _fail(
                "PREFIX_END_MISMATCH",
                f"prefix ends {sessions[CONTEXT_PREFIX_LENGTH - 1]}, "
                f"specification says {spec.prefix_end}",
            )
        if sessions[CONTEXT_PREFIX_LENGTH] != spec.target_start:
            raise _fail(
                "TARGET_START_MISMATCH",
                f"scored suffix starts {sessions[CONTEXT_PREFIX_LENGTH]}, "
                f"specification says {spec.target_start}",
            )
        if sessions[-1] != spec.target_end:
            raise _fail(
                "TARGET_END_MISMATCH",
                f"window ends {sessions[-1]}, specification says {spec.target_end}",
            )

    @staticmethod
    def _assert_shape(
        name: str, array: NDArray[np.generic], expected: tuple[int, ...]
    ) -> None:
        if tuple(array.shape) != expected:
            raise _fail(
                "INVALID_TENSOR_SHAPE",
                f"{name} must have shape {expected}, got {tuple(array.shape)}",
                field=name,
                category=FailureCategory.INVALID_SHAPE,
            )
        if not np.isfinite(np.asarray(array, dtype=np.float64)).all():
            raise _fail(
                "NON_FINITE_TENSOR",
                f"{name} contains non-finite values",
                field=name,
                category=FailureCategory.UNSUPPORTED_NUMERICAL,
            )
