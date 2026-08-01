"""Context-prefix / scored-suffix window construction for Bridge Phase 2.

Each example is exactly 512 candles: a read-only causal context prefix at
positions 0-447 and a scored suffix at positions 448-511. Only the 64-candle
scored suffix must lie wholly inside its assigned partition. The prefix may draw
on earlier chronological history as read-only context.

Partition isolation binds trainable targets, losses, model fitting, preprocessing
fitting, checkpoint selection, threshold selection, and final metrics. It does not
prohibit earlier observations serving as causal context for later targets.

See research/bridge-v0/phase2-amendment-2-context-prefix.yaml.
"""

from __future__ import annotations

import hashlib
from datetime import date
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .errors import BridgeFailure, BridgeTransformError, FailureCategory

__all__ = [
    "CONTEXT_PREFIX_LENGTH",
    "EVALUATION_STRIDE",
    "EXAMPLE_LENGTH",
    "SCORED_SUFFIX_LENGTH",
    "FeasibilityRow",
    "Partition",
    "SequenceSpec",
    "assert_nonoverlapping_targets",
    "assert_preprocessing_fit_partitions",
    "build_scored_sequences",
    "feasibility_row",
    "score_mask",
    "score_mask_sha256",
    "sequence_id",
    "validate_example_window",
]

EXAMPLE_LENGTH = 512
CONTEXT_PREFIX_LENGTH = 448
SCORED_SUFFIX_LENGTH = 64
EVALUATION_STRIDE = 64


class Partition(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    RECONSTRUCTION_TEST = "reconstruction_test"
    EXTERNAL_LATER = "external_later"


#: Partitions whose candles may never appear anywhere inside a training example.
_HELD_OUT_PARTITIONS = frozenset(
    {Partition.VALIDATION, Partition.RECONSTRUCTION_TEST, Partition.EXTERNAL_LATER}
)


def _fail(
    category: FailureCategory,
    code: str,
    message: str,
    *,
    candle_index: int | None = None,
) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(
            category=category,
            code=code,
            candle_index=candle_index,
            message=message,
        )
    )


def score_mask() -> tuple[bool, ...]:
    """Immutable score mask: false over the prefix, true over the scored suffix."""
    return (False,) * CONTEXT_PREFIX_LENGTH + (True,) * SCORED_SUFFIX_LENGTH


def score_mask_sha256() -> str:
    """SHA-256 of the mask encoded as 512 ASCII ``0``/``1`` characters in order."""
    encoded = "".join("1" if flag else "0" for flag in score_mask())
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def sequence_id(
    *,
    symbol: str,
    interval: str,
    partition: Partition,
    target_start: date,
    target_end: date,
) -> str:
    """Deterministic identifier for a scored sequence."""
    canonical = (
        f"openalpha.bridge.sequence.v1|{symbol}|{interval}|{partition.value}"
        f"|{target_start.isoformat()}|{target_end.isoformat()}"
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class SequenceSpec(BaseModel):
    """One scored evaluation sequence and the context window that feeds it."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    schema_version: Literal["openalpha.bridge.sequence.v1"] = "openalpha.bridge.sequence.v1"
    sequence_id: str = Field(min_length=16, max_length=16)
    symbol: str = Field(min_length=1)
    interval: str = Field(min_length=1)
    partition: Partition
    prefix_start: date
    prefix_end: date
    target_start: date
    target_end: date
    prefix_length: int = Field(ge=0)
    suffix_length: int = Field(ge=1)
    target_overlaps_other_target: bool


class FeasibilityRow(BaseModel):
    """Pre-retrieval feasibility for one symbol, interval, and partition."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    schema_version: Literal["openalpha.bridge.feasibility.v1"] = "openalpha.bridge.feasibility.v1"
    symbol: str
    interval: str
    partition: Partition
    declared_start: date
    declared_end: date
    expected_sessions: int = Field(ge=0)
    prefix_sessions_available_before_partition: int = Field(ge=0)
    nonoverlapping_scored_sequences: int = Field(ge=0)
    scored_candles: int = Field(ge=0)
    first_target_start: date | None
    first_target_end: date | None
    last_target_start: date | None
    last_target_end: date | None
    causal_warm_up_available: bool
    unused_tail_sessions: int = Field(ge=0)
    feasible: bool


def build_scored_sequences(
    *,
    symbol: str,
    interval: str,
    partition: Partition,
    partition_sessions: tuple[date, ...],
    history_sessions: tuple[date, ...] = (),
    stride: int = EVALUATION_STRIDE,
) -> tuple[SequenceSpec, ...]:
    """Build nonoverlapping scored sequences for one symbol and partition.

    ``history_sessions`` are chronologically earlier sessions usable as read-only
    warm-up context. Warm-up may also come from inside ``partition_sessions``
    itself when the target starts far enough into the partition, which is how the
    training partition supplies its own context.
    """
    if stride < 1:
        raise _fail(
            FailureCategory.INVALID_CONFIGURATION,
            "INVALID_STRIDE",
            f"stride must be positive, got {stride}",
        )
    if history_sessions and partition_sessions and history_sessions[-1] >= partition_sessions[0]:
        raise _fail(
            FailureCategory.INVALID_INPUT,
            "HISTORY_NOT_STRICTLY_EARLIER",
            "history_sessions must end strictly before partition_sessions begins",
        )

    history_count = len(history_sessions)
    timeline = history_sessions + partition_sessions
    specs: list[SequenceSpec] = []

    offset = 0
    while offset + SCORED_SUFFIX_LENGTH <= len(partition_sessions):
        target_first_global = history_count + offset
        if target_first_global < CONTEXT_PREFIX_LENGTH:
            offset += stride
            continue

        prefix_first_global = target_first_global - CONTEXT_PREFIX_LENGTH
        target_start = partition_sessions[offset]
        target_end = partition_sessions[offset + SCORED_SUFFIX_LENGTH - 1]
        specs.append(
            SequenceSpec(
                sequence_id=sequence_id(
                    symbol=symbol,
                    interval=interval,
                    partition=partition,
                    target_start=target_start,
                    target_end=target_end,
                ),
                symbol=symbol,
                interval=interval,
                partition=partition,
                prefix_start=timeline[prefix_first_global],
                prefix_end=timeline[target_first_global - 1],
                target_start=target_start,
                target_end=target_end,
                prefix_length=CONTEXT_PREFIX_LENGTH,
                suffix_length=SCORED_SUFFIX_LENGTH,
                target_overlaps_other_target=stride < SCORED_SUFFIX_LENGTH,
            )
        )
        offset += stride

    return tuple(specs)


def feasibility_row(
    *,
    symbol: str,
    interval: str,
    partition: Partition,
    declared_start: date,
    declared_end: date,
    partition_sessions: tuple[date, ...],
    history_sessions: tuple[date, ...] = (),
) -> FeasibilityRow:
    """Deterministic pre-retrieval feasibility for one symbol/interval/partition."""
    specs = build_scored_sequences(
        symbol=symbol,
        interval=interval,
        partition=partition,
        partition_sessions=partition_sessions,
        history_sessions=history_sessions,
    )
    scored = len(specs) * SCORED_SUFFIX_LENGTH
    consumed = 0
    if specs:
        last_offset = partition_sessions.index(specs[-1].target_start)
        consumed = last_offset + SCORED_SUFFIX_LENGTH

    return FeasibilityRow(
        symbol=symbol,
        interval=interval,
        partition=partition,
        declared_start=declared_start,
        declared_end=declared_end,
        expected_sessions=len(partition_sessions),
        prefix_sessions_available_before_partition=len(history_sessions),
        nonoverlapping_scored_sequences=len(specs),
        scored_candles=scored,
        first_target_start=specs[0].target_start if specs else None,
        first_target_end=specs[0].target_end if specs else None,
        last_target_start=specs[-1].target_start if specs else None,
        last_target_end=specs[-1].target_end if specs else None,
        causal_warm_up_available=bool(specs),
        unused_tail_sessions=len(partition_sessions) - consumed,
        feasible=bool(specs),
    )


def assert_nonoverlapping_targets(specs: tuple[SequenceSpec, ...]) -> None:
    """Fail when two evaluation sequences share any scored target candle."""
    seen: dict[tuple[str, str, date], str] = {}
    for spec in specs:
        if spec.target_overlaps_other_target:
            raise _fail(
                FailureCategory.INVALID_CONFIGURATION,
                "OVERLAPPING_EVALUATION_TARGET",
                f"sequence {spec.sequence_id} is flagged as overlapping",
            )
        key = (spec.symbol, spec.interval, spec.target_start)
        previous = seen.get(key)
        if previous is not None:
            raise _fail(
                FailureCategory.INVALID_CONFIGURATION,
                "DUPLICATE_EVALUATION_TARGET",
                f"sequences {previous} and {spec.sequence_id} share target {spec.target_start}",
            )
        seen[key] = spec.sequence_id


def assert_preprocessing_fit_partitions(partitions: frozenset[Partition]) -> None:
    """Bridge scalers and statistics may be fitted on the training partition only."""
    leaked = sorted(partition.value for partition in partitions & _HELD_OUT_PARTITIONS)
    if leaked:
        raise _fail(
            FailureCategory.INVALID_CONFIGURATION,
            "PREPROCESSING_FIT_ON_HELD_OUT_PARTITION",
            f"preprocessing may be fitted on train only, saw {', '.join(leaked)}",
        )


def validate_example_window(
    *,
    session_timestamps: tuple[date, ...],
    partition: Partition,
    partition_membership: tuple[Partition, ...],
    mask: tuple[bool, ...] | None = None,
) -> None:
    """Fail closed on any construction, causality, masking, or leakage violation.

    ``partition_membership`` declares, per position, which partition the candle
    at that position belongs to.
    """
    applied_mask = score_mask() if mask is None else mask

    if len(session_timestamps) != EXAMPLE_LENGTH:
        raise _fail(
            FailureCategory.INVALID_SHAPE,
            "INVALID_EXAMPLE_LENGTH",
            f"example must be {EXAMPLE_LENGTH} candles, got {len(session_timestamps)}",
        )
    if len(partition_membership) != EXAMPLE_LENGTH:
        raise _fail(
            FailureCategory.INVALID_SHAPE,
            "INVALID_MEMBERSHIP_LENGTH",
            f"membership must be {EXAMPLE_LENGTH} entries, got {len(partition_membership)}",
        )
    if len(applied_mask) != EXAMPLE_LENGTH:
        raise _fail(
            FailureCategory.INVALID_SHAPE,
            "INVALID_SCORE_MASK_LENGTH",
            f"score mask must be {EXAMPLE_LENGTH} entries, got {len(applied_mask)}",
        )

    scored_count = sum(applied_mask)
    if scored_count != SCORED_SUFFIX_LENGTH:
        raise _fail(
            FailureCategory.INVALID_SHAPE,
            "INVALID_SCORED_SUFFIX_LENGTH",
            f"score mask must select {SCORED_SUFFIX_LENGTH} candles, got {scored_count}",
        )
    if applied_mask != score_mask():
        raise _fail(
            FailureCategory.INVALID_CONFIGURATION,
            "INVALID_SCORE_MASK",
            "score mask must be false over positions 0-447 and true over 448-511",
        )

    # Checked before general monotonicity so that an injected future context
    # candle reports the specific causality violation rather than mere disorder.
    target_start = session_timestamps[CONTEXT_PREFIX_LENGTH]
    for index in range(CONTEXT_PREFIX_LENGTH):
        if session_timestamps[index] >= target_start:
            raise _fail(
                FailureCategory.INVALID_INPUT,
                "FUTURE_CONTEXT_CANDLE",
                (
                    f"context candle {session_timestamps[index].isoformat()} is not strictly "
                    f"earlier than target start {target_start.isoformat()}"
                ),
                candle_index=index,
            )

    for index in range(1, EXAMPLE_LENGTH):
        if session_timestamps[index] <= session_timestamps[index - 1]:
            raise _fail(
                FailureCategory.INVALID_INPUT,
                "NON_MONOTONIC_CAUSAL_ORDER",
                (
                    f"timestamp {session_timestamps[index].isoformat()} does not follow "
                    f"{session_timestamps[index - 1].isoformat()}"
                ),
                candle_index=index,
            )

    for index in range(CONTEXT_PREFIX_LENGTH, EXAMPLE_LENGTH):
        if partition_membership[index] is not partition:
            raise _fail(
                FailureCategory.INVALID_INPUT,
                "SCORED_TARGET_OUTSIDE_PARTITION",
                (
                    f"scored candle {session_timestamps[index].isoformat()} belongs to "
                    f"{partition_membership[index].value}, not {partition.value}"
                ),
                candle_index=index,
            )

    if partition is Partition.TRAIN:
        for index in range(EXAMPLE_LENGTH):
            if partition_membership[index] in _HELD_OUT_PARTITIONS:
                raise _fail(
                    FailureCategory.INVALID_INPUT,
                    "HELD_OUT_CANDLE_IN_TRAINING_EXAMPLE",
                    (
                        f"training example contains a "
                        f"{partition_membership[index].value} candle at "
                        f"{session_timestamps[index].isoformat()}"
                    ),
                    candle_index=index,
                )
