"""Stage A: real feature extraction and its immutable evidence record.

Stage A passes only when extraction actually happened and every locked
invariant held. There is no default, no assumed value, and no pure state
transition: a run with no validated feature artifacts fails closed.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..errors import BridgeFailure, BridgeTransformError, FailureCategory
from ..windowing import (
    CONTEXT_PREFIX_LENGTH,
    EXAMPLE_LENGTH,
    SCORED_SUFFIX_LENGTH,
    SequenceSpec,
    score_mask_sha256,
)
from .cache import CACHE_SCHEMA_VERSION, CacheIdentity, FeatureCache, ShardRef
from .features import ExtractedSequence, FeatureExtractor
from .kronos import BRIDGE_INPUT_DIMENSION, SOURCE_SPEC, ResolvedAssets
from .provider import Candle, MarketSeries

__all__ = ["StageAReport", "StageASequenceRecord", "run_stage_a"]


def _fail(code: str, message: str) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            message=message,
        )
    )


class StageASequenceRecord(BaseModel):
    """Per-sequence evidence. Metadata and hashes only, never raw candles."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    sequence_id: str
    symbol: str
    interval: str
    partition: str
    target_start: str
    target_end: str
    prefix_start: str
    prefix_end: str
    bridge_input_shape: tuple[int, int]
    bridge_input_dtype: str
    canonical_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    shard_relative_path: str
    shard_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    shard_size_bytes: int = Field(ge=0)
    deterministic_replay_matched: bool


class StageAReport(BaseModel):
    """Immutable Stage A artifact. Contains no secret and no raw market data."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    schema_version: Literal["openalpha.bridge.phase2.stage_a.v1"] = (
        "openalpha.bridge.phase2.stage_a.v1"
    )
    run_id: str
    experiment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    amendment_sha256: tuple[str, ...]
    source_commit: str | None
    evidence_class: str
    provider: str
    provider_mode: str
    provider_client_version: str
    kronos_mode: str
    kronos_repository: str
    kronos_revision: str
    kronos_config_sha256: str | None
    kronos_weights_sha256: str | None
    frozen_parameter_sha256: str | None
    official_source_repository: str
    official_source_revision: str
    official_source_file_sha256: dict[str, str]
    cache_schema_version: str
    representation_version: str
    prefix_length: int
    suffix_length: int
    score_mask_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bridge_input_dimension: int
    retrieved_candles: int = Field(ge=0)
    sequences: tuple[StageASequenceRecord, ...]
    completed_at: datetime
    passed: bool

    @property
    def sequence_count(self) -> int:
        return len(self.sequences)


def _window_candles(
    series: MarketSeries, spec: SequenceSpec
) -> tuple[Candle, ...]:
    """Slice the exact 512-candle window the specification describes."""
    by_session = {candle.session: candle for candle in series.candles}
    if spec.prefix_start not in by_session:
        raise _fail(
            "WINDOW_PREFIX_START_UNAVAILABLE",
            f"{spec.symbol} has no candle at prefix start {spec.prefix_start}",
        )
    sessions = sorted(by_session)
    start = sessions.index(spec.prefix_start)
    window = sessions[start : start + EXAMPLE_LENGTH]
    if len(window) != EXAMPLE_LENGTH:
        raise _fail(
            "WINDOW_INCOMPLETE",
            (
                f"{spec.symbol} yields {len(window)} candles from {spec.prefix_start}, "
                f"need {EXAMPLE_LENGTH}"
            ),
        )
    return tuple(by_session[session] for session in window)


def run_stage_a(
    *,
    run_id: str,
    experiment_sha256: str,
    source_commit: str | None,
    evidence_class: str,
    series: MarketSeries,
    specs: tuple[SequenceSpec, ...],
    extractor: FeatureExtractor,
    assets: ResolvedAssets,
    cache: FeatureCache,
    completed_at: datetime,
    amendment_sha256: tuple[str, ...],
    verified_source_file_sha256: dict[str, str],
    verify_determinism: bool = True,
) -> StageAReport:
    """Extract, validate, cache, and record. Any deviation raises.

    ``verify_determinism`` re-runs extraction for each sequence and requires a
    byte-identical canonical hash, which is what makes the frozen path's
    determinism an observed property rather than an assumption.
    """
    if not specs:
        raise _fail("STAGE_A_NO_SEQUENCES", "Stage A requires at least one window")

    records: list[StageASequenceRecord] = []
    source_sha = series.normalized_sha256

    for spec in specs:
        candles = _window_candles(series, spec)
        extracted: ExtractedSequence = extractor.extract(
            spec=spec, candles=candles, source_data_sha256=source_sha
        )

        if extracted.bridge_input.shape != (EXAMPLE_LENGTH, BRIDGE_INPUT_DIMENSION):
            raise _fail(
                "STAGE_A_INVALID_SHAPE",
                f"{spec.sequence_id} produced {extracted.bridge_input.shape}",
            )
        if extracted.bridge_input.dtype != np.float32:
            raise _fail(
                "STAGE_A_INVALID_DTYPE",
                f"{spec.sequence_id} produced dtype {extracted.bridge_input.dtype}",
            )

        canonical = extracted.canonical_sha256()
        replay_matched = True
        if verify_determinism:
            replay = extractor.extract(
                spec=spec, candles=candles, source_data_sha256=source_sha
            )
            replay_matched = replay.canonical_sha256() == canonical
            if not replay_matched:
                raise _fail(
                    "STAGE_A_NONDETERMINISTIC",
                    f"{spec.sequence_id} did not reproduce byte-identical features",
                )

        identity = CacheIdentity(
            run_id=run_id,
            experiment_sha256=experiment_sha256,
            amendment_sha256=amendment_sha256,
            score_mask_sha256=score_mask_sha256(),
            source_commit=source_commit or "unknown",
            evidence_class=evidence_class,
            provider=series.provider,
            provider_client_version=series.client_version,
            symbol=spec.symbol,
            interval=spec.interval,
            partition=spec.partition,
            prefix_start=spec.prefix_start.isoformat(),
            prefix_end=spec.prefix_end.isoformat(),
            target_start=spec.target_start.isoformat(),
            target_end=spec.target_end.isoformat(),
            candle_data_sha256=source_sha,
            official_source_repository=SOURCE_SPEC.repository,
            official_source_revision=SOURCE_SPEC.revision,
            official_source_file_sha256=dict(verified_source_file_sha256),
            tokenizer_repository=assets.repository,
            tokenizer_revision=assets.revision,
            tokenizer_config_sha256=assets.observed_config_sha256,
            tokenizer_weights_sha256=assets.observed_weights_sha256,
            frozen_parameter_sha256=assets.frozen_parameter_sha256,
            feature_schema_version=CACHE_SCHEMA_VERSION,
            representation_version="openalpha.bridge.financial.v1",
            tensor_specification={
                "coarse_ids": "int64[512]",
                "fine_ids": "int64[512]",
                "bipolar_latent": "float32[512,20]",
                "frozen_hidden": "float32[512,256]",
                "causal_features": "float32[512,13]",
                "constrained_targets": "float32[64,5]",
                "bridge_input": "float32[512,269]",
            },
        )
        shard: ShardRef = cache.write(extracted.to_cached_example(identity))
        # Read back through the cache so integrity is proven, not assumed.
        restored = cache.read(shard)
        if restored.sequence_id != extracted.sequence_id:
            raise _fail(
                "STAGE_A_CACHE_IDENTITY_MISMATCH",
                f"cache returned {restored.sequence_id} for {extracted.sequence_id}",
            )
        if restored.source_data_sha256 != source_sha:
            raise _fail(
                "STAGE_A_CACHE_SOURCE_MISMATCH",
                f"{spec.sequence_id} cache does not bind the active source data",
            )

        records.append(
            StageASequenceRecord(
                sequence_id=spec.sequence_id,
                symbol=spec.symbol,
                interval=spec.interval,
                partition=spec.partition.value,
                target_start=spec.target_start.isoformat(),
                target_end=spec.target_end.isoformat(),
                prefix_start=spec.prefix_start.isoformat(),
                prefix_end=spec.prefix_end.isoformat(),
                bridge_input_shape=(EXAMPLE_LENGTH, BRIDGE_INPUT_DIMENSION),
                bridge_input_dtype="float32",
                canonical_sha256=canonical,
                shard_relative_path=shard.relative_path,
                shard_content_sha256=shard.content_sha256,
                shard_size_bytes=shard.size_bytes,
                deterministic_replay_matched=replay_matched,
            )
        )

    return StageAReport(
        run_id=run_id,
        experiment_sha256=experiment_sha256,
        amendment_sha256=amendment_sha256,
        source_commit=source_commit,
        evidence_class=evidence_class,
        provider=series.provider,
        provider_mode=series.provider_mode.value,
        provider_client_version=series.client_version,
        kronos_mode=assets.kronos_mode.value,
        kronos_repository=assets.repository,
        kronos_revision=assets.revision,
        kronos_config_sha256=assets.observed_config_sha256,
        kronos_weights_sha256=assets.observed_weights_sha256,
        frozen_parameter_sha256=assets.frozen_parameter_sha256,
        official_source_repository=SOURCE_SPEC.repository,
        official_source_revision=SOURCE_SPEC.revision,
        official_source_file_sha256=dict(verified_source_file_sha256),
        cache_schema_version=CACHE_SCHEMA_VERSION,
        representation_version="openalpha.bridge.financial.v1",
        prefix_length=CONTEXT_PREFIX_LENGTH,
        suffix_length=SCORED_SUFFIX_LENGTH,
        score_mask_sha256=score_mask_sha256(),
        bridge_input_dimension=BRIDGE_INPUT_DIMENSION,
        retrieved_candles=len(series.candles),
        sequences=tuple(records),
        completed_at=completed_at,
        passed=True,
    )
