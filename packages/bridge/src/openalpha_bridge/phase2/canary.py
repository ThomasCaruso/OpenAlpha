"""Stage A official compatibility canary.

One isolated check that the official frozen path executes and matches the locked
contract on the single amended SPY training window. It deliberately does not go
through the Phase 2 runner, because that runner continues into Stage B and
Stage C. There is no code path here into training, checkpoint creation,
checkpoint selection, metric or gate evaluation, or test opening.

Success is exactly one code: STAGE_A_OFFICIAL_CANARY_PASSED. It is compatibility
evidence, not a reconstruction result, and it authorizes nothing.
"""

from __future__ import annotations

import re
import time
from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..errors import BridgeFailure, BridgeTransformError, FailureCategory
from ..windowing import (
    CONTEXT_PREFIX_LENGTH,
    EXAMPLE_LENGTH,
    SCORED_SUFFIX_LENGTH,
    Partition,
    SequenceSpec,
    score_mask_sha256,
    sequence_id,
)
from .cache import CACHE_SCHEMA_VERSION, CacheIdentity, FeatureCache
from .features import FeatureExtractor
from .identity import (
    AMENDMENT_1_SHA256,
    AMENDMENT_2_SHA256,
    AMENDMENT_3_SHA256,
    EXPERIMENT_SHA256,
    verify_locked_hashes,
)
from .kronos import (
    BRIDGE_INPUT_DIMENSION,
    DECODER_HIDDEN_DIMENSION,
    EFFECTIVE_DECODER_BLOCKS,
    QUANTIZED_LATENT_DIMENSION,
    SOURCE_SPEC,
    TOKENIZER_INPUT_DIMENSION,
    TOKENIZER_SPEC,
    KronosBackend,
    KronosMode,
)
from .observed import ObservedKronosComponents, assert_observed_matches_locks
from .provider import Candle, MarketSeries, Phase2Provider, RetrievalRequest, validate_series
from .states import EvidenceClass

__all__ = [
    "CANARY_EVIDENCE_CLASS",
    "CANARY_SUCCESS_CODE",
    "CANARY_WINDOW",
    "CanaryFailure",
    "CanaryReport",
    "CanaryWindow",
    "run_stage_a_canary",
]

#: Amendment 3 stage_a_official_canary. Its own class with its own storage
#: root: never real_phase2 and never synthetic_pipeline_validation.
CANARY_EVIDENCE_CLASS = EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY
CANARY_SUCCESS_CODE = "STAGE_A_OFFICIAL_CANARY_PASSED"

AMENDMENTS: tuple[str, ...] = (AMENDMENT_1_SHA256, AMENDMENT_2_SHA256, AMENDMENT_3_SHA256)


def _fail(code: str, message: str) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            message=message,
        )
    )


class CanaryWindow(BaseModel):
    """The single amended SPY training window. Nothing else may be retrieved."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    symbol: Literal["SPY"] = "SPY"
    interval: Literal["1d"] = "1d"
    partition: Literal[Partition.TRAIN] = Partition.TRAIN
    sequence_id: Literal["ecd7fd5797a9147a"] = "ecd7fd5797a9147a"
    prefix_start: str = "2015-05-07"
    prefix_end: str = "2017-02-14"
    target_start: str = "2017-02-15"
    target_end: str = "2017-05-17"
    retrieval_start_inclusive: str = "2015-05-07"
    retrieval_end_exclusive: str = "2017-05-18"
    expected_candles: int = EXAMPLE_LENGTH

    def to_spec(self) -> SequenceSpec:
        from datetime import date

        def parse(value: str) -> date:
            return date.fromisoformat(value)

        derived = sequence_id(
            symbol=self.symbol,
            interval=self.interval,
            partition=Partition.TRAIN,
            target_start=parse(self.target_start),
            target_end=parse(self.target_end),
        )
        if derived != self.sequence_id:
            raise _fail(
                "CANARY_SEQUENCE_ID_MISMATCH",
                f"amended sequence {self.sequence_id} does not derive from its dates ({derived})",
            )
        return SequenceSpec(
            sequence_id=derived,
            symbol=self.symbol,
            interval=self.interval,
            partition=Partition.TRAIN,
            prefix_start=parse(self.prefix_start),
            prefix_end=parse(self.prefix_end),
            target_start=parse(self.target_start),
            target_end=parse(self.target_end),
            prefix_length=CONTEXT_PREFIX_LENGTH,
            suffix_length=SCORED_SUFFIX_LENGTH,
            target_overlaps_other_target=False,
        )


CANARY_WINDOW = CanaryWindow()



class CanaryFailure(BaseModel):
    """Immutable terminal record of a canary that did not pass.

    A failure is preserved, never retried and never converted into success.
    """

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    schema_version: Literal["openalpha.bridge.phase2.stage_a_canary_failure.v1"] = (
        "openalpha.bridge.phase2.stage_a_canary_failure.v1"
    )
    outcome: Literal[
        "STAGE_A_OFFICIAL_CANARY_FAILED", "STAGE_A_OFFICIAL_CANARY_OPERATIONAL_FAILURE"
    ]
    evidence_class: EvidenceClass = CANARY_EVIDENCE_CLASS
    authorizes_stage_b: Literal[False] = False
    authorizes_real_run: Literal[False] = False
    scientific_result_available: Literal[False] = False

    run_id: str
    source_commit: str
    experiment_sha256: str
    amendment_sha256: tuple[str, ...]
    failure_stage: str
    failure_code: str
    message: str
    completed_at: datetime


class CanaryReport(BaseModel):
    """Immutable compatibility evidence. Metadata and hashes only."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    schema_version: Literal["openalpha.bridge.phase2.stage_a_canary.v1"] = (
        "openalpha.bridge.phase2.stage_a_canary.v1"
    )
    outcome: str
    evidence_class: EvidenceClass = CANARY_EVIDENCE_CLASS
    authorizes_stage_b: Literal[False] = False
    authorizes_real_run: Literal[False] = False

    source_commit: str
    experiment_sha256: str
    amendment_sha256: tuple[str, ...]
    score_mask_sha256: str

    symbol: str
    sequence_id: str
    prefix_start: str
    prefix_end: str
    target_start: str
    target_end: str
    candle_data_sha256: str
    retrieved_candles: int

    official_source_repository: str
    official_source_revision: str
    official_source_file_sha256: dict[str, str]
    official_dependency_versions: dict[str, str]
    tokenizer_repository: str
    tokenizer_revision: str
    tokenizer_config_sha256: str | None
    tokenizer_weights_sha256: str | None
    frozen_parameter_sha256: str | None
    observed_components: ObservedKronosComponents

    coarse_id_range: tuple[int, int]
    fine_id_range: tuple[int, int]
    bipolar_latent_shape: tuple[int, int]
    frozen_hidden_shape: tuple[int, int]
    bridge_input_shape: tuple[int, int]
    bipolar_latent_dtype: str
    frozen_hidden_dtype: str
    bridge_input_dtype: str

    deterministic_replay_sha256: str
    deterministic_replay_matched: bool
    scored_suffix_causality_holds: bool

    cache_schema_version: str
    shard_relative_path: str
    shard_content_sha256: str
    shard_bytes: int
    cache_read_verified: bool

    peak_gpu_memory_bytes: int | None
    wall_clock_seconds: float
    provider_request_count: int = Field(ge=0)
    downloaded_asset_bytes: int | None
    completed_at: datetime


def _peak_gpu_memory() -> int | None:
    try:
        import torch  # pyright: ignore[reportMissingImports]
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.max_memory_allocated())


_RUN_ID_PATTERN = re.compile(r"^canary_[0-9a-f]{8,32}$")
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def _validate_invocation(run_id: str, source_commit: str, deployed_commit: str | None) -> None:
    """Reject anything that could become a path or a mismatched deployment."""
    if not _COMMIT_PATTERN.fullmatch(source_commit):
        raise _fail(
            "CANARY_INVALID_SOURCE_COMMIT",
            "source_commit must be exactly 40 lowercase hexadecimal characters",
        )
    if deployed_commit is not None and source_commit != deployed_commit:
        raise _fail(
            "CANARY_SOURCE_COMMIT_MISMATCH",
            (
                "source_commit does not match the commit baked into the deployed "
                "image; the canary must run the code that was deployed"
            ),
        )
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise _fail(
            "CANARY_INVALID_RUN_ID",
            "run_id must match canary_<8-32 hex>; slashes, traversal, and whitespace are refused",
        )


def run_stage_a_canary(
    *,
    provider: Phase2Provider,
    backend: KronosBackend,
    cache: FeatureCache,
    research_root: Any,
    source_commit: str,
    run_id: str,
    downloaded_asset_bytes: int | None = None,
    deployed_commit: str | None = None,
    now: datetime | None = None,
) -> CanaryReport:
    """Execute the canary end to end, or fail closed and preserve the negative."""
    _validate_invocation(run_id, source_commit, deployed_commit)
    started = time.perf_counter()

    # 1. locks and amendments
    observed_locks = verify_locked_hashes(research_root)
    if observed_locks.get("phase2-amendment-3-scale-features.yaml") != AMENDMENT_3_SHA256:
        raise _fail("CANARY_AMENDMENT_MISMATCH", "Amendment 3 does not verify")
    if backend.mode is not KronosMode.PINNED_OFFICIAL:
        raise _fail(
            "CANARY_REQUIRES_OFFICIAL_BACKEND",
            f"the canary requires the pinned official backend, got {backend.mode.value}",
        )

    spec = CANARY_WINDOW.to_spec()

    # 2. retrieve ONLY the amended window
    from datetime import date

    series: MarketSeries = provider.fetch(
        RetrievalRequest(
            symbol=CANARY_WINDOW.symbol,
            start=date.fromisoformat(CANARY_WINDOW.retrieval_start_inclusive),
            end=date.fromisoformat(CANARY_WINDOW.retrieval_end_exclusive),
            maximum_candles=EXAMPLE_LENGTH,
        )
    )
    validate_series(series)
    if len(series.candles) != EXAMPLE_LENGTH:
        raise _fail(
            "CANARY_UNEXPECTED_CANDLE_COUNT",
            f"expected {EXAMPLE_LENGTH} candles, retrieved {len(series.candles)}",
        )

    # 3. official assets
    assets = backend.resolve_assets()
    if not assets.revisions_verified:
        raise _fail("CANARY_ASSETS_UNVERIFIED", "official asset revisions did not verify")

    observed_source = dict(getattr(backend, "observed_source_files", {}))
    if observed_source != dict(SOURCE_SPEC.files):
        raise _fail(
            "CANARY_SOURCE_FILES_MISMATCH",
            "the executed official source files do not equal SOURCE_SPEC.files",
        )
    asset_checks = (
        ("repository", assets.repository, TOKENIZER_SPEC.repository),
        ("revision", assets.revision, TOKENIZER_SPEC.revision),
        ("config_sha256", assets.observed_config_sha256, TOKENIZER_SPEC.config_sha256),
        ("weights_sha256", assets.observed_weights_sha256, TOKENIZER_SPEC.weights_sha256),
    )
    drifted = sorted(name for name, got, want in asset_checks if got != want)
    if drifted:
        raise _fail(
            "CANARY_TOKENIZER_IDENTITY_MISMATCH",
            f"resolved tokenizer does not equal TOKENIZER_SPEC: {', '.join(drifted)}",
        )

    # 4. extraction over the official path
    extractor = FeatureExtractor(backend)
    candles: tuple[Candle, ...] = series.candles
    source_sha = series.normalized_sha256
    extracted = extractor.extract(spec=spec, candles=candles, source_data_sha256=source_sha)

    observed = getattr(backend, "observed", None)
    if not isinstance(observed, ObservedKronosComponents):
        raise _fail(
            "CANARY_OBSERVED_MANIFEST_MISSING",
            "the backend produced no runtime-observed component manifest",
        )
    assert_observed_matches_locks(
        observed,
        quantized_latent_dimension=QUANTIZED_LATENT_DIMENSION,
        decoder_hidden_dimension=DECODER_HIDDEN_DIMENSION,
        tokenizer_input_dimension=TOKENIZER_INPUT_DIMENSION,
        expected_decoder_blocks=EFFECTIVE_DECODER_BLOCKS,
        expected_sequence_length=EXAMPLE_LENGTH,
    )

    # 5. byte-identical replay
    replay = extractor.extract(spec=spec, candles=candles, source_data_sha256=source_sha)
    canonical = extracted.canonical_sha256()
    replay_matched = replay.canonical_sha256() == canonical
    if not replay_matched:
        raise _fail(
            "CANARY_NONDETERMINISTIC",
            "repeated extraction did not reproduce byte-identical features",
        )

    # 6. causality at the FIRST scored position, on the official backend.
    #    Perturbing only the final candle would prove almost nothing: causal
    #    attention already makes position 511 the sole position it could reach.
    perturbed = list(candles)
    pivot = CONTEXT_PREFIX_LENGTH
    original = perturbed[pivot]
    perturbed[pivot] = Candle(
        session=original.session,
        open=original.open * 1.03,
        high=original.high * 1.05,
        low=original.low * 0.97,
        close=original.close * 1.04,
        volume=original.volume * 1.5,
        amount=original.amount * 1.5,
    )
    disturbed = extractor.extract(
        spec=spec, candles=tuple(perturbed), source_data_sha256=source_sha
    )

    # The perturbation must actually reach the official representation, or the
    # causality assertion below would hold vacuously.
    effective = (
        not np.array_equal(extracted.coarse_ids[pivot:], disturbed.coarse_ids[pivot:])
        or not np.array_equal(extracted.fine_ids[pivot:], disturbed.fine_ids[pivot:])
        or not np.array_equal(
            extracted.bipolar_latent[pivot:], disturbed.bipolar_latent[pivot:]
        )
        or not np.array_equal(
            extracted.frozen_hidden[pivot:], disturbed.frozen_hidden[pivot:]
        )
    )
    if not effective:
        raise _fail(
            "CANARY_CAUSALITY_PERTURBATION_INEFFECTIVE",
            (
                "perturbing the first scored candle changed no official token, latent, "
                "or hidden value, so causality was not actually exercised"
            ),
        )

    causality_holds = bool(
        np.array_equal(extracted.frozen_hidden[:pivot], disturbed.frozen_hidden[:pivot])
    )
    if not causality_holds:
        raise _fail(
            "CANARY_CAUSALITY_VIOLATED",
            (
                f"perturbing scored position {pivot} changed the frozen hidden state "
                "at an earlier position"
            ),
        )

    # 7. identity-bound cache write, then a verified read
    identity = CacheIdentity(
        run_id=run_id,
        experiment_sha256=EXPERIMENT_SHA256,
        amendment_sha256=AMENDMENTS,
        score_mask_sha256=score_mask_sha256(),
        source_commit=source_commit,
        evidence_class=CANARY_EVIDENCE_CLASS,
        provider=series.provider,
        provider_client_version=series.client_version,
        symbol=spec.symbol,
        interval=spec.interval,
        partition=Partition.TRAIN,
        prefix_start=CANARY_WINDOW.prefix_start,
        prefix_end=CANARY_WINDOW.prefix_end,
        target_start=CANARY_WINDOW.target_start,
        target_end=CANARY_WINDOW.target_end,
        candle_data_sha256=source_sha,
        official_source_repository=SOURCE_SPEC.repository,
        official_source_revision=SOURCE_SPEC.revision,
        official_source_file_sha256=dict(getattr(backend, "observed_source_files", {})),
        tokenizer_repository=assets.repository,
        tokenizer_revision=assets.revision,
        tokenizer_config_sha256=assets.observed_config_sha256,
        tokenizer_weights_sha256=assets.observed_weights_sha256,
        frozen_parameter_sha256=assets.frozen_parameter_sha256,
        feature_schema_version=CACHE_SCHEMA_VERSION,
        representation_version="openalpha.bridge.financial.v1",
        tensor_specification={
            "bipolar_latent": f"float32[{EXAMPLE_LENGTH},{QUANTIZED_LATENT_DIMENSION}]",
            "frozen_hidden": f"float32[{EXAMPLE_LENGTH},{DECODER_HIDDEN_DIMENSION}]",
            "bridge_input": f"float32[{EXAMPLE_LENGTH},{BRIDGE_INPUT_DIMENSION}]",
        },
    )
    shard = cache.write(extracted.to_cached_example(identity))
    restored = cache.read(shard)
    cache_verified = (
        restored.sequence_id == extracted.sequence_id
        and restored.identity.identity_sha256 == identity.identity_sha256
        and np.array_equal(restored.frozen_hidden, extracted.frozen_hidden)
    )
    if not cache_verified:
        raise _fail("CANARY_CACHE_VERIFICATION_FAILED", "the cached shard did not round trip")

    return CanaryReport(
        outcome=CANARY_SUCCESS_CODE,
        source_commit=source_commit,
        experiment_sha256=EXPERIMENT_SHA256,
        amendment_sha256=AMENDMENTS,
        score_mask_sha256=score_mask_sha256(),
        symbol=spec.symbol,
        sequence_id=spec.sequence_id,
        prefix_start=CANARY_WINDOW.prefix_start,
        prefix_end=CANARY_WINDOW.prefix_end,
        target_start=CANARY_WINDOW.target_start,
        target_end=CANARY_WINDOW.target_end,
        candle_data_sha256=source_sha,
        retrieved_candles=len(series.candles),
        official_source_repository=SOURCE_SPEC.repository,
        official_source_revision=SOURCE_SPEC.revision,
        official_source_file_sha256=dict(getattr(backend, "observed_source_files", {})),
        official_dependency_versions=dict(getattr(backend, "dependency_versions", {})),
        tokenizer_repository=assets.repository,
        tokenizer_revision=assets.revision,
        tokenizer_config_sha256=assets.observed_config_sha256,
        tokenizer_weights_sha256=assets.observed_weights_sha256,
        frozen_parameter_sha256=assets.frozen_parameter_sha256,
        observed_components=observed,
        coarse_id_range=(int(extracted.coarse_ids.min()), int(extracted.coarse_ids.max())),
        fine_id_range=(int(extracted.fine_ids.min()), int(extracted.fine_ids.max())),
        bipolar_latent_shape=(EXAMPLE_LENGTH, QUANTIZED_LATENT_DIMENSION),
        frozen_hidden_shape=(EXAMPLE_LENGTH, DECODER_HIDDEN_DIMENSION),
        bridge_input_shape=(EXAMPLE_LENGTH, BRIDGE_INPUT_DIMENSION),
        bipolar_latent_dtype=str(extracted.bipolar_latent.dtype),
        frozen_hidden_dtype=str(extracted.frozen_hidden.dtype),
        bridge_input_dtype=str(extracted.bridge_input.dtype),
        deterministic_replay_sha256=canonical,
        deterministic_replay_matched=replay_matched,
        scored_suffix_causality_holds=causality_holds,
        cache_schema_version=CACHE_SCHEMA_VERSION,
        shard_relative_path=shard.relative_path,
        shard_content_sha256=shard.content_sha256,
        shard_bytes=shard.size_bytes,
        cache_read_verified=cache_verified,
        peak_gpu_memory_bytes=_peak_gpu_memory(),
        wall_clock_seconds=round(time.perf_counter() - started, 3),
        provider_request_count=1,
        downloaded_asset_bytes=downloaded_asset_bytes,
        completed_at=now or datetime.now(UTC),
    )
