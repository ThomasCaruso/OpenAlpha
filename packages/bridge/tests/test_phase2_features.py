"""Stage A feature-extraction tests.

Deterministic test doubles only. No network, no official Kronos asset, and no
real market data; the official integration path is exercised separately and is
explicitly marked.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from openalpha_bridge.errors import BridgeTransformError
from openalpha_bridge.phase2.cache import FeatureCache
from openalpha_bridge.phase2.features import (
    CLIP_RANGE,
    NORMALIZATION_EPSILON,
    ExtractedSequence,
    FeatureExtractor,
    candle_matrix,
    compute_scale_features,
    forward_representation,
    normalize_with_prefix_state,
)
from openalpha_bridge.phase2.kronos import (
    BRIDGE_INPUT_DIMENSION,
    DECODER_HIDDEN_DIMENSION,
    EFFECTIVE_DECODER_BLOCKS,
    QUANTIZED_LATENT_DIMENSION,
    SCALE_FEATURE_COUNT,
    SCALE_FEATURE_ORDER,
    SOURCE_SPEC,
    TOKENIZER_INPUT_DIMENSION,
    DeterministicFakeKronosBackend,
)
from openalpha_bridge.phase2.provider import Candle, MarketSeries, ProviderMode
from openalpha_bridge.phase2.stage_a import run_stage_a
from openalpha_bridge.phase2.stage_a_plan import build_stage_a_plan
from openalpha_bridge.phase2.stage_a_verify import verify_stage_a_report
from openalpha_bridge.windowing import (
    CONTEXT_PREFIX_LENGTH,
    EXAMPLE_LENGTH,
    SCORED_SUFFIX_LENGTH,
    Partition,
    SequenceSpec,
    score_mask_sha256,
    sequence_id,
)

START = date(2020, 1, 1)


def _candles(count: int = EXAMPLE_LENGTH, *, seed: int = 7) -> tuple[Candle, ...]:
    rng = np.random.default_rng(seed)
    out: list[Candle] = []
    level = 100.0
    for index in range(count):
        level = max(5.0, level * float(1.0 + rng.normal(0.0, 0.004)))
        open_ = level
        close = max(5.0, level * float(1.0 + rng.normal(0.0, 0.004)))
        high = max(open_, close) * 1.002
        low = min(open_, close) / 1.002
        volume = 1.0e6 * float(1.0 + rng.random())
        ohlc_mean = (open_ + high + low + close) / 4.0
        out.append(
            Candle(
                session=START + timedelta(days=index),
                open=open_,
                high=high,
                low=low,
                close=close,
                volume=volume,
                amount=volume * ohlc_mean,
            )
        )
        level = close
    return tuple(out)


def _spec(candles: tuple[Candle, ...], partition: Partition = Partition.TRAIN) -> SequenceSpec:
    target_start = candles[CONTEXT_PREFIX_LENGTH].session
    target_end = candles[-1].session
    return SequenceSpec(
        sequence_id=sequence_id(
            symbol="SPY",
            interval="1d",
            partition=partition,
            target_start=target_start,
            target_end=target_end,
        ),
        symbol="SPY",
        interval="1d",
        partition=partition,
        prefix_start=candles[0].session,
        prefix_end=candles[CONTEXT_PREFIX_LENGTH - 1].session,
        target_start=target_start,
        target_end=target_end,
        prefix_length=CONTEXT_PREFIX_LENGTH,
        suffix_length=SCORED_SUFFIX_LENGTH,
        target_overlaps_other_target=False,
    )


def _extract(candles: tuple[Candle, ...], **kwargs: object) -> ExtractedSequence:
    extractor = FeatureExtractor(DeterministicFakeKronosBackend())
    return extractor.extract(
        spec=_spec(candles), candles=candles, source_data_sha256="0" * 64, **kwargs
    )


def _identity(partition: Partition = Partition.TRAIN, **overrides):
    """A complete cache identity for tests."""
    from openalpha_bridge.phase2.cache import CACHE_SCHEMA_VERSION, CacheIdentity

    payload = {
        "run_id": "syn_stage_a",
        "experiment_sha256": "d" * 64,
        "amendment_sha256": ("1" * 64, "2" * 64, "3" * 64),
        "score_mask_sha256": score_mask_sha256(),
        "source_commit": "a" * 40,
        "evidence_class": "development_compatibility_canary",
        "provider": "deterministic_fake",
        "provider_client_version": "fake-1",
        "symbol": "SPY",
        "interval": "1d",
        "partition": partition,
        "prefix_start": "2015-05-07",
        "prefix_end": "2017-02-14",
        "target_start": "2017-02-15",
        "target_end": "2017-05-17",
        "candle_data_sha256": "0" * 64,
        "official_source_repository": "https://github.com/shiyu-coder/Kronos",
        "official_source_revision": "6" * 40,
        "official_source_file_sha256": {"model/kronos.py": "e" * 64},
        "tokenizer_repository": "NeoQuasar/Kronos-Tokenizer-2k",
        "tokenizer_revision": "2" * 40,
        "tokenizer_config_sha256": None,
        "tokenizer_weights_sha256": None,
        "frozen_parameter_sha256": "f" * 64,
        "feature_schema_version": CACHE_SCHEMA_VERSION,
        "representation_version": "openalpha.bridge.financial.v1",
        "tensor_specification": {"bridge_input": "float32[512,269]"},
    }
    payload.update(overrides)
    return CacheIdentity(**payload)


# ------------------------------------------------------------- dimensionality


def test_bridge_input_is_exactly_269_and_composed_as_preregistered() -> None:
    assert DECODER_HIDDEN_DIMENSION + SCALE_FEATURE_COUNT == BRIDGE_INPUT_DIMENSION == 269
    extracted = _extract(_candles())
    assert extracted.bridge_input.shape == (EXAMPLE_LENGTH, 269)
    assert extracted.frozen_hidden.shape == (EXAMPLE_LENGTH, 256)
    assert extracted.causal_features.shape == (EXAMPLE_LENGTH, 13)
    # The first 256 columns are the frozen trunk, the last 13 the scale features.
    assert np.allclose(extracted.bridge_input[:, :256], extracted.frozen_hidden)
    assert np.allclose(extracted.bridge_input[:, 256:], extracted.causal_features)


def test_intermediate_tensor_contract() -> None:
    extracted = _extract(_candles())
    assert extracted.coarse_ids.shape == (EXAMPLE_LENGTH,)
    assert extracted.fine_ids.shape == (EXAMPLE_LENGTH,)
    assert extracted.coarse_ids.dtype == np.int64
    assert extracted.bipolar_latent.shape == (EXAMPLE_LENGTH, QUANTIZED_LATENT_DIMENSION)
    assert extracted.bridge_input.dtype == np.float32
    assert extracted.constrained_targets.shape == (SCORED_SUFFIX_LENGTH, 5)
    assert int(extracted.coarse_ids.min()) >= 0
    assert int(extracted.coarse_ids.max()) <= 1023


def test_locked_constants_match_the_experiment() -> None:
    assert TOKENIZER_INPUT_DIMENSION == 6
    assert EFFECTIVE_DECODER_BLOCKS == 3
    assert QUANTIZED_LATENT_DIMENSION == 20
    assert len(SCALE_FEATURE_ORDER) == SCALE_FEATURE_COUNT == 13


# ------------------------------------------------------------- normalization


def test_normalization_uses_prefix_statistics_only() -> None:
    """Suffix values must not influence the encoder input of the prefix."""
    candles = _candles()
    matrix = candle_matrix(candles)
    baseline, mean, std = normalize_with_prefix_state(matrix)

    perturbed = matrix.copy()
    perturbed[CONTEXT_PREFIX_LENGTH:, :] *= 3.0
    after, mean_after, std_after = normalize_with_prefix_state(perturbed)

    assert np.allclose(mean, mean_after)
    assert np.allclose(std, std_after)
    assert np.array_equal(baseline[:CONTEXT_PREFIX_LENGTH], after[:CONTEXT_PREFIX_LENGTH])


def test_normalized_values_are_clipped_to_the_official_range() -> None:
    candles = _candles()
    matrix = candle_matrix(candles)
    matrix[CONTEXT_PREFIX_LENGTH + 1, 0] = 1.0e9  # a violent outlier in the suffix
    normalized, _, _ = normalize_with_prefix_state(matrix)
    assert float(normalized.min()) >= CLIP_RANGE[0]
    assert float(normalized.max()) <= CLIP_RANGE[1]


def test_normalization_epsilon_prevents_division_by_zero() -> None:
    matrix = np.ones((EXAMPLE_LENGTH, 6), dtype=np.float64)
    normalized, _, std = normalize_with_prefix_state(matrix)
    assert np.allclose(std, 0.0)
    assert np.isfinite(normalized).all()
    assert NORMALIZATION_EPSILON > 0.0


def test_too_short_a_window_fails() -> None:
    with pytest.raises(BridgeTransformError) as excinfo:
        normalize_with_prefix_state(np.ones((10, 6), dtype=np.float64))
    assert excinfo.value.failures[0].code == "SEQUENCE_TOO_SHORT"


# ------------------------------------------------------------ scale features


def test_scale_features_are_prefix_only_and_ordered() -> None:
    candles = _candles()
    matrix = candle_matrix(candles)
    baseline = compute_scale_features(matrix)
    assert baseline.shape == (13,)

    perturbed = matrix.copy()
    perturbed[CONTEXT_PREFIX_LENGTH:, :] *= 5.0
    assert np.array_equal(baseline, compute_scale_features(perturbed))


def test_scale_feature_formulas_follow_the_locked_names() -> None:
    matrix = candle_matrix(_candles())
    values = compute_scale_features(matrix)
    prefix = matrix[:CONTEXT_PREFIX_LENGTH]
    anchor = float(prefix[-1, 3])
    mean = prefix.mean(axis=0)
    std = prefix.std(axis=0, ddof=0)

    assert values[SCALE_FEATURE_ORDER.index("log_anchor_close")] == pytest.approx(
        math.log(anchor), rel=1e-6
    )
    assert values[SCALE_FEATURE_ORDER.index("open_mean_relative_to_anchor")] == pytest.approx(
        mean[0] / anchor, rel=1e-6
    )
    assert values[SCALE_FEATURE_ORDER.index("log1p_close_std_over_anchor")] == pytest.approx(
        math.log1p(std[3] / anchor), rel=1e-6
    )
    assert values[SCALE_FEATURE_ORDER.index("log1p_volume_mean")] == pytest.approx(
        math.log1p(mean[4]), rel=1e-6
    )
    assert values[SCALE_FEATURE_ORDER.index("log1p_amount_std")] == pytest.approx(
        math.log1p(std[5]), rel=1e-6
    )


def test_invalid_anchor_fails_closed() -> None:
    matrix = candle_matrix(_candles())
    matrix[CONTEXT_PREFIX_LENGTH - 1, 3] = 0.0
    with pytest.raises(BridgeTransformError) as excinfo:
        compute_scale_features(matrix)
    assert excinfo.value.failures[0].code == "INVALID_ANCHOR_CLOSE"


# ---------------------------------------------------------------- causality


def test_a_future_candle_cannot_alter_an_earlier_timestep_feature() -> None:
    """The decisive causality property for the frozen path."""
    candles = _candles()
    baseline = _extract(candles)

    perturbed_list = list(candles)
    late = perturbed_list[-1]
    perturbed_list[-1] = Candle(
        session=late.session,
        open=late.open * 1.05,
        high=late.high * 1.08,
        low=late.low * 0.95,
        close=late.close * 1.06,
        volume=late.volume * 2.0,
        amount=late.amount * 2.0,
    )
    perturbed = _extract(tuple(perturbed_list))

    # Every position strictly before the perturbed one is untouched.
    assert np.array_equal(baseline.frozen_hidden[:-1], perturbed.frozen_hidden[:-1])
    assert np.array_equal(baseline.bridge_input[:-1], perturbed.bridge_input[:-1])


def test_prefix_rows_are_unaffected_by_any_suffix_change() -> None:
    candles = _candles()
    baseline = _extract(candles)

    perturbed_list = list(candles)
    for index in range(CONTEXT_PREFIX_LENGTH, EXAMPLE_LENGTH):
        c = perturbed_list[index]
        perturbed_list[index] = Candle(
            session=c.session,
            open=c.open * 1.01,
            high=c.high * 1.01,
            low=c.low * 0.99,
            close=c.close * 1.01,
            volume=c.volume,
            amount=c.amount,
        )
    perturbed = _extract(tuple(perturbed_list))

    assert np.array_equal(
        baseline.bridge_input[:CONTEXT_PREFIX_LENGTH],
        perturbed.bridge_input[:CONTEXT_PREFIX_LENGTH],
    )


def test_score_mask_selects_only_the_scored_suffix() -> None:
    extracted = _extract(_candles())
    example = extracted.to_cached_example(_identity())
    mask = example.score_mask
    assert mask.sum() == SCORED_SUFFIX_LENGTH
    assert not mask[:CONTEXT_PREFIX_LENGTH].any()
    assert mask[CONTEXT_PREFIX_LENGTH:].all()
    assert extracted.score_mask_sha256 == (
        "2fe5b1b3c66dfd7c7e8af2612d69d3c2337a4a0f6dd3896a4d7c2f69911f3711"
    )


# ------------------------------------------------------------- determinism


def test_extraction_is_byte_identical_across_repeats() -> None:
    candles = _candles()
    first, second = _extract(candles), _extract(candles)
    assert first.canonical_sha256() == second.canonical_sha256()
    assert np.array_equal(first.bridge_input, second.bridge_input)


# ------------------------------------------------------- window enforcement


@pytest.mark.parametrize(
    ("count", "code"),
    [(511, "INVALID_EXAMPLE_LENGTH"), (513, "INVALID_EXAMPLE_LENGTH")],
)
def test_wrong_window_length_fails(count: int, code: str) -> None:
    candles = _candles(count)
    extractor = FeatureExtractor(DeterministicFakeKronosBackend())
    spec = _spec(_candles())
    with pytest.raises(BridgeTransformError) as excinfo:
        extractor.extract(spec=spec, candles=candles, source_data_sha256="0" * 64)
    assert excinfo.value.failures[0].code == code


def test_window_must_match_the_specification_boundaries() -> None:
    """A specification describing different sessions must not silently bind."""
    candles = _candles()
    base = _spec(candles)
    shifted = base.model_copy(update={"prefix_start": base.prefix_start + timedelta(days=1)})
    extractor = FeatureExtractor(DeterministicFakeKronosBackend())
    with pytest.raises(BridgeTransformError) as excinfo:
        extractor.extract(spec=shifted, candles=candles, source_data_sha256="0" * 64)
    assert excinfo.value.failures[0].code == "PREFIX_START_MISMATCH"


def test_non_monotonic_sessions_fail() -> None:
    candles = list(_candles())
    candles[10], candles[11] = candles[11], candles[10]
    extractor = FeatureExtractor(DeterministicFakeKronosBackend())
    with pytest.raises(BridgeTransformError) as excinfo:
        extractor.extract(
            spec=_spec(tuple(candles)), candles=tuple(candles), source_data_sha256="0" * 64
        )
    assert excinfo.value.failures[0].code == "NON_MONOTONIC_SESSIONS"


# -------------------------------------------------- forward representation


def test_forward_representation_matches_the_documented_formulas() -> None:
    candles = _candles()
    suffix = candles[CONTEXT_PREFIX_LENGTH:]
    anchor = candles[CONTEXT_PREFIX_LENGTH - 1].close
    rows = forward_representation(suffix, anchor_close=anchor)

    first = suffix[0]
    assert rows[0, 0] == pytest.approx(math.log(first.open) - math.log(anchor), rel=1e-6)
    assert rows[0, 1] == pytest.approx(math.log(first.close) - math.log(first.open), rel=1e-6)
    assert rows[0, 4] == pytest.approx(math.log1p(first.volume), rel=1e-6)
    assert (rows[:, 2] >= 0.0).all()
    assert (rows[:, 3] >= 0.0).all()


def test_forward_representation_rejects_a_wrong_length() -> None:
    with pytest.raises(BridgeTransformError) as excinfo:
        forward_representation(_candles()[:10], anchor_close=100.0)
    assert excinfo.value.failures[0].code == "INVALID_SUFFIX_LENGTH"


# -------------------------------------------------------------- Stage A run


def _series(candles: tuple[Candle, ...]) -> MarketSeries:
    return MarketSeries(
        symbol="SPY",
        interval="1d",
        provider="deterministic_fake",
        provider_mode=ProviderMode.FAKE,
        client_version="fake-1",
        retrieval_timestamp=None,
        candles=candles,
    )


def _assets():
    return DeterministicFakeKronosBackend().resolve_assets()


def test_stage_a_produces_validated_cached_artifacts(tmp_path: Path) -> None:
    candles = _candles()
    cache = FeatureCache(tmp_path / "cache")
    report = run_stage_a(
        run_id="syn_stage_a",
        experiment_sha256="d" * 64,
        source_commit="a" * 40,
        evidence_class="synthetic_pipeline_validation",
        series=_series(candles),
        specs=(_spec(candles),),
        extractor=FeatureExtractor(DeterministicFakeKronosBackend()),
        assets=_assets(),
        cache=cache,
        completed_at=datetime(2026, 8, 2, tzinfo=UTC),
        amendment_sha256=("1" * 64, "2" * 64, "3" * 64),
        verified_source_file_sha256={"model/kronos.py": "e" * 64},
    )

    assert report.passed
    assert report.sequence_count == 1
    assert report.bridge_input_dimension == 269
    assert report.prefix_length == 448
    assert report.suffix_length == 64
    record = report.sequences[0]
    assert record.bridge_input_shape == (512, 269)
    assert record.bridge_input_dtype == "float32"
    assert record.deterministic_replay_matched


def test_stage_a_report_carries_no_raw_candles(tmp_path: Path) -> None:
    candles = _candles()
    report = run_stage_a(
        run_id="syn_stage_a",
        experiment_sha256="d" * 64,
        source_commit=None,
        evidence_class="synthetic_pipeline_validation",
        series=_series(candles),
        specs=(_spec(candles),),
        extractor=FeatureExtractor(DeterministicFakeKronosBackend()),
        assets=_assets(),
        cache=FeatureCache(tmp_path / "cache"),
        completed_at=datetime(2026, 8, 2, tzinfo=UTC),
        amendment_sha256=("1" * 64, "2" * 64, "3" * 64),
        verified_source_file_sha256={"model/kronos.py": "e" * 64},
    )
    payload = report.model_dump_json()
    for candle in candles[:5]:
        assert repr(candle.close) not in payload


def test_stage_a_requires_at_least_one_sequence(tmp_path: Path) -> None:
    with pytest.raises(BridgeTransformError) as excinfo:
        run_stage_a(
            run_id="syn",
            experiment_sha256="d" * 64,
            source_commit=None,
            evidence_class="synthetic_pipeline_validation",
            series=_series(_candles()),
            specs=(),
            extractor=FeatureExtractor(DeterministicFakeKronosBackend()),
            assets=_assets(),
            cache=FeatureCache(tmp_path / "cache"),
            completed_at=datetime(2026, 8, 2, tzinfo=UTC),
            amendment_sha256=("1" * 64, "2" * 64, "3" * 64),
            verified_source_file_sha256={"model/kronos.py": "e" * 64},
        )
    assert excinfo.value.failures[0].code == "STAGE_A_NO_SEQUENCES"


def test_test_partition_shards_stay_blocked_before_the_gate(tmp_path: Path) -> None:
    candles = _candles()
    cache = FeatureCache(tmp_path / "cache")
    spec = _spec(candles, partition=Partition.RECONSTRUCTION_TEST)
    extracted = FeatureExtractor(DeterministicFakeKronosBackend()).extract(
        spec=spec, candles=candles, source_data_sha256="0" * 64
    )
    example = extracted.to_cached_example(_identity(Partition.RECONSTRUCTION_TEST))

    # Materializing test features is itself an access, so the write is blocked.
    with pytest.raises(BridgeTransformError) as excinfo:
        cache.write(example)
    assert excinfo.value.failures[0].code == "TEST_SHARD_WRITE_BEFORE_GATE"

    # After the gate opens, write and read both succeed.
    cache.open_test_gate()
    ref = cache.write(example)
    assert cache.read(ref).partition is Partition.RECONSTRUCTION_TEST


def test_test_partition_reads_are_blocked_before_the_gate(tmp_path: Path) -> None:
    """A shard written under an open gate is unreadable by a fresh, closed cache."""
    candles = _candles()
    root = tmp_path / "cache"
    spec = _spec(candles, partition=Partition.RECONSTRUCTION_TEST)
    extracted = FeatureExtractor(DeterministicFakeKronosBackend()).extract(
        spec=spec, candles=candles, source_data_sha256="0" * 64
    )
    opened = FeatureCache(root, test_gate_open=True)
    ref = opened.write(extracted.to_cached_example(_identity(Partition.RECONSTRUCTION_TEST)))

    closed = FeatureCache(root)
    with pytest.raises(BridgeTransformError) as excinfo:
        closed.read(ref)
    assert excinfo.value.failures[0].code == "TEST_SHARD_LOAD_BEFORE_GATE"


def test_cache_corruption_is_detected(tmp_path: Path) -> None:
    candles = _candles()
    cache = FeatureCache(tmp_path / "cache")
    ref = cache.write(_extract(candles).to_cached_example(_identity()))
    path = cache.root / ref.relative_path
    payload = bytearray(path.read_bytes())
    payload[-2] ^= 0xFF
    path.write_bytes(bytes(payload))
    with pytest.raises(BridgeTransformError) as excinfo:
        cache.read(ref)
    assert excinfo.value.failures[0].code == "SHARD_HASH_MISMATCH"


def test_cache_round_trip_preserves_every_tensor(tmp_path: Path) -> None:
    cache = FeatureCache(tmp_path / "cache")
    extracted = _extract(_candles())
    restored = cache.read(cache.write(extracted.to_cached_example(_identity())))
    assert np.array_equal(restored.frozen_hidden, extracted.frozen_hidden)
    assert np.array_equal(restored.causal_features, extracted.causal_features)
    assert np.array_equal(restored.constrained_targets, extracted.constrained_targets)
    assert restored.initial_previous_close == pytest.approx(extracted.initial_previous_close)


# ------------------------------------------------- Stage A fails closed


def test_stage_a_blocks_without_extraction_evidence(tmp_path: Path) -> None:
    """A run must not reach training with no features ever produced."""
    from openalpha_bridge.phase2.kronos import KronosMode
    from openalpha_bridge.phase2.pipeline import Phase2Config, Phase2Pipeline
    from openalpha_bridge.phase2.provider import DeterministicFakeProvider
    from openalpha_bridge.phase2.states import EvidenceClass, Phase2State
    from openalpha_bridge.phase2.training import NumpyTrainingBackend

    root = Path(__file__).resolve().parents[3]
    pipeline = Phase2Pipeline(
        config=Phase2Config(
            run_directory=tmp_path / "run",
            cache_directory=tmp_path / "cache",
            research_root=root / "research" / "bridge-v0",
            repository_root=root,
            evidence_class=EvidenceClass.SYNTHETIC_PIPELINE_VALIDATION,
            provider_mode=ProviderMode.FAKE,
            kronos_mode=KronosMode.FAKE,
            dry_run=True,
            require_accelerator=False,
            training_symbols=("SPY",),
            unseen_symbols=("IWM",),
        ),
        provider=DeterministicFakeProvider(),
        kronos=DeterministicFakeKronosBackend(),
        training_backend=NumpyTrainingBackend(),
    )
    pipeline.preflight()
    pipeline.retrieve()
    pipeline.validate_data()
    pipeline.build_windows()
    pipeline.coverage_audit()
    pipeline.resolve_assets()

    result = pipeline.stage_a()  # no evidence supplied
    assert result.state is Phase2State.BLOCKED
    assert result.detail["blocker"] == "STAGE_A_EXTRACTION_NOT_PERFORMED"
    assert pipeline.journal.current_state is Phase2State.BLOCKED

    # And Stage B remains unreachable.
    with pytest.raises(BridgeTransformError) as excinfo:
        pipeline.stage_b()
    assert excinfo.value.failures[0].code in {"STAGE_SKIPPED", "TERMINAL_STATE_TRANSITION"}


# ------------------------------------------- cache identity binding (v2)


def _example(partition: Partition = Partition.TRAIN, **identity_overrides):
    extracted = _extract(_candles())
    return extracted.to_cached_example(_identity(partition, **identity_overrides))


def test_cache_schema_is_v2_and_binds_full_identity(tmp_path: Path) -> None:
    from openalpha_bridge.phase2.cache import CACHE_SCHEMA_VERSION

    assert CACHE_SCHEMA_VERSION == "openalpha.bridge.phase2.cache.v2"
    cache = FeatureCache(tmp_path / "cache")
    restored = cache.read(cache.write(_example()))
    identity = restored.identity
    assert identity.experiment_sha256 == "d" * 64
    assert identity.amendment_sha256 == ("1" * 64, "2" * 64, "3" * 64)
    assert identity.score_mask_sha256 == score_mask_sha256()
    assert identity.official_source_revision == "6" * 40
    assert identity.provider_client_version == "fake-1"
    assert identity.tensor_specification["bridge_input"] == "float32[512,269]"
    assert len(identity.identity_sha256) == 64


def test_identical_rewrite_is_accepted(tmp_path: Path) -> None:
    cache = FeatureCache(tmp_path / "cache")
    example = _example()
    first = cache.write(example)
    second = cache.write(example)
    assert first.content_sha256 == second.content_sha256


@pytest.mark.parametrize(
    "override",
    [
        {"amendment_sha256": ("9" * 64,)},
        {"official_source_revision": "0" * 40},
        {"provider": "some_other_provider"},
        {"experiment_sha256": "e" * 64},
        {"run_id": "a_different_run"},
        {"source_commit": "b" * 40},
    ],
    ids=["amendment", "source_revision", "provider", "experiment", "run", "commit"],
)
def test_differing_identity_at_the_same_path_is_a_collision(tmp_path: Path, override: dict) -> None:
    """A stale shard must never be silently reused or overwritten."""
    cache = FeatureCache(tmp_path / "cache")
    cache.write(_example())
    with pytest.raises(BridgeTransformError) as excinfo:
        cache.write(_example(**override))
    assert excinfo.value.failures[0].code == "SHARD_IDENTITY_COLLISION"


@pytest.mark.parametrize(
    "override",
    [{"symbol": "QQQ"}, {"interval": "1h"}, {"candle_data_sha256": "9" * 64}],
    ids=["symbol", "interval", "candle_digest"],
)
def test_identity_must_describe_the_example(tmp_path: Path, override: dict) -> None:
    cache = FeatureCache(tmp_path / "cache")
    with pytest.raises(BridgeTransformError) as excinfo:
        cache.write(_example(**override))
    assert excinfo.value.failures[0].code == "SHARD_IDENTITY_MISMATCH"


def test_identity_partition_must_match_the_example(tmp_path: Path) -> None:
    """A train example carrying a test identity is rejected before any write."""
    cache = FeatureCache(tmp_path / "cache")
    extracted = _extract(_candles())
    example = extracted.to_cached_example(_identity(Partition.VALIDATION))
    with pytest.raises(BridgeTransformError) as excinfo:
        cache.write(example)
    assert excinfo.value.failures[0].code == "SHARD_IDENTITY_MISMATCH"


# ------------------------------------------ Stage A report verification


def _real_report(tmp_path: Path):
    """A genuine Stage A report plus the cache it wrote, for verification."""
    candles = _candles()
    cache = FeatureCache(tmp_path / "cache")
    report = run_stage_a(
        run_id="syn_stage_a",
        experiment_sha256="d" * 64,
        source_commit="a" * 40,
        evidence_class="development_compatibility_canary",
        series=_series(candles),
        specs=(_spec(candles),),
        extractor=FeatureExtractor(DeterministicFakeKronosBackend()),
        assets=_assets(),
        cache=cache,
        completed_at=datetime(2026, 8, 2, tzinfo=UTC),
        amendment_sha256=("1" * 64, "2" * 64, "3" * 64),
        verified_source_file_sha256=dict(SOURCE_SPEC.files),
    )
    return report, cache


def _plan(report, **overrides):
    """The audit expectation.

    In production every field comes from locked configuration and the live run;
    here the genuine report's own values stand in for that run so the tests can
    perturb one field at a time. The verifier never reads the report for these.
    """
    fields = {
        "run_id": report.run_id,
        "experiment_sha256": report.experiment_sha256,
        "amendment_sha256": ("1" * 64, "2" * 64, "3" * 64),
        "evidence_class": report.evidence_class,
        "source_commit": report.source_commit,
        "provider_name": report.provider,
        "provider_client_version": report.provider_client_version,
        "config": _plan_config(),
        "assets": _assets(),
        "expected_sequence_ids": frozenset({report.sequences[0].sequence_id}),
    }
    fields.update(overrides)
    return build_stage_a_plan(**fields)


def _plan_config():
    from openalpha_bridge.phase2.kronos import KronosMode
    from openalpha_bridge.phase2.pipeline import Phase2Config
    from openalpha_bridge.phase2.provider import ProviderMode as _Mode
    from openalpha_bridge.phase2.states import EvidenceClass

    root = Path(".")
    return Phase2Config(
        run_directory=root,
        cache_directory=root,
        research_root=root,
        repository_root=root,
        evidence_class=EvidenceClass.SYNTHETIC_PIPELINE_VALIDATION,
        provider_mode=_Mode.FAKE,
        kronos_mode=KronosMode.FAKE,
    )


def _verify_kwargs(report, cache, **overrides):
    return {"plan": _plan(report, **overrides), "cache": cache}


def test_a_genuine_report_verifies(tmp_path: Path) -> None:
    report, cache = _real_report(tmp_path)
    assert verify_stage_a_report(report, **_verify_kwargs(report, cache)) is report


def test_a_fabricated_object_does_not_pass(tmp_path: Path) -> None:
    """passed=True, one sequence, and dimension 269 must not be enough."""

    class Fake:
        passed = True
        sequences = ("anything",)
        bridge_input_dimension = 269

    report, cache = _real_report(tmp_path)
    with pytest.raises(BridgeTransformError) as excinfo:
        verify_stage_a_report(Fake(), **_verify_kwargs(report, cache))
    assert excinfo.value.failures[0].code == "STAGE_A_REPORT_WRONG_TYPE"


@pytest.mark.parametrize(
    "override",
    [
        {"run_id": "a_different_run"},
        {"experiment_sha256": "e" * 64},
        {"amendment_sha256": ("9" * 64,)},
        {"evidence_class": "real_phase2"},
        {"source_commit": "b" * 40},
        {"provider_name": "another_provider"},
    ],
    ids=["run", "experiment", "amendment", "evidence", "commit", "provider"],
)
def test_report_must_describe_the_active_run(tmp_path: Path, override: dict) -> None:
    report, cache = _real_report(tmp_path)
    with pytest.raises(BridgeTransformError) as excinfo:
        verify_stage_a_report(report, **_verify_kwargs(report, cache, **override))
    assert excinfo.value.failures[0].code == "STAGE_A_REPORT_IDENTITY_MISMATCH"


def test_unexpected_sequence_set_is_rejected(tmp_path: Path) -> None:
    report, cache = _real_report(tmp_path)
    with pytest.raises(BridgeTransformError) as excinfo:
        verify_stage_a_report(
            report, **_verify_kwargs(report, cache, expected_sequence_ids=frozenset({"0" * 16}))
        )
    assert excinfo.value.failures[0].code == "STAGE_A_UNEXPECTED_SEQUENCES"


def test_a_missing_shard_is_detected(tmp_path: Path) -> None:
    report, cache = _real_report(tmp_path)
    (cache.root / report.sequences[0].shard_relative_path).unlink()
    with pytest.raises(BridgeTransformError) as excinfo:
        verify_stage_a_report(report, **_verify_kwargs(report, cache))
    assert excinfo.value.failures[0].code == "MISSING_SHARD"


def test_a_tampered_shard_is_detected(tmp_path: Path) -> None:
    report, cache = _real_report(tmp_path)
    path = cache.root / report.sequences[0].shard_relative_path
    payload = bytearray(path.read_bytes())
    payload[-3] ^= 0xFF
    path.write_bytes(bytes(payload))
    with pytest.raises(BridgeTransformError) as excinfo:
        verify_stage_a_report(report, **_verify_kwargs(report, cache))
    assert excinfo.value.failures[0].code == "SHARD_HASH_MISMATCH"


def test_a_non_training_sequence_is_rejected(tmp_path: Path) -> None:
    """Stage A may not touch validation, reconstruction-test, or external data."""
    report, cache = _real_report(tmp_path)
    tainted = report.model_copy(
        update={
            "sequences": (
                report.sequences[0].model_copy(update={"partition": "reconstruction_test"}),
            )
        }
    )
    with pytest.raises(BridgeTransformError) as excinfo:
        verify_stage_a_report(tainted, **_verify_kwargs(report, cache))
    assert excinfo.value.failures[0].code == "STAGE_A_NON_TRAINING_SEQUENCE"


def test_pipeline_rejects_a_non_report_object(tmp_path: Path) -> None:
    from openalpha_bridge.phase2.kronos import KronosMode
    from openalpha_bridge.phase2.pipeline import Phase2Config, Phase2Pipeline
    from openalpha_bridge.phase2.provider import DeterministicFakeProvider
    from openalpha_bridge.phase2.states import EvidenceClass, Phase2State
    from openalpha_bridge.phase2.training import NumpyTrainingBackend

    class Fake:
        passed = True
        sequences = ("x",)
        bridge_input_dimension = 269

    root = Path(__file__).resolve().parents[3]
    pipeline = Phase2Pipeline(
        config=Phase2Config(
            run_directory=tmp_path / "run",
            cache_directory=tmp_path / "cache2",
            research_root=root / "research" / "bridge-v0",
            repository_root=root,
            evidence_class=EvidenceClass.SYNTHETIC_PIPELINE_VALIDATION,
            provider_mode=ProviderMode.FAKE,
            kronos_mode=KronosMode.FAKE,
            dry_run=True,
            require_accelerator=False,
            training_symbols=("SPY",),
            unseen_symbols=("IWM",),
        ),
        provider=DeterministicFakeProvider(),
        kronos=DeterministicFakeKronosBackend(),
        training_backend=NumpyTrainingBackend(),
    )
    for step in (
        pipeline.preflight,
        pipeline.retrieve,
        pipeline.validate_data,
        pipeline.build_windows,
        pipeline.coverage_audit,
        pipeline.resolve_assets,
    ):
        step()
    result = pipeline.stage_a(Fake())  # type: ignore[arg-type]
    assert result.state is Phase2State.BLOCKED
    assert result.detail["blocker"] == "STAGE_A_REPORT_WRONG_TYPE"


# ------------------- one altered field must block Stage A (item 2)


@pytest.mark.parametrize(
    "field",
    [
        "provider",
        "official_source_file_sha256",
        "score_mask_sha256",
        "target_start",
        "tokenizer_weights_sha256",
        "representation_version",
        "tensor_specification",
    ],
)
def test_one_altered_shard_identity_field_blocks_stage_a(tmp_path: Path, field: str) -> None:
    """A single drifted shard-bound field must stop Stage A advancing."""
    from openalpha_bridge.phase2.cache import CacheIdentity

    candles = _candles()
    cache = FeatureCache(tmp_path / "cache")
    spec = _spec(candles)
    extracted = FeatureExtractor(DeterministicFakeKronosBackend()).extract(
        spec=spec, candles=candles, source_data_sha256="0" * 64
    )

    assets = _assets()
    base = {
        "run_id": "syn_stage_a",
        "experiment_sha256": "d" * 64,
        "amendment_sha256": ("1" * 64, "2" * 64, "3" * 64),
        "score_mask_sha256": score_mask_sha256(),
        "source_commit": "a" * 40,
        "evidence_class": "development_compatibility_canary",
        "provider": "deterministic_fake",
        "provider_client_version": "fake-1",
        "symbol": spec.symbol,
        "interval": spec.interval,
        "partition": Partition.TRAIN,
        "prefix_start": spec.prefix_start.isoformat(),
        "prefix_end": spec.prefix_end.isoformat(),
        "target_start": spec.target_start.isoformat(),
        "target_end": spec.target_end.isoformat(),
        "candle_data_sha256": "0" * 64,
        "official_source_repository": SOURCE_SPEC.repository,
        "official_source_revision": SOURCE_SPEC.revision,
        "official_source_file_sha256": dict(SOURCE_SPEC.files),
        "tokenizer_repository": assets.repository,
        "tokenizer_revision": assets.revision,
        "tokenizer_config_sha256": assets.observed_config_sha256,
        "tokenizer_weights_sha256": assets.observed_weights_sha256,
        "frozen_parameter_sha256": assets.frozen_parameter_sha256,
        "feature_schema_version": "openalpha.bridge.phase2.cache.v2",
        "representation_version": "openalpha.bridge.financial.v1",
        "tensor_specification": {"bridge_input": "float32[512,269]"},
    }
    drift = {
        "provider": "another_provider",
        "official_source_file_sha256": {"model/kronos.py": "9" * 64},
        "score_mask_sha256": "9" * 64,
        "target_start": "1999-01-01",
        "tokenizer_weights_sha256": "9" * 64,
        "representation_version": "openalpha.bridge.financial.v999",
        "tensor_specification": {"bridge_input": "float32[512,268]"},
    }
    # The drifted value must survive the write-time consistency check, which
    # only compares the fields that describe the example itself.
    tainted = dict(base)
    tainted[field] = drift[field]
    ref = cache.write(extracted.to_cached_example(CacheIdentity(**tainted)))

    from openalpha_bridge.phase2.stage_a import StageASequenceRecord

    record = StageASequenceRecord(
        sequence_id=extracted.sequence_id,
        symbol=spec.symbol,
        interval=spec.interval,
        partition="train",
        target_start=spec.target_start.isoformat(),
        target_end=spec.target_end.isoformat(),
        prefix_start=spec.prefix_start.isoformat(),
        prefix_end=spec.prefix_end.isoformat(),
        bridge_input_shape=(512, 269),
        bridge_input_dtype="float32",
        canonical_sha256=extracted.canonical_sha256(),
        shard_relative_path=ref.relative_path,
        shard_content_sha256=ref.content_sha256,
        shard_size_bytes=ref.size_bytes,
        deterministic_replay_matched=True,
    )
    from openalpha_bridge.phase2.stage_a import StageAReport

    report = StageAReport(
        run_id="syn_stage_a",
        experiment_sha256="d" * 64,
        amendment_sha256=("1" * 64, "2" * 64, "3" * 64),
        source_commit="a" * 40,
        evidence_class="development_compatibility_canary",
        provider="deterministic_fake",
        provider_mode="fake",
        provider_client_version="fake-1",
        kronos_mode="fake",
        kronos_repository=assets.repository,
        kronos_revision=assets.revision,
        kronos_config_sha256=assets.observed_config_sha256,
        kronos_weights_sha256=assets.observed_weights_sha256,
        frozen_parameter_sha256=assets.frozen_parameter_sha256,
        official_source_repository=SOURCE_SPEC.repository,
        official_source_revision=SOURCE_SPEC.revision,
        official_source_file_sha256=dict(SOURCE_SPEC.files),
        cache_schema_version="openalpha.bridge.phase2.cache.v2",
        representation_version="openalpha.bridge.financial.v1",
        prefix_length=448,
        suffix_length=64,
        score_mask_sha256=score_mask_sha256(),
        bridge_input_dimension=269,
        retrieved_candles=512,
        sequences=(record,),
        completed_at=datetime(2026, 8, 2, tzinfo=UTC),
        passed=True,
    )

    with pytest.raises(BridgeTransformError) as excinfo:
        verify_stage_a_report(
            report,
            plan=build_stage_a_plan(
                run_id="syn_stage_a",
                experiment_sha256="d" * 64,
                amendment_sha256=("1" * 64, "2" * 64, "3" * 64),
                evidence_class="development_compatibility_canary",
                source_commit="a" * 40,
                provider_name="deterministic_fake",
                provider_client_version="fake-1",
                config=_plan_config(),
                assets=assets,
                expected_sequence_ids=frozenset({extracted.sequence_id}),
            ),
            cache=cache,
            expected_tensor_specification={"bridge_input": "float32[512,269]"},
        )
    assert excinfo.value.failures[0].code in {
        "STAGE_A_SHARD_IDENTITY_MISMATCH",
        "SHARD_IDENTITY_MISMATCH",
    }
