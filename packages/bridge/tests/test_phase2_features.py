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
    TOKENIZER_INPUT_DIMENSION,
    DeterministicFakeKronosBackend,
)
from openalpha_bridge.phase2.provider import Candle, MarketSeries, ProviderMode
from openalpha_bridge.phase2.stage_a import run_stage_a
from openalpha_bridge.windowing import (
    CONTEXT_PREFIX_LENGTH,
    EXAMPLE_LENGTH,
    SCORED_SUFFIX_LENGTH,
    Partition,
    SequenceSpec,
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
    assert values[
        SCALE_FEATURE_ORDER.index("open_mean_relative_to_anchor")
    ] == pytest.approx(mean[0] / anchor, rel=1e-6)
    assert values[
        SCALE_FEATURE_ORDER.index("log1p_close_std_over_anchor")
    ] == pytest.approx(math.log1p(std[3] / anchor), rel=1e-6)
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
    assert np.array_equal(
        baseline.frozen_hidden[:-1], perturbed.frozen_hidden[:-1]
    )
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
    example = extracted.to_cached_example()
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
    shifted = base.model_copy(
        update={"prefix_start": base.prefix_start + timedelta(days=1)}
    )
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
        )
    assert excinfo.value.failures[0].code == "STAGE_A_NO_SEQUENCES"


def test_test_partition_shards_stay_blocked_before_the_gate(tmp_path: Path) -> None:
    candles = _candles()
    cache = FeatureCache(tmp_path / "cache")
    spec = _spec(candles, partition=Partition.RECONSTRUCTION_TEST)
    extracted = FeatureExtractor(DeterministicFakeKronosBackend()).extract(
        spec=spec, candles=candles, source_data_sha256="0" * 64
    )
    ref = cache.write(extracted.to_cached_example())

    with pytest.raises(BridgeTransformError) as excinfo:
        cache.read(ref)
    assert excinfo.value.failures[0].code == "TEST_SHARD_LOAD_BEFORE_GATE"

    cache.open_test_gate()
    assert cache.read(ref).partition is Partition.RECONSTRUCTION_TEST


def test_cache_corruption_is_detected(tmp_path: Path) -> None:
    candles = _candles()
    cache = FeatureCache(tmp_path / "cache")
    ref = cache.write(_extract(candles).to_cached_example())
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
    restored = cache.read(cache.write(extracted.to_cached_example()))
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
