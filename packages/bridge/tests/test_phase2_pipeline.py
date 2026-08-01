"""SYNTHETIC PIPELINE VALIDATION - NOT EMPIRICAL EVIDENCE.

Exercises the real state machine, windowing, cache, metric, bootstrap, and gate
code against fake provider and fake Kronos components. No artifact produced here
belongs to the real Phase 2 evidence namespace.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from openalpha_bridge.errors import BridgeTransformError
from openalpha_bridge.phase2.bootstrap import (
    ConfirmationDecision,
    PairedObservation,
    ResamplingUnit,
    pair_by_unit,
    paired_bootstrap,
)
from openalpha_bridge.phase2.cache import CachedExample, FeatureCache
from openalpha_bridge.phase2.gates import (
    GateOutcome,
    TerminalConclusion,
    evaluate_conclusion,
    locked_gate_specifications,
)
from openalpha_bridge.phase2.identity import AMENDMENT_2_SHA256, RunIdentity, verify_locked_hashes
from openalpha_bridge.phase2.kronos import (
    BRIDGE_INPUT_DIMENSION,
    DeterministicFakeKronosBackend,
    KronosMode,
    OfficialKronosBackend,
    assert_backend_matches_mode,
    assert_bridge_input,
)
from openalpha_bridge.phase2.metrics import (
    ReconstructionMethod,
    compute_reconstruction_metrics,
    scored_view,
)
from openalpha_bridge.phase2.optional import BridgeExtra, require_module
from openalpha_bridge.phase2.pipeline import Phase2Config, Phase2Pipeline
from openalpha_bridge.phase2.preflight import run_preflight
from openalpha_bridge.phase2.provider import (
    DeterministicFakeProvider,
    ProviderMode,
    RetrievalRequest,
    validate_series,
)
from openalpha_bridge.phase2.states import (
    EvidenceClass,
    Phase2State,
    StateJournal,
    assert_transition_allowed,
)
from openalpha_bridge.phase2.testgate import (
    TestOpeningPreconditions,
    is_test_partition_opened,
    open_test_partition,
)
from openalpha_bridge.phase2.training import (
    LOCKED_PARAMETER_COUNT,
    NumpyTrainingBackend,
    TrainingConfig,
    assert_locked_architecture,
    expected_parameter_count,
)
from openalpha_bridge.windowing import (
    CONTEXT_PREFIX_LENGTH,
    EXAMPLE_LENGTH,
    SCORED_SUFFIX_LENGTH,
    Partition,
    score_mask,
    score_mask_sha256,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
RESEARCH_ROOT = REPOSITORY_ROOT / "research" / "bridge-v0"


class _FixedClock:
    def __init__(self) -> None:
        self._now = datetime(2026, 7, 31, 12, 0, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        self._now += timedelta(seconds=1)
        return self._now


def _pipeline(tmp_path: Path, **overrides: object) -> Phase2Pipeline:
    config = Phase2Config(
        run_directory=tmp_path / "run",
        cache_directory=tmp_path / "cache",
        research_root=RESEARCH_ROOT,
        repository_root=REPOSITORY_ROOT,
        evidence_class=EvidenceClass.SYNTHETIC_PIPELINE_VALIDATION,
        provider_mode=ProviderMode.FAKE,
        kronos_mode=KronosMode.FAKE,
        dry_run=True,
        require_accelerator=False,
        training_symbols=("SPY", "QQQ"),
        unseen_symbols=("IWM",),
        **overrides,  # type: ignore[arg-type]
    )
    return Phase2Pipeline(
        config=config,
        provider=DeterministicFakeProvider(),
        kronos=DeterministicFakeKronosBackend(),
        training_backend=NumpyTrainingBackend(),
        clock=_FixedClock(),
    )


# ------------------------------------------------- optional dependency boundary


def test_importing_bridge_does_not_import_torch_or_kronos() -> None:
    code = (
        "import sys; import openalpha_bridge; import openalpha_bridge.phase2; "
        "print('torch' in sys.modules, 'huggingface_hub' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False False"


def test_missing_optional_dependency_is_typed_and_actionable() -> None:
    with pytest.raises(BridgeTransformError) as excinfo:
        require_module("definitely_not_installed_xyz", stage="stage-c", extra=BridgeExtra.BRIDGE_TRAINING)
    failure = excinfo.value.failures[0]
    assert failure.code == "MISSING_OPTIONAL_DEPENDENCY"
    assert failure.field == "definitely_not_installed_xyz"
    assert "bridge-training" in failure.message
    assert "uv sync --extra" in failure.message
    assert "stage=stage-c" in failure.message


def test_base_suite_does_not_require_torch() -> None:
    assert "torch" not in sys.modules


# ------------------------------------------------------------- state machine


def test_stages_cannot_be_skipped() -> None:
    with pytest.raises(BridgeTransformError) as excinfo:
        assert_transition_allowed(Phase2State.PREFLIGHT_PASSED, Phase2State.STAGE_C_TRAINED)
    assert excinfo.value.failures[0].code == "ILLEGAL_STATE_TRANSITION"


def test_test_cannot_open_before_checkpoint_freeze() -> None:
    with pytest.raises(BridgeTransformError) as excinfo:
        assert_transition_allowed(Phase2State.STAGE_C_TRAINED, Phase2State.TEST_OPENED)
    assert excinfo.value.failures[0].code == "ILLEGAL_STATE_TRANSITION"


def test_terminal_states_cannot_transition() -> None:
    with pytest.raises(BridgeTransformError) as excinfo:
        assert_transition_allowed(Phase2State.FINALIZED, Phase2State.TEST_EVALUATED)
    assert excinfo.value.failures[0].code == "TERMINAL_STATE_TRANSITION"


def test_journal_requires_monotonic_time() -> None:
    start = datetime(2026, 7, 31, tzinfo=UTC)
    journal = StateJournal.start(
        run_id="syn_test", evidence_class=EvidenceClass.SYNTHETIC_PIPELINE_VALIDATION,
        occurred_at=start,
    )
    with pytest.raises(BridgeTransformError) as excinfo:
        journal.advance(Phase2State.PREFLIGHT_PASSED, occurred_at=start)
    assert excinfo.value.failures[0].code == "NON_MONOTONIC_TRANSITION_TIME"


# ------------------------------------------------------------- run identity


def test_changing_configuration_changes_run_identity() -> None:
    base = RunIdentity.derive(
        configuration={"device": "cuda"},
        evidence_class=EvidenceClass.REAL_PHASE2,
        provider_mode="real",
        kronos_mode="pinned_official",
    )
    changed = RunIdentity.derive(
        configuration={"device": "cpu"},
        evidence_class=EvidenceClass.REAL_PHASE2,
        provider_mode="real",
        kronos_mode="pinned_official",
    )
    assert base.run_id != changed.run_id
    assert base.amendment_2_sha256 == AMENDMENT_2_SHA256


def test_synthetic_and_real_identities_are_namespaced() -> None:
    synthetic = RunIdentity.derive(
        configuration={},
        evidence_class=EvidenceClass.SYNTHETIC_PIPELINE_VALIDATION,
        provider_mode="fake",
        kronos_mode="fake",
    )
    assert synthetic.run_id.startswith("syn_")
    assert not synthetic.is_real_evidence


def test_locked_hashes_verify_against_the_repository() -> None:
    observed = verify_locked_hashes(RESEARCH_ROOT)
    assert observed["phase2-amendment-2-context-prefix.yaml"] == AMENDMENT_2_SHA256


# ------------------------------------------------------------ mode enforcement


def test_fake_kronos_backend_rejected_in_real_run() -> None:
    with pytest.raises(BridgeTransformError) as excinfo:
        assert_backend_matches_mode(
            DeterministicFakeKronosBackend(), declared=KronosMode.PINNED_OFFICIAL
        )
    assert excinfo.value.failures[0].code == "KRONOS_MODE_MISMATCH"


def test_real_run_rejects_fake_provider(tmp_path: Path) -> None:
    config = Phase2Config(
        run_directory=tmp_path / "run",
        cache_directory=tmp_path / "cache",
        research_root=RESEARCH_ROOT,
        repository_root=REPOSITORY_ROOT,
        evidence_class=EvidenceClass.REAL_PHASE2,
        provider_mode=ProviderMode.FAKE,
        kronos_mode=KronosMode.FAKE,
    )
    with pytest.raises(BridgeTransformError) as excinfo:
        Phase2Pipeline(
            config=config,
            provider=DeterministicFakeProvider(),
            kronos=DeterministicFakeKronosBackend(),
            training_backend=NumpyTrainingBackend(),
        )
    assert excinfo.value.failures[0].code == "FAKE_PROVIDER_IN_REAL_RUN"


def test_official_backend_never_loads_assets_locally() -> None:
    backend = OfficialKronosBackend(stage="unit-test")
    assert backend.mode is KronosMode.PINNED_OFFICIAL
    with pytest.raises(BridgeTransformError) as excinfo:
        backend.frozen_parameter_sha256()
    assert excinfo.value.failures[0].code == "OFFICIAL_BACKEND_NOT_LOADED"


# ------------------------------------------------------------- fake provider


def test_fake_provider_is_deterministic_and_structurally_valid() -> None:
    request = RetrievalRequest(
        symbol="SPY", start=date(2022, 1, 1), end=date(2022, 4, 1), maximum_candles=500
    )
    first = DeterministicFakeProvider().fetch(request)
    second = DeterministicFakeProvider().fetch(request)
    assert first.normalized_sha256 == second.normalized_sha256
    assert first.provider_mode is ProviderMode.FAKE
    assert first.retrieval_timestamp is None
    validate_series(first)


def test_fake_provider_covers_every_declared_period() -> None:
    provider = DeterministicFakeProvider()
    for start, end in (
        (date(2010, 1, 1), date(2022, 1, 1)),
        (date(2022, 1, 1), date(2023, 1, 1)),
        (date(2023, 1, 1), date(2024, 6, 29)),
        (date(2024, 7, 1), date(2025, 7, 1)),
    ):
        series = provider.fetch(
            RetrievalRequest(symbol="IWM", start=start, end=end, maximum_candles=10_000)
        )
        validate_series(series)
        assert series.candles


def test_invalid_series_fails_closed() -> None:
    provider = DeterministicFakeProvider()
    series = provider.fetch(
        RetrievalRequest(
            symbol="SPY", start=date(2022, 1, 1), end=date(2022, 2, 1), maximum_candles=100
        )
    )
    broken = series.model_copy(
        update={"candles": tuple(reversed(series.candles))}
    )
    with pytest.raises(BridgeTransformError) as excinfo:
        validate_series(broken)
    assert excinfo.value.failures[0].code == "NON_MONOTONIC_SESSIONS"


# ---------------------------------------------------------------- feature cache


def _example(partition: Partition, sequence_id: str = "a" * 16) -> CachedExample:
    return CachedExample(
        sequence_id=sequence_id,
        symbol="SPY",
        interval="1d",
        partition=partition,
        coarse_ids=np.zeros(EXAMPLE_LENGTH, dtype=np.int64),
        fine_ids=np.zeros(EXAMPLE_LENGTH, dtype=np.int64),
        bipolar_latent=np.zeros((EXAMPLE_LENGTH, 20), dtype=np.float32),
        frozen_hidden=np.zeros((EXAMPLE_LENGTH, 256), dtype=np.float32),
        causal_features=np.zeros((EXAMPLE_LENGTH, 13), dtype=np.float32),
        constrained_targets=np.zeros((SCORED_SUFFIX_LENGTH, 5), dtype=np.float32),
        initial_previous_close=100.0,
        volume_mask=np.ones(EXAMPLE_LENGTH, dtype=np.bool_),
        sequence_mask=np.ones(EXAMPLE_LENGTH, dtype=np.bool_),
        score_mask=np.asarray(score_mask(), dtype=np.bool_),
        source_data_sha256="0" * 64,
    )


def test_cache_round_trip_and_partition_separation(tmp_path: Path) -> None:
    cache = FeatureCache(tmp_path / "cache")
    ref = cache.write(_example(Partition.TRAIN))
    restored = cache.read(ref)
    assert restored.sequence_id == "a" * 16
    assert restored.partition is Partition.TRAIN
    assert ref.relative_path.startswith("train/")


def test_test_shards_cannot_load_before_the_gate(tmp_path: Path) -> None:
    cache = FeatureCache(tmp_path / "cache")
    ref = cache.write(_example(Partition.RECONSTRUCTION_TEST, "b" * 16))
    with pytest.raises(BridgeTransformError) as excinfo:
        cache.read(ref)
    assert excinfo.value.failures[0].code == "TEST_SHARD_LOAD_BEFORE_GATE"
    cache.open_test_gate()
    assert cache.read(ref).partition is Partition.RECONSTRUCTION_TEST


def test_corrupt_shard_is_detected(tmp_path: Path) -> None:
    cache = FeatureCache(tmp_path / "cache")
    ref = cache.write(_example(Partition.TRAIN))
    path = cache.root / ref.relative_path
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 0xFF
    path.write_bytes(bytes(payload))
    with pytest.raises(BridgeTransformError) as excinfo:
        cache.read(ref)
    assert excinfo.value.failures[0].code == "SHARD_HASH_MISMATCH"


def test_interrupted_writes_are_recovered(tmp_path: Path) -> None:
    cache = FeatureCache(tmp_path / "cache")
    partial = cache.root / "train"
    partial.mkdir(parents=True, exist_ok=True)
    (partial / ".ghost.npz.deadbeef.tmp").write_bytes(b"partial")
    removed = cache.recover_interrupted_writes()
    assert removed == (".ghost.npz.deadbeef.tmp",)


def test_cache_rejects_a_cap_above_the_lock(tmp_path: Path) -> None:
    with pytest.raises(BridgeTransformError) as excinfo:
        FeatureCache(tmp_path / "cache", maximum_bytes=10_737_418_241)
    assert excinfo.value.failures[0].code == "CACHE_CAP_EXCEEDS_LOCK"


def test_cache_rejects_a_bad_score_mask(tmp_path: Path) -> None:
    cache = FeatureCache(tmp_path / "cache")
    broken = _example(Partition.TRAIN).model_copy(
        update={"score_mask": np.ones(EXAMPLE_LENGTH, dtype=np.bool_)}
    )
    with pytest.raises(BridgeTransformError) as excinfo:
        cache.write(broken)
    assert excinfo.value.failures[0].code == "INVALID_CACHED_SCORE_MASK"


# -------------------------------------------------------------------- metrics


def _candles(rows: int, *, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    close = 100.0 + rng.normal(0.0, 1.0, size=rows)
    open_ = close + rng.normal(0.0, 0.2, size=rows)
    high = np.maximum(open_, close) + np.abs(rng.normal(0.0, 0.2, size=rows))
    low = np.minimum(open_, close) - np.abs(rng.normal(0.0, 0.2, size=rows))
    return np.column_stack([open_, high, low, close])


def test_metrics_use_only_the_scored_suffix() -> None:
    actual = _candles(EXAMPLE_LENGTH, seed=1)
    predicted = _candles(EXAMPLE_LENGTH, seed=2)
    baseline = compute_reconstruction_metrics(
        method=ReconstructionMethod.OPENALPHA_BRIDGE_2K,
        predicted=predicted,
        actual=actual,
        score_mask_hash=score_mask_sha256(),
    )
    assert baseline.scored_candles == SCORED_SUFFIX_LENGTH
    assert baseline.prefix_length == CONTEXT_PREFIX_LENGTH


def test_prefix_values_cannot_affect_any_reported_metric() -> None:
    actual = _candles(EXAMPLE_LENGTH, seed=1)
    predicted = _candles(EXAMPLE_LENGTH, seed=2)
    before = compute_reconstruction_metrics(
        method=ReconstructionMethod.OPENALPHA_BRIDGE_2K,
        predicted=predicted,
        actual=actual,
        score_mask_hash=score_mask_sha256(),
    )

    # Corrupt every warm-up row beyond recognition.
    perturbed_actual = actual.copy()
    perturbed_predicted = predicted.copy()
    perturbed_actual[:CONTEXT_PREFIX_LENGTH] *= 1_000.0
    perturbed_predicted[:CONTEXT_PREFIX_LENGTH] -= 500.0

    after = compute_reconstruction_metrics(
        method=ReconstructionMethod.OPENALPHA_BRIDGE_2K,
        predicted=perturbed_predicted,
        actual=perturbed_actual,
        score_mask_hash=score_mask_sha256(),
    )
    assert before.model_dump() == after.model_dump()


def test_scored_view_rejects_a_tampered_mask() -> None:
    values = _candles(EXAMPLE_LENGTH)
    bad = np.ones(EXAMPLE_LENGTH, dtype=bool)
    with pytest.raises(BridgeTransformError) as excinfo:
        scored_view(values, mask=bad)
    assert excinfo.value.failures[0].code == "INVALID_METRIC_MASK"


def test_structural_invalidity_is_detected() -> None:
    actual = _candles(SCORED_SUFFIX_LENGTH, seed=1)
    predicted = actual.copy()
    predicted[0, 1] = predicted[0, 2] - 1.0  # high below low
    metrics = compute_reconstruction_metrics(
        method=ReconstructionMethod.OPENALPHA_BRIDGE_2K,
        predicted=predicted,
        actual=actual,
        score_mask_hash=score_mask_sha256(),
    )
    assert metrics.structurally_invalid_candle_fraction > 0.0


# ------------------------------------------------------------------ bootstrap


def _observations(values: list[tuple[float, float]]):
    return tuple(
        PairedObservation(unit_id=f"seq-{index}", value_a=a, value_b=b)
        for index, (a, b) in enumerate(values)
    )


def test_bootstrap_confirms_a_clear_improvement() -> None:
    result = paired_bootstrap(
        metric="high_low_range_mae",
        method_a=ReconstructionMethod.OPENALPHA_BRIDGE_2K,
        method_b=ReconstructionMethod.TERMINAL_PROJECTION,
        observations=_observations([(0.10, 0.30)] * 40),
        replicates=500,
    )
    assert result.confirmation is ConfirmationDecision.A_BETTER
    assert result.interval_upper < 0.0
    assert result.resampling_unit is ResamplingUnit.SCORED_SEQUENCE


def test_bootstrap_detects_a_clear_degradation() -> None:
    result = paired_bootstrap(
        metric="close_mae",
        method_a=ReconstructionMethod.OPENALPHA_BRIDGE_2K,
        method_b=ReconstructionMethod.OFFICIAL_DECODER,
        observations=_observations([(0.40, 0.10)] * 40),
        replicates=500,
    )
    assert result.confirmation is ConfirmationDecision.B_BETTER


def test_bootstrap_on_exact_ties_is_inconclusive() -> None:
    result = paired_bootstrap(
        metric="close_mae",
        method_a=ReconstructionMethod.OPENALPHA_BRIDGE_2K,
        method_b=ReconstructionMethod.OFFICIAL_DECODER,
        observations=_observations([(0.2, 0.2)] * 30),
        replicates=500,
    )
    assert result.confirmation is ConfirmationDecision.INCONCLUSIVE
    assert result.point_difference == pytest.approx(0.0)


def test_bootstrap_reports_insufficient_sample() -> None:
    result = paired_bootstrap(
        metric="close_mae",
        method_a=ReconstructionMethod.OPENALPHA_BRIDGE_2K,
        method_b=ReconstructionMethod.OFFICIAL_DECODER,
        observations=_observations([(0.2, 0.3)]),
    )
    assert result.confirmation is ConfirmationDecision.NOT_EVALUATED
    assert result.replicates == 0


def test_bootstrap_rejects_missing_pairs() -> None:
    with pytest.raises(BridgeTransformError) as excinfo:
        pair_by_unit(values_a={"s1": 0.1, "s2": 0.2}, values_b={"s1": 0.3})
    assert excinfo.value.failures[0].code == "MISSING_PAIRED_UNIT"


def test_bootstrap_rejects_duplicate_units() -> None:
    duplicated = (
        PairedObservation(unit_id="same", value_a=0.1, value_b=0.2),
        PairedObservation(unit_id="same", value_a=0.3, value_b=0.4),
    )
    with pytest.raises(BridgeTransformError) as excinfo:
        paired_bootstrap(
            metric="close_mae",
            method_a=ReconstructionMethod.OPENALPHA_BRIDGE_2K,
            method_b=ReconstructionMethod.OFFICIAL_DECODER,
            observations=duplicated,
        )
    assert excinfo.value.failures[0].code == "DUPLICATE_RESAMPLING_UNIT"


def test_bootstrap_is_deterministic_under_the_locked_seed() -> None:
    kwargs = {
        "metric": "close_mae",
        "method_a": ReconstructionMethod.OPENALPHA_BRIDGE_2K,
        "method_b": ReconstructionMethod.OFFICIAL_DECODER,
        "observations": _observations([(0.1, 0.2), (0.15, 0.18), (0.12, 0.25)] * 5),
        "replicates": 200,
    }
    assert paired_bootstrap(**kwargs).model_dump() == paired_bootstrap(**kwargs).model_dump()


# ---------------------------------------------------------------- gate table


def test_all_locked_gates_are_encoded() -> None:
    ids = {spec.gate_id for spec in locked_gate_specifications()}
    assert {
        "structurally_invalid_candle_fraction",
        "high_low_range_mae_projection_ratio",
        "high_low_range_paired_bootstrap_upper",
        "close_return_mae_official_multiplier",
        "checkpoint_size_bytes",
    } <= ids


def _passing_measurements() -> dict[str, float | str | None]:
    return {
        "structurally_invalid_candle_fraction": 0.0,
        "token_identifier_parity_fraction": 1.0,
        "frozen_weight_hash_parity_fraction": 1.0,
        "post_output_projection": "false",
        "high_low_range_mae_projection_ratio": 0.80,
        "high_low_range_paired_bootstrap_upper": -0.01,
        "full_ohlc_mae_official_multiplier": 1.00,
        "close_mae_official_multiplier": 1.00,
        "close_return_mae_official_multiplier": 1.00,
        "trainable_parameter_count": LOCKED_PARAMETER_COUNT,
        "checkpoint_size_bytes": 100_000,
        "median_latency_official_multiplier": 1.2,
        "p95_incremental_latency_ms": 10.0,
        "incremental_peak_memory_bytes": 1_000_000,
        "canonical_hash_match_fraction": 1.0,
    }


def test_all_gates_passing_yields_feasible() -> None:
    table = evaluate_conclusion(_passing_measurements())
    assert table.conclusion is TerminalConclusion.BRIDGE_2K_FEASIBLE
    assert table.scientific_result_available
    assert not table.failed


def test_a_structural_failure_blocks_regardless_of_quality() -> None:
    measurements = _passing_measurements()
    measurements["structurally_invalid_candle_fraction"] = 0.01
    table = evaluate_conclusion(measurements)
    assert table.conclusion is TerminalConclusion.OPERATIONALLY_BLOCKED
    assert not table.scientific_result_available


def test_all_quality_gates_failing_yields_tokens_insufficient() -> None:
    measurements = _passing_measurements()
    measurements.update(
        {
            "high_low_range_mae_projection_ratio": 1.4,
            "high_low_range_paired_bootstrap_upper": 0.2,
            "full_ohlc_mae_official_multiplier": 2.0,
            "close_mae_official_multiplier": 2.0,
            "close_return_mae_official_multiplier": 2.0,
        }
    )
    table = evaluate_conclusion(measurements)
    assert table.conclusion is TerminalConclusion.TOKENS_INSUFFICIENT


def test_a_single_quality_failure_is_partially_feasible() -> None:
    measurements = _passing_measurements()
    measurements["close_mae_official_multiplier"] = 2.0
    table = evaluate_conclusion(measurements)
    assert table.conclusion is TerminalConclusion.BRIDGE_2K_PARTIALLY_FEASIBLE


def test_no_measurements_cannot_claim_a_result() -> None:
    table = evaluate_conclusion({})
    assert table.conclusion is TerminalConclusion.OPERATIONALLY_BLOCKED
    assert all(r.outcome is GateOutcome.NOT_EVALUATED for r in table.results)


def test_blocked_reason_short_circuits_the_table() -> None:
    table = evaluate_conclusion(_passing_measurements(), blocked_reason="NO_ACCELERATOR")
    assert table.conclusion is TerminalConclusion.OPERATIONALLY_BLOCKED
    assert not table.scientific_result_available


# ----------------------------------------------------------------- architecture


def test_locked_parameter_count_is_seventeen_thousand_six_hundred_five() -> None:
    assert expected_parameter_count() == LOCKED_PARAMETER_COUNT == 17_605


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"hidden_units": 128}, "ARCHITECTURE_DIMENSION_MISMATCH"),
        ({"activation": "ReLU"}, "ACTIVATION_MISMATCH"),
        ({"trainable_parameters": 17_604}, "PARAMETER_COUNT_MISMATCH"),
        ({"frozen_kronos_parameters_require_grad": True}, "KRONOS_PARAMETERS_NOT_FROZEN"),
        ({"official_head_in_optimizer": True}, "OFFICIAL_HEAD_IN_OPTIMIZER"),
    ],
)
def test_architecture_violations_fail_explicitly(kwargs: dict, code: str) -> None:
    baseline = {
        "input_dimension": BRIDGE_INPUT_DIMENSION,
        "hidden_units": 64,
        "output_units": 5,
        "activation": "SiLU",
        "trainable_parameters": LOCKED_PARAMETER_COUNT,
        "frozen_kronos_parameters_require_grad": False,
        "official_head_in_optimizer": False,
    }
    with pytest.raises(BridgeTransformError) as excinfo:
        assert_locked_architecture(**{**baseline, **kwargs})
    assert excinfo.value.failures[0].code == code


def test_training_config_rejects_an_unlocked_learning_rate() -> None:
    with pytest.raises(BridgeTransformError) as excinfo:
        TrainingConfig(learning_rate=0.01).assert_locked()
    assert excinfo.value.failures[0].code == "LEARNING_RATE_POLICY_MISMATCH"


def test_bridge_input_shape_is_asserted() -> None:
    with pytest.raises(BridgeTransformError) as excinfo:
        assert_bridge_input(np.zeros((512, 268), dtype=np.float32), sequence_length=512)
    assert excinfo.value.failures[0].code == "INVALID_BRIDGE_INPUT_SHAPE"


def test_fake_kronos_builds_a_valid_bridge_tensor() -> None:
    backend = DeterministicFakeKronosBackend()
    coarse, fine = backend.encode_tokens(np.zeros((EXAMPLE_LENGTH, 6), dtype=np.float32))
    latent = backend.bipolar_latent(coarse, fine)
    hidden = backend.decoder_trunk(backend.project_latent(latent))
    tensor = backend.build_bridge_input(
        hidden=hidden,
        scale_features=np.zeros((EXAMPLE_LENGTH, 13), dtype=np.float32),
    )
    assert tensor.shape == (EXAMPLE_LENGTH, BRIDGE_INPUT_DIMENSION)


def test_fake_decoder_trunk_is_causal() -> None:
    backend = DeterministicFakeKronosBackend()
    projected = np.arange(20 * 256, dtype=np.float32).reshape(20, 256)
    baseline = backend.decoder_trunk(projected)
    perturbed = projected.copy()
    perturbed[10:] += 1_000.0
    assert np.allclose(baseline[:10], backend.decoder_trunk(perturbed)[:10])


# ------------------------------------------------------------------ preflight


def test_gpu_preflight_fails_safely_without_an_accelerator(tmp_path: Path) -> None:
    report = run_preflight(
        research_root=RESEARCH_ROOT,
        cache_directory=tmp_path,
        repository_root=REPOSITORY_ROOT,
    )
    assert not report.passed
    assert not report.may_access_provider
    assert report.blocker_code in {"MISSING_OPTIONAL_DEPENDENCY", "NO_ACCELERATOR"}
    assert any(check.check_id == "torch_import" for check in report.checks)


def test_real_gpu_preflight_still_demands_the_full_disk_budget(tmp_path: Path) -> None:
    """The GPU worker's disk requirement is unchanged at 40 GiB."""
    from openalpha_bridge.phase2.preflight import (
        MINIMUM_FREE_DISK_BYTES,
        MINIMUM_FREE_DISK_BYTES_NO_ACCELERATOR,
    )

    assert MINIMUM_FREE_DISK_BYTES == 40 * (1 << 30)
    assert MINIMUM_FREE_DISK_BYTES_NO_ACCELERATOR < MINIMUM_FREE_DISK_BYTES

    report = run_preflight(
        research_root=RESEARCH_ROOT,
        cache_directory=tmp_path,
        repository_root=REPOSITORY_ROOT,
        require_accelerator=True,
    )
    disk = next(c for c in report.checks if c.check_id == "free_disk")
    assert str(MINIMUM_FREE_DISK_BYTES) in disk.requirement


def test_no_accelerator_preflight_uses_the_smaller_disk_budget(tmp_path: Path) -> None:
    """Synthetic validation must not be blocked by the GPU worker's budget."""
    from openalpha_bridge.phase2.preflight import MINIMUM_FREE_DISK_BYTES_NO_ACCELERATOR

    report = run_preflight(
        research_root=RESEARCH_ROOT,
        cache_directory=tmp_path,
        repository_root=REPOSITORY_ROOT,
        require_accelerator=False,
    )
    disk = next(c for c in report.checks if c.check_id == "free_disk")
    assert str(MINIMUM_FREE_DISK_BYTES_NO_ACCELERATOR) in disk.requirement
    assert disk.outcome.value == "PASS"


def test_explicit_minimum_free_bytes_overrides_the_default(tmp_path: Path) -> None:
    report = run_preflight(
        research_root=RESEARCH_ROOT,
        cache_directory=tmp_path,
        repository_root=REPOSITORY_ROOT,
        require_accelerator=False,
        minimum_free_bytes=1 << 60,  # a petabyte: must fail anywhere
    )
    assert not report.passed
    assert report.blocker_code == "INSUFFICIENT_DISK"


def test_preflight_rejects_a_cache_inside_the_repository() -> None:
    report = run_preflight(
        research_root=RESEARCH_ROOT,
        cache_directory=REPOSITORY_ROOT / "cache",
        repository_root=REPOSITORY_ROOT,
        require_accelerator=False,
    )
    cache_check = next(c for c in report.checks if c.check_id == "cache_outside_git")
    assert cache_check.outcome.value == "FAIL"


# --------------------------------------------- synthetic end-to-end pipeline


def test_synthetic_end_to_end_run_reaches_finalized(tmp_path: Path) -> None:
    """SYNTHETIC PIPELINE VALIDATION - NOT EMPIRICAL EVIDENCE."""
    pipeline = _pipeline(tmp_path)

    assert pipeline.preflight().passed
    assert pipeline.retrieve().state is Phase2State.DATA_RETRIEVED
    assert pipeline.validate_data().state is Phase2State.DATA_VALIDATED

    windows = pipeline.build_windows()
    counts = cast("dict[str, int]", windows.detail["sequence_counts"])
    assert counts[Partition.VALIDATION.value] == 3 * 2  # 3 per symbol, 2 symbols
    assert counts[Partition.RECONSTRUCTION_TEST.value] == 5 * 3  # includes the unseen symbol

    assert pipeline.coverage_audit().state is Phase2State.COVERAGE_PASSED
    assert pipeline.resolve_assets().state is Phase2State.ASSETS_RESOLVED
    assert pipeline.stage_a().state is Phase2State.STAGE_A_PASSED
    assert pipeline.stage_b().state is Phase2State.STAGE_B_PASSED
    assert pipeline.stage_c().state is Phase2State.STAGE_C_TRAINED

    frozen = pipeline.freeze_checkpoint()
    assert frozen.state is Phase2State.CHECKPOINT_FROZEN

    opened = pipeline.open_test(operator_command="pytest synthetic")
    assert opened.state is Phase2State.TEST_OPENED

    pipeline.evaluate_test(_passing_measurements())
    pipeline.evaluate_external()
    table = pipeline.finalize(_passing_measurements())

    assert table.conclusion is TerminalConclusion.BRIDGE_2K_FEASIBLE
    assert pipeline.journal.current_state is Phase2State.FINALIZED

    summary = pipeline.summary()
    assert summary["evidence_class"] == EvidenceClass.SYNTHETIC_PIPELINE_VALIDATION.value
    assert summary["score_mask_sha256"] == score_mask_sha256()


def test_synthetic_run_never_writes_the_real_test_opening_record(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    pipeline.preflight()
    pipeline.retrieve()
    pipeline.validate_data()
    pipeline.build_windows()
    pipeline.coverage_audit()
    pipeline.resolve_assets()
    pipeline.stage_a()
    pipeline.stage_b()
    pipeline.stage_c()
    pipeline.freeze_checkpoint()
    pipeline.open_test(operator_command="pytest synthetic")

    run_dir = pipeline.config.run_directory
    assert is_test_partition_opened(run_dir, EvidenceClass.SYNTHETIC_PIPELINE_VALIDATION)
    assert not is_test_partition_opened(run_dir, EvidenceClass.REAL_PHASE2)
    assert (run_dir / "synthetic_test_opening_record.json").is_file()
    assert not (run_dir / "test_opening_record.json").exists()


def test_test_partition_cannot_be_opened_twice(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    pipeline.preflight()
    pipeline.retrieve()
    pipeline.validate_data()
    pipeline.build_windows()
    pipeline.coverage_audit()
    pipeline.resolve_assets()
    pipeline.stage_a()
    pipeline.stage_b()
    pipeline.stage_c()
    pipeline.freeze_checkpoint()
    pipeline.open_test(operator_command="first")

    with pytest.raises(BridgeTransformError) as excinfo:
        open_test_partition(
            run_directory=pipeline.config.run_directory,
            identity=pipeline.identity,
            journal=pipeline.journal,
            preconditions=TestOpeningPreconditions(
                stage_a_passed=True,
                stage_b_passed=True,
                stage_c_completed=True,
                selected_checkpoint_fixed=True,
                checkpoint_sha256="0" * 64,
                frozen_kronos_weights_verified=True,
                validation_selection_report_sealed=True,
                experiment_hash_verified=True,
                preprocessing_state_sha256="0" * 64,
                feature_manifest_sha256="0" * 64,
            ),
            operator_command="second",
            opened_at=datetime(2026, 8, 1, tzinfo=UTC),
        )
    assert excinfo.value.failures[0].code in {
        "TEST_PARTITION_ALREADY_OPENED",
        "TEST_OPENED_BEFORE_CHECKPOINT_FROZEN",
    }


def test_unmet_preconditions_block_the_test_gate(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    pipeline.preflight()
    pipeline.retrieve()
    pipeline.validate_data()
    pipeline.build_windows()
    pipeline.coverage_audit()
    pipeline.resolve_assets()
    pipeline.stage_a()
    pipeline.stage_b()
    pipeline.stage_c()
    pipeline.freeze_checkpoint()

    with pytest.raises(BridgeTransformError) as excinfo:
        open_test_partition(
            run_directory=pipeline.config.run_directory,
            identity=pipeline.identity,
            journal=pipeline.journal,
            preconditions=TestOpeningPreconditions(
                stage_a_passed=True,
                stage_b_passed=False,
                stage_c_completed=True,
                selected_checkpoint_fixed=True,
                checkpoint_sha256="0" * 64,
                frozen_kronos_weights_verified=True,
                validation_selection_report_sealed=True,
                experiment_hash_verified=True,
                preprocessing_state_sha256="0" * 64,
                feature_manifest_sha256="0" * 64,
            ),
            operator_command="premature",
            opened_at=datetime(2026, 8, 1, tzinfo=UTC),
        )
    failure = excinfo.value.failures[0]
    assert failure.code == "TEST_OPENING_PRECONDITION_UNMET"
    assert "stage_b_passed" in failure.message


def test_stage_cannot_run_before_its_predecessor(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    pipeline.preflight()
    with pytest.raises(BridgeTransformError) as excinfo:
        pipeline.stage_c()
    assert excinfo.value.failures[0].code == "STAGE_SKIPPED"


def test_run_resumes_from_the_persisted_journal(tmp_path: Path) -> None:
    first = _pipeline(tmp_path)
    first.preflight()
    first.retrieve()
    first.validate_data()
    assert first.journal.current_state is Phase2State.DATA_VALIDATED

    resumed = _pipeline(tmp_path)
    assert resumed.identity.run_id == first.identity.run_id
    assert resumed.journal.current_state is Phase2State.DATA_VALIDATED

    resumed.build_windows()
    assert resumed.journal.current_state is Phase2State.WINDOWS_BUILT

    journal = json.loads((tmp_path / "run" / "journal.json").read_text(encoding="utf-8"))
    assert journal["transitions"][-1]["state"] == Phase2State.WINDOWS_BUILT.value


def test_changed_configuration_refuses_to_reuse_a_run_directory(tmp_path: Path) -> None:
    _pipeline(tmp_path).preflight()
    with pytest.raises(BridgeTransformError) as excinfo:
        _pipeline(tmp_path, device="cpu")
    assert excinfo.value.failures[0].code == "RUN_IDENTITY_CHANGED"
