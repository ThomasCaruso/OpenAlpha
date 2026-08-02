"""The v2 diagnostic contract, exercised end to end against fakes.

Nothing here touches a network, an official asset, Torch, or any partition
other than the already-retrieved SPY training window.
"""

from __future__ import annotations

import ast
import inspect
import json
import math
import typing
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from diagnostic_fakes import FakeCodec, FakeForecastModel, fake_assets, frozen_digest
from openalpha_bridge.diagnostic import conclusion as conclusion_module
from openalpha_bridge.diagnostic import methods as methods_module
from openalpha_bridge.diagnostic import normalization as normalization_module
from openalpha_bridge.diagnostic import runner as runner_module
from openalpha_bridge.diagnostic import validity as validity_module
from openalpha_bridge.diagnostic.conclusion import RETIRED_LABELS, DiagnosticConclusion
from openalpha_bridge.diagnostic.normalization import (
    CLIP_VALUE,
    EPSILON,
    fit_context_state,
)
from openalpha_bridge.diagnostic.official_input import (
    OFFICIAL_COLUMNS,
    OFFICIAL_STAMP_COLUMNS,
    OfficialRow,
    official_stamp,
)
from openalpha_bridge.diagnostic.runner import (
    DiagnosticArtifact,
    run_frozen_inference_diagnostic,
)
from openalpha_bridge.diagnostic.spec import (
    CONTEXT_CANDLES,
    OFFICIAL_INFERENCE_SETTINGS,
    ROLLOUT_SEEDS,
    TARGET_CANDLES,
    TOTAL_CANDLES,
    V1_SPECIFICATION_SHA256,
    V2_SPECIFICATION_SHA256,
    V3_SPECIFICATION_SHA256,
    V4_SPECIFICATION_SHA256,
    WINDOW,
)
from openalpha_bridge.errors import BridgeTransformError
from openalpha_bridge.phase2.invocation import WorkerInvocation
from openalpha_bridge.phase2.provider import Candle, MarketSeries, ProviderMode

ROOT = Path(__file__).resolve().parents[3]
RESEARCH = ROOT / "research" / "bridge-v0"
NOW = datetime(2026, 8, 2, tzinfo=UTC)
COMMIT = "a" * 40


def _invocation() -> WorkerInvocation:
    return WorkerInvocation.validate_all(
        run_id="canary_0badc0de", source_commit=COMMIT, deployed_commit=COMMIT
    )


def _real_candles() -> tuple[Candle, ...]:
    """512 deterministic, structurally valid daily candles."""
    start = date(2015, 5, 7)
    out: list[Candle] = []
    level = 100.0
    for index in range(TOTAL_CANDLES):
        level = max(5.0, level * (1.0 + 0.0004 * math.cos(index / 9.0)))
        close = level * 1.0002
        out.append(
            Candle(
                session=start + timedelta(days=index),
                open=level,
                high=max(level, close) * 1.004,
                low=min(level, close) * 0.996,
                close=close,
                volume=1.0e6 + index,
                amount=(1.0e6 + index) * close,
            )
        )
        level = close
    return tuple(out)


class _SingleWindowProvider:
    name = "deterministic_fake"
    mode = ProviderMode.FAKE
    client_version = "fake-1"

    def __init__(self) -> None:
        self.requests: list[str] = []

    def fetch(self, request):
        self.requests.append(f"{request.symbol}:{request.start}:{request.end}")
        return MarketSeries(
            symbol=request.symbol,
            interval="1d",
            provider=self.name,
            provider_mode=self.mode,
            client_version=self.client_version,
            retrieval_timestamp=None,
            candles=_real_candles(),
        )


def _as_rows(candles: tuple[Candle, ...]) -> tuple[OfficialRow, ...]:
    return tuple(
        OfficialRow(
            session=c.session,
            open=c.open,
            high=c.high,
            low=c.low,
            close=c.close,
            volume=c.volume,
            amount=c.amount,
        )
        for c in candles
    )


ALL_ROWS = _as_rows(_real_candles())
TRUE_TARGET = ALL_ROWS[CONTEXT_CANDLES:]


def _shift(path: tuple[OfficialRow, ...], factor: float) -> tuple[OfficialRow, ...]:
    return tuple(
        r.model_copy(
            update={
                "open": r.open * factor,
                "high": r.high * factor,
                "low": r.low * factor,
                "close": r.close * factor,
            }
        )
        for r in path
    )


def _make_invalid(path: tuple[OfficialRow, ...]) -> tuple[OfficialRow, ...]:
    """Same closes, but high and low crossed: invalid, equally accurate."""
    return tuple(r.model_copy(update={"high": r.low * 0.9, "low": r.high * 1.1}) for r in path)


def _position(seed: int) -> int:
    return seed - ROLLOUT_SEEDS[0]


def _run(policy, *, codec: FakeCodec | None = None, assets=None, digest=None):
    codec = codec or FakeCodec()
    model = FakeForecastModel(codec=codec, path_for=policy)
    provider = _SingleWindowProvider()
    artifact = run_frozen_inference_diagnostic(
        provider=provider,
        codec=codec,
        model=model,
        assets=assets or fake_assets(),
        invocation=_invocation(),
        research_root=RESEARCH,
        parameter_digest=digest or frozen_digest(),
        now=NOW,
    )
    return artifact, provider, codec, model


# ============================================== full official input identity


def test_the_input_carries_all_six_channels_in_the_official_order() -> None:
    artifact, _, _, _ = _run(lambda seed, ctx: _shift(TRUE_TARGET, 1.01))
    assert artifact.ordered_columns == ("open", "high", "low", "close", "volume", "amount")
    assert len(artifact.ordered_columns) == 6
    assert OFFICIAL_COLUMNS == artifact.ordered_columns


def test_a_five_channel_reduction_is_impossible() -> None:
    """Every row exposes six channels; there is no four-price-plus-volume form."""
    row = ALL_ROWS[0]
    assert len(row.channels()) == 6
    assert row.channels()[5] == row.amount


def test_every_row_keeps_its_session() -> None:
    artifact, _, _, _ = _run(lambda seed, ctx: _shift(TRUE_TARGET, 1.01))
    assert artifact.first_session == "2015-05-07"
    assert artifact.last_session == ALL_ROWS[-1].session.isoformat()
    # Decoded output keeps timestamps too, so a path can be aligned to sessions.
    assert artifact.method_b.raw_decoded[0].session == TRUE_TARGET[0].session
    assert artifact.method_b.raw_decoded[-1].session == TRUE_TARGET[-1].session
    assert len(artifact.method_a.reconstruction) == TOTAL_CANDLES


def test_the_stamp_features_match_the_official_derivation() -> None:
    stamp = official_stamp(date(2016, 3, 9))
    assert OFFICIAL_STAMP_COLUMNS == ("minute", "hour", "weekday", "day", "month")
    assert stamp.values() == (0, 0, 2, 9, 3)  # 2016-03-09 is a Wednesday
    assert len(stamp.values()) == 5


def test_the_model_receives_one_stamp_per_context_and_target_row() -> None:
    _, _, _, model = _run(lambda seed, ctx: _shift(TRUE_TARGET, 1.01))
    assert set(model.stamp_shapes) == {(CONTEXT_CANDLES, TARGET_CANDLES)}
    assert set(model.context_lengths) == {CONTEXT_CANDLES}


def test_the_column_presence_mask_is_recorded() -> None:
    artifact, _, _, _ = _run(lambda seed, ctx: _shift(TRUE_TARGET, 1.01))
    mask = artifact.column_presence.as_mask()
    assert set(mask) == set(OFFICIAL_COLUMNS)
    assert artifact.column_presence.all_retrieved is True


def test_frequency_calendar_and_boundary_are_recorded() -> None:
    artifact, _, _, _ = _run(lambda seed, ctx: _shift(TRUE_TARGET, 1.01))
    assert artifact.frequency == "1d"
    assert artifact.calendar == "XNYS"
    assert artifact.context_target_boundary == CONTEXT_CANDLES == 448
    assert artifact.context_candles == 448
    assert artifact.target_candles == 64


# ================================================ context-only normalization


def test_the_state_is_fitted_from_the_448_context_rows_only() -> None:
    artifact, _, _, _ = _run(lambda seed, ctx: _shift(TRUE_TARGET, 1.01))
    state = artifact.normalization_state
    assert state.fitted_candle_count == CONTEXT_CANDLES == 448
    assert state.columns == OFFICIAL_COLUMNS
    assert state.epsilon == EPSILON == 1e-5
    assert state.clip_value == CLIP_VALUE == 5.0
    assert state.clip_policy == "symmetric_clip_after_standardization"
    assert state.standard_deviation_ddof == 0

    # Fitting on all 512 would give a different state, which is the leak the
    # contract forbids.
    leaked = fit_context_state(ALL_ROWS)
    assert leaked.state_sha256 != state.state_sha256


def test_the_state_matches_the_official_formulas() -> None:
    """Compared against the pinned NumPy expression, in float32.

    A float64 Python computation of the same statistics differs from this by
    roughly a part in 10^5, which is the whole reason the implementation was
    changed; the comparison here is against what the source actually computes.
    """
    import numpy as np

    context = ALL_ROWS[:CONTEXT_CANDLES]
    state = fit_context_state(context)

    matrix = np.array([row.channels() for row in context], dtype=np.float32)
    expected_mean = np.mean(matrix, axis=0)
    expected_std = np.std(matrix, axis=0)

    assert state.mean_array().tobytes() == expected_mean.astype("<f4").tobytes()
    assert state.std_array().tobytes() == expected_std.astype("<f4").tobytes()

    expected_normalized = np.clip((matrix - expected_mean) / (expected_std + 1e-5), -5, 5)
    observed = state.normalize_array(matrix)
    assert observed.dtype == np.float32
    assert observed.tobytes() == expected_normalized.tobytes()


def test_clipping_is_symmetric_and_applied_after_standardization() -> None:
    context = ALL_ROWS[:CONTEXT_CANDLES]
    state = fit_context_state(context)
    extreme = context[0].model_copy(update={"close": context[0].close * 1e6})
    normalized = state.normalize((extreme,))
    index = OFFICIAL_COLUMNS.index("close")
    assert normalized[0][index] == CLIP_VALUE


def test_every_method_is_handed_the_same_state_and_records_its_hash() -> None:
    artifact, _, codec, model = _run(lambda seed, ctx: _shift(TRUE_TARGET, 1.01))
    digest = artifact.normalization_state.state_sha256
    assert artifact.method_a.normalization_state_sha256 == digest
    assert artifact.method_b.normalization_state_sha256 == digest
    assert artifact.method_d.normalization_state_sha256 == digest
    # The adapters were told, every single time.
    assert set(codec.states_seen) == {digest}
    assert set(model.states_seen) == {digest}


def test_no_adapter_retains_an_implicit_last_normalization() -> None:
    """The protocols require the state as a keyword argument on every call."""
    from openalpha_bridge.diagnostic.backends import ForecastModel, TokenizerCodec

    for protocol, method in ((TokenizerCodec, "encode"), (TokenizerCodec, "decode")):
        assert "state" in inspect.signature(getattr(protocol, method)).parameters
    assert "state" in inspect.signature(ForecastModel.generate).parameters


# ==================================================== corrected Method A metrics


def test_method_a_reports_all_five_required_quantities() -> None:
    artifact, _, _, _ = _run(lambda seed, ctx: _shift(TRUE_TARGET, 1.01))
    a = artifact.method_a
    assert a.full_sequence_error.defined
    assert a.full_sequence_error.full_sequence_ohlcva_mae is not None
    assert set(a.full_sequence_error.per_column_mae or {}) == set(OFFICIAL_COLUMNS)
    assert a.full_sequence_error.internal_return_mae is not None
    assert a.full_sequence_error.transitions_scored == TOTAL_CANDLES - 1 == 511
    assert a.target_suffix_error.rows_scored == TARGET_CANDLES
    assert a.validity_all.candle_count == TOTAL_CANDLES
    assert a.validity_target_suffix.candle_count == TARGET_CANDLES


def test_method_a_uses_no_external_anchor() -> None:
    """Transitions are internal to the sequence, 1 through 511.

    Checked structurally: reconstruction_error has nowhere to receive an
    external close, and run_method_a never mentions one. forecast_error keeps
    its anchor, which is legitimate because the model genuinely had that value.
    """
    from openalpha_bridge.diagnostic import metrics as metrics_module

    assert "anchor" not in inspect.getsource(methods_module.run_method_a)
    reconstruction_parameters = inspect.signature(metrics_module.reconstruction_error).parameters
    assert set(reconstruction_parameters) == {"reconstructed", "actual"}
    assert "anchor_close" in inspect.signature(metrics_module.forecast_error).parameters

    # The metric it computes really is transition-internal.
    perfect = metrics_module.reconstruction_error(ALL_ROWS, ALL_ROWS)
    assert perfect.internal_return_mae == pytest.approx(0.0)
    assert perfect.transitions_scored == len(ALL_ROWS) - 1


def test_any_invalid_round_trip_candle_is_a_strict_observation() -> None:
    """One invalid candle in 512 is below the 1 per cent threshold and still reported."""
    codec = FakeCodec(corrupt_round_trip=True, corrupt_count=1)
    artifact, _, _, _ = _run(lambda seed, ctx: _shift(TRUE_TARGET, 1.01), codec=codec)
    a = artifact.method_a
    assert a.validity_all.invalid_candle_count == 1
    assert a.validity_all.invalid_candle_fraction < 0.01  # immaterial
    assert a.structural_invalidity_observed is True
    # v3: recorded as a finding, never primary, and it no longer suppresses
    # the later rules. Every one of them is still evaluated.
    assert (
        DiagnosticConclusion.ROUNDTRIP_STRUCTURAL_INVALIDITY_OBSERVED in artifact.matched_findings
    )
    assert artifact.conclusion is not (
        DiagnosticConclusion.ROUNDTRIP_STRUCTURAL_INVALIDITY_OBSERVED
    )
    later = {"R3", "R4", "R5", "R6", "R7", "R8"}
    evaluated = {e.rule_id for e in artifact.decision.evaluations}
    assert later <= evaluated


def test_material_round_trip_invalidity_is_separately_labelled() -> None:
    codec = FakeCodec(corrupt_round_trip=True, corrupt_fraction=0.5)
    artifact, _, _, _ = _run(lambda seed, ctx: _shift(TRUE_TARGET, 1.01), codec=codec)
    assert artifact.method_a.validity_all.invalid_candle_fraction > 0.01
    assert DiagnosticConclusion.ROUNDTRIP_MATERIAL_INVALIDITY in artifact.matched_findings
    # R1 may be primary, but the later findings still had to be evaluated.
    later = {"R3", "R4", "R5", "R6", "R7", "R8"}
    assert later <= {e.rule_id for e in artifact.decision.evaluations}


# ===================================================== narrow conclusion vocabulary


@pytest.mark.parametrize("retired", sorted(RETIRED_LABELS))
def test_retired_causal_labels_cannot_be_produced(retired: str) -> None:
    assert retired not in {c.value for c in DiagnosticConclusion}
    assert RETIRED_LABELS[retired]


def test_the_vocabulary_is_exactly_the_v4_labels() -> None:
    assert {c.value for c in DiagnosticConclusion} == {
        "DIAGNOSTIC_OPERATIONAL_FAILURE",
        "REPRODUCIBILITY_FAILURE",
        "ROUNDTRIP_MATERIAL_INVALIDITY",
        "ROUNDTRIP_STRUCTURAL_INVALIDITY_OBSERVED",
        "ROUNDTRIP_INVALIDITY_ON_UNCLIPPED_INPUTS",
        "ROUNDTRIP_INVALIDITY_CONFINED_TO_CLIPPED_INPUTS",
        "MATERIAL_CLIPPING_EXPOSURE_OBSERVED",
        "ROUNDTRIP_INTERPRETATION_CONFOUNDED_BY_CLIPPING",
        "NO_VALID_ROLLOUTS_OBSERVED",
        "LOW_VALID_ROLLOUT_FRACTION",
        "VALID_ONLY_ENSEMBLE_MEETS_IMPROVEMENT_THRESHOLD",
        "PROJECTION_RESTORES_VALIDITY_WITHOUT_THRESHOLD_IMPROVEMENT",
        "NO_SKILL_AGAINST_PERSISTENCE",
        "SKILL_AGAINST_PERSISTENCE_OBSERVED",
        "NO_PREREGISTERED_EFFECT_DETECTED",
        "DIAGNOSTIC_INCONCLUSIVE",
    }


def test_the_default_final_rule_is_not_causal() -> None:
    """A wildly wrong but valid forecast beats no baseline, and says so."""
    wrong_but_valid = _shift(TRUE_TARGET, 1.30)
    artifact, _, _, _ = _run(lambda seed, ctx: wrong_but_valid)
    assert artifact.decision.forecast_origins == 1
    assert DiagnosticConclusion.NO_SKILL_AGAINST_PERSISTENCE in artifact.matched_findings
    assert artifact.conclusion is DiagnosticConclusion.NO_SKILL_AGAINST_PERSISTENCE
    assert artifact.recommended_next_experiment.value == ("ABANDON_STRUCTURAL_VALIDITY_DIRECTION")


def test_no_valid_rollouts_is_typed_and_nothing_is_substituted() -> None:
    artifact, _, _, _ = _run(lambda seed, ctx: _make_invalid(_shift(TRUE_TARGET, 1.01)))
    assert DiagnosticConclusion.NO_VALID_ROLLOUTS_OBSERVED in artifact.matched_findings
    assert artifact.method_d.valid_only_ensemble is None
    # k == 0, so there is no size to match and every control statistic is null.
    controls = artifact.method_d.size_matched_controls
    assert controls.k == 0
    assert controls.scored_repetitions == 0
    assert controls.mean_primary_error is None
    assert controls.undefined_reason is not None
    assert artifact.method_d.valid_only_ensemble_error is None
    assert artifact.method_d.manual_seeded_ensemble is not None


def test_valid_only_ensemble_meeting_the_threshold_is_stated_as_such() -> None:
    accurate_valid = _shift(TRUE_TARGET, 1.0005)
    inaccurate_invalid = _make_invalid(_shift(TRUE_TARGET, 1.35))
    artifact, _, _, _ = _run(
        lambda seed, ctx: accurate_valid if _position(seed) % 2 == 0 else inaccurate_invalid
    )
    assert (
        DiagnosticConclusion.VALID_ONLY_ENSEMBLE_MEETS_IMPROVEMENT_THRESHOLD
        in artifact.matched_findings
    )
    # The comparison is against size-matched controls, not against all 64.
    controls = artifact.method_d.size_matched_controls
    assert controls.k == artifact.method_d.valid_rollout_count
    assert controls.mean_primary_error is not None
    assert not controls.degenerate


def test_a_low_valid_fraction_is_described_not_diagnosed_as_a_support_defect() -> None:
    valid_path = _shift(TRUE_TARGET, 1.02)
    invalid_path = _make_invalid(_shift(TRUE_TARGET, 1.02))
    artifact, _, _, _ = _run(lambda seed, ctx: valid_path if _position(seed) < 3 else invalid_path)
    assert DiagnosticConclusion.LOW_VALID_ROLLOUT_FRACTION in artifact.matched_findings
    assert artifact.method_d.valid_rollout_fraction < 0.25
    assert artifact.method_d.size_matched_controls.k == 3


def test_projection_without_threshold_improvement_is_described_locally() -> None:
    invalid_close = _make_invalid(_shift(TRUE_TARGET, 1.001))
    valid_close = _shift(TRUE_TARGET, 1.001)

    def policy(seed: int, _ctx):
        position = _position(seed)
        if position == 0:
            return invalid_close
        return valid_close if position % 2 == 0 else invalid_close

    artifact, _, _, _ = _run(policy)
    assert artifact.method_c.restores_validity
    assert (
        DiagnosticConclusion.PROJECTION_RESTORES_VALIDITY_WITHOUT_THRESHOLD_IMPROVEMENT
        in artifact.matched_findings
    )


def test_every_rule_is_recorded_even_when_it_did_not_match() -> None:
    artifact, _, _, _ = _run(lambda seed, ctx: _shift(TRUE_TARGET, 1.30))
    ids = [e.rule_id for e in artifact.decision.evaluations]
    assert ids == [
        "R0",
        "R0b",
        "R1",
        "R2",
        "R2a",
        "R2b",
        "R2c",
        "R2d",
        "R3",
        "R4",
        "R5",
        "R6",
        "R7",
        "R8",
        "R9",
        "R10",
    ]
    # v3 permits several simultaneous findings, so this is no longer one.
    matched = [e for e in artifact.decision.evaluations if e.matched]
    assert len(matched) >= 1
    assert artifact.decision.primary_conclusion in artifact.matched_findings


# ======================================================= probability definition


def test_stored_probabilities_state_exactly_what_they_are() -> None:
    artifact, _, _, _ = _run(lambda seed, ctx: _shift(TRUE_TARGET, 1.01))
    assert artifact.sampling_probability_definition == (
        "log_softmax_of_temperature_scaled_then_top_k_top_p_filtered_logits_renormalized"
    )
    step = artifact.method_b.sampling[0]
    assert step.measured_after_temperature is True
    assert step.measured_after_top_k_top_p_filtering is True
    assert step.filtered_distribution_renormalized is True
    assert step.pair_log_probability == pytest.approx(
        step.coarse_log_probability + step.fine_conditional_log_probability
    )
    assert len(artifact.method_b.sampling) == TARGET_CANDLES
    assert artifact.method_b.total_path_sampling_log_probability == pytest.approx(
        sum(s.pair_log_probability for s in artifact.method_b.sampling)
    )


def test_no_stored_quantity_is_called_a_likelihood() -> None:
    for module in (methods_module, runner_module, conclusion_module):
        source = inspect.getsource(module).lower()
        assert "likelihood" not in source or "not a model likelihood" in source
    from openalpha_bridge.diagnostic import backends

    fields = set(backends.GeneratedPath.model_fields) | set(backends.StepSampling.model_fields)
    assert not any("likelihood" in name for name in fields)
    assert any("sampling_log_probability" in name for name in fields)


# ========================================================== ensemble definition


def test_the_manual_ensemble_is_never_called_official() -> None:
    artifact, _, _, _ = _run(lambda seed, ctx: _shift(TRUE_TARGET, 1.01))
    d = artifact.method_d
    assert d.manual_seeded_ensemble is not None
    assert d.manual_ensemble_is_official is False
    hints = typing.get_type_hints(methods_module.MethodDResult, include_extras=True)
    assert hints["manual_ensemble_is_official"] == typing.Literal[False]
    # No field claims to be the official sample_count ensemble.
    assert not any(
        "official_ensemble" in name for name in methods_module.MethodDResult.model_fields
    )


def test_the_forecast_origin_count_is_recorded_as_one() -> None:
    artifact, _, _, _ = _run(lambda seed, ctx: _shift(TRUE_TARGET, 1.01))
    assert artifact.method_d.forecast_origins == 1


# ============================================================ prohibitions


def test_no_parameter_is_modified() -> None:
    artifact, _, _, model = _run(lambda seed, ctx: _shift(TRUE_TARGET, 1.01))
    assert artifact.parameter_sha256_before == artifact.parameter_sha256_after
    assert artifact.parameters_unmodified is True
    assert model.parameter_writes == 0


def test_a_changed_parameter_digest_fails_closed() -> None:
    digests = iter(["a" * 64, "b" * 64])
    with pytest.raises(BridgeTransformError) as excinfo:
        _run(lambda seed, ctx: _shift(TRUE_TARGET, 1.01), digest=lambda: next(digests))
    assert excinfo.value.failures[0].code == "DIAGNOSTIC_PARAMETERS_MODIFIED"


def test_trainable_parameters_are_refused() -> None:
    with pytest.raises(BridgeTransformError) as excinfo:
        _run(lambda seed, ctx: _shift(TRUE_TARGET, 1.01), assets=fake_assets(trainable=17_605))
    assert excinfo.value.failures[0].code == "DIAGNOSTIC_PARAMETERS_NOT_FROZEN"


def _diagnostic_modules():
    return (
        runner_module,
        methods_module,
        conclusion_module,
        validity_module,
        normalization_module,
    )


def _call_graph(module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(inspect.getsource(module))):
        if isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Name):
                names.add(target.id)
            elif isinstance(target, ast.Attribute):
                names.add(target.attr)
    return names


def _imported_names(module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(inspect.getsource(module))):
        if isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names.update(alias.name for alias in node.names)
            if node.module:
                names.update(node.module.split("."))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
                names.update(alias.name.split("."))
    return names


@pytest.mark.parametrize(
    "forbidden",
    [
        "backward",
        "zero_grad",
        "Adam",
        "AdamW",
        "SGD",
        "optimizer",
        "run_training",
        "TrainingBackend",
        "select_checkpoint",
        "freeze_checkpoint",
        "CheckpointRecord",
        "evaluate_conclusion",
        "GateTable",
        "open_test_partition",
        "open_cloud_test_partition",
        "BridgeDecoder",
        "CloudRunner",
    ],
)
def test_no_optimizer_or_training_code_is_reachable(forbidden: str) -> None:
    for module in _diagnostic_modules():
        assert forbidden not in _call_graph(module)
        assert forbidden not in _imported_names(module)


def test_the_diagnostic_imports_no_training_module() -> None:
    for module in _diagnostic_modules():
        imported = _imported_names(module)
        for banned in ("training", "gates", "testgate", "pipeline", "torch"):
            assert not any(name == banned or name.endswith(f".{banned}") for name in imported)


def test_importing_the_diagnostic_loads_no_torch() -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, openalpha_bridge.diagnostic; print('torch' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False"


def test_only_one_provider_retrieval_occurs() -> None:
    artifact, provider, _, _ = _run(lambda seed, ctx: _shift(TRUE_TARGET, 1.01))
    assert provider.requests == ["SPY:2015-05-07:2017-05-18"]
    assert artifact.provider_request_count == 1
    assert artifact.retrieved_sessions == TOTAL_CANDLES


def test_no_held_out_partition_is_opened() -> None:
    artifact, provider, _, _ = _run(lambda seed, ctx: _shift(TRUE_TARGET, 1.01))
    assert artifact.held_out_partition_opened is False
    assert provider.requests == ["SPY:2015-05-07:2017-05-18"]
    for module in _diagnostic_modules():
        source = inspect.getsource(module)
        for partition in ("reconstruction_test", "external_later", "holdout"):
            assert partition not in source
    assert WINDOW.start_inclusive == "2015-05-07"
    assert WINDOW.end_exclusive == "2017-05-18"


def test_the_model_never_sees_the_target() -> None:
    seen: list[int] = []

    def policy(seed: int, context):
        seen.append(len(context))
        return _shift(TRUE_TARGET, 1.01)

    _run(policy)
    assert set(seen) == {CONTEXT_CANDLES}


def test_beam_search_is_not_implemented() -> None:
    for module in _diagnostic_modules():
        assert "beam" not in inspect.getsource(module).lower()


# ============================================================ the artifact


def test_the_artifact_authorizes_nothing_by_type() -> None:
    artifact, _, _, _ = _run(lambda seed, ctx: _shift(TRUE_TARGET, 1.01))
    hints = typing.get_type_hints(DiagnosticArtifact, include_extras=True)
    for field in (
        "training_performed",
        "optimizer_constructed",
        "authorizes_training",
        "authorizes_stage_b",
        "authorizes_stage_c",
        "authorizes_test_opening",
        "authorizes_production_inference",
        "authorizes_trading_claims",
        "scientific_result_available",
        "held_out_partition_opened",
    ):
        assert hints[field] == typing.Literal[False]
        assert getattr(artifact, field) is False


def test_all_four_specifications_are_verified_and_v4_is_operative() -> None:
    artifact, _, _, _ = _run(lambda seed, ctx: _shift(TRUE_TARGET, 1.01))
    assert artifact.specification_v1_sha256 == V1_SPECIFICATION_SHA256
    assert artifact.specification_v2_sha256 == V2_SPECIFICATION_SHA256
    assert artifact.specification_v3_sha256 == V3_SPECIFICATION_SHA256
    assert artifact.specification_v4_sha256 == V4_SPECIFICATION_SHA256
    assert artifact.operative_specification == "phase2-frozen-inference-diagnostic-v4.yaml"
    assert artifact.schema_version == "openalpha.bridge.diagnostic.frozen_inference.v4"


def test_a_tampered_specification_fails_closed(tmp_path: Path) -> None:
    fake_root = tmp_path / "research"
    fake_root.mkdir()
    for name in (
        "phase2-frozen-inference-diagnostic.yaml",
        "phase2-frozen-inference-diagnostic-v2.yaml",
        "phase2-frozen-inference-diagnostic-v3.yaml",
        "phase2-frozen-inference-diagnostic-v4.yaml",
    ):
        (fake_root / name).write_text("schema: tampered\n", encoding="utf-8")
    codec = FakeCodec()
    with pytest.raises(BridgeTransformError) as excinfo:
        run_frozen_inference_diagnostic(
            provider=_SingleWindowProvider(),
            codec=codec,
            model=FakeForecastModel(codec=codec, path_for=lambda s, c: _shift(TRUE_TARGET, 1.0)),
            assets=fake_assets(),
            invocation=_invocation(),
            research_root=fake_root,
            parameter_digest=frozen_digest(),
            now=NOW,
        )
    assert excinfo.value.failures[0].code == "DIAGNOSTIC_SPECIFICATION_HASH_MISMATCH"


def test_the_inference_settings_come_from_predict_defaults() -> None:
    settings = OFFICIAL_INFERENCE_SETTINGS
    assert settings.temperature == 1.0
    assert settings.top_k == 0
    assert settings.top_p == 0.9
    assert settings.sample_count_per_call == 1
    assert settings.prediction_length == 64
    assert settings.max_context == 512
    assert settings.clip == 5.0
    assert settings.gradient_mode == "inference_mode"


def test_raw_tokens_and_raw_six_channel_outputs_are_preserved() -> None:
    artifact, _, _, _ = _run(lambda seed, ctx: _shift(TRUE_TARGET, 1.01))
    assert len(artifact.method_a.coarse_token_ids) == TOTAL_CANDLES
    assert len(artifact.method_b.coarse_token_ids) == TARGET_CANDLES
    assert len(artifact.method_d.rollouts) == 64
    for rollout in artifact.method_d.rollouts:
        assert len(rollout.raw_decoded) == TARGET_CANDLES
        assert all(len(row.channels()) == 6 for row in rollout.raw_decoded)
    payload = json.loads(artifact.model_dump_json())
    assert payload["ordered_columns"] == list(OFFICIAL_COLUMNS)
    assert payload["normalization_state"]["fitted_candle_count"] == 448
