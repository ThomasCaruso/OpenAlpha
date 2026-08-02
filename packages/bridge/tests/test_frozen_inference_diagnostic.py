"""The diagnostic must distinguish the situations it exists to tell apart.

Every test runs the real diagnostic end to end against fakes. Nothing here
touches a network, an official asset, Torch, or any partition other than the
already-retrieved SPY training window.
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
from openalpha_bridge.diagnostic import runner as runner_module
from openalpha_bridge.diagnostic import validity as validity_module
from openalpha_bridge.diagnostic.conclusion import DiagnosticConclusion
from openalpha_bridge.diagnostic.runner import (
    DiagnosticArtifact,
    run_frozen_inference_diagnostic,
)
from openalpha_bridge.diagnostic.spec import (
    CONTEXT_CANDLES,
    ROLLOUT_SEEDS,
    TARGET_CANDLES,
    TOTAL_CANDLES,
    WINDOW,
)
from openalpha_bridge.diagnostic.validity import DecodedCandle
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


# --------------------------------------------------------------- the window


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
                volume=1.0e6,
                amount=1.0e6 * close,
            )
        )
        level = close
    return tuple(out)


class _SingleWindowProvider:
    """Serves the amended window once and counts every call."""

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


def _as_decoded(candles: tuple[Candle, ...]) -> tuple[DecodedCandle, ...]:
    return tuple(
        DecodedCandle(open=c.open, high=c.high, low=c.low, close=c.close, volume=c.volume)
        for c in candles
    )


TRUE_TARGET = _as_decoded(_real_candles())[CONTEXT_CANDLES:]


def _shift(path: tuple[DecodedCandle, ...], factor: float) -> tuple[DecodedCandle, ...]:
    """A structurally valid path displaced from the truth by ``factor``."""
    return tuple(
        DecodedCandle(
            open=c.open * factor,
            high=c.high * factor,
            low=c.low * factor,
            close=c.close * factor,
            volume=c.volume,
        )
        for c in path
    )


def _make_invalid(path: tuple[DecodedCandle, ...]) -> tuple[DecodedCandle, ...]:
    """Same closes, but high and low crossed: invalid, equally accurate."""
    return tuple(
        DecodedCandle(
            open=c.open, high=c.low * 0.9, low=c.high * 1.1, close=c.close, volume=c.volume
        )
        for c in path
    )


def _position(seed: int) -> int:
    """Rollout position for a seed. Method B shares seed 0 with rollout 0."""
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


# ============================================================ the six cases


def test_1_in_distribution_round_trip_invalidity_is_detected() -> None:
    """Case 1: the tokenizer decoder mangles its own encoder's tokens."""
    codec = FakeCodec(corrupt_round_trip=True, corrupt_fraction=0.5)
    artifact, _, _, _ = _run(lambda seed, ctx: _shift(TRUE_TARGET, 1.01), codec=codec)

    assert artifact.conclusion is DiagnosticConclusion.TOKENIZER_DECODER_DEFECT
    assert artifact.decision.matched_rule_id == "R1"
    assert artifact.method_a.validity.invalid_candle_fraction > 0.01
    assert artifact.method_a.validity.path_is_invalid
    assert "high_below_low" in artifact.method_a.validity.violation_counts


def test_2_valid_round_trip_but_invalid_generated_decoding() -> None:
    """Case 2: round trip is clean; generated tokens decode outside validity.

    Valid and invalid rollouts are made equally accurate, so the finding is
    about where invalidity enters rather than about accuracy.
    """
    valid_path = _shift(TRUE_TARGET, 1.02)
    invalid_path = _make_invalid(_shift(TRUE_TARGET, 1.02))

    def policy(seed: int, _context):
        # 3 valid out of 64 -> below the 0.25 support minimum, but not zero.
        return valid_path if _position(seed) < 3 else invalid_path

    artifact, _, _, _ = _run(policy)

    assert artifact.method_a.validity.invalid_candle_fraction == 0.0
    assert artifact.conclusion is DiagnosticConclusion.GENERATED_TOKEN_SUPPORT_DEFECT
    assert artifact.decision.matched_rule_id == "R4"
    assert artifact.method_d.valid_rollout_count == 3
    assert artifact.method_d.valid_rollout_fraction < 0.25


def test_3_projection_fixes_validity_without_improving_accuracy() -> None:
    """Case 3: geometry is repairable, and repair buys no accuracy."""
    invalid_but_close = _make_invalid(_shift(TRUE_TARGET, 1.001))
    valid_same_accuracy = _shift(TRUE_TARGET, 1.001)

    def policy(seed: int, _context):
        # Rollout 0, which Method B shares, is invalid. Half the remaining
        # rollouts are valid so R4 does not fire, and both groups have identical
        # accuracy so R3 does not either.
        position = _position(seed)
        if position == 0:
            return invalid_but_close
        return valid_same_accuracy if position % 2 == 0 else invalid_but_close

    artifact, _, _, _ = _run(policy)

    assert artifact.method_b.validity.path_is_invalid
    assert artifact.method_c.restores_validity
    assert not artifact.method_c.validity_after.path_is_invalid
    # Closes are untouched by the projection, so the primary metric cannot move.
    assert artifact.method_c.primary_error_improvement == pytest.approx(0.0, abs=1e-12)
    assert artifact.conclusion is DiagnosticConclusion.VALIDITY_REPAIR_ONLY
    assert artifact.decision.matched_rule_id == "R5"


def test_4_valid_rollouts_outperform_invalid_rollouts() -> None:
    """Case 4: conditioning on validity actually helps."""
    accurate_valid = _shift(TRUE_TARGET, 1.0005)
    inaccurate_invalid = _make_invalid(_shift(TRUE_TARGET, 1.35))

    def policy(seed: int, _context):
        return accurate_valid if _position(seed) % 2 == 0 else inaccurate_invalid

    artifact, _, _, _ = _run(policy)

    assert artifact.conclusion is DiagnosticConclusion.VALIDITY_FILTER_IMPROVES_FORECAST
    assert artifact.decision.matched_rule_id == "R3"
    valid_error = artifact.method_d.valid_only_ensemble_error
    all_error = artifact.method_d.all_rollout_ensemble_error
    assert valid_error is not None and all_error is not None
    valid_primary = valid_error.close_return_mae
    all_primary = all_error.close_return_mae
    assert valid_primary is not None and all_primary is not None
    assert valid_primary < all_primary


def test_5_no_valid_rollout_exists_is_typed_not_silently_repaired() -> None:
    """Case 5: zero valid paths, and nothing is substituted for them."""
    artifact, _, _, _ = _run(lambda seed, ctx: _make_invalid(_shift(TRUE_TARGET, 1.01)))

    assert artifact.conclusion is DiagnosticConclusion.NO_VALID_ROLLOUTS
    assert artifact.decision.matched_rule_id == "R2"
    assert artifact.method_d.valid_rollout_count == 0
    # Explicitly absent rather than projected, substituted, or zero-filled.
    assert artifact.method_d.valid_only_ensemble is None
    assert artifact.method_d.valid_only_ensemble_error is None
    assert artifact.method_d.valid_group_mean_primary_error is None
    # The all-rollout ensemble is still reported, so the official procedure's
    # answer remains visible next to the refusal to use it.
    assert artifact.method_d.all_rollout_ensemble is not None


def test_6_structurally_valid_forecasts_that_remain_inaccurate() -> None:
    """Case 6: everything is valid and everything is still wrong."""
    wrong_but_valid = _shift(TRUE_TARGET, 1.30)
    artifact, _, _, _ = _run(lambda seed, ctx: wrong_but_valid)

    assert artifact.method_a.validity.invalid_candle_fraction == 0.0
    assert not artifact.method_b.validity.path_is_invalid
    assert artifact.method_d.valid_rollout_fraction == 1.0
    assert artifact.conclusion is DiagnosticConclusion.INVALIDITY_NOT_CAUSAL_TO_FORECAST_ERROR
    assert artifact.decision.matched_rule_id == "R6"
    error = artifact.method_d.valid_only_ensemble_error
    assert error is not None and error.close_return_mae is not None


# ================================================= prohibitions and boundary


def test_7_no_parameter_is_modified() -> None:
    """Measured by hashing before and after, not asserted."""
    artifact, _, _, model = _run(lambda seed, ctx: _shift(TRUE_TARGET, 1.01))
    assert artifact.parameter_sha256_before == artifact.parameter_sha256_after
    assert artifact.parameters_unmodified is True
    assert artifact.training_performed is False
    assert model.parameter_writes == 0


def test_7b_a_changed_parameter_digest_fails_closed() -> None:
    """If a parameter did move, the diagnostic refuses to report a result."""
    digests = iter(["a" * 64, "b" * 64])

    with pytest.raises(BridgeTransformError) as excinfo:
        _run(lambda seed, ctx: _shift(TRUE_TARGET, 1.01), digest=lambda: next(digests))
    assert excinfo.value.failures[0].code == "DIAGNOSTIC_PARAMETERS_MODIFIED"


def test_7c_trainable_parameters_are_refused() -> None:
    with pytest.raises(BridgeTransformError) as excinfo:
        _run(lambda seed, ctx: _shift(TRUE_TARGET, 1.01), assets=fake_assets(trainable=17_605))
    assert excinfo.value.failures[0].code == "DIAGNOSTIC_PARAMETERS_NOT_FROZEN"


def _diagnostic_modules():
    return (runner_module, methods_module, conclusion_module, validity_module)


def _call_graph(module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(inspect.getsource(module))):
        if isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Name):
                names.add(target.id)
            elif isinstance(target, ast.Attribute):
                names.add(target.attr)
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.name for alias in node.names)
            names.add(node.module or "")
    return names


@pytest.mark.parametrize(
    "forbidden",
    [
        "backward",
        "zero_grad",
        "step",
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
        "stage_b",
        "stage_c",
    ],
)
def test_8_no_optimizer_or_training_code_is_reachable(forbidden: str) -> None:
    """Checked structurally rather than by grep.

    A line scan cannot be used here: `step` is an ordinary loop variable and the
    module docstrings name the very things they promise not to do. What matters
    is whether anything is called or imported, which the AST answers exactly.
    """
    for module in _diagnostic_modules():
        assert forbidden not in _call_graph(module), f"{module.__name__} calls {forbidden}"
        assert forbidden not in _imported_names(module), f"{module.__name__} imports {forbidden}"


def _imported_names(module) -> set[str]:
    """Every module and symbol the module imports, at any depth."""
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


def test_8c_the_diagnostic_imports_no_training_module() -> None:
    """The training, gate and testgate modules are never pulled in at all."""
    for module in _diagnostic_modules():
        imported = _imported_names(module)
        for banned in ("training", "gates", "testgate", "pipeline", "torch", "runner"):
            assert not any(name == banned or name.endswith(f".{banned}") for name in imported), (
                f"{module.__name__} imports {banned}"
            )


def test_8b_importing_the_diagnostic_loads_no_torch() -> None:
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


def test_9_only_one_provider_retrieval_occurs() -> None:
    artifact, provider, _, _ = _run(lambda seed, ctx: _shift(TRUE_TARGET, 1.01))
    assert len(provider.requests) == 1
    assert provider.requests[0] == "SPY:2015-05-07:2017-05-18"
    assert artifact.provider_request_count == 1
    assert artifact.retrieved_candles == TOTAL_CANDLES


def test_10_no_held_out_partition_is_opened() -> None:
    artifact, provider, _, _ = _run(lambda seed, ctx: _shift(TRUE_TARGET, 1.01))
    assert artifact.held_out_partition_opened is False
    assert provider.requests == ["SPY:2015-05-07:2017-05-18"]
    for module in _diagnostic_modules():
        source = inspect.getsource(module)
        for partition in ("reconstruction_test", "external_later", "RECONSTRUCTION", "holdout"):
            assert partition not in source
    # The window never reaches beyond the amended training range.
    assert WINDOW.start_inclusive == "2015-05-07"
    assert WINDOW.end_exclusive == "2017-05-18"


# ======================================================= the decision artifact


def test_the_window_is_split_448_context_and_64_target() -> None:
    artifact, _, _, _ = _run(lambda seed, ctx: _shift(TRUE_TARGET, 1.01))
    assert artifact.context_candles == CONTEXT_CANDLES == 448
    assert artifact.target_candles == TARGET_CANDLES == 64
    assert artifact.context_candles + artifact.target_candles == TOTAL_CANDLES


def test_the_model_never_sees_the_target() -> None:
    """The context handed to the model is exactly the first 448 candles."""
    seen: list[int] = []

    def policy(seed: int, context):
        seen.append(len(context))
        return _shift(TRUE_TARGET, 1.01)

    _run(policy)
    assert set(seen) == {CONTEXT_CANDLES}


def test_the_fixed_seed_set_is_used_exactly() -> None:
    _, _, _, model = _run(lambda seed, ctx: _shift(TRUE_TARGET, 1.01))
    # 64 rollouts for Method D, plus the single Method B forecast.
    assert model.seeds_used[0] == ROLLOUT_SEEDS[0]
    assert model.seeds_used[1:] == ROLLOUT_SEEDS
    assert len(ROLLOUT_SEEDS) == 64
    assert len(set(ROLLOUT_SEEDS)) == 64


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


def test_the_claim_boundary_is_carried_and_fixed() -> None:
    artifact, _, _, _ = _run(lambda seed, ctx: _shift(TRUE_TARGET, 1.01))
    assert artifact.claim_boundary == "DEVELOPMENT DIAGNOSTIC - NOT HOLDOUT OR TRADING EVIDENCE"
    payload = json.loads(artifact.model_dump_json())
    assert payload["claim_boundary"] == artifact.claim_boundary
    assert payload["conclusion"] == artifact.conclusion.value


def test_the_conclusion_cannot_be_supplied_by_a_caller() -> None:
    """It is computed from the rules, never passed in."""
    parameters = inspect.signature(run_frozen_inference_diagnostic).parameters
    assert "conclusion" not in parameters
    assert "decision" not in parameters
    for name in parameters:
        assert "conclusion" not in name


def test_every_rule_is_recorded_even_when_it_did_not_match() -> None:
    artifact, _, _, _ = _run(lambda seed, ctx: _shift(TRUE_TARGET, 1.30))
    ids = [e.rule_id for e in artifact.decision.evaluations]
    assert ids == ["R0", "R1", "R2", "R3", "R4", "R5", "R6"]
    matched = [e for e in artifact.decision.evaluations if e.matched]
    assert len(matched) == 1
    assert matched[0].rule_id == artifact.decision.matched_rule_id


def test_the_specification_hash_is_verified_and_recorded() -> None:
    artifact, _, _, _ = _run(lambda seed, ctx: _shift(TRUE_TARGET, 1.01))
    assert artifact.specification_name == "phase2-frozen-inference-diagnostic.yaml"
    assert artifact.specification_sha256 == (
        "c909e156a61ac5b5323de9e794aad20521839c0f43f491fb356c1968f730101b"
    )


def test_a_tampered_specification_fails_closed(tmp_path: Path) -> None:
    fake_root = tmp_path / "research"
    fake_root.mkdir()
    (fake_root / "phase2-frozen-inference-diagnostic.yaml").write_text(
        "schema: tampered\n", encoding="utf-8"
    )
    codec = FakeCodec()
    with pytest.raises(BridgeTransformError) as excinfo:
        run_frozen_inference_diagnostic(
            provider=_SingleWindowProvider(),
            codec=codec,
            model=FakeForecastModel(
                codec=codec, path_for=lambda seed, ctx: _shift(TRUE_TARGET, 1.0)
            ),
            assets=fake_assets(),
            invocation=_invocation(),
            research_root=fake_root,
            parameter_digest=frozen_digest(),
            now=NOW,
        )
    assert excinfo.value.failures[0].code == "DIAGNOSTIC_SPECIFICATION_HASH_MISMATCH"


def test_raw_tokens_and_raw_outputs_are_preserved() -> None:
    artifact, _, _, _ = _run(lambda seed, ctx: _shift(TRUE_TARGET, 1.01))
    assert len(artifact.method_a.coarse_token_ids) == TOTAL_CANDLES
    assert len(artifact.method_b.coarse_token_ids) == TARGET_CANDLES
    assert len(artifact.method_b.step_log_probabilities) == TARGET_CANDLES
    assert len(artifact.method_b.raw_decoded) == TARGET_CANDLES
    assert len(artifact.method_d.rollouts) == 64
    for rollout in artifact.method_d.rollouts:
        assert len(rollout.coarse_token_ids) == TARGET_CANDLES
        assert len(rollout.raw_decoded) == TARGET_CANDLES


def test_beam_search_is_not_implemented() -> None:
    """Contingent on the diagnostic's result; building it now would be speculative."""
    for module in _diagnostic_modules():
        source = inspect.getsource(module).lower()
        assert "beam" not in source
