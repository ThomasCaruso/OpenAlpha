"""The runtime probe: a compatibility observation that authorizes nothing.

Executed with doubles. No network, no official asset, no Torch, no market
data.
"""

from __future__ import annotations

import ast
import inspect
import typing
from datetime import UTC, datetime
from pathlib import Path

import pytest
from diagnostic_fakes import FakeCodec, FakeForecastModel, fake_assets, frozen_digest
from openalpha_bridge.diagnostic import runtime_probe as probe_module
from openalpha_bridge.diagnostic.artifact import (
    DIAGNOSTIC_ARTIFACT_NAME,
    DIAGNOSTIC_SUCCESS_CODE,
    diagnostic_artifact_key,
)
from openalpha_bridge.diagnostic.conclusion import DiagnosticConclusion
from openalpha_bridge.diagnostic.official_input import OFFICIAL_COLUMNS, OfficialRow
from openalpha_bridge.diagnostic.runner import DIAGNOSTIC_SCHEMA_VERSION
from openalpha_bridge.diagnostic.runtime_probe import (
    PROBE_ARTIFACT_NAME,
    PROBE_OUTCOME_PASSED,
    PROBE_SCHEMA_VERSION,
    RuntimeEnvironment,
    RuntimeProbeResult,
    run_frozen_inference_runtime_probe,
    synthetic_probe_context,
)
from openalpha_bridge.errors import BridgeTransformError

APP = Path(__file__).resolve().parents[3] / "cloud" / "modal" / "bridge_phase2_app.py"
NOW = datetime(2026, 8, 2, tzinfo=UTC)
COMMIT = "a" * 40


def _environment(**overrides) -> RuntimeEnvironment:
    fields = {
        "torch_version": "2.13.0+cu126",
        "torch_cuda_version": "12.6",
        "cuda_available": True,
        "device_name": "NVIDIA A10G",
        "device_capability": "8.6",
        "numpy_version": "2.5.1",
    }
    fields.update(overrides)
    return RuntimeEnvironment(**fields)


def _run(*, environment=None, assets=None, digest=None, steps: int = 1):
    codec = FakeCodec()
    model = FakeForecastModel(
        codec=codec,
        path_for=lambda seed, ctx: tuple(
            OfficialRow(
                session=ctx[-1].session,
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=1.0e6,
                amount=1.005e8,
            )
            for _ in range(steps)
        ),
    )
    return run_frozen_inference_runtime_probe(
        codec=codec,
        model=model,
        assets=assets or fake_assets(),
        parameter_digest=digest or frozen_digest(),
        environment=environment or _environment(),
        run_id="runtime_probe",
        deployed_commit=COMMIT,
        steps=steps,
        now=NOW,
    )


# ============================================================ what it checks


def test_the_probe_passes_and_reports_the_environment() -> None:
    result = _run()
    assert result.outcome == PROBE_OUTCOME_PASSED
    assert result.schema_version == PROBE_SCHEMA_VERSION
    assert result.environment.torch_version == "2.13.0+cu126"
    assert result.environment.torch_cuda_version == "12.6"
    assert result.environment.device_name == "NVIDIA A10G"
    assert result.environment.cuda_available is True


def test_the_probe_reports_the_source_and_asset_hashes() -> None:
    result = _run()
    assert result.assets.source_revision == "67b630e67f6a18c9e9be918d9b4337c960db1e9a"
    assert set(result.assets.source_as_committed_sha256) == {
        "model/kronos.py",
        "model/module.py",
    }
    assert set(result.assets.source_crlf_normalized_sha256) == {
        "model/kronos.py",
        "model/module.py",
    }
    assert result.assets.model_weights_sha256
    assert result.assets.tokenizer_weights_sha256


def test_absent_cuda_fails_closed() -> None:
    with pytest.raises(BridgeTransformError) as excinfo:
        _run(environment=_environment(cuda_available=False, device_name=None))
    assert excinfo.value.failures[0].code == "RUNTIME_PROBE_NO_CUDA"


def test_trainable_parameters_fail_closed() -> None:
    with pytest.raises(BridgeTransformError) as excinfo:
        _run(assets=fake_assets(trainable=17_605))
    assert excinfo.value.failures[0].code == "RUNTIME_PROBE_PARAMETERS_NOT_FROZEN"


def test_a_moved_parameter_fails_closed() -> None:
    digests = iter(["a" * 64, "b" * 64])
    with pytest.raises(BridgeTransformError) as excinfo:
        _run(digest=lambda: next(digests))
    assert excinfo.value.failures[0].code == "RUNTIME_PROBE_PARAMETERS_MODIFIED"


def test_the_parameter_hash_is_unchanged_on_success() -> None:
    result = _run()
    assert result.parameter_sha256_before == result.parameter_sha256_after
    assert result.parameters_unmodified is True
    assert result.trainable_parameter_count == 0
    assert result.total_parameter_count > 0


def test_the_whole_path_is_exercised() -> None:
    result = _run()
    assert result.synthetic_context_rows == 64
    assert result.ordered_columns == OFFICIAL_COLUMNS
    assert result.normalization_state.fitted_candle_count == 64
    assert result.encoded_token_count == 64
    assert result.generated_steps == 1
    assert result.decoded_window_length == 65
    assert result.decoded_suffix_rows == 1
    assert result.decoded_channels == 6


def test_token_bounds_are_verified() -> None:
    result = _run()
    low, high = result.coarse_token_range
    assert 0 <= low <= high < 1024
    low, high = result.fine_token_range
    assert 0 <= low <= high < 1024


def test_measurements_are_recorded() -> None:
    result = _run()
    assert result.elapsed_seconds >= 0.0
    assert result.gpu_measurement.cuda_available in (True, False)
    assert result.completed_at == NOW


# =========================================================== what it is not


def test_the_probe_retrieves_no_market_data() -> None:
    result = _run()
    assert result.market_data_retrieved is False
    source = inspect.getsource(probe_module)
    for forbidden in ("Phase2Provider", "RetrievalRequest", "YahooDaily", "fetch("):
        assert forbidden not in source


def test_the_context_is_synthetic_and_deterministic() -> None:
    first = synthetic_probe_context()
    second = synthetic_probe_context()
    assert first == second
    assert len(first) == 64
    assert all(len(row.channels()) == 6 for row in first)


def test_the_probe_writes_no_diagnostic_conclusion() -> None:
    result = _run()
    assert result.diagnostic_conclusion_written is False
    assert not hasattr(result, "conclusion")
    assert "conclusion" not in RuntimeProbeResult.model_fields
    source = inspect.getsource(probe_module)
    assert "DiagnosticConclusion" not in source
    assert "decide(" not in source


def test_the_probe_authorizes_nothing_including_the_diagnostic() -> None:
    result = _run()
    hints = typing.get_type_hints(RuntimeProbeResult, include_extras=True)
    for field in (
        "market_data_retrieved",
        "diagnostic_conclusion_written",
        "scientific_result_available",
        "authorizes_training",
        "authorizes_stage_b",
        "authorizes_stage_c",
        "authorizes_test_opening",
        "authorizes_the_frozen_diagnostic",
        "authorizes_production_inference",
        "authorizes_trading_claims",
    ):
        assert hints[field] == typing.Literal[False]
        assert getattr(result, field) is False


def test_the_probe_carries_its_own_claim_boundary() -> None:
    result = _run()
    assert result.claim_boundary == (
        "RUNTIME COMPATIBILITY PROBE - NOT A DIAGNOSTIC AND NOT EMPIRICAL EVIDENCE"
    )
    assert "DEVELOPMENT DIAGNOSTIC" not in result.claim_boundary


# ============================== it can never be confused with the diagnostic


def test_the_probe_outcome_is_not_a_diagnostic_outcome() -> None:
    assert PROBE_OUTCOME_PASSED != DIAGNOSTIC_SUCCESS_CODE
    assert PROBE_OUTCOME_PASSED not in {c.value for c in DiagnosticConclusion}


def test_the_probe_schema_is_not_the_diagnostic_schema() -> None:
    assert PROBE_SCHEMA_VERSION != DIAGNOSTIC_SCHEMA_VERSION


def test_the_probe_artifact_name_is_not_the_diagnostic_name() -> None:
    assert PROBE_ARTIFACT_NAME != DIAGNOSTIC_ARTIFACT_NAME
    assert PROBE_ARTIFACT_NAME not in diagnostic_artifact_key("canary_0badc0de")


def test_the_probe_never_writes_the_diagnostic_key() -> None:
    source = inspect.getsource(probe_module)
    assert "diagnostic_artifact_key" not in source
    assert "publish_diagnostic_artifact" not in source
    assert "put_json" not in source
    # It returns a result; storing it is the caller's decision.
    assert "ObjectStore" not in source


def test_no_training_surface_is_reachable() -> None:
    names: set[str] = set()
    for node in ast.walk(ast.parse(inspect.getsource(probe_module))):
        if isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Name):
                names.add(target.id)
            elif isinstance(target, ast.Attribute):
                names.add(target.attr)
    for forbidden in ("backward", "zero_grad", "step", "Adam", "run_training", "CloudRunner"):
        assert forbidden not in names


# ================================================== the Modal function


def _app_source() -> str:
    return APP.read_text(encoding="utf-8")


def _probe_function() -> str:
    source = _app_source()
    tree = ast.parse(source)
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "verify_frozen_inference_runtime"
    )
    return ast.get_source_segment(source, function) or ""


def test_the_modal_probe_exists_and_is_its_own_function() -> None:
    tree = ast.parse(_app_source())
    names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert "verify_frozen_inference_runtime" in names
    assert "frozen_inference_diagnostic" in names
    assert "verify_deployment" in names


def test_the_modal_probe_restricts_both_downloads() -> None:
    body = _probe_function()
    assert body.count("snapshot_download(") == 2
    assert body.count("allow_patterns=wanted") == 2
    assert 'wanted = ["config.json", "model.safetensors"]' in body


def test_the_modal_probe_requires_cuda_before_doing_anything() -> None:
    body = _probe_function()
    assert "torch.cuda.is_available()" in body
    assert "RUNTIME_PROBE_NO_CUDA" in body
    assert body.index("torch.cuda.is_available()") < body.index("snapshot_download(")


def test_the_modal_probe_verifies_assets_and_freezes() -> None:
    body = _probe_function()
    assert "load_official_components(" in body
    assert "isolated_official_source(source_root)" in body
    assert "parameter_digest(tokenizer, model)" in body


def test_the_modal_probe_retrieves_no_market_data() -> None:
    body = _probe_function()
    for forbidden in ("YahooDailyProvider", "RetrievalRequest", "run_diagnostic_worker"):
        assert forbidden not in body


def test_the_modal_probe_writes_no_artifact() -> None:
    body = _probe_function()
    assert "_build_store()" not in body
    assert "put_json" not in body
    assert "publish_diagnostic_artifact" not in body
    assert 'return result.model_dump(mode="json")' in body
