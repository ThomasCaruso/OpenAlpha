"""Worker-level tests: the canary worker executed, not merely read.

Every test here runs the real ``run_canary_worker`` with a fake provider and a
fake official backend. No network, no official assets, no real market data, and
no held-out partition is touched.
"""

from __future__ import annotations

import ast
import inspect
import json
import typing
from datetime import UTC, datetime
from pathlib import Path

import pytest
from openalpha_bridge.cloud.objectstore import InMemoryObjectStore, run_prefix
from openalpha_bridge.errors import BridgeTransformError
from openalpha_bridge.phase2 import canary_worker as worker_module
from openalpha_bridge.phase2.canary import CANARY_SUCCESS_CODE
from openalpha_bridge.phase2.canary_artifact import (
    CANARY_FAILURE_CODE,
    CANARY_OPERATIONAL_FAILURE_CODE,
    terminal_artifact_key,
)
from openalpha_bridge.phase2.canary_worker import CanaryWorkerResult, run_canary_worker
from openalpha_bridge.phase2.kronos import DeterministicFakeKronosBackend
from openalpha_bridge.phase2.states import EvidenceClass
from test_stage_a_canary import RESEARCH, _CountingProvider, _PseudoOfficialBackend

NOW = datetime(2026, 8, 2, tzinfo=UTC)
COMMIT = "a" * 40
RUN_ID = "canary_0badc0de"

MODAL_APP = Path(__file__).resolve().parents[3] / "cloud" / "modal" / "bridge_phase2_app.py"


def _run(tmp_path: Path, **overrides):
    store = overrides.pop("store", None) or InMemoryObjectStore()
    provider = overrides.pop("provider", None) or _CountingProvider()
    result = run_canary_worker(
        store=store,
        provider=provider,
        backend=overrides.pop("backend", None) or _PseudoOfficialBackend(),
        feature_cache_root=tmp_path / "cache",
        asset_cache_root=overrides.pop("asset_cache_root", tmp_path / "hf"),
        research_root=RESEARCH,
        source_commit=overrides.pop("source_commit", COMMIT),
        deployed_commit=overrides.pop("deployed_commit", COMMIT),
        run_id=overrides.pop("run_id", RUN_ID),
        now=NOW,
        **overrides,
    )
    return store, provider, result


# ------------------------------------------------------- evidence and roots


def test_the_worker_succeeds_and_classifies_its_evidence(tmp_path: Path) -> None:
    _, _, result = _run(tmp_path)
    assert result.outcome == CANARY_SUCCESS_CODE
    assert result.evidence_class is EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY
    assert result.already_existed is False
    assert result.deployed_commit == COMMIT


def test_the_artifact_lands_under_the_compatibility_root_only(tmp_path: Path) -> None:
    store, _, result = _run(tmp_path)
    expected = run_prefix(RUN_ID, EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY)
    assert result.artifact_key == terminal_artifact_key(RUN_ID)
    assert result.artifact_key.startswith(expected + "/")
    assert store.list_keys("openalpha/") == ()
    assert store.list_keys("openalpha-synthetic/") == ()
    assert len(store.list_keys("openalpha-compatibility/")) == 1


def test_the_result_authorizes_nothing(tmp_path: Path) -> None:
    _, _, result = _run(tmp_path)
    assert result.authorizes_stage_b is False
    assert result.authorizes_real_run is False
    assert result.scientific_result_available is False

    hints = typing.get_type_hints(CanaryWorkerResult, include_extras=True)
    assert hints["authorizes_stage_b"] == typing.Literal[False]
    assert hints["authorizes_real_run"] == typing.Literal[False]
    assert hints["scientific_result_available"] == typing.Literal[False]
    assert hints["evidence_class"] == typing.Literal[EvidenceClass.DEVELOPMENT_COMPATIBILITY_CANARY]


# --------------------------------------------------------- deployed commit


@pytest.mark.parametrize("commit", ["", "HEAD", "a" * 39, "a" * 41, "A" * 40, "g" * 40])
def test_the_worker_refuses_a_malformed_deployed_commit(tmp_path: Path, commit: str) -> None:
    with pytest.raises(BridgeTransformError) as excinfo:
        _run(tmp_path, deployed_commit=commit)
    assert excinfo.value.failures[0].code == "CANARY_WORKER_INVALID_DEPLOYED_COMMIT"


def test_the_deployed_commit_cannot_be_omitted() -> None:
    binding = inspect.signature(run_canary_worker).parameters["deployed_commit"]
    assert binding.default is inspect.Parameter.empty
    assert binding.kind is inspect.Parameter.KEYWORD_ONLY


def test_a_commit_that_does_not_match_the_image_fails_and_is_preserved(tmp_path: Path) -> None:
    store, _, result = _run(tmp_path, source_commit="b" * 40, deployed_commit=COMMIT)
    assert result.outcome == CANARY_FAILURE_CODE
    assert result.report["failure_code"] == "CANARY_SOURCE_COMMIT_MISMATCH"
    assert store.exists(result.artifact_key)


# ------------------------------------------------------------ failure paths


def test_a_contract_failure_is_stored_as_a_typed_failure(tmp_path: Path) -> None:
    """A non-official backend is a contract violation, not an outage."""
    store, _, result = _run(tmp_path, backend=DeterministicFakeKronosBackend())
    assert result.outcome == CANARY_FAILURE_CODE
    assert result.report["failure_code"] == "CANARY_REQUIRES_OFFICIAL_BACKEND"
    assert store.exists(result.artifact_key)


def test_an_outage_is_stored_as_an_operational_failure(tmp_path: Path) -> None:
    class _DeadProvider(_CountingProvider):
        def fetch(self, request):
            raise ConnectionResetError("provider unreachable")

    store, _, result = _run(tmp_path, provider=_DeadProvider())
    assert result.outcome == CANARY_OPERATIONAL_FAILURE_CODE
    assert result.outcome != CANARY_FAILURE_CODE
    assert store.exists(result.artifact_key)


def test_a_failed_worker_still_leaves_exactly_one_artifact(tmp_path: Path) -> None:
    store, _, result = _run(tmp_path, backend=DeterministicFakeKronosBackend())
    assert len(store.list_keys("openalpha-compatibility/")) == 1
    assert result.report["authorizes_stage_b"] is False


# ------------------------------------------------------- duplicate invocation


def test_a_second_invocation_verifies_and_does_not_overwrite(tmp_path: Path) -> None:
    store, _, first = _run(tmp_path)
    original = store.get(first.artifact_key).body

    _, _, second = _run(tmp_path, store=store, backend=DeterministicFakeKronosBackend())

    assert second.already_existed is True
    assert second.artifact_key == first.artifact_key
    assert second.content_sha256 == first.content_sha256
    assert second.outcome == CANARY_SUCCESS_CODE
    assert store.get(first.artifact_key).body == original
    assert len(store.list_keys("openalpha-compatibility/")) == 1


def test_a_duplicate_does_not_re_run_a_failure_into_a_pass(tmp_path: Path) -> None:
    store, _, first = _run(tmp_path, backend=DeterministicFakeKronosBackend())
    _, _, second = _run(tmp_path, store=store)
    assert first.outcome == CANARY_FAILURE_CODE
    assert second.outcome == CANARY_FAILURE_CODE
    assert second.already_existed is True


# --------------------------------------------------- measurements are stored


def test_the_stored_report_carries_the_measured_provider_count(tmp_path: Path) -> None:
    store, provider, result = _run(tmp_path)
    stored = json.loads(store.get(result.artifact_key).body)
    assert stored["provider_request_count"] == len(provider.requests) == 1


def test_the_observed_retrievals_survive_serialization(tmp_path: Path) -> None:
    store, _, result = _run(tmp_path)
    calls = json.loads(store.get(result.artifact_key).body)["provider_calls"]
    assert calls == [
        {
            "symbol": "SPY",
            "interval": "1d",
            "start": "2015-05-07",
            "end_exclusive": "2017-05-18",
            "maximum_candles": 512,
            "returned_candles": 512,
        }
    ]


def test_the_stored_report_carries_measurements_not_estimates(tmp_path: Path) -> None:
    asset_root = tmp_path / "hf"
    asset_root.mkdir()
    (asset_root / "preexisting.bin").write_bytes(b"\x00" * 2048)
    store, _, result = _run(tmp_path, asset_cache_root=asset_root)
    stored = json.loads(store.get(result.artifact_key).body)

    measured = stored["cache_measurement"]
    assert measured["asset_cache_bytes_before"] == 2048
    assert measured["feature_cache_bytes_after"] > 0
    assert measured["new_shard_bytes"] == stored["shard_bytes"]

    timing = stored["timing"]
    assert timing["estimated_monetary_cost"] is None
    assert timing["total_worker_wall_seconds"] >= 0.0
    assert stored["gpu_measurement"]["cuda_available"] in (True, False)


def test_the_observed_component_manifest_survives_serialization(tmp_path: Path) -> None:
    store, _, result = _run(tmp_path)
    observed = json.loads(store.get(result.artifact_key).body)["observed_components"]
    assert observed["projection"]["in_features"] == 20
    assert observed["projection"]["out_features"] == 256
    assert observed["decoder_block_count"] == 3
    assert observed["trainable_parameters"] == 0
    assert observed["encode"]["batched_shapes"] == [[1, 512], [1, 512]]


# ------------------------------------------------ no path into later stages


def _worker_call_graph() -> set[str]:
    """Every name called anywhere in the worker module."""
    tree = ast.parse(inspect.getsource(worker_module))
    names: set[str] = set()
    for node in ast.walk(tree):
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
        "stage_b",
        "stage_c",
        "run_training",
        "select_checkpoint",
        "freeze_checkpoint",
        "evaluate_conclusion",
        "open_test_partition",
        "open_cloud_test_partition",
        "GateTable",
        "TestOpeningPreconditions",
        "CloudRunner",
        "Phase2Pipeline",
    ],
)
def test_the_worker_has_no_path_into_later_stages(forbidden: str) -> None:
    assert forbidden not in _worker_call_graph()
    # `authorizes_stage_b` legitimately names a stage it refuses to authorize,
    # so the source check ignores the authorization fields and looks for a real
    # reference: a call, an import, or an attribute access.
    source = inspect.getsource(worker_module)
    for line in source.splitlines():
        if "authorizes_" in line or "scientific_result_available" in line:
            continue
        assert forbidden not in line, f"worker references {forbidden}: {line.strip()}"


def test_the_worker_never_reaches_a_held_out_partition() -> None:
    source = inspect.getsource(worker_module)
    for partition in ("validation", "reconstruction_test", "external_later", "RECONSTRUCTION"):
        assert partition not in source


def test_the_run_leaves_no_training_or_checkpoint_artifact(tmp_path: Path) -> None:
    store, _, _ = _run(tmp_path)
    keys = store.list_keys("")
    assert len(keys) == 1
    for fragment in ("checkpoint", "gate", "test_opening", "stage_b", "stage_c"):
        assert not any(fragment in key for key in keys)


# ---------------------------------------------------- the Modal shell is thin


def test_the_modal_function_only_delegates() -> None:
    """If logic drifts back into the deployment file, it stops being tested."""
    source = MODAL_APP.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "stage_a_official_canary"
    )
    assert "run_canary_worker(" in source
    # Nothing that belongs in the tested worker may live in the shell.
    body = ast.get_source_segment(source, function) or ""
    for moved in ("publish_terminal_artifact", "run_stage_a_canary", "FeatureCache("):
        assert moved not in body
