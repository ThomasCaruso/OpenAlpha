from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PHASE2 = ROOT / "research" / "bridge-v0" / "phase2"
ORIGINAL_LOCK_SHA256 = "d52a9be733f4ec331e081346e64ab7415ac0f9510e494733746f975f114f47b2"
AMENDMENT_SHA256 = "4c60846e8cb791d922c29a5e704fe3a473740171dd2bbb7b3390796ceeeef3c8"
JSON_ARTIFACTS = (
    "data_manifest.json",
    "representation_coverage.json",
    "feature_manifest.json",
    "training_manifest.json",
    "selected_checkpoint.json",
    "reconstruction_metrics.json",
    "paired_bootstrap.json",
    "slice_analysis.json",
    "operational_metrics.json",
    "compatibility_manifest.json",
)


def _load_json(name: str) -> dict[str, Any]:
    value = json.loads((PHASE2 / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_phase2_terminal_artifacts_are_complete_and_do_not_claim_execution() -> None:
    expected = {*JSON_ARTIFACTS, "training_history.jsonl", "report.md"}
    assert PHASE2.is_dir()
    assert expected <= {path.name for path in PHASE2.iterdir() if path.is_file()}

    artifacts = {name: _load_json(name) for name in JSON_ARTIFACTS}
    for name, artifact in artifacts.items():
        assert artifact["terminal_conclusion"] == "OPERATIONALLY_BLOCKED", name
        assert artifact["scientific_result_available"] is False, name
        assert artifact["original_experiment_sha256"] == ORIGINAL_LOCK_SHA256, name
        assert artifact["phase2_amendment_sha256"] == AMENDMENT_SHA256, name

    data = artifacts["data_manifest.json"]
    assert data["provider_requests"] == 0
    assert data["retrieved_complete_candles"] == 0
    assert data["raw_or_normalized_market_files_created"] == 0

    feature = artifacts["feature_manifest.json"]
    assert feature["kronos_checkpoint_loads"] == 0
    assert feature["tokenizer_encode_calls"] == 0
    assert feature["feature_cache_created"] is False

    training = artifacts["training_manifest.json"]
    assert training["training_started"] is False
    assert training["optimizer_steps"] == 0
    assert training["expected_trainable_parameter_count"] == 17605
    assert training["observed_trainable_parameter_count"] is None

    checkpoint = artifacts["selected_checkpoint.json"]
    assert checkpoint["checkpoint_selected"] is False
    assert checkpoint["checkpoint_sha256"] is None
    assert checkpoint["checkpoint_size_bytes"] == 0

    metrics = artifacts["reconstruction_metrics.json"]
    assert metrics["test_partition_opened"] is False
    assert metrics["evaluated_sequence_count"] == 0
    assert all(value is None for value in metrics["method_metrics"].values())

    bootstrap = artifacts["paired_bootstrap.json"]
    assert bootstrap["configured_resamples"] == 10000
    assert bootstrap["executed_resamples"] == 0
    assert bootstrap["confirmation_result"] is None

    operations = artifacts["operational_metrics.json"]
    assert operations["training_runtime_seconds"] == 0.0
    assert operations["gpu_hours"] == 0.0
    assert operations["peak_training_memory_bytes"] is None

    history = (PHASE2 / "training_history.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(history) == 1
    event = json.loads(history[0])
    assert event["event"] == "PRE_DATA_GATE_TERMINAL"
    assert event["optimizer_steps"] == 0
    assert event["terminal_conclusion"] == "OPERATIONALLY_BLOCKED"

    report = (PHASE2 / "report.md").read_text(encoding="utf-8")
    assert "OPERATIONALLY_BLOCKED" in report
    assert "No market data was retrieved" in report
    assert "No Kronos checkpoint was loaded" in report
    assert "not a Bridge reconstruction-quality result" in report


def test_phase2_terminal_artifacts_reference_byte_identical_locks() -> None:
    original = ROOT / "research" / "bridge-v0" / "experiment.yaml"
    amendment = ROOT / "research" / "bridge-v0" / "phase2-preregistration-amendment.yaml"
    assert hashlib.sha256(original.read_bytes()).hexdigest() == ORIGINAL_LOCK_SHA256
    assert hashlib.sha256(amendment.read_bytes()).hexdigest() == AMENDMENT_SHA256
