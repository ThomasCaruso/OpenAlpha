from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest


def _worker() -> ModuleType:
    path = Path("scripts/kronos_token_manifold_worker.py")
    spec = importlib.util.spec_from_file_location("kronos_token_manifold_worker", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load token-manifold worker")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _observations() -> list[dict[str, object]]:
    return [
        {
            "timestamp": f"2024-01-{(index % 28) + 1:02d}T21:00:00+00:00",
            "open": 100.0,
            "high": 102.0,
            "low": 99.0,
            "close": 101.0,
            "volume": 10.0,
        }
        for index in range(512)
    ]


def _request() -> dict[str, object]:
    return {
        "schema_version": "sentinel-kronos-token-manifold-worker-v1",
        "operation": "compatibility",
        "experiment_sha256": (
            "6dacd9fd0912a77cce9d373d910b7bff"
            "9d58b44076d814960a191cdbe183bd76"
        ),
        "origin_id": "sentinel-v1-SPY-2024-07-05-h5",
        "symbol": "SPY",
        "cutoff": "2024-07-05",
        "model_repository": "NeoQuasar/Kronos-mini",
        "model_revision": "f4e68697d9d5aed55cef5c96aabc3376bcad9f81",
        "tokenizer_repository": "NeoQuasar/Kronos-Tokenizer-2k",
        "tokenizer_revision": "26966d0035065a0cae0ebad7af8ece35bc1fb51c",
        "source_revision": "67b630e67f6a18c9e9be918d9b4337c960db1e9a",
        "context_length": 512,
        "forecast_horizon": 5,
        "sampling_seed": 1729,
        "temperature": 1.0,
        "top_p": 0.9,
        "top_k": 0,
        "sample_count": 1,
        "candidate_budgets": [64, 256, 1024],
        "support_coarse_counts": {"1": 2},
        "support_fine_counts": {"2": 2},
        "support_pair_counts": {"1:2": 2},
        "forecast_sessions": [
            "2024-07-08",
            "2024-07-09",
            "2024-07-10",
            "2024-07-11",
            "2024-07-12",
        ],
        "observations": _observations(),
    }


def _roundtrip_request() -> dict[str, Any]:
    windows = []
    for symbol_index, symbol in enumerate(
        ("SPY", "QQQ", "IWM", "DIA", "TLT", "HYG", "GLD", "EFA", "EEM", "XLF")
    ):
        for window_index in range(3):
            windows.append(
                {
                    "window_id": f"{symbol}-{window_index + 1}",
                    "symbol": symbol,
                    "observations": _observations(),
                }
            )
    return {
        "schema_version": "sentinel-kronos-token-manifold-worker-v1",
        "operation": "roundtrip",
        "experiment_sha256": (
            "6dacd9fd0912a77cce9d373d910b7bff"
            "9d58b44076d814960a191cdbe183bd76"
        ),
        "source_revision": "67b630e67f6a18c9e9be918d9b4337c960db1e9a",
        "tokenizer_repository": "NeoQuasar/Kronos-Tokenizer-2k",
        "tokenizer_revision": "26966d0035065a0cae0ebad7af8ece35bc1fb51c",
        "windows": windows,
    }


def test_worker_rejects_unlocked_budgets_and_unknown_fields() -> None:
    worker = _worker()
    bad_budget = _request()
    bad_budget["candidate_budgets"] = [2048]
    with pytest.raises(ValueError, match="candidate budgets"):
        worker._validate_request(bad_budget)

    unknown = {**_request(), "cutoff_override": "2025-07-01"}
    with pytest.raises(ValueError, match="unknown fields"):
        worker._validate_request(unknown)


def test_worker_requires_the_pinned_model_tokenizer_pairing() -> None:
    worker = _worker()
    request = _request()
    request["model_repository"] = "NeoQuasar/Kronos-small"
    request["model_revision"] = "901c26c1332695a2a8f243eb2f37243a37bea320"
    with pytest.raises(ValueError, match="pairing"):
        worker._validate_request(request)


def test_roundtrip_worker_requires_exactly_thirty_locked_windows() -> None:
    worker = _worker()
    request = _roundtrip_request()
    worker._validate_request(request)

    request["windows"] = request["windows"][:-1]
    with pytest.raises(ValueError, match="30"):
        worker._validate_request(request)


def test_process_boundary_preserves_typed_failure_without_model_loading(
    tmp_path: Path,
) -> None:
    worker = _worker()

    result = worker.run_payload(
        {"schema_version": "wrong"},
        source_path=tmp_path,
        cache_path=tmp_path,
    )

    assert result["status"] == "failure"
    assert result["failure"]["code"] == "WORKER_FAILED"
    assert "schema" in result["failure"]["message"].lower()


def test_compatibility_batch_is_limited_to_one_origin_and_three_seeds(
    tmp_path: Path,
) -> None:
    worker = _worker()
    requests = []
    for seed in (1729, 2027, 7919):
        request = _request()
        request["sampling_seed"] = seed
        requests.append(request)
    worker._validate_payload(
        {
            "schema_version": "sentinel-kronos-token-manifold-batch-v1",
            "requests": requests,
        }
    )

    with pytest.raises(ValueError, match="three"):
        worker._validate_payload(
            {
                "schema_version": "sentinel-kronos-token-manifold-batch-v1",
                "requests": [*requests, requests[0]],
            }
        )

    other = _request()
    other["origin_id"] = "sentinel-v1-QQQ-2024-07-05-h5"
    with pytest.raises(ValueError, match="one origin"):
        worker._validate_payload(
            {
                "schema_version": "sentinel-kronos-token-manifold-batch-v1",
                "requests": [requests[0], other],
            }
        )


def test_feature_preparation_is_named_and_derives_amount() -> None:
    worker = _worker()
    values = worker._feature_values(_observations()[:1], np)

    assert values.dtype == np.float32
    assert values.shape == (1, 6)
    assert values[0].tolist() == pytest.approx(
        [100.0, 102.0, 99.0, 101.0, 10.0, 1005.0]
    )


def test_top_p_distribution_is_normalized_and_truncated() -> None:
    worker = _worker()
    probabilities = worker._top_p_probabilities(
        np.asarray([3.0, 2.0, 1.0, -5.0]),
        top_p=0.8,
        np=np,
    )

    assert probabilities.sum() == pytest.approx(1.0)
    assert np.count_nonzero(probabilities) == 2
    assert probabilities[0] > probabilities[1] > 0.0


def test_auxiliary_fraction_replays_exactly() -> None:
    worker = _worker()
    first = worker._auxiliary_fraction(
        experiment_sha256="a" * 64,
        origin_id="sentinel-v1-SPY-2024-07-05-h5",
        seed=1729,
        step=1,
    )
    second = worker._auxiliary_fraction(
        experiment_sha256="a" * 64,
        origin_id="sentinel-v1-SPY-2024-07-05-h5",
        seed=1729,
        step=1,
    )

    assert first == second
    assert 0.0 <= first < 1.0


def test_candidate_grid_preserves_joint_probability_support_and_validity() -> None:
    worker = _worker()
    records, considered_mass = worker._candidate_grid_from_arrays(
        coarse_probabilities=np.asarray([0.6, 0.4]),
        fine_probabilities_by_coarse={
            0: np.asarray([0.75, 0.25]),
            1: np.asarray([0.50, 0.50]),
        },
        decoded_by_pair={
            "0:0": [100.0, 102.0, 99.0, 101.0, 1.0, 1.0],
            "0:1": [100.0, 100.0, 99.0, 101.0, 1.0, 1.0],
            "1:0": [100.0, 103.0, 98.0, 102.0, 1.0, 1.0],
            "1:1": [100.0, 102.0, 98.0, 101.0, 1.0, 1.0],
        },
        pair_counts={"0:0": 3, "0:1": 1, "1:0": 0, "1:1": 2},
        coarse_limit=2,
        fine_limit=2,
        np=np,
    )

    assert considered_mass == pytest.approx(1.0)
    assert [item["joint_probability"] for item in records] == pytest.approx(
        [0.45, 0.20, 0.20, 0.15]
    )
    assert records[0]["rank"] == 1
    assert records[0]["valid"] is True
    assert records[-1]["valid"] is False
    assert records[-1]["violations"] == ["HIGH_BELOW_CLOSE"]


def test_candidate_grid_uses_exact_topk_ids_when_zero_probabilities_tie() -> None:
    worker = _worker()
    valid = [100.0, 102.0, 99.0, 101.0, 1.0, 1.0]
    records, _ = worker._candidate_grid_from_arrays(
        coarse_probabilities=np.asarray([0.0, 0.4, 0.6, 0.0]),
        coarse_ids=(2, 1, 3),
        fine_probabilities_by_coarse={
            2: np.asarray([1.0]),
            1: np.asarray([1.0]),
            3: np.asarray([1.0]),
        },
        decoded_by_pair={
            "2:0": valid,
            "1:0": valid,
            "3:0": valid,
        },
        pair_counts={},
        coarse_limit=3,
        fine_limit=1,
        np=np,
    )

    assert {item["coarse_token"] for item in records} == {1, 2, 3}


def test_nested_candidate_grids_reuse_one_decoded_superset() -> None:
    worker = _worker()
    valid = [100.0, 102.0, 99.0, 101.0, 1.0, 1.0]
    grids = worker._nested_candidate_grids_from_arrays(
        coarse_probabilities=np.asarray([0.6, 0.4]),
        coarse_ids=(0, 1),
        fine_probabilities_by_coarse={
            0: np.asarray([0.7, 0.3]),
            1: np.asarray([0.8, 0.2]),
        },
        decoded_by_pair={
            "0:0": valid,
            "0:1": valid,
            "1:0": valid,
            "1:1": valid,
        },
        pair_counts={},
        sides=(1, 2),
        np=np,
    )

    assert len(grids[1][0]) == 1
    assert len(grids[2][0]) == 4
    assert grids[1][1] == pytest.approx(0.42)
    assert grids[2][1] == pytest.approx(1.0)


def test_worker_support_selection_has_no_fallback() -> None:
    worker = _worker()
    records = [
        {"valid": False, "exact_pair_count": 5, "joint_probability": 0.5},
        {"valid": True, "exact_pair_count": 1, "joint_probability": 0.3},
        {"valid": True, "exact_pair_count": 2, "joint_probability": 0.2},
    ]

    selected = worker._select_supported_record(
        records,
        auxiliary_fraction=0.25,
        minimum_pair_count=2,
    )
    assert selected is records[2]

    with pytest.raises(ValueError, match="no valid supported"):
        worker._select_supported_record(
            records[:2],
            auxiliary_fraction=0.25,
            minimum_pair_count=2,
        )
