from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

from openalpha_sentinel.contracts import OHLCVObservation
from openalpha_sentinel.phase3b import (
    EXPERIMENT_SHA256,
    build_constrained_worker_requests,
    locked_phase3b_origins,
)


def test_phase3b_origins_are_fixed_evenly_distributed_and_pre_holdout() -> None:
    origins = locked_phase3b_origins()

    assert len(origins) == 12
    assert {item.asset for item in origins} == {"SPY", "QQQ"}
    assert {item.cutoff.isoformat() for item in origins} == {
        "2024-07-05",
        "2024-09-13",
        "2024-11-22",
        "2025-01-31",
        "2025-04-11",
        "2025-06-20",
    }
    assert all(item.forecast_sessions[-1] < date(2025, 7, 1) for item in origins)
    assert len(EXPERIMENT_SHA256) == 64


def test_worker_requests_use_only_last_512_causal_rows_and_three_seeds() -> None:
    origin = locked_phase3b_origins()[0]
    observations = tuple(
        OHLCVObservation(
            session=date(2022, 1, 1).fromordinal(date(2022, 1, 1).toordinal() + index),
            timestamp=datetime.fromordinal(
                date(2022, 1, 1).toordinal() + index
            ).replace(tzinfo=UTC),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1_000.0,
        )
        for index in range(520)
    )
    observations = (
        *observations[:-1],
        observations[-1].model_copy(deep=True),
    )
    final = observations[-1]
    observations = (
        *observations[:-1],
        OHLCVObservation(
            session=origin.cutoff,
            timestamp=datetime.combine(origin.cutoff, datetime.min.time(), tzinfo=UTC),
            open=final.open,
            high=final.high,
            low=final.low,
            close=final.close,
            volume=final.volume,
        ),
    )

    requests = build_constrained_worker_requests(
        origin=origin,
        observations=observations,
    )

    assert tuple(item["sampling_seed"] for item in requests) == (1729, 2027, 7919)
    assert all(len(item["observations"]) == 512 for item in requests)
    assert all(
        item["observations"][-1]["timestamp"].startswith(origin.cutoff.isoformat())
        for item in requests
    )
    assert all(
        all("amount" not in row for row in item["observations"]) for item in requests
    )
    assert requests[0]["official_parity_probe"] is True
    assert requests[1]["official_parity_probe"] is False
    assert requests[2]["official_parity_probe"] is False


def test_phase3b_manifest_and_compact_origin_summaries_verify() -> None:
    root = Path(__file__).resolve().parents[3]
    manifest = json.loads(
        (root / "research/sentinel-v1/feasibility/manifest.json").read_text(
            encoding="utf-8"
        )
    )

    for item in (*manifest["artifacts"], *manifest["implementation"]):
        payload = (root / item["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == item["sha256"]

    summary_path = root / "research/sentinel-v1/feasibility/origin_summaries.jsonl"
    summaries = [json.loads(line) for line in summary_path.read_text().splitlines()]
    assert len(summaries) == 12
    assert all(item["holdout_accessed"] is False for item in summaries)
    assert all(
        "path" not in record
        for item in summaries
        for record in item["method_records"]
    )
