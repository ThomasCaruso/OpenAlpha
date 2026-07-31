from pathlib import Path

import pytest
from openalpha_sentinel.development_manifest import build_development_manifest
from openalpha_sentinel.development_runner import (
    DevelopmentRunner,
    OriginExecutionSummary,
    PilotOriginRecord,
    evaluate_pilot,
)


def _summary(origin_id: str) -> OriginExecutionSummary:
    return OriginExecutionSummary(
        schema_version='sentinel-phase3a-origin-execution-v1',
        origin_id=origin_id,
        terminal_status='completed',
        request_success_count=9,
        request_count=9,
        ensemble_latency_seconds=12.0,
        cache_bytes=1_024,
        process_stable=True,
        terminal_record_sha256='a' * 64,
    )


def test_runner_resumes_without_repeating_terminal_origins(tmp_path: Path) -> None:
    origins = build_development_manifest().origins[:12]
    runner = DevelopmentRunner(tmp_path, terminal_verifier=lambda _origin, _marker: True)
    calls: list[str] = []

    def execute(origin):
        calls.append(origin.origin_id)
        return _summary(origin.origin_id)

    first = runner.run_origins(origins, execute, max_new=5)
    assert len(first) == 5
    assert calls == [origin.origin_id for origin in origins[:5]]

    resumed = runner.run_origins(origins, execute)
    assert len(resumed) == 12
    assert calls == [origin.origin_id for origin in origins]
    assert runner.verify_terminal_markers(origins) is True


def test_pilot_gate_enforces_locked_operational_limits() -> None:
    records = tuple(
        PilotOriginRecord(
            origin_id=f'origin-{index}',
            request_success_count=9,
            request_count=9,
            ensemble_latency_seconds=120.0,
            process_stable=True,
        )
        for index in range(10)
    )
    gate = evaluate_pilot(
        records,
        cache_bytes=32_283_678,
        hosted_cost_per_cutoff=0.0,
        deterministic_replay_supported=True,
    )
    assert gate.request_success_rate >= 0.95
    assert gate.median_ensemble_latency_seconds <= 600
    assert gate.cache_bytes <= 2_147_483_648
    assert gate.continue_automatically is True

    assert evaluate_pilot(
        records[:-1]
        + (
            PilotOriginRecord(
                origin_id='failed',
                request_success_count=0,
                request_count=20,
                ensemble_latency_seconds=120.0,
                process_stable=True,
            ),
        ),
        cache_bytes=32_283_678,
        hosted_cost_per_cutoff=0.0,
        deterministic_replay_supported=True,
    ).continue_automatically is False
    assert evaluate_pilot(
        records,
        cache_bytes=2_147_483_649,
        hosted_cost_per_cutoff=0.0,
        deterministic_replay_supported=True,
    ).continue_automatically is False
    slow = tuple(
        item.model_copy() if index else PilotOriginRecord(
            origin_id=item.origin_id,
            request_success_count=9,
            request_count=9,
            ensemble_latency_seconds=601.0,
            process_stable=True,
        )
        for index, item in enumerate(records)
    )
    assert evaluate_pilot(
        slow,
        cache_bytes=32_283_678,
        hosted_cost_per_cutoff=0.0,
        deterministic_replay_supported=True,
    ).continue_automatically is True
    unstable = tuple(
        PilotOriginRecord(
            origin_id=item.origin_id,
            request_success_count=item.request_success_count,
            request_count=item.request_count,
            ensemble_latency_seconds=item.ensemble_latency_seconds,
            process_stable=index != 0,
        )
        for index, item in enumerate(records)
    )
    assert evaluate_pilot(
        unstable,
        cache_bytes=32_283_678,
        hosted_cost_per_cutoff=0.0,
        deterministic_replay_supported=True,
    ).continue_automatically is False


def test_pilot_runs_first_ten_pending_then_continues_automatically(tmp_path: Path) -> None:
    origins = build_development_manifest().origins[:12]
    runner = DevelopmentRunner(tmp_path, terminal_verifier=lambda _origin, _marker: True)
    calls: list[str] = []

    def execute(origin):
        calls.append(origin.origin_id)
        return _summary(origin.origin_id)

    receipt = runner.run_pilot(
        origins,
        execute,
        cache_bytes=lambda: 32_283_678,
        hosted_cost_per_cutoff=0.0,
        deterministic_replay_supported=True,
    )

    assert receipt.pilot_origin_ids == tuple(origin.origin_id for origin in origins[:10])
    assert receipt.gate.continue_automatically is True
    assert receipt.continued_automatically is True
    assert receipt.terminal_origin_count == 12
    assert calls == [origin.origin_id for origin in origins]


def test_runner_refuses_to_resume_an_unverified_terminal_marker(tmp_path: Path) -> None:
    origin = build_development_manifest().origins[0]
    verified = True

    def verify_terminal(_origin, _marker):
        return verified

    runner = DevelopmentRunner(tmp_path, terminal_verifier=verify_terminal)
    runner.run_origins((origin,), lambda item: _summary(item.origin_id))
    verified = False

    with pytest.raises(ValueError, match='terminal record failed verification'):
        runner.run_origins((origin,), lambda item: _summary(item.origin_id))
