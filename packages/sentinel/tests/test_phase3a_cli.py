from pathlib import Path

import pytest

from scripts import run_sentinel_phase3a


def test_cli_exposes_only_locked_operations_and_no_date_override() -> None:
    parser = run_sentinel_phase3a.build_parser()
    operation = next(
        action for action in parser._actions if action.dest == 'operation'
    )
    assert operation.choices is not None
    assert tuple(operation.choices) == (
        'preflight',
        'pilot',
        'run',
        'analyze',
        'freeze',
        'report',
        'verify',
    )
    with pytest.raises(SystemExit):
        parser.parse_args(['preflight', '--cutoff', '2025-07-01'])
    with pytest.raises(SystemExit):
        parser.parse_args(['preflight', '--period', 'holdout'])


def test_verify_is_offline_and_does_not_build_real_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_sentinel_phase3a.preflight(tmp_path)

    def fail_executor(*args, **kwargs):
        raise AssertionError('verify attempted to construct network or inference dependencies')

    monkeypatch.setattr(run_sentinel_phase3a, 'build_real_executor', fail_executor)
    result = run_sentinel_phase3a.dispatch(
        run_sentinel_phase3a.build_parser().parse_args(
            ['verify', '--state-root', str(tmp_path)]
        )
    )

    assert result['manifest_verified'] is True
    assert result['network_accessed'] is False
    assert result['inference_accessed'] is False


def test_analysis_freeze_and_report_are_offline_operations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fail_executor(*args, **kwargs):
        raise AssertionError('offline operation attempted to construct real dependencies')

    monkeypatch.setattr(run_sentinel_phase3a, 'build_real_executor', fail_executor)
    monkeypatch.setattr(
        run_sentinel_phase3a,
        'analyze_development_state',
        lambda root: calls.append('analyze') or {'operation': 'analyze'},
    )
    monkeypatch.setattr(
        run_sentinel_phase3a,
        'freeze_development_state',
        lambda root: calls.append('freeze') or {'operation': 'freeze'},
    )
    monkeypatch.setattr(
        run_sentinel_phase3a,
        'report_development_state',
        lambda root: calls.append('report') or {'operation': 'report'},
    )

    for operation in ('analyze', 'freeze', 'report'):
        result = run_sentinel_phase3a.dispatch(
            run_sentinel_phase3a.build_parser().parse_args(
                [operation, '--state-root', str(tmp_path)]
            )
        )
        assert result['operation'] == operation
    assert calls == ['analyze', 'freeze', 'report']
