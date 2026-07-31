from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from openalpha_sentinel.contracts import SOURCE_REVISION
from openalpha_sentinel.development_execution import (
    RealDevelopmentExecutor,
    run_reproducibility_probe,
)
from openalpha_sentinel.development_manifest import build_development_manifest
from openalpha_sentinel.development_pipeline import (
    analyze_development_state,
    freeze_development_state,
    report_development_state,
    verify_development_state,
)
from openalpha_sentinel.development_runner import (
    DevelopmentRunner,
    PilotGate,
    preflight,
    verify_offline_state,
)
from openalpha_sentinel.inference_cache import InferenceCache
from openalpha_sentinel.market_data import default_yfinance_provider
from openalpha_sentinel.providers.kronos import KronosSubprocessClient

OPERATIONS = (
    'preflight',
    'pilot',
    'run',
    'analyze',
    'freeze',
    'report',
    'verify',
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run the locked Sentinel v0 Phase 3A development sample'
    )
    parser.add_argument('operation', choices=OPERATIONS)
    parser.add_argument(
        '--state-root',
        type=Path,
        default=Path.home() / '.cache' / 'openalpha-sentinel' / 'phase3a',
    )
    parser.add_argument(
        '--inference-python',
        type=Path,
        default=(
            Path.home()
            / '.cache'
            / 'openalpha-sentinel'
            / 'phase2'
            / 'venv'
            / 'Scripts'
            / 'python.exe'
        ),
    )
    parser.add_argument(
        '--source-path',
        type=Path,
        default=Path.home() / '.cache' / 'openalpha-sentinel' / 'phase2' / 'kronos-src',
    )
    parser.add_argument(
        '--model-cache',
        type=Path,
        default=Path.home() / '.cache' / 'openalpha-sentinel' / 'phase2' / 'hf',
    )
    return parser


def build_real_executor(args: argparse.Namespace):
    if args.operation not in {'pilot', 'run'}:
        raise ValueError('real executor is limited to pilot and run operations')
    verify_offline_state(args.state_root)
    repository_root = Path(__file__).resolve().parents[1]
    required_paths = (
        args.inference_python,
        args.source_path,
        args.model_cache,
        repository_root / 'scripts' / 'kronos_inference_worker.py',
    )
    if any(not path.exists() for path in required_paths):
        missing = [str(path) for path in required_paths if not path.exists()]
        raise ValueError(f'missing real inference dependency: {missing}')
    source_revision = subprocess.run(
        ['git', '-C', str(args.source_path), 'rev-parse', 'HEAD'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if source_revision != SOURCE_REVISION:
        raise ValueError('cached official Kronos source revision does not match lock')
    provider = default_yfinance_provider()
    client = KronosSubprocessClient(
        python_executable=args.inference_python,
        worker_script=repository_root / 'scripts' / 'kronos_inference_worker.py',
        source_path=args.source_path,
        cache_path=args.model_cache,
    )
    inference_cache = InferenceCache(args.state_root / 'inference-cache')
    manifest = build_development_manifest()

    def execute() -> dict[str, object]:
        probe = run_reproducibility_probe(
            origin=manifest.origins[0],
            state_root=args.state_root,
            market_provider=provider,
            inference_client=client,
            inference_cache=inference_cache,
            clock=lambda: datetime.now(UTC),
        )
        origin_executor = RealDevelopmentExecutor(
            state_root=args.state_root,
            repository_root=repository_root,
            market_provider=provider,
            inference_client=client,
            inference_cache=inference_cache,
            inference_environment=probe.environment,
            model_cache=args.model_cache,
            source_revision=source_revision,
        )
        runner = DevelopmentRunner(
            args.state_root,
            terminal_verifier=origin_executor.verify_terminal_summary,
        )
        if args.operation == 'pilot':
            receipt = runner.run_pilot(
                manifest.origins,
                origin_executor,
                cache_bytes=origin_executor.cache_bytes,
                hosted_cost_per_cutoff=0.0,
                deterministic_replay_supported=(
                    probe.deterministic_replay_supported
                ),
            )
            return {
                'operation': 'pilot',
                'probe_receipt_sha256': probe.receipt_sha256,
                **receipt.model_dump(mode='json'),
            }
        gate_path = args.state_root / 'pilot-gate.json'
        if not gate_path.is_file():
            raise ValueError('run requires a completed operational pilot gate')
        gate = PilotGate.model_validate_json(gate_path.read_bytes())
        if not gate.continue_automatically:
            raise ValueError('run is prohibited because the pilot gate failed')
        summaries = runner.run_origins(manifest.origins, origin_executor)
        return {
            'operation': 'run',
            'probe_receipt_sha256': probe.receipt_sha256,
            'terminal_origin_count': len(summaries),
            'completed_origin_count': sum(
                item.terminal_status == 'completed' for item in summaries
            ),
            'failed_origin_count': sum(
                item.terminal_status == 'failed' for item in summaries
            ),
        }

    return execute


def dispatch(args: argparse.Namespace) -> dict[str, object]:
    if args.operation == 'preflight':
        return preflight(args.state_root).model_dump(mode='json')
    if args.operation == 'analyze':
        return analyze_development_state(args.state_root)
    if args.operation == 'freeze':
        return freeze_development_state(args.state_root)
    if args.operation == 'report':
        return report_development_state(args.state_root)
    if args.operation == 'verify':
        if (args.state_root / 'analysis-receipt.json').is_file():
            return verify_development_state(args.state_root)
        return verify_offline_state(args.state_root)
    executor = build_real_executor(args)
    return executor()


def main() -> int:
    result = dispatch(build_parser().parse_args())
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
