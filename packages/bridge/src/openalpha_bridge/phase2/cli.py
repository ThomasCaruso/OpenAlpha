"""Phase 2 execution CLI.

    python -m openalpha_bridge.phase2 preflight  --run-dir ... --cache-dir ...
    python -m openalpha_bridge.phase2 stage-a    --run-dir ... --cache-dir ...
    python -m openalpha_bridge.phase2 run        --run-dir ... --cache-dir ...

Every command supports structured JSON output, human-readable logging, explicit
device selection, dry-run, and resume. No command opens the reconstruction-test
partition without the explicit ``--open-test-partition`` transition.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from ..errors import BridgeTransformError
from .gates import TerminalConclusion
from .kronos import DeterministicFakeKronosBackend, KronosMode, OfficialKronosBackend
from .pipeline import Phase2Config, Phase2Pipeline
from .preflight import run_preflight
from .provider import DeterministicFakeProvider, ProviderMode, YahooDailyProvider
from .states import EvidenceClass
from .training import NumpyTrainingBackend, TorchTrainingBackend

__all__ = ["build_parser", "main"]

_LOGGER = logging.getLogger("openalpha.bridge.phase2")

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_BLOCKED = 2
EXIT_USAGE = 64


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openalpha-bridge-phase2",
        description="OpenAlpha Bridge-2K Phase 2 execution pipeline",
    )
    parser.add_argument("command", choices=(
        "preflight", "stage-a", "stage-b", "stage-c", "evaluate", "verify", "run",
    ))  # fmt: skip
    parser.add_argument("--experiment", type=Path, default=Path("research/bridge-v0"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path())
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true", help="emit structured JSON")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--provider-mode",
        choices=tuple(mode.value for mode in ProviderMode),
        default=ProviderMode.REAL.value,
    )
    parser.add_argument(
        "--kronos-mode",
        choices=tuple(mode.value for mode in KronosMode),
        default=KronosMode.PINNED_OFFICIAL.value,
    )
    parser.add_argument(
        "--evidence-class",
        choices=tuple(item.value for item in EvidenceClass),
        default=EvidenceClass.REAL_PHASE2.value,
    )
    parser.add_argument(
        "--open-test-partition",
        action="store_true",
        help="explicit one-time transition that opens the reconstruction-test partition",
    )
    parser.add_argument("--operator", default="unspecified-operator")
    return parser


def _build_pipeline(args: argparse.Namespace) -> Phase2Pipeline:
    evidence_class = EvidenceClass(args.evidence_class)
    provider_mode = ProviderMode(args.provider_mode)
    kronos_mode = KronosMode(args.kronos_mode)

    provider = (
        YahooDailyProvider(stage=args.command)
        if provider_mode is ProviderMode.REAL
        else DeterministicFakeProvider()
    )
    kronos = (
        OfficialKronosBackend(stage=args.command, device=args.device)
        if kronos_mode is KronosMode.PINNED_OFFICIAL
        else DeterministicFakeKronosBackend()
    )
    backend = (
        TorchTrainingBackend(stage=args.command, device=args.device)
        if kronos_mode is KronosMode.PINNED_OFFICIAL
        else NumpyTrainingBackend()
    )

    config = Phase2Config(
        run_directory=args.run_dir,
        cache_directory=args.cache_dir,
        research_root=args.experiment,
        repository_root=args.repository_root,
        evidence_class=evidence_class,
        provider_mode=provider_mode,
        kronos_mode=kronos_mode,
        device=args.device,
        dry_run=args.dry_run,
    )
    return Phase2Pipeline(
        config=config, provider=provider, kronos=kronos, training_backend=backend
    )


def _emit(payload: dict[str, object], *, as_json: bool) -> None:
    if as_json:
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        for key, value in sorted(payload.items()):
            sys.stdout.write(f"{key}: {value}\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
    )

    try:
        if args.command == "preflight":
            report = run_preflight(
                research_root=args.experiment,
                cache_directory=args.cache_dir,
                repository_root=args.repository_root,
            )
            _emit(report.model_dump(mode="json"), as_json=args.json)
            return EXIT_OK if report.passed else EXIT_BLOCKED

        pipeline = _build_pipeline(args)

        if args.command == "verify":
            _emit(pipeline.summary(), as_json=args.json)
            return EXIT_OK

        if args.command == "run":
            if not args.open_test_partition:
                _LOGGER.error(
                    "run requires --open-test-partition; the test partition never opens implicitly"
                )
                return EXIT_USAGE
            table = pipeline.run(measurements={}, operator_command=args.operator)
            _emit(
                {"conclusion": table.conclusion.value, **pipeline.summary()},
                as_json=args.json,
            )
            return (
                EXIT_BLOCKED
                if table.conclusion is TerminalConclusion.OPERATIONALLY_BLOCKED
                else EXIT_OK
            )

        report = pipeline.preflight()
        if not report.passed:
            _emit(report.model_dump(mode="json"), as_json=args.json)
            return EXIT_BLOCKED

        stage_map = {
            "stage-a": pipeline.stage_a,
            "stage-b": pipeline.stage_b,
            "stage-c": pipeline.stage_c,
        }
        if args.command in stage_map:
            pipeline.retrieve()
            pipeline.validate_data()
            pipeline.build_windows()
            pipeline.coverage_audit()
            pipeline.resolve_assets()
            result = stage_map[args.command]()
            _emit({"state": result.state.value, **pipeline.summary()}, as_json=args.json)
            return EXIT_OK

        if args.command == "evaluate":
            _LOGGER.error(
                "evaluate requires a frozen checkpoint and an explicit test-opening transition"
            )
            return EXIT_USAGE

        return EXIT_USAGE

    except BridgeTransformError as error:
        failure = error.failures[0]
        _emit(
            {
                "error": failure.code,
                "category": failure.category.value,
                "message": failure.message,
            },
            as_json=args.json,
        )
        return EXIT_BLOCKED if failure.code == "MISSING_OPTIONAL_DEPENDENCY" else EXIT_FAILED


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
