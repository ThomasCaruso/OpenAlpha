from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the one-origin Sentinel v0 Phase 2 proof")
    parser.add_argument("operation", choices=("create", "resolve"))
    parser.add_argument(
        "--state-root",
        type=Path,
        default=Path.home()
        / ".cache"
        / "openalpha-sentinel"
        / "phase2"
        / "run-spy-20240705",
    )
    parser.add_argument(
        "--inference-python",
        type=Path,
        default=Path.home()
        / ".cache"
        / "openalpha-sentinel"
        / "phase2"
        / "venv"
        / "Scripts"
        / "python.exe",
    )
    parser.add_argument(
        "--source-path",
        type=Path,
        default=Path.home()
        / ".cache"
        / "openalpha-sentinel"
        / "phase2"
        / "kronos-src",
    )
    parser.add_argument(
        "--model-cache-path",
        type=Path,
        default=Path.home() / ".cache" / "openalpha-sentinel" / "phase2" / "hf",
    )
    args = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    if args.operation == "create":
        from openalpha_sentinel.origin_create import create_phase_2_origin

        result = create_phase_2_origin(
            repository_root=repository_root,
            state_root=args.state_root,
            inference_python=args.inference_python,
            source_path=args.source_path,
            model_cache_path=args.model_cache_path,
        )
    else:
        from openalpha_sentinel.outcome_resolver import resolve_phase_2_origin

        result = resolve_phase_2_origin(
            repository_root=repository_root,
            state_root=args.state_root,
        )
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
