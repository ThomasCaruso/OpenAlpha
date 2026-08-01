"""Thin authenticated client for the OpenAlpha Bridge Phase 2 control API.

This installs nothing, retrieves no market data, loads no Kronos asset, and
trains nothing. It only calls the managed cloud API. All empirical work happens
in the Modal worker.

    python scripts/bridge_phase2_cloud.py start  --operator you --mode synthetic
    python scripts/bridge_phase2_cloud.py status --run-id syn_...
    python scripts/bridge_phase2_cloud.py logs   --run-id syn_...
    python scripts/bridge_phase2_cloud.py artifacts --run-id syn_...
    python scripts/bridge_phase2_cloud.py resume --run-id run_... --operator you
    python scripts/bridge_phase2_cloud.py cancel --run-id run_... --operator you

Configuration comes from the environment, never from arguments:

    OPENALPHA_API_BASE_URL   base URL of the deployed control API
    OPENALPHA_API_TOKEN      bearer token (never logged or echoed)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

LOCKED_EXPERIMENT_SHA256 = (
    "d52a9be733f4ec331e081346e64ab7415ac0f9510e494733746f975f114f47b2"
)

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_BLOCKED = 2
EXIT_USAGE = 64


def _config() -> tuple[str, str]:
    base_url = os.environ.get("OPENALPHA_API_BASE_URL")
    token = os.environ.get("OPENALPHA_API_TOKEN")
    if not base_url or not token:
        sys.stderr.write(
            "set OPENALPHA_API_BASE_URL and OPENALPHA_API_TOKEN before using this client\n"
        )
        raise SystemExit(EXIT_USAGE)
    return base_url.rstrip("/"), token


def _call(method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    base_url, token = _config()
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=body,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        # The token lives only in the header and is never included here.
        sys.stderr.write(f"HTTP {error.code} from {path}: {detail}\n")
        raise SystemExit(EXIT_BLOCKED if error.code in (401, 403, 409, 429) else EXIT_FAILED)
    except urllib.error.URLError as error:
        sys.stderr.write(f"could not reach the control API: {error.reason}\n")
        raise SystemExit(EXIT_FAILED) from error


def _emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bridge_phase2_cloud",
        description="Authenticated client for the Bridge Phase 2 control API",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start", help="create a run")
    start.add_argument("--operator", required=True)
    start.add_argument("--mode", choices=("synthetic", "real"), default="synthetic")
    start.add_argument("--experiment-hash", default=LOCKED_EXPERIMENT_SHA256)
    start.add_argument("--source-commit", required=True)
    start.add_argument("--idempotency-key")
    start.add_argument("--confirm-real-evidence", action="store_true")
    start.add_argument("--confirm-open-test-partition", action="store_true")

    for name in ("status", "logs", "artifacts"):
        command = sub.add_parser(name, help=f"read {name}")
        command.add_argument("--run-id", required=True)

    resume = sub.add_parser("resume", help="resume a blocked or failed run")
    resume.add_argument("--run-id", required=True)
    resume.add_argument("--operator", required=True)

    cancel = sub.add_parser("cancel", help="cancel a run")
    cancel.add_argument("--run-id", required=True)
    cancel.add_argument("--operator", required=True)
    cancel.add_argument("--reason")

    args = parser.parse_args(argv)

    if args.command == "start":
        if args.mode == "real" and not args.confirm_real_evidence:
            sys.stderr.write("a real run requires --confirm-real-evidence\n")
            return EXIT_USAGE
        payload: dict[str, Any] = {
            "experiment_hash": args.experiment_hash,
            "source_commit": args.source_commit,
            "operator": args.operator,
            "execution_mode": args.mode,
            "confirm_real_evidence": bool(args.confirm_real_evidence),
            "confirm_open_test_partition": bool(args.confirm_open_test_partition),
        }
        if args.idempotency_key:
            payload["idempotency_key"] = args.idempotency_key
        _emit(_call("POST", "/v1/bridge/phase2/runs", payload))
        return EXIT_OK

    base = f"/v1/bridge/phase2/runs/{args.run_id}"
    match args.command:
        case "status":
            _emit(_call("GET", base))
        case "logs":
            _emit(_call("GET", f"{base}/logs"))
        case "artifacts":
            _emit(_call("GET", f"{base}/artifacts"))
        case "resume":
            _emit(_call("POST", f"{base}/resume", {"operator": args.operator}))
        case "cancel":
            body: dict[str, Any] = {"operator": args.operator}
            if args.reason:
                body["reason"] = args.reason
            _emit(_call("POST", f"{base}/cancel", body))
        case _:  # pragma: no cover - argparse enforces the choices
            return EXIT_USAGE
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
