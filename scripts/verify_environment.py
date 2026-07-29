from __future__ import annotations

import os
import re
import subprocess
import sys

VERSION_PATTERN = re.compile(r"(?:^|\s)v?\d+(?:\.\d+)+(?:\s|$)")


def parse_major_version(version_output: str) -> int | None:
    match = re.fullmatch(r"v?(\d+)(?:\.\d+)*", version_output.strip())
    return int(match.group(1)) if match else None


def version_command(
    command: str,
    *,
    os_name: str = os.name,
    command_processor: str | None = None,
) -> list[str]:
    if os_name == "nt" and command == "npm":
        processor = command_processor or os.environ.get("COMSPEC") or "cmd.exe"
        return [processor, "/d", "/s", "/c", "npm --version"]
    return [command, "--version"]


def probe_version(
    command: str,
    runner=subprocess.run,
    *,
    os_name: str = os.name,
    command_processor: str | None = None,
) -> tuple[str | None, str | None]:
    reinstall = f"Reinstall {command} and ensure it is on PATH."
    invocation = version_command(
        command,
        os_name=os_name,
        command_processor=command_processor,
    )
    try:
        result = runner(
            invocation,
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError as error:
        if invocation[0] != command:
            return (
                None,
                (
                    f"{command} could not be executed through {invocation[0]}: {error}. "
                    f"Ensure {invocation[0]} is available, then {reinstall[0].lower()}{reinstall[1:]}"
                ),
            )
        return None, f"{command} could not be executed: {error}. {reinstall}"

    output = (result.stdout or result.stderr).strip()
    if result.returncode != 0:
        detail = output or "no diagnostic output"
        return (
            None,
            (
                f"{command} --version failed with exit code {result.returncode}: "
                f"{detail}. {reinstall}"
            ),
        )
    if not VERSION_PATTERN.search(output):
        return (
            None,
            f"{command} --version returned an unrecognized version: {output!r}. {reinstall}",
        )
    return output, None


def environment_errors(
    python_version: tuple[int, int],
    uv_available: bool,
    node_version_output: str | None,
    npm_available: bool,
    *,
    uv_error: str | None = None,
    node_error: str | None = None,
    npm_error: str | None = None,
) -> list[str]:
    errors: list[str] = []
    python_major, python_minor = python_version

    if python_version != (3, 13):
        errors.append(
            "Python 3.13 is required; "
            f"found {python_major}.{python_minor}. Install Python 3.13 and rerun this script."
        )
    if not uv_available:
        errors.append(
            uv_error or "uv was not found on PATH. Install it from https://docs.astral.sh/uv/."
        )

    node_major = parse_major_version(node_version_output) if node_version_output else None
    if node_major is None:
        errors.append(node_error or "Node.js was not found on PATH. Install Node.js 22 LTS.")
    elif node_major < 22:
        errors.append(
            f"Node.js 22 or newer is required; found {node_version_output}. Install Node.js 22 LTS."
        )

    if not npm_available:
        errors.append(npm_error or "npm was not found on PATH. Install npm with Node.js 22 LTS.")

    return errors


def main(runner=subprocess.run) -> int:
    uv_version, uv_error = probe_version("uv", runner)
    node_version, node_error = probe_version("node", runner)
    npm_version, npm_error = probe_version("npm", runner)

    errors = environment_errors(
        (sys.version_info.major, sys.version_info.minor),
        uv_version is not None,
        node_version,
        npm_version is not None,
        uv_error=uv_error,
        node_error=node_error,
        npm_error=npm_error,
    )

    print(f"Python: {sys.version.split()[0]}")
    print(f"uv: {uv_version or 'unavailable'}")
    print(f"Node.js: {node_version or 'unavailable'}")
    print(f"npm: {npm_version or 'unavailable'}")

    if errors:
        print("\nEnvironment validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("Environment is ready for OpenAlpha development.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
