from __future__ import annotations

import re
import shutil
import subprocess
import sys


def parse_major_version(version_output: str) -> int | None:
    match = re.fullmatch(r'v?(\d+)(?:\.\d+)*', version_output.strip())
    return int(match.group(1)) if match else None


def environment_errors(
    python_version: tuple[int, int],
    uv_available: bool,
    node_version_output: str | None,
    npm_available: bool,
) -> list[str]:
    errors: list[str] = []
    python_major, python_minor = python_version

    if python_version != (3, 13):
        errors.append(
            'Python 3.13 is required; '
            f'found {python_major}.{python_minor}. Install Python 3.13 and rerun this script.'
        )
    if not uv_available:
        errors.append('uv was not found on PATH. Install it from https://docs.astral.sh/uv/.')

    node_major = parse_major_version(node_version_output) if node_version_output else None
    if node_major is None:
        errors.append('Node.js was not found on PATH. Install Node.js 22 LTS.')
    elif node_major < 22:
        errors.append(
            'Node.js 22 or newer is required; '
            f'found {node_version_output}. Install Node.js 22 LTS.'
        )

    if not npm_available:
        errors.append('npm was not found on PATH. Install npm with Node.js 22 LTS.')

    return errors


def main() -> int:
    uv_path = shutil.which('uv')
    node_path = shutil.which('node')
    npm_path = shutil.which('npm')
    node_version_output: str | None = None

    if node_path:
        result = subprocess.run(
            [node_path, '--version'], capture_output=True, check=False, text=True
        )
        if result.returncode == 0:
            node_version_output = result.stdout.strip()

    errors = environment_errors(
        (sys.version_info.major, sys.version_info.minor),
        uv_path is not None,
        node_version_output,
        npm_path is not None,
    )

    print(f'Python: {sys.version.split()[0]}')
    print(f'uv: {'found' if uv_path else 'missing'}')
    print(f'Node.js: {node_version_output or 'missing'}')
    print(f'npm: {'found' if npm_path else 'missing'}')

    if errors:
        print('\nEnvironment validation failed:', file=sys.stderr)
        for error in errors:
            print(f'- {error}', file=sys.stderr)
        return 1

    print('Environment is ready for OpenAlpha development.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
