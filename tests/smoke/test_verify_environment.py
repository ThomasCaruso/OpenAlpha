from scripts.verify_environment import environment_errors, parse_major_version


def test_parse_major_version_accepts_node_and_npm_output() -> None:
    assert parse_major_version("v22.20.0") == 22
    assert parse_major_version("11.12.1") == 11


def test_environment_errors_accepts_supported_toolchain() -> None:
    assert environment_errors((3, 13), True, "v22.20.0", True) == []


def test_environment_errors_returns_actionable_diagnostics() -> None:
    assert environment_errors((3, 12), False, "v20.19.0", False) == [
        "Python 3.13 is required; found 3.12. Install Python 3.13 and rerun this script.",
        "uv was not found on PATH. Install it from https://docs.astral.sh/uv/.",
        "Node.js 22 or newer is required; found v20.19.0. Install Node.js 22 LTS.",
        "npm was not found on PATH. Install npm with Node.js 22 LTS.",
    ]
