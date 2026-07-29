import subprocess

from scripts import verify_environment


def result(command: str, returncode: int, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess([command, "--version"], returncode, stdout, stderr)


def requested_tool(command: list[str]) -> str:
    return "npm" if command[-1] == "npm --version" else command[0]


def test_probe_version_reports_successful_tool_versions() -> None:
    versions = {
        "uv": result("uv", 0, "uv 0.11.19\n"),
        "node": result("node", 0, "v22.20.0\n"),
        "npm": result("npm", 0, "11.12.1\n"),
    }

    def runner(command, **_):
        return versions[requested_tool(command)]

    assert verify_environment.probe_version("uv", runner) == ("uv 0.11.19", None)
    assert verify_environment.probe_version("node", runner) == ("v22.20.0", None)
    assert verify_environment.probe_version("npm", runner) == ("11.12.1", None)


def test_probe_version_handles_broken_shim() -> None:
    def runner(command, **_):
        raise OSError("broken shim")

    assert verify_environment.probe_version("uv", runner) == (
        None,
        "uv could not be executed: broken shim. Reinstall uv and ensure it is on PATH.",
    )


def test_version_command_routes_windows_npm_through_cmd() -> None:
    assert verify_environment.version_command("npm", os_name="nt", command_processor="cmd.exe") == [
        "cmd.exe",
        "/d",
        "/s",
        "/c",
        "npm --version",
    ]
    assert verify_environment.version_command("uv", os_name="nt", command_processor="cmd.exe") == [
        "uv",
        "--version",
    ]


def test_probe_version_handles_missing_windows_command_processor() -> None:
    def runner(command, **_):
        raise OSError("cmd.exe missing")

    assert verify_environment.probe_version(
        "npm",
        runner,
        os_name="nt",
        command_processor="cmd.exe",
    ) == (
        None,
        (
            "npm could not be executed through cmd.exe: cmd.exe missing. "
            "Ensure cmd.exe is available, then reinstall npm and ensure it is on PATH."
        ),
    )


def test_probe_version_handles_nonzero_exit() -> None:
    def runner(command, **_):
        return result(command[0], 1, stderr="broken shim\n")

    assert verify_environment.probe_version("npm", runner) == (
        None,
        (
            "npm --version failed with exit code 1: broken shim. "
            "Reinstall npm and ensure it is on PATH."
        ),
    )


def test_probe_version_handles_malformed_output() -> None:
    def runner(command, **_):
        return result(command[0], 0, stdout="not a version\n")

    assert verify_environment.probe_version("npm", runner) == (
        None,
        (
            "npm --version returned an unrecognized version: 'not a version'. "
            "Reinstall npm and ensure it is on PATH."
        ),
    )


def test_main_prints_executed_tool_versions(capsys) -> None:
    versions = {
        "uv": result("uv", 0, "uv 0.11.19\n"),
        "node": result("node", 0, "v22.20.0\n"),
        "npm": result("npm", 0, "11.12.1\n"),
    }

    def runner(command, **_):
        return versions[requested_tool(command)]

    assert verify_environment.main(runner) == 0
    output = capsys.readouterr().out
    assert "uv: uv 0.11.19" in output
    assert "Node.js: v22.20.0" in output
    assert "npm: 11.12.1" in output
