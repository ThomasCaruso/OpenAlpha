"""Deployment-manifest validation for GitHub Actions workflows.

These are static checks only. They start no run, contact no cloud provider, and
need no optional dependency.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = ROOT / ".github" / "workflows"

#: Keys that GitHub evaluates before a job begins. The `inputs` context is not
#: available in any of them on `workflow_dispatch`; only `github.event.inputs`
#: is. Using `inputs` there makes GitHub reject the file, and the workflow
#: silently never registers.
_WORKFLOW_LEVEL_KEYS = ("concurrency:", "run-name:")


def _workflows() -> list[Path]:
    found = sorted(WORKFLOW_DIR.glob("*.yml")) + sorted(WORKFLOW_DIR.glob("*.yaml"))
    assert found, "no workflow files found"
    return found


def _preamble(text: str) -> str:
    """The portion of a workflow before the first top-level `jobs:` key."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("jobs:"):
            return "\n".join(lines[:index])
    return text


@pytest.mark.parametrize("path", _workflows(), ids=lambda p: p.name)
def test_workflow_level_keys_do_not_use_the_inputs_context(path: Path) -> None:
    preamble = _preamble(path.read_text(encoding="utf-8"))
    offenders = [
        line.strip()
        for line in preamble.splitlines()
        if "${{ inputs." in line or "${{inputs." in line
    ]
    assert not offenders, (
        f"{path.name} uses the `inputs` context above `jobs:`, which GitHub "
        f"rejects on workflow_dispatch. Use `github.event.inputs` instead. "
        f"Offending lines: {offenders}"
    )


@pytest.mark.parametrize("path", _workflows(), ids=lambda p: p.name)
def test_workflow_declares_a_name_and_trigger(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("name:"), f"{path.name} must start with a `name:` key"
    assert "\non:" in text, f"{path.name} must declare an `on:` trigger block"


def test_phase2_workflows_are_present() -> None:
    names = {path.name for path in _workflows()}
    assert "deploy-bridge-phase2-cloud.yml" in names
    assert "start-bridge-phase2-run.yml" in names


def test_run_workflow_is_manual_only_and_requires_confirmations() -> None:
    """Starting a real run must never be automatic."""
    text = (WORKFLOW_DIR / "start-bridge-phase2-run.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch:" in text
    for automatic_trigger in ("\n  push:", "\n  schedule:", "\n  pull_request:"):
        assert automatic_trigger not in text, (
            "the run-start workflow must be manual only"
        )
    assert "confirm_real_evidence" in text
    assert "confirm_open_test_partition" in text


def test_deploy_workflow_never_starts_a_run() -> None:
    text = (WORKFLOW_DIR / "deploy-bridge-phase2-cloud.yml").read_text(encoding="utf-8")
    assert "phase2_gpu_worker" not in text
    assert "/v1/bridge/phase2/runs" not in text
    assert "modal deploy" in text


@pytest.mark.parametrize("path", _workflows(), ids=lambda p: p.name)
def test_full_suite_runs_sync_every_group_they_need(path: Path) -> None:
    """A workflow running the whole suite must install the whole suite's deps.

    `uv run pytest` with no path argument collects the Sentinel tests, which
    import scikit-learn and scipy from the `sentinel-phase3` group. Syncing only
    `dev` collects them and then fails on import.
    """
    text = path.read_text(encoding="utf-8")
    runs_full_suite = any(
        line.strip().startswith("run: uv run pytest")
        and "tests/" not in line
        and "packages/" not in line
        for line in text.splitlines()
    )
    if not runs_full_suite:
        pytest.skip(f"{path.name} does not run the unscoped suite")

    sync_lines = [line for line in text.splitlines() if "uv sync" in line]
    assert sync_lines, f"{path.name} runs the full suite without syncing dependencies"
    assert any("sentinel-phase3" in line for line in sync_lines), (
        f"{path.name} runs the unscoped suite but never syncs the "
        f"sentinel-phase3 group, so scikit-learn and scipy will be missing"
    )


def test_no_workflow_embeds_a_credential_value() -> None:
    """Credentials may be referenced as secrets, never written literally."""
    for path in _workflows():
        text = path.read_text(encoding="utf-8")
        for marker in ("AKIA", "ghp_", "hf_", "sk-", "BEGIN PRIVATE KEY"):
            assert marker not in text, f"{path.name} appears to embed a credential"
