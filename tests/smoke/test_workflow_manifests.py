"""Deployment-manifest validation for GitHub Actions workflows.

These are static checks only. They start no run, contact no cloud provider, and
need no optional dependency.
"""

from __future__ import annotations

import ast
import copy
import shlex
from pathlib import Path
from typing import Any

import pytest
import yaml

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


def _load_workflow(text: str) -> dict[str, Any]:
    document = yaml.load(text, Loader=yaml.BaseLoader)
    assert isinstance(document, dict)
    return document


def _walk_yaml(value: object) -> list[tuple[str | None, object]]:
    walked: list[tuple[str | None, object]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            walked.append((str(key), item))
            walked.extend(_walk_yaml(item))
    elif isinstance(value, list):
        for item in value:
            walked.extend(_walk_yaml(item))
    return walked


def _torch_absence_assertion(script: str) -> bool:
    try:
        tree = ast.parse(script)
    except SyntaxError:
        return False
    assertions = [node for node in ast.walk(tree) if isinstance(node, ast.Assert)]
    if len(assertions) != 1:
        return False
    test = assertions[0].test
    if not isinstance(test, ast.Compare) or len(test.ops) != 1 or len(test.comparators) != 1:
        return False
    if not isinstance(test.ops[0], ast.Is) or not (
        isinstance(test.comparators[0], ast.Constant) and test.comparators[0].value is None
    ):
        return False
    call = test.left
    return (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "find_spec"
        and isinstance(call.func.value, ast.Attribute)
        and call.func.value.attr == "util"
        and isinstance(call.func.value.value, ast.Name)
        and call.func.value.value.id == "importlib"
        and len(call.args) == 1
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == "torch"
    )


def _run_signature(command: object) -> str | None:
    if not isinstance(command, str):
        return None
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    exact_commands = {
        ("uv", "python", "install", "3.13"): "python",
        (
            "uv",
            "sync",
            "--locked",
            "--group",
            "dev",
            "--group",
            "sentinel-phase3",
        ): "sync",
        (
            "uv",
            "run",
            "pytest",
            "-q",
            "-m",
            "not network and not kronos",
        ): "pytest",
        (
            "uv",
            "run",
            "python",
            "scripts/verify_specifications.py",
        ): "verify-specifications",
        ("uv", "run", "python", "scripts/verify_artifacts.py"): "verify-artifacts",
        (
            "uv",
            "run",
            "ruff",
            "check",
            "packages",
            "cloud",
            "tests",
            "scripts",
        ): "ruff",
        ("uv", "run", "pyright"): "pyright",
    }
    signature = exact_commands.get(tuple(tokens))
    if signature is not None:
        return signature
    if len(tokens) == 5 and tokens[:4] == ["uv", "run", "python", "-c"]:
        return "torch-absent" if _torch_absence_assertion(tokens[4]) else None
    return None


def _research_integrity_violations(document: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    if document.get("name") != "Research Integrity":
        violations.append("workflow name")

    triggers = document.get("on")
    if not isinstance(triggers, dict) or set(triggers) != {"push", "pull_request"}:
        violations.append("trigger set")
    if document.get("permissions") != {"contents": "read"}:
        violations.append("workflow permissions")

    jobs = document.get("jobs")
    if not isinstance(jobs, dict) or set(jobs) != {"verify"}:
        violations.append("job set")
        return violations
    verify = jobs.get("verify")
    if not isinstance(verify, dict):
        return [*violations, "verify job"]
    if "environment" in verify or "permissions" in verify:
        violations.append("job privilege override")
    if verify.get("runs-on") != "ubuntu-latest":
        violations.append("runner")

    steps = verify.get("steps")
    if not isinstance(steps, list) or not all(isinstance(step, dict) for step in steps):
        return [*violations, "steps"]
    uses = [step.get("uses") for step in steps if "uses" in step]
    if uses.count("actions/checkout@v4") != 1:
        violations.append("checkout step")
    if uses.count("astral-sh/setup-uv@v6") != 1:
        violations.append("setup-uv step")
    if len(uses) != 2:
        violations.append("unexpected action step")

    checkout = next(
        (step for step in steps if step.get("uses") == "actions/checkout@v4"), None
    )
    if checkout is None or checkout.get("with") != {"persist-credentials": "false"}:
        violations.append("checkout credentials")
    setup_uv = next(
        (step for step in steps if step.get("uses") == "astral-sh/setup-uv@v6"), None
    )
    if setup_uv is None or setup_uv.get("with") != {"enable-cache": "true"}:
        violations.append("setup-uv configuration")

    signatures = [_run_signature(step.get("run")) for step in steps if "run" in step]
    expected_signatures = {
        "python",
        "sync",
        "torch-absent",
        "pytest",
        "verify-specifications",
        "verify-artifacts",
        "ruff",
        "pyright",
    }
    if len(signatures) != len(expected_signatures) or set(signatures) != expected_signatures:
        violations.append("verification commands")

    forbidden_keys = {"environment", "environments", "secret", "secrets"}
    forbidden_text = (
        "workflow_dispatch",
        "repository_dispatch",
        "workflow_call",
        "schedule",
        "${{ secrets.",
    )
    for key, value in _walk_yaml(document):
        if key is not None and key.casefold() in forbidden_keys:
            violations.append(f"forbidden key: {key}")
        if isinstance(value, str) and any(
            marker in value.casefold() for marker in forbidden_text
        ):
            violations.append("forbidden workflow surface")
    return violations


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


def test_research_integrity_workflow_is_complete_and_verification_only() -> None:
    path = WORKFLOW_DIR / "research-integrity.yml"
    assert path.is_file(), "the research-integrity workflow is missing"
    document = _load_workflow(path.read_text(encoding="utf-8"))

    assert _research_integrity_violations(document) == []


@pytest.mark.parametrize(
    "mutation",
    (
        "write-all",
        "added-schedule",
        "removed-pull-request",
        "missing-checkout-hardening",
        "job-permissions",
        "job-environment",
        "secret-reference",
        "provider-command",
        "extra-job",
    ),
)
def test_research_integrity_validator_rejects_privilege_and_trigger_mutations(
    mutation: str,
) -> None:
    document = _load_workflow(
        (WORKFLOW_DIR / "research-integrity.yml").read_text(encoding="utf-8")
    )
    mutated = copy.deepcopy(document)
    if mutation == "write-all":
        mutated["permissions"] = "write-all"
    elif mutation == "added-schedule":
        mutated["on"]["schedule"] = [{"cron": "0 0 * * *"}]
    elif mutation == "removed-pull-request":
        del mutated["on"]["pull_request"]
    elif mutation == "missing-checkout-hardening":
        checkout = mutated["jobs"]["verify"]["steps"][0]
        checkout.pop("with", None)
    elif mutation == "job-permissions":
        mutated["jobs"]["verify"]["permissions"] = {"contents": "write"}
    elif mutation == "job-environment":
        mutated["jobs"]["verify"]["environment"] = "production"
    elif mutation == "secret-reference":
        mutated["jobs"]["verify"]["env"] = {"TOKEN": "${{ secrets.TOKEN }}"}
    elif mutation == "provider-command":
        mutated["jobs"]["verify"]["steps"].append({"run": "modal run study.py"})
    else:
        mutated["jobs"]["deploy"] = {"runs-on": "ubuntu-latest", "steps": []}

    assert _research_integrity_violations(mutated)


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


@pytest.mark.parametrize("path", _workflows(), ids=lambda p: p.name)
def test_no_workflow_installs_into_the_system_python(path: Path) -> None:
    """GitHub runners ship an externally managed Python (PEP 668).

    `uv pip install --system` fails there. Tools must run through `uvx` or a
    virtual environment instead.
    """
    offenders = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        # A comment naming the antipattern is documentation, not an invocation.
        if not line.strip().startswith("#") and "uv pip install --system" in line
    ]
    assert not offenders, (
        f"{path.name} installs into the runner's system Python, which PEP 668 "
        f"rejects. Use `uvx --from <pkg> <cmd>` instead. Offending: {offenders}"
    )


def test_no_workflow_embeds_a_credential_value() -> None:
    """Credentials may be referenced as secrets, never written literally."""
    for path in _workflows():
        text = path.read_text(encoding="utf-8")
        for marker in ("AKIA", "ghp_", "hf_", "sk-", "BEGIN PRIVATE KEY"):
            assert marker not in text, f"{path.name} appears to embed a credential"
