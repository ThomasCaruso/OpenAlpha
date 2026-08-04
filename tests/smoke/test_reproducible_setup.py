import json
import re
from pathlib import Path

ROOT = Path(__file__).parents[2]

#: Every documented entry point must exist and must actually do something. The
#: earlier set also required `dev`, `demo`, `benchmark`, `report`, `security` and
#: `reproduce`, which were placeholders for phases of the Bridge product plan.
#: That plan was closed by this repository's own research, and stub targets whose
#: only behaviour was `exit 1` are worse than no target at all -- a reader trying
#: `make demo` learns nothing except that the repository is broken.
REQUIRED_MAKE_TARGETS = {
    "help",
    "setup",
    "test",
    "test-smoke",
    "lint",
    "typecheck",
    "verify",
    "check",
}

#: Targets that must never be reintroduced as failing stubs.
FORBIDDEN_STUB_MARKER = "@exit 1"


def make_recipe(makefile: str, target: str) -> list[str]:
    lines = makefile.splitlines()
    start = lines.index(f"{target}:") + 1
    recipe: list[str] = []
    for line in lines[start:]:
        if line.startswith("\t"):
            recipe.append(line.removeprefix("\t"))
        elif line:
            break
    return recipe


def test_lockfiles_exist() -> None:
    assert (ROOT / "uv.lock").is_file()
    assert (ROOT / "package-lock.json").is_file()


def declared_targets(makefile: str) -> set[str]:
    """Target names, including targets that declare prerequisites.

    `check: test lint typecheck verify` is an ordinary aggregate target, so
    matching only lines that end in a colon would miss it.
    """
    return {
        match.group(1)
        for match in re.finditer(r"^([A-Za-z][\w.-]*):(?![=])", makefile, re.MULTILINE)
    }


def test_makefile_exposes_required_targets() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert REQUIRED_MAKE_TARGETS <= declared_targets(makefile)


def test_no_make_target_is_a_failing_stub() -> None:
    """Every advertised target must run something, not announce its own absence."""
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert FORBIDDEN_STUB_MARKER not in makefile
    assert "is unavailable until" not in makefile


def test_the_full_test_target_installs_what_the_suite_imports() -> None:
    """`make setup && make test` must work from a clean checkout.

    The shared `setup` recipe is a cross-file contract and installs only the dev
    group, but the Sentinel risk-model tests import scipy and scikit-learn from
    the sentinel-phase3 group. The test target therefore resolves that group
    itself rather than relying on a setup step that cannot provide it.
    """
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert make_recipe(makefile, "test") == ["uv run --group sentinel-phase3 pytest -q"]


def test_setup_commands_are_portable_and_lock_enforcing() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))
    bootstrap = (ROOT / "scripts" / "bootstrap.ps1").read_text(encoding="utf-8")
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert make_recipe(makefile, "setup") == ["uv sync --locked --group dev", "npm ci"]
    assert package["scripts"]["setup"] == "uv sync --locked --group dev && npm ci"
    assert "& uv sync --locked --group dev" in bootstrap
    assert "& npm ci" in bootstrap
    assert 'throw "npm ci failed with exit code $LASTEXITCODE."' in bootstrap
    assert "run: uv sync --locked --group dev" in ci
    assert "run: npm ci" in ci


def test_repository_enforces_lf_line_endings() -> None:
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()
    assert "* text=auto eol=lf" in attributes
