import json
from pathlib import Path

ROOT = Path(__file__).parents[2]
REQUIRED_MAKE_TARGETS = {
    "setup",
    "dev",
    "test",
    "demo",
    "benchmark",
    "report",
    "lint",
    "typecheck",
    "security",
    "reproduce",
}


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


def test_makefile_exposes_required_targets() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    targets = {line.removesuffix(":") for line in makefile.splitlines() if line.endswith(":")}
    assert REQUIRED_MAKE_TARGETS <= targets


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
