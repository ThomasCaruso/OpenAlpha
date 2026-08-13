"""The image must know what code it was built from.

These tests exercise cloud/modal/deployed_commit.py against real throwaway git
repositories. Nothing here deploys, launches, or contacts Modal.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODAL_DIR = ROOT / "cloud" / "modal"
APP = MODAL_DIR / "kronos_research.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location(
        "openalpha_modal_deployed_commit_under_test", MODAL_DIR / "deployed_commit.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


binding = _load()


def _git(root: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        check=True,
        text=True,
        timeout=60,
    )


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    root = tmp_path / "worktree"
    root.mkdir()
    _git(root, "init", "--quiet")
    _git(root, "config", "user.email", "canary@example.invalid")
    _git(root, "config", "user.name", "Canary")
    _git(root, "config", "commit.gpgsign", "false")
    (root / "tracked.txt").write_text("locked\n", encoding="utf-8")
    _git(root, "add", "tracked.txt")
    _git(root, "commit", "--quiet", "-m", "initial")
    return root


def test_a_clean_worktree_resolves_to_exact_forty_hex(repository: Path) -> None:
    resolved = binding.resolve_deploying_commit(repository)
    assert binding.COMMIT_PATTERN.fullmatch(resolved)
    assert len(resolved) == 40
    assert resolved == resolved.lower()


def test_a_dirty_tracked_file_refuses_to_deploy(repository: Path) -> None:
    (repository / "tracked.txt").write_text("edited after the commit\n", encoding="utf-8")
    with pytest.raises(binding.DeploymentBindingError) as excinfo:
        binding.resolve_deploying_commit(repository)
    message = str(excinfo.value)
    assert "DEPLOYED_COMMIT_WORKTREE_DIRTY" in message
    assert "tracked.txt" in message


def test_a_staged_change_also_refuses_to_deploy(repository: Path) -> None:
    (repository / "added.txt").write_text("new\n", encoding="utf-8")
    _git(repository, "add", "added.txt")
    with pytest.raises(binding.DeploymentBindingError) as excinfo:
        binding.resolve_deploying_commit(repository)
    assert "DEPLOYED_COMMIT_WORKTREE_DIRTY" in str(excinfo.value)


def test_an_untracked_file_alone_does_not_block_deployment(repository: Path) -> None:
    """Scratch files are not part of the deployed image and do not change it."""
    (repository / "scratch.log").write_text("noise\n", encoding="utf-8")
    assert binding.COMMIT_PATTERN.fullmatch(binding.resolve_deploying_commit(repository))


def test_a_non_repository_is_unresolvable(tmp_path: Path) -> None:
    with pytest.raises(binding.DeploymentBindingError) as excinfo:
        binding.resolve_deploying_commit(tmp_path)
    assert "DEPLOYED_COMMIT_UNRESOLVABLE" in str(excinfo.value)


def test_a_repository_with_no_commits_is_unresolvable(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    root.mkdir()
    _git(root, "init", "--quiet")
    with pytest.raises(binding.DeploymentBindingError) as excinfo:
        binding.resolve_deploying_commit(root)
    assert "DEPLOYED_COMMIT_UNRESOLVABLE" in str(excinfo.value)


# ------------------------------------------------- inspection is not a bypass


def test_deploying_is_strict_by_default(repository: Path) -> None:
    """No flag set: a dirty worktree stops the deployment."""
    (repository / "tracked.txt").write_text("edited\n", encoding="utf-8")
    with pytest.raises(binding.DeploymentBindingError):
        binding.bind_for_image(repository, environment={})


def test_inspection_bakes_a_sentinel_rather_than_a_commit(repository: Path) -> None:
    (repository / "tracked.txt").write_text("edited\n", encoding="utf-8")
    baked = binding.bind_for_image(repository, environment={binding.INSPECTION_VARIABLE: "1"})
    assert baked == binding.INSPECTION_SENTINEL


def test_the_inspection_sentinel_can_never_pass_the_worker_check() -> None:
    """The flag is not a relaxation: an image built under it is inert."""
    assert not binding.COMMIT_PATTERN.fullmatch(binding.INSPECTION_SENTINEL)
    namespace = _extract_validator()
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("OPENALPHA_DEPLOYED_COMMIT", binding.INSPECTION_SENTINEL)
        with pytest.raises(RuntimeError) as excinfo:
            namespace["_require_deployed_commit"]()
    assert "DEPLOYED_COMMIT_MALFORMED" in str(excinfo.value)


def test_inspection_still_reports_a_clean_commit_honestly(repository: Path) -> None:
    """The flag substitutes only when it must; it does not blind the tooling."""
    baked = binding.bind_for_image(repository, environment={binding.INSPECTION_VARIABLE: "1"})
    # A clean tree under inspection still yields the sentinel rather than a real
    # commit, because an inspection image must never be mistaken for deployable.
    assert baked == binding.INSPECTION_SENTINEL


# ------------------------------------------------------- the worker contract


def _app_source() -> str:
    return APP.read_text(encoding="utf-8")


def test_the_app_and_the_helper_name_the_same_variable() -> None:
    """The container has no sibling module, so the name is duplicated. It must agree."""
    tree = ast.parse(_app_source())
    duplicated = next(
        node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "DEPLOYED_COMMIT_VARIABLE"
        and isinstance(node.value, ast.Constant)
    )
    assert duplicated == binding.COMMIT_ENVIRONMENT_VARIABLE == "OPENALPHA_DEPLOYED_COMMIT"


def test_the_image_bakes_the_binding_at_build_time() -> None:
    source = _app_source()
    assert "binding.bind_for_image(root, environment=os.environ)" in source
    assert "COMMIT_ENVIRONMENT_VARIABLE" in source


def test_the_worker_requires_the_binding_and_passes_it_through() -> None:
    source = _app_source()
    assert source.count("deployed_commit=_require_deployed_commit()") == 8
    for name in (
        "run_mini_structural_validity",
        "run_base_structural_validity",
        "run_zero_shot_benchmark",
    ):
        assert f"def {name}(source_commit: str, run_id: str)" in source
    assert 'os.environ.get(DEPLOYED_COMMIT_VARIABLE, "")' in source
    assert "setdefault(DEPLOYED_COMMIT_VARIABLE" not in source


def test_every_completed_study_entrypoint_requires_the_binding() -> None:
    source = _app_source()
    assert source.count("deployed_commit=_require_deployed_commit()") == 8


@pytest.mark.parametrize(
    "value", ["", "HEAD", "a" * 39, "a" * 41, "A" * 40, "g" * 40, " " + "a" * 40]
)
def test_the_worker_validator_rejects_anything_but_forty_hex(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    namespace = _extract_validator()
    monkeypatch.setenv("OPENALPHA_DEPLOYED_COMMIT", value)
    with pytest.raises(RuntimeError) as excinfo:
        namespace["_require_deployed_commit"]()
    expected = "DEPLOYED_COMMIT_MISSING" if value == "" else "DEPLOYED_COMMIT_MALFORMED"
    assert expected in str(excinfo.value)


def test_the_worker_validator_accepts_exact_forty_hex(monkeypatch: pytest.MonkeyPatch) -> None:
    namespace = _extract_validator()
    monkeypatch.setenv("OPENALPHA_DEPLOYED_COMMIT", "0" * 39 + "f")
    assert namespace["_require_deployed_commit"]() == "0" * 39 + "f"


def _extract_validator() -> dict[str, Any]:
    """Execute just the validator, without importing modal.

    The app module imports modal at the top level, which is deliberately absent
    from the base development environment. Lifting the one function out keeps
    this a real behavioural test rather than a source-string assertion.
    """
    tree = ast.parse(_app_source())
    wanted: list[ast.stmt] = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name == "_require_deployed_commit")
        or (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "DEPLOYED_COMMIT_VARIABLE"
        )
    ]
    assert len(wanted) == 2, "expected the constant and the validator to both be present"
    namespace: dict[str, Any] = {"os": os}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(APP), "exec"), namespace)  # noqa: S102
    return namespace


# --------------------------------- untracked files that would enter the image


def _shipped_file(repository: Path, relative: str) -> Path:
    path = repository / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def test_an_untracked_file_under_a_shipped_package_blocks_deployment(
    repository: Path,
) -> None:
    """add_local_dir copies directories, not the git index.

    An untracked .py file under a shipped package is built into the image and
    imported at runtime, while the image still claims to be HEAD.
    """
    _shipped_file(repository, "packages/kronos-research/src/openalpha_kronos/sneaky.py").write_text(
        "SHIPPED = True\n", encoding="utf-8"
    )
    with pytest.raises(binding.DeploymentBindingError) as excinfo:
        binding.resolve_deploying_commit(repository)
    message = str(excinfo.value)
    assert "DEPLOYED_COMMIT_UNTRACKED_IN_IMAGE" in message
    assert "sneaky.py" in message


def test_an_untracked_modal_entrypoint_blocks_deployment(repository: Path) -> None:
    _shipped_file(repository, "cloud/modal/kronos_research.py").write_text(
        "SHIPPED = True\n", encoding="utf-8"
    )
    with pytest.raises(binding.DeploymentBindingError) as excinfo:
        binding.resolve_deploying_commit(repository)
    message = str(excinfo.value)
    assert "DEPLOYED_COMMIT_UNTRACKED_IN_IMAGE" in message
    assert "cloud/modal/kronos_research.py" in message.replace(chr(92), "/")


def test_an_untracked_file_under_the_research_directory_blocks_deployment(
    repository: Path,
) -> None:
    _shipped_file(repository, "research/bridge-v0/extra.yaml").write_text(
        "unexpected: true\n", encoding="utf-8"
    )
    with pytest.raises(binding.DeploymentBindingError) as excinfo:
        binding.resolve_deploying_commit(repository)
    assert "DEPLOYED_COMMIT_UNTRACKED_IN_IMAGE" in str(excinfo.value)


def test_an_untracked_file_in_a_nested_untracked_directory_is_still_seen(
    repository: Path,
) -> None:
    """--untracked-files=all, so a new directory is not collapsed to one entry."""
    _shipped_file(
        repository, "packages/research-core/src/openalpha_research/newpkg/mod.py"
    ).write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(binding.DeploymentBindingError) as excinfo:
        binding.resolve_deploying_commit(repository)
    assert "newpkg/mod.py" in str(excinfo.value).replace("\\", "/")


def test_untracked_bytecode_does_not_block_deployment(repository: Path) -> None:
    """add_local_dir ignores these, so they cannot reach the image."""
    _shipped_file(
        repository, "packages/kronos-research/src/openalpha_kronos/__pycache__/mod.cpython-313.pyc"
    ).write_bytes(b"\x00")
    _shipped_file(
        repository, "packages/kronos-research/src/openalpha_kronos/stale.pyc"
    ).write_bytes(b"\x00")
    assert binding.COMMIT_PATTERN.fullmatch(binding.resolve_deploying_commit(repository))


def test_untracked_files_outside_the_shipped_roots_do_not_block(repository: Path) -> None:
    """Scratch files, notes and local tooling are not copied into the image."""
    for relative in ("scratch.log", "docs/notes.md", "tests/local_probe.py", "cloud/tmp.txt"):
        _shipped_file(repository, relative).write_text("noise\n", encoding="utf-8")
    assert binding.COMMIT_PATTERN.fullmatch(binding.resolve_deploying_commit(repository))


def test_the_binding_never_suppresses_untracked_files() -> None:
    # Comments dropped: the docstring names the old flag to explain why it was
    # wrong, which is worth keeping. Only the executed call is policed.
    lines = (MODAL_DIR / "deployed_commit.py").read_text(encoding="utf-8").splitlines()
    executable = "\n".join(
        line for line in lines if not line.lstrip().startswith(("#", '"""', "``"))
    )
    assert '"--untracked-files=no"' not in executable
    assert '"--untracked-files=all"' in executable


def test_the_shipped_roots_match_what_the_image_actually_copies() -> None:
    """IMAGE_SOURCE_ROOTS is duplicated from the app; it must not drift."""
    tree = ast.parse(_app_source())
    packages = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "LOCAL_PACKAGES"
        and node.value is not None
    )
    copied = {local for local, _ in packages}
    copied.add("cloud/modal/kronos_research.py")
    copied.add("research/bridge-v0")
    assert set(binding.IMAGE_SOURCE_ROOTS) == copied


def test_the_ignore_list_matches_the_image() -> None:
    source = _app_source()
    ignore = next(
        ast.literal_eval(node.value)
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "_IGNORE"
    )
    assert binding.IMAGE_IGNORED_DIRECTORY == "__pycache__"
    for suffix in binding.IMAGE_IGNORED_SUFFIXES:
        assert f"*{suffix}" in ignore
