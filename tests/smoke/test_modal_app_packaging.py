"""Packaging guards for the Modal application.

`modal deploy` runs from an isolated environment in which the workspace packages
are not installed. Anything that resolves a package by importing it therefore
fails at deploy time with "openalpha_bridge has no spec - might not be
installed?", which no local test catches because the development environment has
the workspace installed.

These are static and subprocess checks. They deploy nothing and contact no cloud
provider.
"""

from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
APP_PATH = ROOT / "cloud" / "modal" / "bridge_phase2_app.py"
BRIDGE_SRC = ROOT / "packages" / "bridge" / "src"


def _source() -> str:
    return APP_PATH.read_text(encoding="utf-8")


def _declared_local_packages() -> tuple[tuple[str, str], ...]:
    """Read LOCAL_PACKAGES without importing the module, which needs modal."""
    tree = ast.parse(_source())
    for node in tree.body:
        if isinstance(node, ast.AnnAssign | ast.Assign):
            targets = [node.target] if isinstance(node, ast.AnnAssign) else node.targets
            names = [t.id for t in targets if isinstance(t, ast.Name)]
            if "LOCAL_PACKAGES" in names and node.value is not None:
                return tuple(ast.literal_eval(node.value))
    pytest.fail("LOCAL_PACKAGES is not declared in the Modal application")


def test_app_does_not_use_add_local_python_source() -> None:
    """That API resolves packages by import, which the deploy runner cannot do."""
    offenders = [
        line.strip()
        for line in _source().splitlines()
        if not line.strip().startswith("#") and "add_local_python_source" in line
    ]
    assert not offenders, (
        "add_local_python_source requires the package to be installed in the "
        "deploying environment. Use add_local_dir with an explicit path. "
        f"Offending: {offenders}"
    )


def test_declared_local_packages_exist_and_are_real_packages() -> None:
    declared = _declared_local_packages()
    assert declared, "no local packages declared"
    for local_path, remote_path in declared:
        source = ROOT / local_path
        assert source.is_dir(), f"declared source directory is missing: {local_path}"
        assert (source / "__init__.py").is_file(), f"{local_path} is not a Python package"
        assert remote_path.startswith("/root/"), (
            f"{remote_path} must land under /root so it is importable"
        )
        assert remote_path.rsplit("/", 1)[-1] == source.name, (
            f"{local_path} must be mounted under its own package name, got {remote_path}"
        )


def test_the_three_workspace_packages_are_mounted() -> None:
    mounted = {remote.rsplit("/", 1)[-1] for _, remote in _declared_local_packages()}
    assert {"openalpha_bridge", "openalpha_sentinel", "openalpha_research"} <= mounted


def test_every_sibling_package_the_bridge_imports_is_mounted() -> None:
    """A runtime import of an unmounted package would fail only in the cloud."""
    pattern = re.compile(r"^\s*(?:from|import)\s+(openalpha_[a-z_]+)", re.MULTILINE)
    imported: set[str] = set()
    for path in BRIDGE_SRC.rglob("*.py"):
        imported.update(pattern.findall(path.read_text(encoding="utf-8")))
    siblings = {name for name in imported if name != "openalpha_bridge"}

    mounted = {remote.rsplit("/", 1)[-1] for _, remote in _declared_local_packages()}
    missing = sorted(siblings - mounted)
    assert not missing, f"openalpha_bridge imports unmounted package(s): {missing}"


def test_local_paths_are_resolved_against_the_repository_root() -> None:
    """A relative path would depend on the directory `modal deploy` runs from."""
    source = _source()
    assert "def _repo_root()" in source
    assert "Path(__file__).resolve()" in source
    assert "root / local_path" in source


def test_repository_root_lookup_tolerates_a_shallow_path() -> None:
    """Modal flattens the module to /root/<name>.py, which has only two parents.

    Indexing parents[2] unconditionally raises IndexError there and every
    container crash-loops on startup.
    """
    source = _source()
    assert "len(here.parents) < 3" in source, (
        "the repository-root lookup must tolerate a flattened container path"
    )
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "REPO_ROOT" in names:
                pytest.fail(
                    "REPO_ROOT must not be computed at module level; the "
                    "container re-imports this module from a flattened path"
                )


def test_research_lock_directory_is_mounted() -> None:
    source = _source()
    assert '"research" / "bridge-v0"' in source
    assert "/root/research/bridge-v0" in source


def test_bytecode_is_excluded_from_the_image() -> None:
    """Host bytecode must not shadow the image's own modules."""
    tree = ast.parse(_source())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "_IGNORE" in names:
                patterns = ast.literal_eval(node.value)
                assert any("__pycache__" in p for p in patterns)
                assert any(p.endswith(".pyc") for p in patterns)
                return
    pytest.fail("the image does not declare an ignore list for bytecode")


@pytest.mark.network
def test_app_imports_where_the_workspace_is_not_installed() -> None:
    """Reproduce the deploy runner: modal present, workspace packages absent."""
    if shutil.which("uvx") is None and shutil.which("uv") is None:
        pytest.skip("uv is required to build an isolated environment")

    probe = "\n".join(
        (
            "import importlib.util",
            (
                "assert importlib.util.find_spec('openalpha_bridge') is None, "
                "'workspace leaked into the isolated environment'"
            ),
            f"spec = importlib.util.spec_from_file_location('app', r'{APP_PATH}')",
            "mod = importlib.util.module_from_spec(spec)",
            "spec.loader.exec_module(mod)",
            "print('OK', len(mod.LOCAL_PACKAGES))",
        )
    )

    result = subprocess.run(
        ["uvx", "--from", "modal>=0.64,<2", "python", "-c", probe],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=600,
        check=False,
        # Importing to inspect, not to deploy. A developer worktree is dirty
        # almost always, and the deploy-time binding refuses a dirty tree. Under
        # this flag the image is built with an inert sentinel instead.
        env={**os.environ, "OPENALPHA_MODAL_INSPECTION": "1"},
    )
    assert result.returncode == 0, (
        "the Modal app could not be imported without the workspace installed:\n"
        f"{result.stdout}\n{result.stderr}"
    )
    assert "OK 3" in result.stdout


@pytest.mark.network
def test_app_imports_when_flattened_like_the_container(tmp_path: Path) -> None:
    """Reproduce the container layout exactly: the module alone at /root.

    The previous isolated-import test ran the file from its repository location,
    where parents[2] resolves, so it could not catch the flattening failure.
    """
    if shutil.which("uvx") is None and shutil.which("uv") is None:
        pytest.skip("uv is required to build an isolated environment")

    # A temporary directory is far too deep to reproduce this: the container
    # path /root/<name>.py has exactly two parents, which is what makes
    # parents[2] raise. Rebinding the module's __file__ reproduces that depth
    # faithfully on both Linux and Windows.
    probe = "\n".join(
        (
            "import importlib.util",
            f"spec = importlib.util.spec_from_file_location('app', r'{APP_PATH}')",
            "mod = importlib.util.module_from_spec(spec)",
            "spec.loader.exec_module(mod)",
            "mod.__file__ = '/root/bridge_phase2_app.py'",
            "assert mod._repo_root() is None, 'a flattened module must find no repository'",
            "img = mod._build_image()",
            "assert img is not None, 'image construction must survive flattening'",
            "print('FLATTENED OK')",
        )
    )

    result = subprocess.run(
        ["uvx", "--from", "modal>=0.64,<2", "python", "-c", probe],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=600,
        check=False,
        env={**os.environ, "OPENALPHA_MODAL_INSPECTION": "1"},
    )
    assert result.returncode == 0, (
        "the Modal app crashes when imported from a flattened container path, "
        "which makes every container crash-loop on startup:\n"
        f"{result.stdout}\n{result.stderr}"
    )
    assert "FLATTENED OK" in result.stdout


def test_app_module_imports_nothing_from_the_workspace_at_module_level() -> None:
    """Module-level workspace imports would break the deploy runner outright."""
    tree = ast.parse(_source())
    offenders: list[str] = []
    for node in tree.body:  # module level only
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("openalpha_"):
            offenders.append(f"line {node.lineno}: from {node.module}")
        if isinstance(node, ast.Import):
            offenders.extend(
                f"line {node.lineno}: import {alias.name}"
                for alias in node.names
                if alias.name.startswith("openalpha_")
            )
    assert not offenders, f"workspace imports must stay inside functions, found: {offenders}"


def test_image_pins_and_verifies_the_official_kronos_source() -> None:
    """The image must clone, detach, and verify before anything can run."""
    source = _source()
    assert "git clone" in source
    assert "git checkout --detach" in source
    assert "67b630e67f6a18c9e9be918d9b4337c960db1e9a" in source
    assert "KRONOS_SOURCE_REVISION_MISMATCH" in source
    assert "KRONOS_WORKTREE_DIRTY" in source
    assert "KRONOS_SOURCE_HASH_MISMATCH" in source
    assert "OPENALPHA_KRONOS_SOURCE_PATH" in source


def test_the_image_normalizes_line_endings_before_hashing_the_source() -> None:
    """The sealed digests are over CRLF-normalized content.

    `sha256sum -c` on the checked-out bytes was wrong and would have failed
    every image build: model/kronos.py is committed upstream with LF, and the
    digest experiment.yaml seals is the digest of its CRLF form. Normalizing to
    LF first keeps model/module.py correct, which is committed with CRLF and
    must not be doubled.
    """
    source = _source()
    assert "sha256sum -c" not in source
    # Anchored without backslashes: the shell snippet is escaped twice over
    # (Python source, then f-string), which makes a literal match unreadable.
    assert "sed -e " in source
    assert source.count("sed -e ") == 1
    assert "sha256sum | cut -d' ' -f1" in source


def test_image_pins_official_runtime_dependencies_explicitly() -> None:
    """The official source imports these; none may be transitive."""
    source = _source()
    for package in ("pandas", "tqdm", "einops"):
        assert f'"{package}' in source, f"{package} is not pinned in the image"


def test_image_source_constants_match_the_locked_spec() -> None:
    """The Modal constants must not drift from openalpha_bridge SOURCE_SPEC."""
    tree = ast.parse(_source())
    found: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.AnnAssign | ast.Assign):
            targets = [node.target] if isinstance(node, ast.AnnAssign) else node.targets
            for target in targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id.startswith("KRONOS_SOURCE")
                    and node.value is not None
                ):
                    found[target.id] = ast.literal_eval(node.value)

    assert found["KRONOS_SOURCE_REVISION"] == "67b630e67f6a18c9e9be918d9b4337c960db1e9a"
    assert found["KRONOS_SOURCE_REPOSITORY"] == "https://github.com/shiyu-coder/Kronos"
    files = found["KRONOS_SOURCE_FILES"]
    assert isinstance(files, dict)
    assert files == {
        "model/kronos.py": ("638a56e035856c600c9848b368be087cb706a61603a0790124968c95b8c69f3a"),
        "model/module.py": ("a07edbadc0e96804c8158c021bbc6063bb7cc43b34d7fc470d5c8ff2005a409f"),
    }


def _image_pins() -> dict[str, str]:
    """Read CANARY_RUNTIME_PINS from the Modal app without importing modal."""
    tree = ast.parse(_source())
    for node in tree.body:
        if isinstance(node, ast.AnnAssign | ast.Assign):
            targets = [node.target] if isinstance(node, ast.AnnAssign) else node.targets
            names = [t.id for t in targets if isinstance(t, ast.Name)]
            if "CANARY_RUNTIME_PINS" in names and node.value is not None:
                return ast.literal_eval(node.value)
    pytest.fail("the Modal app declares no CANARY_RUNTIME_PINS")


def test_image_pins_match_the_runtime_manifest() -> None:
    """The image literals and the package manifest must never drift."""
    import sys

    sys.path.insert(0, str(ROOT / "packages" / "bridge" / "src"))
    try:
        from openalpha_bridge.phase2.runtime_pins import CANARY_RUNTIME_PINS
    finally:
        sys.path.pop(0)
    assert _image_pins() == CANARY_RUNTIME_PINS


def test_every_canary_dependency_is_pinned_exactly() -> None:
    pins = _image_pins()
    required = {
        "torch",
        "numpy",
        "pandas",
        "tqdm",
        "einops",
        "huggingface-hub",
        "safetensors",
        "yfinance",
        "pydantic",
    }
    assert required <= set(pins)
    for name, version in pins.items():
        assert version[0].isdigit(), f"{name} pin {version!r} is not an exact version"
        for operator in (">=", "<=", "~=", ">", "<", "*"):
            assert operator not in version, f"{name} pin {version!r} is a range"


def test_image_installs_no_ranged_canary_dependency() -> None:
    """A ranged specifier for a canary package would defeat the pinning."""
    source = _source()
    for name in ("numpy", "pandas", "tqdm", "einops", "safetensors", "yfinance"):
        assert f'"{name}>=' not in source, f"{name} is installed with a range"


def test_pins_match_the_lockfile() -> None:
    """uv.lock is the source of authority; the manifest must agree with it."""
    import re

    text = (ROOT / "uv.lock").read_text(encoding="utf-8")
    resolved: dict[str, str] = {}
    for block in text.split("[[package]]"):
        name = re.search(r'^name = "([^"]+)"', block, re.MULTILINE)
        version = re.search(r'^version = "([^"]+)"', block, re.MULTILINE)
        if name and version:
            resolved[name.group(1)] = version.group(1)

    for name, pinned in _image_pins().items():
        assert name in resolved, f"{name} is pinned but absent from uv.lock"
        assert resolved[name] == pinned, (
            f"{name} pinned {pinned} but uv.lock resolves {resolved[name]}"
        )
