"""Packaging guards for the Modal application.

`modal deploy` runs from an isolated environment in which the workspace packages
are not installed. The research shell must therefore copy its workspace
packages by explicit path without importing them in the deploying process.

These are static and subprocess checks. They deploy nothing and contact no cloud
provider.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[2]
APP_PATH = ROOT / "cloud" / "modal" / "kronos_research.py"
PACKAGE_SOURCES = (
    ROOT / "packages" / "kronos-research" / "src",
    ROOT / "packages" / "research-core" / "src",
)
PACKAGE_MANIFESTS = (
    ROOT / "packages" / "research-core" / "pyproject.toml",
    ROOT / "packages" / "kronos-research" / "pyproject.toml",
)
SELECTED_KRONOS_EXTRAS = frozenset({"yahoo", "runtime", "gpu", "modal"})
PLATFORM_SDK_EXEMPTIONS = frozenset({"modal"})
S3_CLIENT_STACK = frozenset(
    {
        "boto3",
        "botocore",
        "s3transfer",
        "jmespath",
        "python-dateutil",
        "urllib3",
        "six",
    }
)


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


def test_exactly_the_two_research_workspace_packages_are_mounted() -> None:
    mounted = {remote.rsplit("/", 1)[-1] for _, remote in _declared_local_packages()}
    assert mounted == {"openalpha_kronos", "openalpha_research"}


def test_every_workspace_package_the_research_code_imports_is_mounted() -> None:
    """A runtime import of an unmounted package would fail only in the cloud."""
    pattern = re.compile(r"^\s*(?:from|import)\s+(openalpha_[a-z_]+)", re.MULTILINE)
    imported: set[str] = set()
    for source_root in PACKAGE_SOURCES:
        for path in source_root.rglob("*.py"):
            imported.update(pattern.findall(path.read_text(encoding="utf-8")))
    siblings = set(imported)

    mounted = {remote.rsplit("/", 1)[-1] for _, remote in _declared_local_packages()}
    missing = sorted(siblings - mounted)
    assert not missing, f"research code imports unmounted package(s): {missing}"


def test_only_completed_study_secrets_are_attached() -> None:
    tree = ast.parse(_source())
    names = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "SECRET_NAMES"
    )
    assert names == ("openalpha-storage", "openalpha-huggingface")


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


def test_app_imports_where_the_workspace_is_not_installed() -> None:
    """Import with a local Modal double and no workspace module import."""

    probe = "\n".join(
        (
            "import importlib.util, sys, types",
            "assert importlib.util.find_spec('openalpha_kronos') is None",
            "assert importlib.util.find_spec('openalpha_research') is None",
            "class Fake:",
            "    def __getattr__(self, name): return self",
            "    def __call__(self, *args, **kwargs):",
            "        return args[0] if len(args) == 1 and callable(args[0]) and not kwargs else self",
            "fake = Fake()",
            "modal = types.ModuleType('modal')",
            "modal.Image = modal.App = modal.Volume = modal.Secret = fake",
            "sys.modules['modal'] = modal",
            f"spec = importlib.util.spec_from_file_location('app', r'{APP_PATH}')",
            "mod = importlib.util.module_from_spec(spec)",
            "spec.loader.exec_module(mod)",
            "print('OK', len(mod.LOCAL_PACKAGES))",
        )
    )

    result = subprocess.run(
        [sys.executable, "-S", "-c", probe],
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
    assert "OK 2" in result.stdout


def test_app_imports_when_flattened_like_the_container(tmp_path: Path) -> None:
    """Reproduce the container layout exactly: the module alone at /root.

    The previous isolated-import test ran the file from its repository location,
    where parents[2] resolves, so it could not catch the flattening failure.
    """
    # A temporary directory is far too deep to reproduce this: the container
    # path /root/<name>.py has exactly two parents, which is what makes
    # parents[2] raise. Rebinding the module's __file__ reproduces that depth
    # faithfully on both Linux and Windows.
    probe = "\n".join(
        (
            "import importlib.util, sys, types",
            "assert importlib.util.find_spec('openalpha_kronos') is None",
            "assert importlib.util.find_spec('openalpha_research') is None",
            "class Fake:",
            "    def __getattr__(self, name): return self",
            "    def __call__(self, *args, **kwargs):",
            "        return args[0] if len(args) == 1 and callable(args[0]) and not kwargs else self",
            "fake = Fake()",
            "modal = types.ModuleType('modal')",
            "modal.Image = modal.App = modal.Volume = modal.Secret = fake",
            "sys.modules['modal'] = modal",
            f"spec = importlib.util.spec_from_file_location('app', r'{APP_PATH}')",
            "mod = importlib.util.module_from_spec(spec)",
            "spec.loader.exec_module(mod)",
            "mod.__file__ = '/root/kronos_research.py'",
            "assert mod._repo_root() is None, 'a flattened module must find no repository'",
            "img = mod._build_image()",
            "assert img is not None, 'image construction must survive flattening'",
            "print('FLATTENED OK')",
        )
    )

    result = subprocess.run(
        [sys.executable, "-S", "-c", probe],
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
    pinned = {
        ast.literal_eval(node.args[0])
        for node in ast.walk(ast.parse(_source()))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_pin"
        and len(node.args) == 1
    }
    for package in ("pandas", "tqdm", "einops"):
        assert package in pinned, f"{package} is not pinned in the image"


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
    """Read RESEARCH_RUNTIME_PINS without importing the Modal SDK."""
    tree = ast.parse(_source())
    for node in tree.body:
        if isinstance(node, ast.AnnAssign | ast.Assign):
            targets = [node.target] if isinstance(node, ast.AnnAssign) else node.targets
            names = [t.id for t in targets if isinstance(t, ast.Name)]
            if "RESEARCH_RUNTIME_PINS" in names and node.value is not None:
                return ast.literal_eval(node.value)
    pytest.fail("the Modal app declares no RESEARCH_RUNTIME_PINS")


def _manifest_requirements() -> dict[str, list[Requirement]]:
    """Resolve direct requirements through selected workspace-package extras."""
    manifests: dict[str, dict[str, object]] = {}
    for path in PACKAGE_MANIFESTS:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
        project = document["project"]
        assert isinstance(project, dict)
        manifests[canonicalize_name(str(project["name"]))] = document

    root = canonicalize_name("openalpha-kronos-research")
    selected: dict[str, set[str]] = {root: set(SELECTED_KRONOS_EXTRAS)}
    external: dict[str, list[Requirement]] = {}
    changed = True
    while changed:
        changed = False
        external.clear()
        for package_name, extras in tuple(selected.items()):
            project = manifests[package_name]["project"]
            assert isinstance(project, dict)
            raw = list(project.get("dependencies", []))
            optional = project.get("optional-dependencies", {})
            assert isinstance(optional, dict)
            for extra in extras:
                raw.extend(optional[extra])
            for text in raw:
                requirement = Requirement(str(text))
                name = canonicalize_name(requirement.name)
                if name not in manifests:
                    external.setdefault(name, []).append(requirement)
                    continue
                prior = selected.setdefault(name, set())
                discovered = set(requirement.extras) - prior
                if discovered:
                    prior.update(discovered)
                    changed = True
    return external


def _installed_pin_names() -> set[str]:
    tree = ast.parse(_source())
    builder = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_build_image"
    )
    return {
        canonicalize_name(ast.literal_eval(node.args[0]))
        for node in ast.walk(builder)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_pin"
        and len(node.args) == 1
    }


def _lock_versions() -> dict[str, str]:
    document = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    return {
        canonicalize_name(package["name"]): package["version"]
        for package in document["package"]
    }


def test_selected_manifest_dependency_closure_is_installed_and_satisfied() -> None:
    requirements = _manifest_requirements()
    assert set(requirements) == {
        "boto3",
        "einops",
        "exchange-calendars",
        "huggingface-hub",
        "modal",
        "numpy",
        "pandas",
        "pydantic",
        "safetensors",
        "torch",
        "tqdm",
        "yfinance",
    }

    special = PLATFORM_SDK_EXEMPTIONS | {"torch"}
    required_pins = set(requirements) - special
    pins = _image_pins()
    installed = _installed_pin_names()
    assert not required_pins - installed, (
        f"selected manifest dependencies missing from image: {required_pins - installed}"
    )
    for name in required_pins:
        version = Version(pins[name])
        assert all(version in requirement.specifier for requirement in requirements[name])


def test_modal_is_a_platform_sdk_exemption_not_an_in_image_pip_install() -> None:
    """Modal supplies its own SDK; installing Modal inside its image is redundant."""
    assert "modal" in _manifest_requirements()
    assert "modal" not in _image_pins()
    assert "modal" not in _installed_pin_names()
    assert "import modal" in _source()


def test_torch_uses_the_hashed_wheel_at_the_locked_version() -> None:
    assert "torch" in _manifest_requirements()
    lock_version = _lock_versions()["torch"]
    match = re.search(r"torch-([0-9.]+)%2B", _source())
    assert match is not None
    assert match.group(1) == lock_version == _image_pins()["torch"]
    assert all(
        Version(lock_version) in requirement.specifier
        for requirement in _manifest_requirements()["torch"]
    )
    assert ".pip_install(TORCH_WHEEL_SPECIFIER)" in _source()
    assert "#sha256=" in _source()
    assert _source().index(".pip_install(TORCH_WHEEL_SPECIFIER)") < _source().index(
        ".pip_install(", _source().index(".pip_install(TORCH_WHEEL_SPECIFIER)") + 1
    )


def test_s3_client_stack_is_explicitly_frozen_without_a_ranged_literal() -> None:
    assert S3_CLIENT_STACK <= _installed_pin_names()
    ranged_boto3_literals = [
        node.value
        for node in ast.walk(ast.parse(_source()))
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith("boto3")
        and node.value != "boto3"
    ]
    assert not ranged_boto3_literals


def test_every_explicit_runtime_pin_is_exact_installed_and_matches_lock() -> None:
    pins = _image_pins()
    installed = _installed_pin_names()
    assert installed == set(pins) - {"torch"}
    lock = _lock_versions()
    for name, pinned in pins.items():
        Version(pinned)
        assert pinned == lock[name], (
            f"{name} pinned {pinned} but uv.lock resolves {lock[name]}"
        )
