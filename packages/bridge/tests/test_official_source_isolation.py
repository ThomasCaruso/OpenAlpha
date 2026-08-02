"""The unverified upstream `model/__init__.py` must never execute."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from openalpha_bridge.errors import BridgeTransformError
from openalpha_bridge.phase2.kronos import SOURCE_SPEC
from openalpha_bridge.phase2.official_source import (
    OFFICIAL_RUNTIME_DEPENDENCIES,
    SYNTHETIC_PACKAGE,
    load_official_kronos,
    official_init_was_executed,
    verify_official_source,
)

MARKER = "OPENALPHA_INIT_EXECUTED"


def _fake_source(root: Path, *, kronos_body: str = "", init_body: str = "") -> Path:
    model = root / "model"
    model.mkdir(parents=True, exist_ok=True)
    # An __init__.py that would be obvious if it ever ran.
    (model / "__init__.py").write_text(
        init_body or f"import os\nos.environ['{MARKER}'] = '1'\n", encoding="utf-8"
    )
    (model / "module.py").write_text("MODULE_MARKER = 'module'\n", encoding="utf-8")
    (model / "kronos.py").write_text(
        kronos_body
        or (
            "from .module import MODULE_MARKER\n\n\n"
            "class KronosTokenizer:\n"
            "    marker = MODULE_MARKER\n\n"
            "    @classmethod\n"
            "    def from_pretrained(cls, path):\n"
            "        return cls()\n"
        ),
        encoding="utf-8",
    )
    return root


def _hashes(root: Path) -> dict[str, str]:
    import hashlib

    return {
        rel: hashlib.sha256((root / rel).read_bytes()).hexdigest()
        for rel in ("model/kronos.py", "model/module.py")
    }


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in list(sys.modules):
        if name.startswith(SYNTHETIC_PACKAGE):
            del sys.modules[name]
    monkeypatch.delenv(MARKER, raising=False)
    yield
    for name in list(sys.modules):
        if name.startswith(SYNTHETIC_PACKAGE):
            del sys.modules[name]


def test_upstream_init_is_never_executed(tmp_path: Path) -> None:
    import os

    root = _fake_source(tmp_path)
    module, observed, versions = load_official_kronos(root, _hashes(root), required_dependencies=("numpy",))
    assert module.KronosTokenizer.marker == "module"
    assert os.environ.get(MARKER) is None, "the unverified __init__.py executed"
    assert not official_init_was_executed()
    assert set(observed) == {"model/kronos.py", "model/module.py"}
    assert set(versions) >= {"numpy"}


def test_relative_imports_resolve_inside_the_synthetic_package(tmp_path: Path) -> None:
    root = _fake_source(tmp_path)
    load_official_kronos(root, _hashes(root), required_dependencies=("numpy",))
    assert f"{SYNTHETIC_PACKAGE}.module" in sys.modules
    assert f"{SYNTHETIC_PACKAGE}.kronos" in sys.modules
    assert "model" not in sys.modules


def test_a_tampered_locked_file_is_rejected_before_execution(tmp_path: Path) -> None:
    import os

    root = _fake_source(tmp_path)
    hashes = _hashes(root)
    (root / "model" / "kronos.py").write_text("raise RuntimeError('should never run')\n", encoding="utf-8")
    with pytest.raises(BridgeTransformError) as excinfo:
        load_official_kronos(root, hashes, required_dependencies=("numpy",))
    assert excinfo.value.failures[0].code == "KRONOS_SOURCE_HASH_MISMATCH"
    assert os.environ.get(MARKER) is None


def test_a_missing_locked_file_is_rejected(tmp_path: Path) -> None:
    root = _fake_source(tmp_path)
    hashes = _hashes(root)
    (root / "model" / "module.py").unlink()
    with pytest.raises(BridgeTransformError) as excinfo:
        verify_official_source(root, hashes)
    assert excinfo.value.failures[0].code == "KRONOS_SOURCE_FILE_MISSING"


def test_a_missing_source_root_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(BridgeTransformError) as excinfo:
        verify_official_source(tmp_path / "absent", {"model/kronos.py": "0" * 64})
    assert excinfo.value.failures[0].code == "KRONOS_SOURCE_UNAVAILABLE"


def test_source_without_a_tokenizer_is_rejected(tmp_path: Path) -> None:
    root = _fake_source(tmp_path, kronos_body="NOTHING = 1\n")
    with pytest.raises(BridgeTransformError) as excinfo:
        load_official_kronos(root, _hashes(root), required_dependencies=("numpy",))
    assert excinfo.value.failures[0].code == "KRONOS_TOKENIZER_NOT_FOUND"


def test_a_missing_runtime_dependency_is_named(tmp_path: Path) -> None:
    root = _fake_source(tmp_path)
    with pytest.raises(BridgeTransformError) as excinfo:
        load_official_kronos(
            root, _hashes(root), required_dependencies=("definitely_absent_pkg",)
        )
    failure = excinfo.value.failures[0]
    assert failure.code == "MISSING_OFFICIAL_RUNTIME_DEPENDENCY"
    assert "definitely_absent_pkg" in failure.message


def test_runtime_dependencies_are_pinned_explicitly() -> None:
    """The official source imports these; none may be relied on transitively."""
    assert set(OFFICIAL_RUNTIME_DEPENDENCIES) >= {
        "torch",
        "numpy",
        "pandas",
        "tqdm",
        "einops",
    }


def test_synthetic_package_is_not_named_model() -> None:
    """A real `model` package on sys.path must never satisfy the import."""
    assert SYNTHETIC_PACKAGE != "model"
    assert SYNTHETIC_PACKAGE.startswith("openalpha")


def test_locked_source_spec_matches_the_experiment() -> None:
    assert SOURCE_SPEC.revision == "67b630e67f6a18c9e9be918d9b4337c960db1e9a"
    assert set(SOURCE_SPEC.files) == {"model/kronos.py", "model/module.py"}
