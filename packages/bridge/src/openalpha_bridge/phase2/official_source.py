"""Isolated loader for the pinned official Kronos source.

The upstream package is `model/` with `__init__.py`, `kronos.py`, and
`module.py`. Only the latter two are hash-locked by the experiment, so a plain
`import model` would execute `__init__.py`, which is unverified code, and would
also drag in whatever else that file chooses to import.

This loader builds a synthetic package whose `__path__` points at the source
directory and executes **only** the two verified files inside it. Relative
imports such as `from .module import ...` resolve within the synthetic package,
so no upstream `__init__.py` ever runs.

Every runtime dependency of the official source is pinned explicitly and checked
before execution; nothing is relied on transitively.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

from ..errors import BridgeFailure, BridgeTransformError, FailureCategory

__all__ = [
    "OFFICIAL_RUNTIME_DEPENDENCIES",
    "SYNTHETIC_PACKAGE",
    "load_official_kronos",
    "verify_official_source",
]

#: The synthetic package name. Deliberately not "model", so an unverified
#: upstream package on sys.path can never satisfy the import instead.
SYNTHETIC_PACKAGE = "openalpha_kronos_official"

#: Executed in dependency order: kronos.py imports from module.py.
_VERIFIED_MODULES: tuple[str, ...] = ("module", "kronos")

#: Explicit runtime dependencies of the pinned official source. Pinned here
#: rather than assumed transitively, so a missing one is a named failure.
OFFICIAL_RUNTIME_DEPENDENCIES: tuple[str, ...] = (
    "torch",
    "numpy",
    "pandas",
    "tqdm",
    "einops",
)


def _fail(code: str, message: str, *, field: str | None = None) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            field=field,
            message=message,
        )
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_official_source(root: Path, expected: dict[str, str]) -> dict[str, str]:
    """Verify every locked file byte-for-byte before any of it executes."""
    if not root.is_dir():
        raise _fail(
            "KRONOS_SOURCE_UNAVAILABLE",
            f"pinned official source directory not found: {root}",
        )
    observed: dict[str, str] = {}
    for relative, want in expected.items():
        path = root.joinpath(*relative.split("/"))
        if not path.is_file():
            raise _fail(
                "KRONOS_SOURCE_FILE_MISSING",
                f"pinned source file not found: {relative} under {root}",
                field=relative,
            )
        got = _file_sha256(path)
        observed[relative] = got
        if got != want:
            raise _fail(
                "KRONOS_SOURCE_HASH_MISMATCH",
                f"{relative} expected {want}, observed {got}",
                field=relative,
            )
    return observed


def _require_runtime_dependencies(
    names: tuple[str, ...] = OFFICIAL_RUNTIME_DEPENDENCIES,
) -> dict[str, str]:
    """Fail with the missing dependency named, never on a transitive accident."""
    versions: dict[str, str] = {}
    missing: list[str] = []
    for name in names:
        try:
            module = importlib.import_module(name)
        except ImportError:
            missing.append(name)
            continue
        versions[name] = str(getattr(module, "__version__", "unknown"))
    if missing:
        raise _fail(
            "MISSING_OFFICIAL_RUNTIME_DEPENDENCY",
            (
                "the pinned official source requires "
                f"{', '.join(missing)}; pin them in the execution image"
            ),
        )
    return versions


def load_official_kronos(
    root: Path,
    expected_files: dict[str, str],
    *,
    required_dependencies: tuple[str, ...] = OFFICIAL_RUNTIME_DEPENDENCIES,
) -> tuple[Any, dict[str, str], dict[str, str]]:
    """Load the verified official modules in isolation.

    Returns the loaded ``kronos`` module, the observed source-file hashes, and
    the observed runtime dependency versions.
    """
    observed = verify_official_source(root, expected_files)
    dependency_versions = _require_runtime_dependencies(required_dependencies)

    model_dir = root / "model"
    if not model_dir.is_dir():
        raise _fail(
            "KRONOS_SOURCE_LAYOUT_UNEXPECTED",
            f"expected a model/ directory under {root}",
        )

    # The pinned kronos.py does NOT use a relative import. It contains:
    #     sys.path.append("../")
    #     from model.module import *
    # So an absolute `model.module` must resolve. Rather than let Python find
    # the upstream package, which would execute the unverified model/__init__.py,
    # a non-executable alias package is installed for the duration of the load
    # and mapped to the module we have already verified and executed ourselves.
    for occupied in ("model", "model.module"):
        if occupied in sys.modules:
            raise _fail(
                "KRONOS_MODEL_NAMESPACE_OCCUPIED",
                (
                    f"sys.modules already contains {occupied!r}; refusing to alias over "
                    "an unrelated package"
                ),
                field=occupied,
            )

    package = sys.modules.get(SYNTHETIC_PACKAGE)
    if package is None:
        package = types.ModuleType(SYNTHETIC_PACKAGE)
        package.__path__ = [str(model_dir)]  # type: ignore[attr-defined]
        package.__doc__ = (
            "Synthetic package holding only hash-verified official Kronos modules."
        )
        sys.modules[SYNTHETIC_PACKAGE] = package

    def _execute(name: str) -> Any:
        qualified = f"{SYNTHETIC_PACKAGE}.{name}"
        if qualified in sys.modules:
            return sys.modules[qualified]
        relative = f"model/{name}.py"
        if relative not in observed:
            raise _fail(
                "KRONOS_SOURCE_FILE_NOT_LOCKED",
                f"{relative} is executed but not hash-locked; refusing to run it",
                field=relative,
            )
        spec = importlib.util.spec_from_file_location(qualified, model_dir / f"{name}.py")
        if spec is None or spec.loader is None:
            raise _fail(
                "KRONOS_SOURCE_IMPORT_FAILED",
                f"could not build an import spec for {relative}",
                field=relative,
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        try:
            spec.loader.exec_module(module)
        except Exception as error:
            sys.modules.pop(qualified, None)
            raise _fail(
                "KRONOS_SOURCE_IMPORT_FAILED",
                f"executing {relative} failed: {type(error).__name__}: {error}",
                field=relative,
            ) from error
        return module

    # The pinned kronos.py does `sys.path.append("../")`. That mutation must
    # not survive the load, so the entry list is snapshotted and restored.
    original_sys_path = list(sys.path)
    try:
        verified_module = _execute("module")
    except BaseException:
        sys.path[:] = original_sys_path
        raise

    # A bare namespace object with no loader and no __path__: nothing can be
    # imported *through* it, and its body is never executed.
    alias = types.ModuleType("model")
    alias.__doc__ = "Temporary alias to hash-verified official modules. Not the upstream package."
    sys.modules["model"] = alias
    sys.modules["model.module"] = verified_module
    alias.module = verified_module  # type: ignore[attr-defined]
    try:
        loaded: Any = _execute("kronos")
    finally:
        # The alias exists only for the duration of the verified execution, and
        # sys.path is restored exactly, on success and on failure alike.
        sys.modules.pop("model.module", None)
        sys.modules.pop("model", None)
        sys.path[:] = original_sys_path

    if loaded is None or not hasattr(loaded, "KronosTokenizer"):
        available = sorted(n for n in dir(loaded) if not n.startswith("_")) if loaded else []
        raise _fail(
            "KRONOS_TOKENIZER_NOT_FOUND",
            f"the verified official source exposes no KronosTokenizer; found {available}",
        )
    return loaded, observed, dependency_versions


def official_init_was_executed() -> bool:
    """True when an upstream `model` package was imported, which must not happen."""
    return "model" in sys.modules
