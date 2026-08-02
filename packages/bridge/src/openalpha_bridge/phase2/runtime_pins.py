"""Exact runtime pins for the canary execution image.

Every version here was read from ``uv.lock`` after the project declared the
package, so the lockfile is the source of authority rather than a guess. The
Modal image imports these constants, and a packaging test asserts the image and
this manifest cannot drift apart.

Regenerating: declare the package in ``packages/bridge/pyproject.toml``, run
``uv lock``, then copy the resolved version here. Never hand-pick a version.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "CANARY_RUNTIME_PINS",
    "PIN_AUTHORITY",
    "TORCH_WHEEL_FILENAME",
    "TORCH_WHEEL_INDEX",
    "TORCH_WHEEL_MINIMUM_DRIVER",
    "TORCH_WHEEL_MINIMUM_GLIBC",
    "TORCH_WHEEL_PLATFORM_TAG",
    "TORCH_WHEEL_PYTHON_TAG",
    "TORCH_WHEEL_SHA256",
    "TORCH_WHEEL_SIZE_BYTES",
    "TORCH_WHEEL_SPECIFIER",
    "TORCH_WHEEL_URL",
    "pin_specifiers",
]

#: Where each pin came from. Recorded so a reviewer can re-derive it.
PIN_AUTHORITY: Final[str] = (
    "uv.lock, resolved from packages/bridge/pyproject.toml optional-dependencies"
)

#: Exact versions, lockfile-resolved. Keys are PyPI distribution names.
CANARY_RUNTIME_PINS: Final[dict[str, str]] = {
    "torch": "2.13.0",
    "numpy": "2.5.1",
    "pandas": "3.0.5",
    "tqdm": "4.70.0",
    "einops": "0.8.2",
    "huggingface-hub": "0.36.2",
    "safetensors": "0.8.0",
    "yfinance": "1.5.2",
    "pydantic": "2.13.4",
}

#: The exact Torch build the image installs, identified as an artifact rather
#: than as a requirement.
#:
#: This was ``torch==2.13.0`` resolved against the cu124 index, which does not
#: work: that index stops at torch 2.6.0, so the requirement fell through to the
#: default index and installed a different build of 2.13.0 than the one that was
#: named. Installing by exact URL removes the possibility. pip verifies the
#: ``#sha256=`` fragment on a direct URL, and a direct URL cannot be satisfied by
#: any other artifact, so there is no fallback to reason about.
#:
#: Resolved from the official index listing at
#: https://download.pytorch.org/whl/cu126/torch/
TORCH_WHEEL_PYTHON_TAG: Final[str] = "cp313"  # CPython 3.13
TORCH_WHEEL_PLATFORM_TAG: Final[str] = "manylinux_2_28_x86_64"  # Linux x86_64, glibc >= 2.28
TORCH_WHEEL_LOCAL_VERSION: Final[str] = "cu126"  # CUDA 12.6 runtime bundled in the wheel
TORCH_WHEEL_FILENAME: Final[str] = (
    f"torch-{CANARY_RUNTIME_PINS['torch']}%2B{TORCH_WHEEL_LOCAL_VERSION}"
    f"-{TORCH_WHEEL_PYTHON_TAG}-{TORCH_WHEEL_PYTHON_TAG}-{TORCH_WHEEL_PLATFORM_TAG}.whl"
)
TORCH_WHEEL_INDEX: Final[str] = f"https://download.pytorch.org/whl/{TORCH_WHEEL_LOCAL_VERSION}"
TORCH_WHEEL_URL: Final[str] = f"{TORCH_WHEEL_INDEX}/{TORCH_WHEEL_FILENAME}"

#: Published by the official index alongside the wheel. pip checks the artifact
#: against this before installing it.
TORCH_WHEEL_SHA256: Final[str] = "4198c8d7478ab47ad2569309387d88b21fb553a1cf8ab06260fbd5a6ab9b9712"
TORCH_WHEEL_SIZE_BYTES: Final[int] = 843_741_728

#: What .pip_install receives. The fragment is what makes the install verified.
TORCH_WHEEL_SPECIFIER: Final[str] = f"{TORCH_WHEEL_URL}#sha256={TORCH_WHEEL_SHA256}"

#: Debian bookworm ships glibc 2.36, so manylinux_2_28 is satisfied. Recorded so
#: the compatibility argument is checkable rather than assumed.
TORCH_WHEEL_MINIMUM_GLIBC: Final[str] = "2.28"

#: CUDA 12.6 runs on any driver supporting CUDA 12.x under minor version
#: compatibility, which is every driver Modal provisions for its GPU classes.
TORCH_WHEEL_MINIMUM_DRIVER: Final[str] = "525.60.13"

#: Import name -> distribution name, where they differ.
_IMPORT_TO_DISTRIBUTION: Final[dict[str, str]] = {
    "huggingface_hub": "huggingface-hub",
}


def distribution_for(import_name: str) -> str:
    return _IMPORT_TO_DISTRIBUTION.get(import_name, import_name)


def pin_specifiers(*names: str) -> tuple[str, ...]:
    """Exact ``name==version`` specifiers, in the order requested."""
    missing = [name for name in names if name not in CANARY_RUNTIME_PINS]
    if missing:
        raise KeyError(f"no lockfile-resolved pin for: {', '.join(sorted(missing))}")
    return tuple(f"{name}=={CANARY_RUNTIME_PINS[name]}" for name in names)
