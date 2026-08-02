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
    "TORCH_CUDA_INDEX",
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

#: Torch is installed from the CUDA wheel index. The version above is still the
#: lockfile's; only the wheel variant differs.
TORCH_CUDA_INDEX: Final[str] = "https://download.pytorch.org/whl/cu124"

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
