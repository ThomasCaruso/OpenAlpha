"""Kronos integration contract against the pinned compatibility boundary.

Interfaces are typed here; no real asset is loaded during local preparation.
Torch and ``huggingface_hub`` are resolved lazily by the official backend only.

A real run must declare ``kronos_mode: pinned_official``. A fake backend is
rejected outright in that case, so synthetic components can never produce real
evidence.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from ..errors import BridgeFailure, BridgeTransformError, FailureCategory
from ..windowing import EXAMPLE_LENGTH
from .optional import require_module

__all__ = [
    "BRIDGE_INPUT_DIMENSION",
    "DECODER_HIDDEN_DIMENSION",
    "QUANTIZED_LATENT_DIMENSION",
    "SCALE_FEATURE_COUNT",
    "SCALE_FEATURE_ORDER",
    "TOKENIZER_SPEC",
    "DeterministicFakeKronosBackend",
    "KronosBackend",
    "KronosMode",
    "OfficialKronosBackend",
    "PinnedAssetSpec",
    "ResolvedAssets",
    "assert_backend_matches_mode",
    "assert_bridge_input",
]

BRIDGE_INPUT_DIMENSION = 269
DECODER_HIDDEN_DIMENSION = 256
QUANTIZED_LATENT_DIMENSION = 20
SCALE_FEATURE_COUNT = 13
COARSE_VOCABULARY_SIZE = 1024
FINE_VOCABULARY_SIZE = 1024

SCALE_FEATURE_ORDER: tuple[str, ...] = (
    "log_anchor_close",
    "open_mean_relative_to_anchor",
    "high_mean_relative_to_anchor",
    "low_mean_relative_to_anchor",
    "close_mean_relative_to_anchor",
    "log1p_open_std_over_anchor",
    "log1p_high_std_over_anchor",
    "log1p_low_std_over_anchor",
    "log1p_close_std_over_anchor",
    "log1p_volume_mean",
    "log1p_volume_std",
    "log1p_amount_mean",
    "log1p_amount_std",
)


class KronosMode(StrEnum):
    PINNED_OFFICIAL = "pinned_official"
    FAKE = "fake"


def _fail(code: str, message: str, *, field: str | None = None) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            field=field,
            message=message,
        )
    )


class PinnedAssetSpec(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    repository: str
    revision: str = Field(min_length=40, max_length=40)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    weights_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


#: Pinned in research/bridge-v0/experiment.yaml.
TOKENIZER_SPEC = PinnedAssetSpec(
    repository="NeoQuasar/Kronos-Tokenizer-2k",
    revision="26966d0035065a0cae0ebad7af8ece35bc1fb51c",
    config_sha256="0b30a443affb03e05a876a083857de9164f899feb7b4d261da02c485c9a3e3b6",
    weights_sha256="b97ec46b3b72160509e289183eaf7bdf5f0dac5bb9b49522f6d46638a99a8717",
)


class ResolvedAssets(BaseModel):
    """Observed identities of the resolved Kronos assets."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    schema_version: Literal["openalpha.bridge.phase2.assets.v1"] = (
        "openalpha.bridge.phase2.assets.v1"
    )
    kronos_mode: KronosMode
    repository: str
    revision: str
    observed_config_sha256: str | None
    observed_weights_sha256: str | None
    frozen_parameter_sha256: str | None
    revisions_verified: bool


class KronosBackend(Protocol):
    """Frozen tokenizer, projection, and decoder-trunk operations."""

    @property
    def mode(self) -> KronosMode: ...

    def resolve_assets(self) -> ResolvedAssets: ...

    def encode_tokens(
        self, features: NDArray[np.float32]
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]: ...

    def bipolar_latent(
        self, coarse_ids: NDArray[np.int64], fine_ids: NDArray[np.int64]
    ) -> NDArray[np.float32]: ...

    def project_latent(self, latent: NDArray[np.float32]) -> NDArray[np.float32]: ...

    def decoder_trunk(self, projected: NDArray[np.float32]) -> NDArray[np.float32]: ...

    def official_decode(self, hidden: NDArray[np.float32]) -> NDArray[np.float32]: ...

    def frozen_parameter_sha256(self) -> str: ...


def assert_backend_matches_mode(backend: KronosBackend, *, declared: KronosMode) -> None:
    """A real run must use the pinned official backend."""
    if backend.mode is not declared:
        raise _fail(
            "KRONOS_MODE_MISMATCH",
            f"declared kronos_mode={declared.value} but backend reports {backend.mode.value}",
        )
    if declared is KronosMode.PINNED_OFFICIAL and isinstance(
        backend, DeterministicFakeKronosBackend
    ):
        raise _fail(
            "FAKE_BACKEND_IN_REAL_RUN",
            "a fake Kronos backend cannot serve a pinned_official run",
        )


def assert_bridge_input(tensor: NDArray[np.float32], *, sequence_length: int) -> None:
    """Assert rank, dimensions, and dtype of the assembled 269-feature tensor."""
    if tensor.ndim != 2:
        raise _fail(
            "INVALID_BRIDGE_INPUT_RANK",
            f"bridge input must be rank 2, got rank {tensor.ndim}",
        )
    if tensor.shape != (sequence_length, BRIDGE_INPUT_DIMENSION):
        raise _fail(
            "INVALID_BRIDGE_INPUT_SHAPE",
            (
                f"bridge input must be ({sequence_length}, {BRIDGE_INPUT_DIMENSION}), "
                f"got {tensor.shape}"
            ),
        )
    if tensor.dtype != np.float32:
        raise _fail(
            "INVALID_BRIDGE_INPUT_DTYPE",
            f"bridge input must be float32, got {tensor.dtype}",
        )
    if not np.isfinite(tensor).all():
        raise _fail("NON_FINITE_BRIDGE_INPUT", "bridge input contains non-finite values")


def assert_token_ranges(coarse: NDArray[np.int64], fine: NDArray[np.int64]) -> None:
    for name, values, limit in (
        ("coarse_ids", coarse, COARSE_VOCABULARY_SIZE),
        ("fine_ids", fine, FINE_VOCABULARY_SIZE),
    ):
        if values.dtype != np.int64:
            raise _fail("INVALID_TOKEN_DTYPE", f"{name} must be int64, got {values.dtype}")
        if values.size and (int(values.min()) < 0 or int(values.max()) >= limit):
            raise _fail(
                "TOKEN_ID_OUT_OF_RANGE",
                f"{name} must lie in [0, {limit - 1}]",
            )


class OfficialKronosBackend:
    """Pinned official tokenizer and frozen decoder trunk.

    Torch and ``huggingface_hub`` are imported lazily. Never instantiated during
    local pipeline preparation.
    """

    def __init__(self, *, stage: str, device: str = "cuda", cache_dir: str | None = None) -> None:
        self._stage = stage
        self._device = device
        self._cache_dir = cache_dir
        self._model = None

    @property
    def mode(self) -> KronosMode:
        return KronosMode.PINNED_OFFICIAL

    def resolve_assets(self) -> ResolvedAssets:
        hub = require_module("huggingface_hub", stage=self._stage)
        require_module("torch", stage=self._stage)

        config_path = hub.hf_hub_download(
            repo_id=TOKENIZER_SPEC.repository,
            filename="config.json",
            revision=TOKENIZER_SPEC.revision,
            cache_dir=self._cache_dir,
        )
        weights_path = hub.hf_hub_download(
            repo_id=TOKENIZER_SPEC.repository,
            filename="model.safetensors",
            revision=TOKENIZER_SPEC.revision,
            cache_dir=self._cache_dir,
        )
        observed_config = _file_sha256(config_path)
        observed_weights = _file_sha256(weights_path)

        if observed_config != TOKENIZER_SPEC.config_sha256:
            raise _fail(
                "TOKENIZER_CONFIG_HASH_MISMATCH",
                f"expected {TOKENIZER_SPEC.config_sha256}, observed {observed_config}",
            )
        if observed_weights != TOKENIZER_SPEC.weights_sha256:
            raise _fail(
                "TOKENIZER_WEIGHTS_HASH_MISMATCH",
                f"expected {TOKENIZER_SPEC.weights_sha256}, observed {observed_weights}",
            )

        return ResolvedAssets(
            kronos_mode=self.mode,
            repository=TOKENIZER_SPEC.repository,
            revision=TOKENIZER_SPEC.revision,
            observed_config_sha256=observed_config,
            observed_weights_sha256=observed_weights,
            frozen_parameter_sha256=None,
            revisions_verified=True,
        )

    def encode_tokens(
        self, features: NDArray[np.float32]
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        raise _fail(
            "OFFICIAL_BACKEND_NOT_LOADED",
            "call resolve_assets and load the pinned tokenizer before encoding",
        )

    def bipolar_latent(
        self, coarse_ids: NDArray[np.int64], fine_ids: NDArray[np.int64]
    ) -> NDArray[np.float32]:
        raise _fail("OFFICIAL_BACKEND_NOT_LOADED", "pinned tokenizer is not loaded")

    def project_latent(self, latent: NDArray[np.float32]) -> NDArray[np.float32]:
        raise _fail("OFFICIAL_BACKEND_NOT_LOADED", "pinned projection is not loaded")

    def decoder_trunk(self, projected: NDArray[np.float32]) -> NDArray[np.float32]:
        raise _fail("OFFICIAL_BACKEND_NOT_LOADED", "pinned decoder trunk is not loaded")

    def official_decode(self, hidden: NDArray[np.float32]) -> NDArray[np.float32]:
        raise _fail("OFFICIAL_BACKEND_NOT_LOADED", "pinned output head is not loaded")

    def frozen_parameter_sha256(self) -> str:
        raise _fail("OFFICIAL_BACKEND_NOT_LOADED", "pinned weights are not loaded")


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DeterministicFakeKronosBackend:
    """Reproducible stand-in for the frozen Kronos path.

    Shapes, dtypes, and token ranges match the pinned contract so orchestration,
    caching, metric, and gate code can be exercised without Torch. Values carry
    no market meaning and are rejected by a real run.
    """

    def __init__(self, *, seed: int = 1729) -> None:
        self._seed = seed

    @property
    def mode(self) -> KronosMode:
        return KronosMode.FAKE

    def resolve_assets(self) -> ResolvedAssets:
        return ResolvedAssets(
            kronos_mode=self.mode,
            repository="fake/kronos-tokenizer-2k",
            revision="0" * 40,
            observed_config_sha256=None,
            observed_weights_sha256=None,
            frozen_parameter_sha256=self.frozen_parameter_sha256(),
            revisions_verified=False,
        )

    def _generator(self, salt: str, length: int) -> np.random.Generator:
        raw = hashlib.sha256(f"{self._seed}|{salt}|{length}".encode()).digest()[:8]
        return np.random.default_rng(int.from_bytes(raw, "big"))

    def encode_tokens(
        self, features: NDArray[np.float32]
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        length = features.shape[0]
        rng = self._generator("tokens", length)
        coarse = rng.integers(0, COARSE_VOCABULARY_SIZE, size=length, dtype=np.int64)
        fine = rng.integers(0, FINE_VOCABULARY_SIZE, size=length, dtype=np.int64)
        assert_token_ranges(coarse, fine)
        return coarse, fine

    def bipolar_latent(
        self, coarse_ids: NDArray[np.int64], fine_ids: NDArray[np.int64]
    ) -> NDArray[np.float32]:
        assert_token_ranges(coarse_ids, fine_ids)
        length = coarse_ids.shape[0]
        bits = np.empty((length, QUANTIZED_LATENT_DIMENSION), dtype=np.float32)
        for position in range(10):
            bits[:, position] = ((coarse_ids >> position) & 1) * 2.0 - 1.0
            bits[:, 10 + position] = ((fine_ids >> position) & 1) * 2.0 - 1.0
        return bits / np.sqrt(QUANTIZED_LATENT_DIMENSION, dtype=np.float32)

    def project_latent(self, latent: NDArray[np.float32]) -> NDArray[np.float32]:
        rng = self._generator("projection", QUANTIZED_LATENT_DIMENSION)
        weights = rng.standard_normal(
            (QUANTIZED_LATENT_DIMENSION, DECODER_HIDDEN_DIMENSION)
        ).astype(np.float32)
        return (latent @ weights).astype(np.float32)

    def decoder_trunk(self, projected: NDArray[np.float32]) -> NDArray[np.float32]:
        # Causal cumulative mean stands in for three causal decoder blocks: the
        # value at position t depends only on positions <= t.
        weights = np.arange(1, projected.shape[0] + 1, dtype=np.float32)[:, None]
        return (np.cumsum(projected, axis=0) / weights).astype(np.float32)

    def official_decode(self, hidden: NDArray[np.float32]) -> NDArray[np.float32]:
        rng = self._generator("official_head", DECODER_HIDDEN_DIMENSION)
        weights = rng.standard_normal((DECODER_HIDDEN_DIMENSION, 6)).astype(np.float32)
        return (hidden @ weights).astype(np.float32)

    def frozen_parameter_sha256(self) -> str:
        return hashlib.sha256(f"fake-frozen-{self._seed}".encode()).hexdigest()

    def build_bridge_input(
        self,
        *,
        hidden: NDArray[np.float32],
        scale_features: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Concatenate the 256 hidden dims with the 13 causal scale features."""
        if hidden.shape[1] != DECODER_HIDDEN_DIMENSION:
            raise _fail(
                "INVALID_HIDDEN_DIMENSION",
                f"hidden must have {DECODER_HIDDEN_DIMENSION} columns, got {hidden.shape[1]}",
            )
        if scale_features.shape[1] != SCALE_FEATURE_COUNT:
            raise _fail(
                "INVALID_SCALE_FEATURE_COUNT",
                f"scale features must have {SCALE_FEATURE_COUNT} columns, "
                f"got {scale_features.shape[1]}",
            )
        tensor = np.concatenate([hidden, scale_features], axis=1).astype(np.float32)
        assert_bridge_input(tensor, sequence_length=hidden.shape[0])
        return tensor


def default_sequence_length() -> int:
    return EXAMPLE_LENGTH
