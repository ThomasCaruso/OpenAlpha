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
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from ..errors import BridgeFailure, BridgeTransformError, FailureCategory
from ..windowing import EXAMPLE_LENGTH
from .observed import (
    ObservedEncode,
    ObservedKronosComponents,
    ObservedLinear,
    ObservedModule,
    describe_module,
)
from .optional import require_module

__all__ = [
    "BRIDGE_INPUT_DIMENSION",
    "DECODER_HIDDEN_DIMENSION",
    "EFFECTIVE_DECODER_BLOCKS",
    "QUANTIZED_LATENT_DIMENSION",
    "SCALE_FEATURE_COUNT",
    "SCALE_FEATURE_ORDER",
    "SOURCE_SPEC",
    "TOKENIZER_INPUT_DIMENSION",
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

#: experiment.yaml official_tokenizer.input_dimension
TOKENIZER_INPUT_DIMENSION = 6
#: configured_decoder_layers is 4, but the implementation builds range(n - 1).
EFFECTIVE_DECODER_BLOCKS = 3

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


class PinnedSourceSpec(BaseModel):
    """The pinned official source checkout, verified file by file."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    repository: str
    revision: str = Field(min_length=40, max_length=40)
    files: dict[str, str]


#: experiment.yaml official_source
SOURCE_SPEC = PinnedSourceSpec(
    repository="https://github.com/shiyu-coder/Kronos",
    revision="67b630e67f6a18c9e9be918d9b4337c960db1e9a",
    files={
        "model/kronos.py": (
            "638a56e035856c600c9848b368be087cb706a61603a0790124968c95b8c69f3a"
        ),
        "model/module.py": (
            "a07edbadc0e96804c8158c021bbc6063bb7cc43b34d7fc470d5c8ff2005a409f"
        ),
    },
)


def _assert_linear(module: Any, in_features: int, out_features: int) -> None:
    """Confirm a resolved component really is the pinned projection."""
    observed_in = getattr(module, "in_features", None)
    observed_out = getattr(module, "out_features", None)
    if observed_in != in_features or observed_out != out_features:
        raise _fail(
            "UNEXPECTED_COMPONENT_SHAPE",
            (
                f"expected Linear({in_features}, {out_features}), observed "
                f"Linear({observed_in}, {observed_out})"
            ),
        )


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

    def __init__(
        self,
        *,
        stage: str,
        device: str = "cuda",
        cache_dir: str | None = None,
        source_path: str | None = None,
    ) -> None:
        self._stage = stage
        self._device = device
        self._cache_dir = cache_dir
        self._source_path = source_path
        self._model = None
        self._observed_source_files: dict[str, str] = {}
        self._dependency_versions: dict[str, str] = {}
        self._observed: ObservedKronosComponents | None = None
        self._resolved_attributes: dict[str, str] = {}

    @property
    def observed(self) -> ObservedKronosComponents | None:
        """Runtime-observed component manifest, populated on resolve/encode."""
        return self._observed

    @property
    def observed_source_files(self) -> dict[str, str]:
        """Hashes of the official files actually executed; empty until resolved."""
        return dict(self._observed_source_files)

    @property
    def dependency_versions(self) -> dict[str, str]:
        """Observed versions of the pinned official runtime dependencies."""
        return dict(self._dependency_versions)

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

        # Execute only the hash-verified files, inside a synthetic package, so
        # the unverified upstream model/__init__.py never runs.
        import os
        from pathlib import Path as _Path

        from .official_source import load_official_kronos

        root = self._source_path or os.environ.get("OPENALPHA_KRONOS_SOURCE_PATH")
        if not root:
            raise _fail(
                "KRONOS_SOURCE_UNAVAILABLE",
                (
                    "the pinned official source checkout is required; set "
                    "OPENALPHA_KRONOS_SOURCE_PATH or pass source_path"
                ),
            )
        official, self._observed_source_files, self._dependency_versions = (
            load_official_kronos(_Path(root), dict(SOURCE_SPEC.files))
        )
        KronosTokenizer = official.KronosTokenizer

        torch = require_module("torch", stage=self._stage)
        snapshot = os.path.dirname(weights_path)
        tokenizer = KronosTokenizer.from_pretrained(snapshot)
        tokenizer = tokenizer.to(self._device)
        tokenizer.eval()
        # Nothing in Bridge may train a Kronos parameter.
        for parameter in tokenizer.parameters():
            parameter.requires_grad_(False)
        self._model = tokenizer

        self._observed = self._observe(tokenizer)
        _ = torch  # the import is required for device placement above

        return ResolvedAssets(
            kronos_mode=self.mode,
            repository=TOKENIZER_SPEC.repository,
            revision=TOKENIZER_SPEC.revision,
            observed_config_sha256=observed_config,
            observed_weights_sha256=observed_weights,
            frozen_parameter_sha256=self.frozen_parameter_sha256(),
            revisions_verified=True,
        )

    # ------------------------------------------------------------ components

    def _tokenizer(self) -> Any:
        if self._model is None:
            raise _fail(
                "OFFICIAL_BACKEND_NOT_LOADED",
                "call resolve_assets before using the official numerical path",
            )
        return self._model

    def _component(self, *names: str) -> Any:
        """Resolve a pinned submodule, failing closed with what was found."""
        tokenizer = self._tokenizer()
        for name in names:
            found = getattr(tokenizer, name, None)
            if found is not None:
                self._resolved_attributes[names[0]] = name
                return found
        available = sorted(n for n in dir(tokenizer) if not n.startswith("_"))
        raise _fail(
            "KRONOS_COMPONENT_NOT_FOUND",
            (
                f"none of {names} exist on the pinned tokenizer; the checkpoint "
                f"layout does not match the audited boundary. Available: {available}"
            ),
        )

    @staticmethod
    def _as_float32(tensor: Any) -> NDArray[np.float32]:
        array = tensor.detach().to("cpu").float().numpy().astype(np.float32)
        if not np.isfinite(array).all():
            raise _fail("NON_FINITE_KRONOS_OUTPUT", "official component emitted non-finite values")
        return array

    # ------------------------------------------------------------- numerical


    def _observe(self, tokenizer: Any) -> ObservedKronosComponents:
        """Read the manifest off the live tokenizer. Nothing is hardcoded."""

        def _linear(key: str, *names: str) -> ObservedLinear:
            module = self._component(*names)
            cls, mod = describe_module(module)
            return ObservedLinear(
                attribute=self._resolved_attributes.get(key, names[0]),
                class_name=cls,
                module=mod,
                in_features=int(getattr(module, "in_features", -1)),
                out_features=int(getattr(module, "out_features", -1)),
                has_bias=getattr(module, "bias", None) is not None,
            )

        projection = _linear("post_quant_embed", "post_quant_embed")
        head = _linear("head", "head")

        self._component("indices_to_bits")
        blocks = self._component("decoder", "decoder_blocks", "dec_blocks")
        container_cls, container_mod = describe_module(blocks)
        observed_blocks = []
        for block in blocks:
            cls, mod = describe_module(block)
            observed_blocks.append(
                ObservedModule(attribute="block", class_name=cls, module=mod)
            )

        tokenizer_cls, tokenizer_mod = describe_module(tokenizer)
        parameters = list(tokenizer.parameters())
        return ObservedKronosComponents(
            tokenizer_class=tokenizer_cls,
            tokenizer_module=tokenizer_mod,
            indices_to_bits_attribute=self._resolved_attributes.get(
                "indices_to_bits", "indices_to_bits"
            ),
            projection=projection,
            decoder_attribute=self._resolved_attributes.get("decoder", "decoder"),
            decoder_container_class=container_cls,
            decoder_container_module=container_mod,
            decoder_block_count=len(observed_blocks),
            decoder_blocks=tuple(observed_blocks),
            head=head,
            encode=None,
            device=str(self._device),
            training_mode=bool(getattr(tokenizer, "training", False)),
            total_parameters=sum(int(p.numel()) for p in parameters),
            trainable_parameters=sum(
                int(p.numel()) for p in parameters if bool(p.requires_grad)
            ),
        )

    def encode_tokens(
        self, features: NDArray[np.float32]
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Official ``KronosTokenizer.encode(x, half=True)``.

        Input is the normalized, clipped six-channel window; output is the pair
        of official 10-bit identifier streams.
        """
        torch = require_module("torch", stage=self._stage)
        tokenizer = self._tokenizer()

        if features.ndim != 2 or features.shape[1] != TOKENIZER_INPUT_DIMENSION:
            raise _fail(
                "INVALID_TOKENIZER_INPUT_SHAPE",
                f"encoder input must be [T, {TOKENIZER_INPUT_DIMENSION}], got {features.shape}",
            )
        batch = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
        batch = batch.unsqueeze(0).to(self._device)

        with torch.no_grad():
            encoded = tokenizer.encode(batch, half=True)
        if not isinstance(encoded, (tuple, list)) or len(encoded) != 2:
            raise _fail(
                "UNEXPECTED_ENCODE_RESULT",
                "official encode(half=True) must return exactly two identifier tensors",
            )
        coarse = encoded[0].detach().to("cpu").reshape(-1).to(torch.int64).numpy()
        fine = encoded[1].detach().to("cpu").reshape(-1).to(torch.int64).numpy()
        assert_token_ranges(coarse, fine)

        if self._observed is not None and self._observed.encode is None:
            self._observed = self._observed.model_copy(
                update={
                    "encode": ObservedEncode(
                        return_container=type(encoded).__name__,
                        tensor_count=len(encoded),
                        batched_shapes=tuple(
                            tuple(int(d) for d in tensor.shape) for tensor in encoded
                        ),
                        batched_dtypes=tuple(str(tensor.dtype) for tensor in encoded),
                        coarse_flat_shape=tuple(int(d) for d in coarse.shape),
                        fine_flat_shape=tuple(int(d) for d in fine.shape),
                        coarse_flat_dtype=str(coarse.dtype),
                        fine_flat_dtype=str(fine.dtype),
                    )
                }
            )
        return coarse, fine

    def bipolar_latent(
        self, coarse_ids: NDArray[np.int64], fine_ids: NDArray[np.int64]
    ) -> NDArray[np.float32]:
        """Official ``indices_to_bits(..., half=True)``: [T, 20] scaled by 1/sqrt(20)."""
        torch = require_module("torch", stage=self._stage)
        assert_token_ranges(coarse_ids, fine_ids)
        convert = self._component("indices_to_bits")

        coarse = torch.from_numpy(np.ascontiguousarray(coarse_ids)).unsqueeze(0).to(self._device)
        fine = torch.from_numpy(np.ascontiguousarray(fine_ids)).unsqueeze(0).to(self._device)
        with torch.no_grad():
            bits = convert([coarse, fine], half=True)
        latent = self._as_float32(bits).reshape(-1, QUANTIZED_LATENT_DIMENSION)
        return latent

    def project_latent(self, latent: NDArray[np.float32]) -> NDArray[np.float32]:
        """Official learned ``post_quant_embed``: Linear(20, 256)."""
        torch = require_module("torch", stage=self._stage)
        projection = self._component("post_quant_embed")
        _assert_linear(projection, QUANTIZED_LATENT_DIMENSION, DECODER_HIDDEN_DIMENSION)

        tensor = torch.from_numpy(np.ascontiguousarray(latent, dtype=np.float32))
        tensor = tensor.unsqueeze(0).to(self._device)
        with torch.no_grad():
            projected = projection(tensor)
        return self._as_float32(projected).reshape(-1, DECODER_HIDDEN_DIMENSION)

    def decoder_trunk(self, projected: NDArray[np.float32]) -> NDArray[np.float32]:
        """The three official causal decoder blocks, applied to the ordered window.

        The whole window is processed at once. Attention is causal, so position
        t depends only on positions <= t; decoding pairs in isolation would be
        incompatible with the audited architecture.
        """
        torch = require_module("torch", stage=self._stage)
        blocks = self._component("decoder", "decoder_blocks", "dec_blocks")
        try:
            block_count = len(blocks)
        except TypeError as error:
            raise _fail(
                "KRONOS_DECODER_NOT_ITERABLE",
                "the official decoder trunk is not an iterable block sequence",
            ) from error
        if block_count != EFFECTIVE_DECODER_BLOCKS:
            raise _fail(
                "UNEXPECTED_DECODER_BLOCK_COUNT",
                f"expected {EFFECTIVE_DECODER_BLOCKS} frozen decoder blocks, found {block_count}",
            )

        tensor = torch.from_numpy(np.ascontiguousarray(projected, dtype=np.float32))
        hidden = tensor.unsqueeze(0).to(self._device)
        with torch.no_grad():
            for block in blocks:
                hidden = block(hidden)
        return self._as_float32(hidden).reshape(-1, DECODER_HIDDEN_DIMENSION)

    def official_decode(self, hidden: NDArray[np.float32]) -> NDArray[np.float32]:
        """The official unrestricted ``head``: Linear(256, 6). Baseline only."""
        torch = require_module("torch", stage=self._stage)
        head = self._component("head")
        _assert_linear(head, DECODER_HIDDEN_DIMENSION, TOKENIZER_INPUT_DIMENSION)

        tensor = torch.from_numpy(np.ascontiguousarray(hidden, dtype=np.float32))
        tensor = tensor.unsqueeze(0).to(self._device)
        with torch.no_grad():
            decoded = head(tensor)
        return self._as_float32(decoded).reshape(-1, TOKENIZER_INPUT_DIMENSION)

    def frozen_parameter_sha256(self) -> str:
        """Hash over every frozen parameter, for before/after parity checks."""
        tokenizer = self._tokenizer()
        digest = hashlib.sha256()
        for name, parameter in sorted(tokenizer.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(parameter.detach().to("cpu").float().numpy().tobytes())
        return digest.hexdigest()


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
        """Content-sensitive, causal, and deterministic.

        Row t is a pure function of rows 0..t, so it mirrors the official causal
        encoder: a change at position p moves rows p onward and leaves earlier
        rows byte-identical. Deriving tokens from the sequence length alone
        would make any causality test pass vacuously.
        """
        length = features.shape[0]
        contiguous = np.ascontiguousarray(features, dtype=np.float32)
        coarse = np.empty(length, dtype=np.int64)
        fine = np.empty(length, dtype=np.int64)
        running = hashlib.sha256(f"{self._seed}|tokens".encode())
        for index in range(length):
            running.update(contiguous[index].tobytes())
            digest = running.digest()
            coarse[index] = int.from_bytes(digest[:8], "big") % COARSE_VOCABULARY_SIZE
            fine[index] = int.from_bytes(digest[8:16], "big") % FINE_VOCABULARY_SIZE
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
