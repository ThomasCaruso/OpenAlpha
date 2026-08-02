# pyright: reportMissingImports=false
# Torch is deliberately absent from the base development environment: the
# ordinary test suite must run without it, and this module imports it lazily
# so that importing the package costs nothing. It is present in the deployed
# image, pinned to an exact hash-verified wheel.
"""The official frozen backend, over the pinned source and pinned assets.

Nothing here trains, optimizes, or writes a parameter. Every identity is
verified before a weight is loaded, the isolated import restores ``sys.path``
and ``sys.modules`` on the way out, and every generation runs under
``torch.inference_mode()``.

Torch and the official source are imported lazily, so importing this module
costs nothing and the ordinary test suite never downloads an asset.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any

from ..errors import BridgeFailure, BridgeTransformError, FailureCategory
from .backends import GeneratedPath, ResolvedDiagnosticAssets, StepSampling, TokenPair
from .normalization import NormalizationState
from .official_input import OfficialRow, TimeStamp
from .source_conformance import SOURCE_REVISION, verify_source_files
from .spec import KRONOS_MINI_SPEC

__all__ = [
    "OfficialForecastModel",
    "OfficialTokenizerCodec",
    "isolated_official_source",
    "load_official_components",
    "parameter_digest",
    "verify_asset_file",
]


def _fail(code: str, message: str, *, field: str | None = None) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            field=field,
            message=message,
        )
    )


def verify_asset_file(path: Path | str, expected_sha256: str, *, label: str) -> str:
    """Hash a resolved asset and compare, or fail closed.

    Read in chunks: the tokenizer weights are large and a diagnostic that runs
    out of memory verifying its own inputs has failed for the wrong reason.
    """
    target = Path(path)
    if not target.is_file():
        raise _fail("OFFICIAL_ASSET_MISSING", f"{label} not found at {target}", field=label)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    observed = digest.hexdigest()
    if observed != expected_sha256:
        raise _fail(
            "OFFICIAL_ASSET_HASH_MISMATCH",
            f"{label} hashes to {observed}, expected {expected_sha256}",
            field=label,
        )
    return observed


@contextmanager
def isolated_official_source(source_root: Path | str) -> Iterator[Any]:
    """Import the verified official source, then put the interpreter back.

    The official ``kronos.py`` does ``sys.path.append("../")`` and
    ``from model.module import *``, so it can only be imported with a ``model``
    package on the path. Both ``sys.path`` and ``sys.modules`` are snapshotted
    and restored, so a diagnostic run leaves no import-system residue that a
    later import could pick up instead of the real thing.
    """
    root = Path(source_root).resolve()
    verify_source_files(root)

    path_snapshot = list(sys.path)
    modules_snapshot = dict(sys.modules)
    try:
        sys.path.insert(0, str(root))
        package_spec = importlib.util.spec_from_file_location(
            "model", root / "model" / "__init__.py"
        )
        if package_spec is None:
            # The upstream repository has no model/__init__.py; a namespace
            # package over the verified directory is enough for the star import.
            import types

            package = types.ModuleType("model")
            package.__path__ = [str(root / "model")]  # type: ignore[attr-defined]
            sys.modules["model"] = package

        module_spec = importlib.util.spec_from_file_location(
            "model.module", root / "model" / "module.py"
        )
        if module_spec is None or module_spec.loader is None:
            raise _fail("OFFICIAL_SOURCE_UNIMPORTABLE", "cannot load model/module.py")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules["model.module"] = module
        module_spec.loader.exec_module(module)

        kronos_spec = importlib.util.spec_from_file_location(
            "model.kronos", root / "model" / "kronos.py"
        )
        if kronos_spec is None or kronos_spec.loader is None:
            raise _fail("OFFICIAL_SOURCE_UNIMPORTABLE", "cannot load model/kronos.py")
        kronos = importlib.util.module_from_spec(kronos_spec)
        sys.modules["model.kronos"] = kronos
        kronos_spec.loader.exec_module(kronos)
        yield kronos
    finally:
        sys.path[:] = path_snapshot
        for name in set(sys.modules) - set(modules_snapshot):
            del sys.modules[name]
        sys.modules.update(modules_snapshot)


def parameter_digest(*modules: Any) -> str:
    """One hash over every parameter of every supplied module.

    Order is fixed by sorted parameter name so the digest is stable across
    runs, and the tensor bytes are taken on CPU so device placement does not
    change the value.
    """
    import torch

    digest = hashlib.sha256()
    with torch.inference_mode():
        for module in modules:
            for name, tensor in sorted(module.named_parameters(), key=lambda item: item[0]):
                digest.update(name.encode("utf-8"))
                digest.update(str(tuple(tensor.shape)).encode("utf-8"))
                digest.update(tensor.detach().to("cpu").contiguous().numpy().tobytes())
    return digest.hexdigest()


def _freeze(module: Any, label: str) -> tuple[int, int]:
    """eval(), requires_grad=False everywhere, then verify both took."""
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    if module.training:
        raise _fail("OFFICIAL_MODULE_IN_TRAINING_MODE", f"{label} is still in training mode")
    if trainable != 0:
        raise _fail(
            "OFFICIAL_MODULE_NOT_FROZEN",
            f"{label} still reports {trainable} trainable parameters",
        )
    if total <= 0:
        raise _fail("OFFICIAL_MODULE_HAS_NO_PARAMETERS", f"{label} reports no parameters")
    return total, trainable


def load_official_components(
    *,
    source_root: Path | str,
    tokenizer_directory: Path | str,
    model_directory: Path | str,
    tokenizer_spec: Any,
    device: str = "cuda",
) -> tuple[Any, Any, ResolvedDiagnosticAssets]:
    """Verify every identity, then load and freeze both official modules.

    Returns the tokenizer, the model and the observed asset identities. No
    weight is read until after every digest has been checked.
    """
    tokenizer_dir = Path(tokenizer_directory)
    model_dir = Path(model_directory)

    tokenizer_config = verify_asset_file(
        tokenizer_dir / "config.json", tokenizer_spec.config_sha256, label="tokenizer config"
    )
    tokenizer_weights = verify_asset_file(
        tokenizer_dir / "model.safetensors",
        tokenizer_spec.weights_sha256,
        label="tokenizer weights",
    )
    model_config = verify_asset_file(
        model_dir / "config.json", KRONOS_MINI_SPEC.config_sha256, label="model config"
    )
    model_weights = verify_asset_file(
        model_dir / KRONOS_MINI_SPEC.weights_file,
        KRONOS_MINI_SPEC.weights_sha256,
        label="model weights",
    )

    source_digests = verify_source_files(source_root)

    with isolated_official_source(source_root) as official:
        tokenizer = official.KronosTokenizer.from_pretrained(str(tokenizer_dir))
        model = official.Kronos.from_pretrained(str(model_dir))
        tokenizer = tokenizer.to(device)
        model = model.to(device)
        tokenizer_total, tokenizer_trainable = _freeze(tokenizer, "tokenizer")
        model_total, model_trainable = _freeze(model, "model")
        digest = parameter_digest(tokenizer, model)

    assets = ResolvedDiagnosticAssets(
        tokenizer_repository=tokenizer_spec.repository,
        tokenizer_revision=tokenizer_spec.revision,
        tokenizer_config_sha256=tokenizer_config,
        tokenizer_weights_sha256=tokenizer_weights,
        model_repository=KRONOS_MINI_SPEC.repository,
        model_revision=KRONOS_MINI_SPEC.revision,
        model_config_sha256=model_config,
        model_weights_sha256=model_weights,
        source_revision=SOURCE_REVISION,
        source_as_committed_sha256={d.relative_path: d.as_committed_sha256 for d in source_digests},
        source_crlf_normalized_sha256={
            d.relative_path: d.crlf_normalized_sha256 for d in source_digests
        },
        parameter_sha256=digest,
        trainable_parameter_count=tokenizer_trainable + model_trainable,
        total_parameter_count=tokenizer_total + model_total,
    )
    return tokenizer, model, assets


def _rows_to_tensor(rows: tuple[OfficialRow, ...], state: NormalizationState, device: str) -> Any:
    """Six normalized channels, shaped (1, T, 6), in the official order."""
    import torch

    normalized = state.normalize(rows)
    tensor = torch.tensor(normalized, dtype=torch.float32, device=device).unsqueeze(0)
    if tensor.shape != (1, len(rows), 6):
        raise _fail(
            "OFFICIAL_TENSOR_SHAPE_MISMATCH",
            f"expected (1, {len(rows)}, 6), built {tuple(tensor.shape)}",
        )
    return tensor


def _stamps_to_tensor(stamps: tuple[TimeStamp, ...], device: str) -> Any:
    """Five stamp features, shaped (1, T, 5), in the official order."""
    import torch

    tensor = torch.tensor(
        [list(stamp.values()) for stamp in stamps], dtype=torch.float32, device=device
    ).unsqueeze(0)
    if tensor.shape != (1, len(stamps), 5):
        raise _fail(
            "OFFICIAL_STAMP_SHAPE_MISMATCH",
            f"expected (1, {len(stamps)}, 5), built {tuple(tensor.shape)}",
        )
    return tensor


def _assert_token_bounds(ids: Any, vocabulary: int, label: str) -> None:
    minimum = int(ids.min())
    maximum = int(ids.max())
    if minimum < 0 or maximum >= vocabulary:
        raise _fail(
            "OFFICIAL_TOKEN_OUT_OF_VOCABULARY",
            f"{label} ids span [{minimum}, {maximum}], outside [0, {vocabulary - 1}]",
        )


class OfficialTokenizerCodec:
    """The official tokenizer, used only to encode and decode.

    Holds no normalization state of its own: the state arrives on every call
    and is used exactly as handed over.
    """

    def __init__(self, *, tokenizer: Any, official: Any, device: str = "cuda") -> None:
        self._tokenizer = tokenizer
        self._official = official
        self._device = device

    def encode(
        self, rows: tuple[OfficialRow, ...], *, state: NormalizationState
    ) -> tuple[TokenPair, ...]:
        import torch

        tensor = _rows_to_tensor(rows, state, self._device)
        with torch.inference_mode():
            encoded = self._tokenizer.encode(tensor, half=True)
        if not isinstance(encoded, (list, tuple)) or len(encoded) != 2:
            raise _fail(
                "OFFICIAL_ENCODE_SHAPE_MISMATCH",
                "encode(half=True) must return exactly two index tensors",
            )
        coarse, fine = encoded
        for name, ids, vocabulary in (
            ("coarse", coarse, KRONOS_MINI_SPEC.coarse_vocabulary),
            ("fine", fine, KRONOS_MINI_SPEC.fine_vocabulary),
        ):
            if tuple(ids.shape) != (1, len(rows)):
                raise _fail(
                    "OFFICIAL_ENCODE_SHAPE_MISMATCH",
                    f"{name} ids are {tuple(ids.shape)}, expected (1, {len(rows)})",
                )
            _assert_token_bounds(ids, vocabulary, name)
        return tuple(
            TokenPair(coarse=int(c), fine=int(f))
            for c, f in zip(coarse[0].tolist(), fine[0].tolist(), strict=True)
        )

    def decode(
        self,
        tokens: tuple[TokenPair, ...],
        *,
        state: NormalizationState,
        sessions: tuple,
    ) -> tuple[OfficialRow, ...]:
        import torch

        if len(sessions) != len(tokens):
            raise _fail(
                "OFFICIAL_DECODE_SESSION_MISMATCH",
                f"{len(tokens)} tokens against {len(sessions)} sessions",
            )
        coarse = torch.tensor([[t.coarse for t in tokens]], dtype=torch.long, device=self._device)
        fine = torch.tensor([[t.fine for t in tokens]], dtype=torch.long, device=self._device)
        with torch.inference_mode():
            decoded = self._tokenizer.decode([coarse, fine], half=True)
        if tuple(decoded.shape) != (1, len(tokens), 6):
            raise _fail(
                "OFFICIAL_DECODE_SHAPE_MISMATCH",
                f"decode produced {tuple(decoded.shape)}, expected (1, {len(tokens)}, 6)",
            )
        normalized = tuple(tuple(float(v) for v in row) for row in decoded[0].tolist())
        restored = state.invert(normalized)
        return tuple(
            OfficialRow(
                session=session,
                open=values[0],
                high=values[1],
                low=values[2],
                close=values[3],
                volume=values[4],
                amount=values[5],
            )
            for session, values in zip(sessions, restored, strict=True)
        )


class OfficialForecastModel:
    """The pinned Kronos-mini, generating under inference mode only.

    The autoregressive loop mirrors ``auto_regressive_inference`` for
    sample_count = 1 and calls the official ``top_k_top_p_filtering`` so the
    filtering is the official implementation rather than a reimplementation.
    The selected-token probabilities are read off the same softmax the official
    sampler draws from, which is what makes them post-temperature,
    post-filtering and renormalized.
    """

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        official: Any,
        device: str = "cuda",
        max_context: int = 512,
        clip: float = 5.0,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._official = official
        self._device = device
        self._max_context = max_context
        self._clip = clip

    def generate(
        self,
        context: tuple[OfficialRow, ...],
        *,
        context_stamps: tuple[TimeStamp, ...],
        target_stamps: tuple[TimeStamp, ...],
        target_sessions: tuple[date, ...],
        state: NormalizationState,
        steps: int,
        seed: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> GeneratedPath:
        import torch
        import torch.nn.functional as F

        if len(context_stamps) != len(context):
            raise _fail(
                "OFFICIAL_STAMP_SHAPE_MISMATCH",
                f"{len(context_stamps)} context stamps for {len(context)} rows",
            )
        if len(target_stamps) != steps:
            raise _fail(
                "OFFICIAL_STAMP_SHAPE_MISMATCH",
                f"{len(target_stamps)} target stamps for {steps} steps",
            )
        if len(target_sessions) != steps:
            raise _fail(
                "OFFICIAL_STAMP_SHAPE_MISMATCH",
                f"{len(target_sessions)} target sessions for {steps} steps",
            )
        if len(context) + steps > self._max_context:
            raise _fail(
                "OFFICIAL_CONTEXT_WOULD_TRUNCATE",
                (
                    f"{len(context)} context rows plus {steps} steps exceeds max_context "
                    f"{self._max_context}; the diagnostic is specified for a window that "
                    "never truncates"
                ),
            )

        x = _rows_to_tensor(context, state, self._device)
        x = torch.clip(x, -self._clip, self._clip)
        x_stamp = _stamps_to_tensor(context_stamps, self._device)
        y_stamp = _stamps_to_tensor(target_stamps, self._device)
        full_stamp = torch.cat([x_stamp, y_stamp], dim=1)

        # The official source neither seeds nor exposes a seed, so determinism
        # is imposed here and recorded as caller-side in the artifact.
        torch.manual_seed(seed)

        tokens: list[TokenPair] = []
        sampling: list[StepSampling] = []

        with torch.inference_mode():
            encoded = self._tokenizer.encode(x, half=True)
            context_coarse, context_fine = encoded[0], encoded[1]
            if tuple(context_coarse.shape) != (1, len(context)):
                raise _fail(
                    "OFFICIAL_ENCODE_SHAPE_MISMATCH",
                    f"context coarse ids are {tuple(context_coarse.shape)}, expected "
                    f"(1, {len(context)})",
                )
            # The running buffers start as the context tokens and grow. The
            # originals are kept unmodified for the final concatenation.
            coarse_ids = context_coarse.clone()
            fine_ids = context_fine.clone()
            initial = x.size(1)

            def draw(logits: Any) -> tuple[int, float]:
                scaled = logits / temperature
                if top_k > 0 or top_p < 1.0:
                    scaled = self._official.top_k_top_p_filtering(scaled, top_k=top_k, top_p=top_p)
                probabilities = F.softmax(scaled, dim=-1)
                chosen = torch.multinomial(probabilities, num_samples=1)
                index = int(chosen.squeeze(-1)[0])
                probability = float(probabilities[0, index])
                return index, probability

            for step in range(steps):
                window = initial + step
                current = full_stamp[:, :window, :].contiguous()
                s1_logits, decoded_context = self._model.decode_s1(coarse_ids, fine_ids, current)
                coarse_index, coarse_probability = draw(s1_logits[:, -1, :])

                sampled_coarse = torch.tensor(
                    [[coarse_index]], dtype=torch.long, device=self._device
                )
                s2_logits = self._model.decode_s2(decoded_context, sampled_coarse)
                fine_index, fine_probability = draw(s2_logits[:, -1, :])

                if not 0 <= coarse_index < KRONOS_MINI_SPEC.coarse_vocabulary:
                    raise _fail(
                        "OFFICIAL_TOKEN_OUT_OF_VOCABULARY",
                        f"coarse token {coarse_index} outside the vocabulary",
                    )
                if not 0 <= fine_index < KRONOS_MINI_SPEC.fine_vocabulary:
                    raise _fail(
                        "OFFICIAL_TOKEN_OUT_OF_VOCABULARY",
                        f"fine token {fine_index} outside the vocabulary",
                    )

                tokens.append(TokenPair(coarse=coarse_index, fine=fine_index))
                sampling.append(
                    StepSampling(
                        coarse_log_probability=_safe_log(coarse_probability),
                        fine_conditional_log_probability=_safe_log(fine_probability),
                    )
                )
                coarse_ids = torch.cat([coarse_ids, sampled_coarse], dim=1)
                fine_ids = torch.cat(
                    [fine_ids, torch.tensor([[fine_index]], dtype=torch.long, device=self._device)],
                    dim=1,
                )

        if len(tokens) != steps:
            raise _fail(
                "OFFICIAL_TOKEN_LENGTH_MISMATCH",
                f"generated {len(tokens)} steps, expected {steps}",
            )

        raw_suffix = self._decode_official_window(
            context_coarse=context_coarse,
            context_fine=context_fine,
            tokens=tokens,
            steps=steps,
            state=state,
            target_sessions=target_sessions,
        )
        return GeneratedPath(
            seed=seed,
            tokens=tuple(tokens),
            sampling=tuple(sampling),
            total_path_sampling_log_probability=sum(s.pair_log_probability for s in sampling),
            raw_decoded_suffix=raw_suffix,
        )

    def _decode_official_window(
        self,
        *,
        context_coarse: Any,
        context_fine: Any,
        tokens: list[TokenPair],
        steps: int,
        state: NormalizationState,
        target_sessions: tuple[date, ...],
    ) -> tuple[OfficialRow, ...]:
        """Decode the way auto_regressive_inference does, in that order.

        1. concatenate the context tokens with the generated ones
        2. select the final max_context window
        3. decode that complete window
        4. slice the last ``steps`` rows
        5. inverse normalize with the supplied state

        Decoding the generated suffix on its own would restart the decoder at
        position zero with no preceding tokens. The decoder is causal, so its
        output at a position depends on everything before it; the suffix alone
        is a different computation and can decode to different candles.
        """
        import torch

        with torch.inference_mode():
            generated_coarse = torch.tensor(
                [[t.coarse for t in tokens]], dtype=torch.long, device=self._device
            )
            generated_fine = torch.tensor(
                [[t.fine for t in tokens]], dtype=torch.long, device=self._device
            )

            # 1. concatenate
            full_coarse = torch.cat([context_coarse, generated_coarse], dim=1)
            full_fine = torch.cat([context_fine, generated_fine], dim=1)

            total = int(full_coarse.size(1))
            expected_total = int(context_coarse.size(1)) + steps
            if total != expected_total:
                raise _fail(
                    "OFFICIAL_TOKEN_LENGTH_MISMATCH",
                    f"concatenated window is {total} tokens, expected {expected_total}",
                )

            # 2. select the final max_context window
            window_start = max(0, total - self._max_context)
            window_coarse = full_coarse[:, window_start:total].contiguous()
            window_fine = full_fine[:, window_start:total].contiguous()
            window_length = int(window_coarse.size(1))

            # 3. decode the complete window
            decoded = self._tokenizer.decode([window_coarse, window_fine], half=True)
            if tuple(decoded.shape) != (1, window_length, 6):
                raise _fail(
                    "OFFICIAL_DECODE_SHAPE_MISMATCH",
                    f"decode produced {tuple(decoded.shape)}, expected (1, {window_length}, 6)",
                )
            # In this diagnostic's configuration no truncation occurs, so the
            # decoded window is the whole context plus the generated steps.
            if window_length != expected_total:
                raise _fail(
                    "OFFICIAL_CONTEXT_WOULD_TRUNCATE",
                    (
                        f"the decoded window is {window_length} tokens rather than "
                        f"{expected_total}; the diagnostic is specified for a window "
                        "that never truncates"
                    ),
                )

            # 4. slice the final steps rows
            suffix = decoded[0, -steps:, :]
            if tuple(suffix.shape) != (steps, 6):
                raise _fail(
                    "OFFICIAL_DECODE_SHAPE_MISMATCH",
                    f"the sliced suffix is {tuple(suffix.shape)}, expected ({steps}, 6)",
                )
            normalized = tuple(tuple(float(v) for v in row) for row in suffix.tolist())

        # 5. inverse normalize with the supplied context-only state
        restored = state.invert(normalized)
        return tuple(
            OfficialRow(
                session=session,
                open=values[0],
                high=values[1],
                low=values[2],
                close=values[3],
                volume=values[4],
                amount=values[5],
            )
            for session, values in zip(target_sessions, restored, strict=True)
        )


def _safe_log(probability: float) -> float:
    """log of a sampled token's probability.

    A sampled token always had positive probability, so a zero here means the
    number came from somewhere other than the distribution it was drawn from.
    """
    import math

    if not math.isfinite(probability) or probability <= 0.0:
        raise _fail(
            "OFFICIAL_SAMPLING_PROBABILITY_INVALID",
            (
                f"a sampled token reports probability {probability}; a token drawn from "
                "a distribution cannot have had zero or non-finite probability under it"
            ),
        )
    return math.log(probability)
