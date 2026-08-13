"""What the pinned official source actually does, locked and checkable.

Every value here was read out of ``model/kronos.py`` at revision
67b630e67f6a18c9e9be918d9b4337c960db1e9a. The source and the digests sealed in
experiment.yaml are the authority; the README is not.

The constants are duplicated deliberately: the diagnostic needs them as values,
and the conformance tests re-derive them from the source text so a drift
between what the code believes and what the source says fails a test rather
than surfacing as a wrong number in an artifact.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final, Literal

from openalpha_research.failures import FailureCategory, ResearchFailure, ResearchFailureError
from pydantic import BaseModel, ConfigDict, Field

from .assets import OFFICIAL_SOURCE_FILES

__all__ = [
    "SOURCE_CONTRACT",
    "SOURCE_REVISION",
    "SourceContract",
    "SourceFileDigests",
    "crlf_normalize",
    "digest_pair",
    "verify_source_files",
]

SOURCE_REVISION: Final[str] = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"


def _fail(code: str, message: str, *, field: str | None = None) -> ResearchFailureError:
    return ResearchFailureError(
        ResearchFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            field=field,
            message=message,
        )
    )


def crlf_normalize(data: bytes) -> bytes:
    """Line endings as the sealed digests were computed over.

    Normalizes to LF first so a file that is already CRLF is not doubled.
    """
    return data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")


class SourceFileDigests(BaseModel):
    """Both digests for one file, observed rather than assumed."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    relative_path: str
    as_committed_sha256: str
    crlf_normalized_sha256: str
    size_bytes: int


def digest_pair(relative_path: str, data: bytes) -> SourceFileDigests:
    return SourceFileDigests(
        relative_path=relative_path,
        as_committed_sha256=hashlib.sha256(data).hexdigest(),
        crlf_normalized_sha256=hashlib.sha256(crlf_normalize(data)).hexdigest(),
        size_bytes=len(data),
    )


def verify_source_files(root: Path | str) -> tuple[SourceFileDigests, ...]:
    """Verify every pinned file against both locked digests, or fail closed.

    ``root`` is a directory containing ``model/kronos.py`` and
    ``model/module.py``. Both checks must pass: the as-committed digest proves
    the upstream bytes are what was recorded, and the CRLF-normalized digest
    proves they still satisfy the value sealed in experiment.yaml.
    """
    base = Path(root)
    observed: list[SourceFileDigests] = []
    for expected in OFFICIAL_SOURCE_FILES:
        path = base / expected.relative_path
        if not path.is_file():
            raise _fail(
                "OFFICIAL_SOURCE_FILE_MISSING",
                f"pinned source file not found: {path}",
                field=expected.relative_path,
            )
        digests = digest_pair(expected.relative_path, path.read_bytes())
        if digests.as_committed_sha256 != expected.as_committed_sha256:
            raise _fail(
                "OFFICIAL_SOURCE_AS_COMMITTED_MISMATCH",
                (
                    f"{expected.relative_path} hashes to {digests.as_committed_sha256} "
                    f"as committed, expected {expected.as_committed_sha256}"
                ),
                field=expected.relative_path,
            )
        if digests.crlf_normalized_sha256 != expected.sealed_sha256_crlf_normalized:
            raise _fail(
                "OFFICIAL_SOURCE_SEALED_MISMATCH",
                (
                    f"{expected.relative_path} normalizes to "
                    f"{digests.crlf_normalized_sha256}, but experiment.yaml seals "
                    f"{expected.sealed_sha256_crlf_normalized}"
                ),
                field=expected.relative_path,
            )
        if digests.size_bytes != expected.as_committed_bytes:
            raise _fail(
                "OFFICIAL_SOURCE_SIZE_MISMATCH",
                (
                    f"{expected.relative_path} is {digests.size_bytes} bytes, expected "
                    f"{expected.as_committed_bytes}"
                ),
                field=expected.relative_path,
            )
        observed.append(digests)
    return tuple(observed)


class SourceContract(BaseModel):
    """The behaviour of the pinned source, as read from it."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    revision: str = Field(pattern=r"^[0-9a-f]{40}$")

    # --- predictor entry point -------------------------------------------
    predictor_class: str = "KronosPredictor"
    predict_method: str = "predict"
    predict_parameters: tuple[str, ...] = (
        "self",
        "df",
        "x_timestamp",
        "y_timestamp",
        "pred_len",
        "T",
        "top_k",
        "top_p",
        "sample_count",
        "verbose",
    )
    #: predict() defaults. auto_regressive_inference declares different ones
    #: (top_p 0.99, sample_count 5); predict() is the entry point and governs.
    predict_default_temperature: float = 1.0
    predict_default_top_k: int = 0
    predict_default_top_p: float = 0.9
    predict_default_sample_count: int = 1
    predict_default_verbose: bool = True
    inner_default_top_p: float = 0.99
    inner_default_sample_count: int = 5

    # --- constructor defaults ---------------------------------------------
    default_max_context: int = 512
    default_clip: int = 5

    # --- input columns -----------------------------------------------------
    price_columns: tuple[str, ...] = ("open", "high", "low", "close")
    volume_column: str = "volume"
    amount_column: str = "amount"
    ordered_columns: tuple[str, ...] = (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    )
    time_columns: tuple[str, ...] = ("minute", "hour", "weekday", "day", "month")

    # --- missing column behaviour -----------------------------------------
    missing_volume_fills_volume_with: float = 0.0
    missing_volume_also_fills_amount_with: float = 0.0
    missing_amount_formula: str = "volume * mean(open, high, low, close) per row"
    nan_in_required_columns: str = "raises ValueError"

    # --- normalization ------------------------------------------------------
    normalization_epsilon: float = 1e-5
    normalization_std_ddof: Literal[0] = 0
    forward_formula: str = "clip((x - mean) / (std + 1e-5), -clip, +clip)"
    inverse_formula: str = "x * (std + 1e-5) + mean"
    clip_applied_after_standardization: bool = True
    clip_applied_again_inside_inference: bool = True

    # --- tokenizer shapes ---------------------------------------------------
    tokenizer_encode_returns: str = "two index tensors, s1 and s2, each (batch, seq_len)"
    tokenizer_decode_returns: str = "(batch, seq_len, d_in) with d_in = 6"
    tokenizer_half_flag: bool = True

    # --- sampling -----------------------------------------------------------
    coarse_logits_source: str = "Kronos.decode_s1, last position"
    fine_logits_source: str = "Kronos.decode_s2(context, sampled_s1), last position"
    fine_is_conditional_on_sampled_coarse: bool = True
    sampling_order: tuple[str, ...] = (
        "divide_logits_by_temperature",
        "top_k_or_top_p_filtering",
        "softmax",
        "multinomial",
    )
    #: top_k_top_p_filtering returns immediately after the top-k branch, so
    #: top_p only executes when top_k == 0.
    top_k_and_top_p_mutually_exclusive: bool = True
    top_k_takes_precedence: bool = True
    filter_value: str = "-inf"
    sampling_uses_multinomial: bool = True

    # --- multi-sample and averaging ----------------------------------------
    sample_count_batching: str = "context repeated along batch, reshaped to (B*sample_count, ...)"
    ensemble_average_domain: str = "tokenizer-decoded normalized domain, before inverse"
    ensemble_average_op: str = "np.mean over the sample_count axis"
    inverse_is_affine_so_order_is_arithmetically_equivalent: bool = True

    # --- context truncation --------------------------------------------------
    truncation_rule: str = "rolling buffer of max_context, keeping the most recent tokens"
    #: 448 context + 64 steps reaches 511 at the last step, below max_context
    #: 512, so no roll occurs in the diagnostic's configuration.
    rolls_in_diagnostic_configuration: bool = False
    final_decode_window: str = "last max_context tokens of context+generated, then slice pred_len"

    # --- gradients and RNG ---------------------------------------------------
    inference_guard: str = "torch.no_grad()"
    source_seeds_rng: bool = False
    source_exposes_seed_parameter: bool = False
    rng_source: str = "global torch RNG via torch.multinomial"
    determinism_requires_caller_seeding: bool = True


SOURCE_CONTRACT: Final[SourceContract] = SourceContract(revision=SOURCE_REVISION)
