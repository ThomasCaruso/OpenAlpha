"""The forecast suffix must be decoded from the complete official window.

The pinned source never decodes the generated tokens on their own. It
concatenates the context tokens with the generated ones, selects the final
max_context window, decodes that whole window, and only then slices the last
pred_len rows.

The tokenizer decoder is causal, so its output at a position depends on every
token before it. Decoding the 64-token suffix alone restarts it at position
zero with no context, which is a different computation. These tests use a
tokenizer whose decode is genuinely position- and history-dependent, so a
regression to generated-only decoding changes the numbers and fails.
"""

from __future__ import annotations

import ast
import inspect
from datetime import date, timedelta
from pathlib import Path

import pytest
from openalpha_kronos.model import official as official_backend
from openalpha_kronos.model.contracts import (
    GeneratedPath,
    StepSampling,
    TokenPair,
)
from openalpha_kronos.model.input import (
    OFFICIAL_COLUMNS,
    ColumnPresence,
    OfficialRow,
    OfficialSeries,
)
from openalpha_kronos.model.normalization import NormalizationState, fit_context_state
from openalpha_kronos.studies.structural_validity.common import methods as methods_module
from openalpha_kronos.studies.structural_validity.common.methods import run_method_b, run_method_d
from openalpha_kronos.studies.structural_validity.mini.spec import (
    CONTEXT_CANDLES,
    TARGET_CANDLES,
    TOTAL_CANDLES,
)
from openalpha_research.failures import ResearchFailureError

ROOT = Path(__file__).resolve().parents[3]
VENDOR = ROOT / "vendor" / "kronos" / "67b630e6" / "model" / "kronos.py"


# ----------------------------------------------------------- a causal codec


class CausalFakeCodec:
    """A decoder whose output at a position depends on everything before it.

    Each decoded close is a running function of every preceding token in the
    window it was handed. Decoding a suffix in isolation therefore yields
    different numbers than decoding it as the tail of the full window, which is
    exactly the property the real tokenizer decoder has and the property a
    generated-only decode destroys.
    """

    def __init__(self) -> None:
        self.decode_windows: list[int] = []

    def encode(
        self, rows: tuple[OfficialRow, ...], *, state: NormalizationState
    ) -> tuple[TokenPair, ...]:
        state.normalize(rows)
        return tuple(
            TokenPair(coarse=index % 1024, fine=(index * 3) % 1024) for index, _ in enumerate(rows)
        )

    def decode(
        self, tokens: tuple[TokenPair, ...], *, state: NormalizationState, sessions: tuple
    ) -> tuple[OfficialRow, ...]:
        self.decode_windows.append(len(tokens))
        if len(sessions) != len(tokens):
            raise AssertionError("one session per token is required")

        rows: list[OfficialRow] = []
        running = 0.0
        for position, (pair, session) in enumerate(zip(tokens, sessions, strict=True)):
            # Depends on the token, its absolute position, and the running
            # history. All three are lost when a suffix is decoded alone.
            running = running * 0.5 + (pair.coarse + pair.fine + position) / 1000.0
            base = 100.0 + running
            rows.append(
                OfficialRow(
                    session=session,
                    open=base,
                    high=base * 1.01,
                    low=base * 0.99,
                    close=base,
                    volume=1.0e6,
                    amount=1.0e6 * base,
                )
            )
        return tuple(rows)


class WindowDecodingModel:
    """A model that decodes the way the official path does."""

    def __init__(self, *, codec: CausalFakeCodec, max_context: int = TOTAL_CANDLES) -> None:
        self._codec = codec
        self._max_context = max_context
        self.suffix_returned: tuple[OfficialRow, ...] = ()

    def generate(
        self,
        context: tuple[OfficialRow, ...],
        *,
        context_stamps,
        target_stamps,
        target_sessions,
        state: NormalizationState,
        steps: int,
        seed: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> GeneratedPath:
        generated = tuple(
            TokenPair(coarse=(seed + step) % 1024, fine=(seed * 2 + step) % 1024)
            for step in range(steps)
        )
        context_tokens = self._codec.encode(context, state=state)

        full = (*context_tokens, *generated)
        sessions = (*(row.session for row in context), *target_sessions)
        start = max(0, len(full) - self._max_context)
        decoded = self._codec.decode(full[start:], state=state, sessions=sessions[start:])
        suffix = decoded[-steps:]
        self.suffix_returned = suffix

        sampling = tuple(
            StepSampling(
                coarse_log_probability=-0.3,
                fine_conditional_log_probability=-0.2,
            )
            for _ in range(steps)
        )
        return GeneratedPath(
            seed=seed,
            tokens=generated,
            sampling=sampling,
            total_path_sampling_log_probability=sum(s.pair_log_probability for s in sampling),
            raw_decoded_suffix=suffix,
        )


def _series() -> OfficialSeries:
    start = date(2015, 5, 7)
    rows = tuple(
        OfficialRow(
            session=start + timedelta(days=index),
            open=100.0 + index * 0.01,
            high=101.0 + index * 0.01,
            low=99.0 + index * 0.01,
            close=100.5 + index * 0.01,
            volume=1.0e6,
            amount=1.0e6 * (100.5 + index * 0.01),
        )
        for index in range(TOTAL_CANDLES)
    )
    return OfficialSeries(
        symbol="SPY",
        frequency="1d",
        calendar="XNYS",
        columns=OFFICIAL_COLUMNS,
        column_presence=ColumnPresence(volume=True, amount=True),
        rows=rows,
        context_target_boundary=CONTEXT_CANDLES,
    )


# ============================================ the causal fake is really causal


def test_the_fake_decoder_is_position_and_history_dependent() -> None:
    """Without this the rest of the file would prove nothing."""
    codec = CausalFakeCodec()
    series = _series()
    state = fit_context_state(series.context)

    context_tokens = codec.encode(series.context, state=state)
    generated = tuple(
        TokenPair(coarse=(500 + step) % 1024, fine=(700 + step) % 1024)
        for step in range(TARGET_CANDLES)
    )
    sessions = tuple(row.session for row in series.target)

    full_window = codec.decode(
        (*context_tokens, *generated),
        state=state,
        sessions=(*(r.session for r in series.context), *sessions),
    )
    from_full = full_window[-TARGET_CANDLES:]
    from_suffix_alone = codec.decode(generated, state=state, sessions=sessions)

    assert len(from_full) == len(from_suffix_alone) == TARGET_CANDLES
    assert from_full != from_suffix_alone, "the fake decoder is not causal"
    assert from_full[0].close != from_suffix_alone[0].close


# ============================================ methods use the official suffix


def test_method_b_uses_the_backend_suffix() -> None:
    codec = CausalFakeCodec()
    model = WindowDecodingModel(codec=codec)
    series = _series()
    state = fit_context_state(series.context)

    result = run_method_b(model=model, codec=codec, series=series, state=state)

    assert result.raw_decoded == model.suffix_returned
    assert len(result.raw_decoded) == TARGET_CANDLES
    # The only decode was of the full window, never of 64 tokens alone.
    assert codec.decode_windows == [TOTAL_CANDLES]


def test_method_b_would_differ_under_generated_only_decoding() -> None:
    """The regression this commit exists to prevent is detectable."""
    codec = CausalFakeCodec()
    model = WindowDecodingModel(codec=codec)
    series = _series()
    state = fit_context_state(series.context)

    result = run_method_b(model=model, codec=codec, series=series, state=state)

    suffix_alone = codec.decode(
        result_tokens(result), state=state, sessions=tuple(r.session for r in series.target)
    )
    assert result.raw_decoded != suffix_alone
    assert result.raw_decoded[0].close != suffix_alone[0].close


def result_tokens(result) -> tuple[TokenPair, ...]:
    return tuple(
        TokenPair(coarse=c, fine=f)
        for c, f in zip(result.coarse_token_ids, result.fine_token_ids, strict=True)
    )


def test_method_d_uses_the_backend_suffix_for_every_rollout() -> None:
    codec = CausalFakeCodec()
    model = WindowDecodingModel(codec=codec)
    series = _series()
    state = fit_context_state(series.context)

    seeds = (11, 22, 33)
    result = run_method_d(model=model, codec=codec, series=series, state=state, seeds=seeds)

    assert result.rollout_count == 3
    # One full-window decode per rollout, and nothing narrower.
    assert codec.decode_windows == [TOTAL_CANDLES] * 3
    assert all(len(r.raw_decoded) == TARGET_CANDLES for r in result.rollouts)
    # Distinct seeds give distinct suffixes, so the rows really came from the
    # per-rollout decode rather than from a shared cache.
    closes = {r.raw_decoded[0].close for r in result.rollouts}
    assert len(closes) == 3


def test_a_backend_returning_the_wrong_suffix_length_fails_closed() -> None:
    codec = CausalFakeCodec()
    series = _series()
    state = fit_context_state(series.context)

    class ShortSuffixModel(WindowDecodingModel):
        def generate(self, *args, **kwargs):
            path = super().generate(*args, **kwargs)
            return path.model_copy(update={"raw_decoded_suffix": path.raw_decoded_suffix[:-1]})

    with pytest.raises(ResearchFailureError) as excinfo:
        run_method_b(model=ShortSuffixModel(codec=codec), codec=codec, series=series, state=state)
    assert excinfo.value.failures[0].code == "DIAGNOSTIC_DECODED_SUFFIX_LENGTH_MISMATCH"


# ================================= no generated-only decode remains in B or D


def _decode_calls_on_generated_tokens(function) -> list[str]:
    """Any codec.decode(...) whose first argument is the generated tokens."""
    tree = ast.parse(inspect.getsource(function).lstrip())
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if isinstance(target, ast.Attribute) and target.attr == "decode":
            rendered = ast.unparse(node)
            if "generated.tokens" in rendered or "generated_tokens" in rendered:
                offenders.append(rendered)
    return offenders


@pytest.mark.parametrize("function", [run_method_b, run_method_d])
def test_no_generated_only_decode_remains(function) -> None:
    assert _decode_calls_on_generated_tokens(function) == []
    source = inspect.getsource(function)
    assert "codec.decode(generated.tokens" not in source
    assert "generated.raw_decoded_suffix" in source


def test_method_a_still_decodes_its_own_full_window() -> None:
    """A is a round trip over one complete 512-token window, so it keeps the codec."""
    source = inspect.getsource(methods_module.run_method_a)
    assert "codec.encode(rows, state=state)" in source
    assert "codec.decode(tokens, state=state, sessions=series.sessions)" in source


# ====================================== the backend order matches the source


def test_the_backend_decodes_in_the_pinned_source_order() -> None:
    source = inspect.getsource(official_backend.OfficialForecastModel._decode_official_window)
    concatenate = source.index("full_coarse = torch.cat")
    select = source.index("window_start = max(0, total - self._max_context)")
    decode = source.index("self._tokenizer.decode([window_coarse, window_fine], half=True)")
    slice_suffix = source.index("suffix = decoded[0, -steps:, :]")
    invert = source.index("state.invert(normalized)")
    assert concatenate < select < decode < slice_suffix < invert


def test_the_pinned_source_uses_that_same_order() -> None:
    """Read off the vendored file, so the claim is checkable, not asserted."""
    text = VENDOR.read_text(encoding="utf-8")
    body = text[text.index("def auto_regressive_inference") : text.index("def calc_time_stamps")]
    concatenate = body.index("full_pre = torch.cat([x_token[0], generated_pre], dim=1)")
    select = body.index("context_start = max(0, total_seq_len - max_context)")
    decode = body.index("z = tokenizer.decode(input_tokens, half=True)")
    assert concatenate < select < decode
    # The suffix slice and the inverse both happen after, in generate/predict.
    assert "preds = preds[:, -pred_len:, :]" in text
    assert text.index("z = tokenizer.decode(input_tokens, half=True)") < text.index(
        "preds = preds[:, -pred_len:, :]"
    )
    assert text.index("preds = preds[:, -pred_len:, :]") < text.index(
        "preds = preds * (x_std + 1e-5) + x_mean"
    )


def test_the_backend_preserves_the_context_tokens_for_the_concatenation() -> None:
    source = inspect.getsource(official_backend.OfficialForecastModel.generate)
    # The running buffers are clones, so the originals survive untouched.
    assert "coarse_ids = context_coarse.clone()" in source
    assert "fine_ids = context_fine.clone()" in source
    assert "context_coarse=context_coarse" in source
    assert "context_fine=context_fine" in source


def test_the_backend_requires_the_untruncated_window_shape() -> None:
    source = inspect.getsource(official_backend.OfficialForecastModel._decode_official_window)
    assert "(1, window_length, 6)" in source
    assert "OFFICIAL_CONTEXT_WOULD_TRUNCATE" in source
    assert "expected_total = int(context_coarse.size(1)) + steps" in source
