"""Every locked inference value, re-derived from the pinned source.

The source files and the digests sealed in experiment.yaml are the authority.
The README is not consulted anywhere in this file.

The source is read from the vendored verbatim copy under vendor/kronos, whose
digests are verified here first. Nothing is downloaded, nothing is imported,
and no weight is loaded.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest
from openalpha_kronos.model import source as source_contract_module
from openalpha_kronos.model.assets import OFFICIAL_SOURCE_FILES, SOURCE_SPEC
from openalpha_kronos.model.source import (
    SOURCE_CONTRACT,
    SOURCE_REVISION,
    crlf_normalize,
    digest_pair,
    verify_source_files,
)
from openalpha_research.failures import ResearchFailureError

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "vendor" / "kronos" / "67b630e6"
KRONOS = VENDOR / "model" / "kronos.py"
MODULE = VENDOR / "model" / "module.py"


@pytest.fixture(scope="module")
def source() -> str:
    return KRONOS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def tree(source: str) -> ast.Module:
    return ast.parse(source)


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in the pinned source")


def _method(tree: ast.Module, class_name: str, method: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method:
                    return child
    raise AssertionError(f"{class_name}.{method} not found in the pinned source")


def _defaults(function: ast.FunctionDef) -> dict[str, object]:
    """Parameter name to default value, for parameters that have one."""
    args = function.args.args
    defaults = function.args.defaults
    named = args[len(args) - len(defaults) :]
    return {
        arg.arg: ast.literal_eval(default) for arg, default in zip(named, defaults, strict=True)
    }


# ============================================================ file identity


def test_the_vendored_source_matches_both_locked_digests() -> None:
    digests = verify_source_files(VENDOR)
    assert {d.relative_path for d in digests} == {"model/kronos.py", "model/module.py"}


def test_the_sealed_digest_is_the_crlf_normalized_one() -> None:
    """One rule explains both sealed values in experiment.yaml."""
    for expected in OFFICIAL_SOURCE_FILES:
        data = (VENDOR / expected.relative_path).read_bytes()
        observed = digest_pair(expected.relative_path, data)
        assert observed.crlf_normalized_sha256 == expected.sealed_sha256_crlf_normalized
        assert observed.crlf_normalized_sha256 == SOURCE_SPEC.files[expected.relative_path]
        assert observed.as_committed_sha256 == expected.as_committed_sha256


def test_kronos_py_is_committed_with_lf_and_module_py_with_crlf() -> None:
    """The reason the two sealed digests looked inconsistent."""
    kronos = KRONOS.read_bytes()
    module = MODULE.read_bytes()
    assert b"\r\n" not in kronos
    assert b"\r\n" in module
    # kronos.py's sealed digest is therefore not its upstream digest.
    assert hashlib.sha256(kronos).hexdigest() != SOURCE_SPEC.files["model/kronos.py"]
    assert (
        hashlib.sha256(crlf_normalize(kronos)).hexdigest() == (SOURCE_SPEC.files["model/kronos.py"])
    )


def test_crlf_normalization_is_idempotent() -> None:
    for path in (KRONOS, MODULE):
        once = crlf_normalize(path.read_bytes())
        assert crlf_normalize(once) == once


def test_a_tampered_source_file_fails_closed(tmp_path: Path) -> None:
    fake = tmp_path / "model"
    fake.mkdir()
    (fake / "kronos.py").write_bytes(b"# not the pinned source\n")
    (fake / "module.py").write_bytes(MODULE.read_bytes())
    with pytest.raises(ResearchFailureError) as excinfo:
        verify_source_files(tmp_path)
    assert excinfo.value.failures[0].code == "OFFICIAL_SOURCE_AS_COMMITTED_MISMATCH"


def test_the_revision_agrees_with_the_sealed_specification() -> None:
    module_tree = ast.parse(Path(source_contract_module.__file__).read_text(encoding="utf-8"))
    assignment = next(
        node
        for node in module_tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "SOURCE_REVISION"
    )
    assert assignment.value is not None
    assert ast.unparse(assignment.value) == "SOURCE_SPEC.revision"
    assert SOURCE_REVISION is SOURCE_SPEC.revision
    assert SOURCE_CONTRACT.revision == SOURCE_SPEC.revision


# ==================================================== predictor entry point


def test_the_predict_signature_is_what_was_locked(tree: ast.Module) -> None:
    predict = _method(tree, "KronosPredictor", "predict")
    names = tuple(arg.arg for arg in predict.args.args)
    assert names == SOURCE_CONTRACT.predict_parameters


def test_predict_defaults_are_what_was_locked(tree: ast.Module) -> None:
    defaults = _defaults(_method(tree, "KronosPredictor", "predict"))
    assert defaults["T"] == SOURCE_CONTRACT.predict_default_temperature == 1.0
    assert defaults["top_k"] == SOURCE_CONTRACT.predict_default_top_k == 0
    assert defaults["top_p"] == SOURCE_CONTRACT.predict_default_top_p == 0.9
    assert defaults["sample_count"] == SOURCE_CONTRACT.predict_default_sample_count == 1
    assert defaults["verbose"] == SOURCE_CONTRACT.predict_default_verbose is True


def test_the_inner_function_has_different_defaults(tree: ast.Module) -> None:
    """Recorded so nobody derives settings from the wrong entry point."""
    defaults = _defaults(_function(tree, "auto_regressive_inference"))
    assert defaults["top_p"] == SOURCE_CONTRACT.inner_default_top_p == 0.99
    assert defaults["sample_count"] == SOURCE_CONTRACT.inner_default_sample_count == 5
    assert defaults["top_p"] != SOURCE_CONTRACT.predict_default_top_p


def test_constructor_defaults_are_what_was_locked(tree: ast.Module) -> None:
    defaults = _defaults(_method(tree, "KronosPredictor", "__init__"))
    assert defaults["max_context"] == SOURCE_CONTRACT.default_max_context == 512
    assert defaults["clip"] == SOURCE_CONTRACT.default_clip == 5


# ========================================================== input columns


def test_the_column_lists_are_what_was_locked(source: str) -> None:
    assert "self.price_cols = ['open', 'high', 'low', 'close']" in source
    assert "self.vol_col = 'volume'" in source
    assert "self.amt_vol = 'amount'" in source
    assert SOURCE_CONTRACT.price_columns == ("open", "high", "low", "close")
    assert SOURCE_CONTRACT.ordered_columns == (
        *SOURCE_CONTRACT.price_columns,
        SOURCE_CONTRACT.volume_column,
        SOURCE_CONTRACT.amount_column,
    )


def test_the_tensor_is_assembled_in_that_exact_order(source: str) -> None:
    assert (
        "x = df[self.price_cols + [self.vol_col, self.amt_vol]].values.astype(np.float32)" in source
    )


def test_the_time_columns_are_what_was_locked(tree: ast.Module, source: str) -> None:
    calc = _function(tree, "calc_time_stamps")
    assigned = [
        node.slice.value
        for node in ast.walk(calc)
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant)
    ]
    assert tuple(assigned) == SOURCE_CONTRACT.time_columns
    assert SOURCE_CONTRACT.time_columns == ("minute", "hour", "weekday", "day", "month")
    assert len(SOURCE_CONTRACT.time_columns) == 5


# ================================================= missing column behaviour


def test_missing_volume_zeroes_both_volume_and_amount(source: str) -> None:
    assert "df[self.vol_col] = 0.0" in source
    assert "df[self.amt_vol] = 0.0" in source
    assert SOURCE_CONTRACT.missing_volume_fills_volume_with == 0.0
    assert SOURCE_CONTRACT.missing_volume_also_fills_amount_with == 0.0


def test_missing_amount_is_derived_from_volume_and_mean_price(source: str) -> None:
    assert "df[self.amt_vol] = df[self.vol_col] * df[self.price_cols].mean(axis=1)" in source


def test_a_nan_in_the_required_columns_raises(source: str) -> None:
    assert "Input DataFrame contains NaN values in price or volume columns." in source
    assert SOURCE_CONTRACT.nan_in_required_columns == "raises ValueError"


# ========================================================== normalization


def test_the_forward_normalization_formula_is_what_was_locked(source: str) -> None:
    assert "x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)" in source
    assert "x = (x - x_mean) / (x_std + 1e-5)" in source
    assert "x = np.clip(x, -self.clip, self.clip)" in source
    assert SOURCE_CONTRACT.normalization_epsilon == 1e-5
    # np.std defaults to ddof = 0, which is what the population formula uses.
    assert SOURCE_CONTRACT.normalization_std_ddof == 0
    assert "ddof" not in source


def test_the_inverse_normalization_formula_is_what_was_locked(source: str) -> None:
    assert "preds = preds * (x_std + 1e-5) + x_mean" in source


def test_the_clip_is_applied_again_inside_inference(source: str) -> None:
    assert "x = torch.clip(x, -clip, clip)" in source
    assert SOURCE_CONTRACT.clip_applied_again_inside_inference is True


# ======================================================= tokenizer shapes


def test_encode_returns_quantized_indices(source: str) -> None:
    assert "def encode(self, x, half=False):" in source
    assert "bsq_loss, quantized, z_indices = self.tokenizer(z, half=half" in source
    assert "return z_indices" in source


def test_decode_returns_the_input_space(source: str) -> None:
    assert "def decode(self, x, half=False):" in source
    assert "Reconstructed output tensor of shape (batch_size, seq_len, d_in)" in source


def test_the_diagnostic_uses_the_half_flag(source: str) -> None:
    assert "x_token = tokenizer.encode(x, half=True)" in source
    assert "z = tokenizer.decode(input_tokens, half=True)" in source
    assert SOURCE_CONTRACT.tokenizer_half_flag is True


# ============================================================== sampling


def test_coarse_then_conditional_fine(source: str) -> None:
    assert "s1_logits, context = model.decode_s1(input_tokens[0], input_tokens[1]" in source
    assert "sample_pre = sample_from_logits(s1_logits" in source
    # The fine head is conditioned on the token just sampled, not on a
    # distribution over coarse tokens.
    assert "s2_logits = model.decode_s2(context, sample_pre)" in source
    assert "sample_post = sample_from_logits(s2_logits" in source
    assert SOURCE_CONTRACT.fine_is_conditional_on_sampled_coarse is True


def test_the_sampling_order_is_temperature_then_filter_then_softmax(source: str) -> None:
    body = source[source.index("def sample_from_logits") : source.index("def auto_regressive")]
    temperature_at = body.index("logits = logits / temperature")
    filter_at = body.index("logits = top_k_top_p_filtering(")
    softmax_at = body.index("probs = F.softmax(logits, dim=-1)")
    sample_at = body.index("torch.multinomial(probs")
    assert temperature_at < filter_at < softmax_at < sample_at
    assert SOURCE_CONTRACT.sampling_order == (
        "divide_logits_by_temperature",
        "top_k_or_top_p_filtering",
        "softmax",
        "multinomial",
    )


def test_top_k_and_top_p_are_mutually_exclusive(tree: ast.Module) -> None:
    """The top-k branch returns, so top_p only runs when top_k == 0.

    Checked on the AST: the docstring quotes both conditions verbatim, so
    searching the text finds the prose before the code.
    """
    filtering = _function(tree, "top_k_top_p_filtering")
    branches = [node for node in filtering.body if isinstance(node, ast.If)]
    assert len(branches) == 2, "expected exactly a top-k branch and a top-p branch"

    top_k_branch, top_p_branch = branches
    assert ast.unparse(top_k_branch.test) == "top_k > 0"
    assert ast.unparse(top_p_branch.test) == "top_p < 1.0"

    # The top-k branch returns, so control never reaches the top-p branch.
    assert isinstance(top_k_branch.body[-1], ast.Return)
    assert isinstance(top_p_branch.body[-1], ast.Return)
    assert SOURCE_CONTRACT.top_k_and_top_p_mutually_exclusive is True
    assert SOURCE_CONTRACT.top_k_takes_precedence is True


def test_filtered_entries_become_negative_infinity(source: str) -> None:
    assert 'filter_value: float = -float("Inf")' in source
    assert "logits[indices_to_remove] = filter_value" in source
    # softmax over a vector containing -inf gives those entries probability
    # zero and renormalizes the survivors. That is what makes the stored
    # probabilities "renormalized over the kept set".
    assert SOURCE_CONTRACT.filter_value == "-inf"


# =============================================== multi-sample and averaging


def test_sample_count_repeats_the_context_along_the_batch(source: str) -> None:
    assert "x = x.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(" in source
    assert "x_stamp = x_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1)" in source


def test_the_official_average_happens_before_inverse_normalization(source: str) -> None:
    inference = source[
        source.index("def auto_regressive_inference") : source.index("def calc_time_stamps")
    ]
    assert "z = tokenizer.decode(input_tokens, half=True)" in inference
    assert "preds = np.mean(preds, axis=1)" in inference
    # The inverse happens later, in predict(). Inverse is affine, so the two
    # orders agree arithmetically, which is why the manual ensemble can be
    # computed after inversion without changing the number.
    assert "preds = preds * (x_std + 1e-5) + x_mean" not in inference
    assert SOURCE_CONTRACT.inverse_is_affine_so_order_is_arithmetically_equivalent is True


# ========================================================= context truncation


def test_the_buffer_keeps_the_most_recent_max_context_tokens(source: str) -> None:
    assert "buffer_len = min(initial_seq_len, max_context)" in source
    assert "start_idx = max(0, initial_seq_len - max_context)" in source
    assert "pre_buffer.copy_(torch.roll(pre_buffer, shifts=-1, dims=1))" in source


def test_no_roll_occurs_in_the_diagnostic_configuration() -> None:
    """448 context plus 64 steps reaches 511, below max_context 512."""
    from openalpha_kronos.studies.structural_validity.mini.spec import (
        CONTEXT_CANDLES,
        TARGET_CANDLES,
        TOTAL_CANDLES,
    )

    final_running_length = CONTEXT_CANDLES + TARGET_CANDLES - 1
    assert final_running_length == 511
    assert final_running_length < TOTAL_CANDLES == SOURCE_CONTRACT.default_max_context
    assert SOURCE_CONTRACT.rolls_in_diagnostic_configuration is False


def test_the_final_decode_covers_the_whole_window_then_slices(source: str) -> None:
    assert "context_start = max(0, total_seq_len - max_context)" in source
    assert "preds = preds[:, -pred_len:, :]" in source


# ============================================================ gradients and RNG


def test_inference_runs_under_no_grad(source: str) -> None:
    assert "with torch.no_grad():" in source
    assert SOURCE_CONTRACT.inference_guard == "torch.no_grad()"


def test_the_source_neither_seeds_nor_exposes_a_seed(source: str) -> None:
    """Determinism is imposed by the caller, and recorded as such."""
    assert "manual_seed" not in source
    assert "torch.Generator" not in source
    assert "seed" not in source.replace("sample_logits", "")
    assert SOURCE_CONTRACT.source_seeds_rng is False
    assert SOURCE_CONTRACT.source_exposes_seed_parameter is False
    assert SOURCE_CONTRACT.determinism_requires_caller_seeding is True


def test_sampling_draws_from_the_global_rng(source: str) -> None:
    assert "x = torch.multinomial(probs, num_samples=1)" in source
    assert SOURCE_CONTRACT.sampling_uses_multinomial is True


# ================================================ the derived settings agree


def test_the_diagnostic_settings_match_the_source_defaults() -> None:
    from openalpha_kronos.studies.structural_validity.mini.spec import (
        OFFICIAL_INFERENCE_SETTINGS as settings,
    )

    assert settings.temperature == SOURCE_CONTRACT.predict_default_temperature
    assert settings.top_k == SOURCE_CONTRACT.predict_default_top_k
    assert settings.top_p == SOURCE_CONTRACT.predict_default_top_p
    assert settings.sample_count_per_call == SOURCE_CONTRACT.predict_default_sample_count
    assert settings.max_context == SOURCE_CONTRACT.default_max_context
    assert settings.clip == float(SOURCE_CONTRACT.default_clip)


def test_the_diagnostic_columns_match_the_source_columns() -> None:
    from openalpha_kronos.model.input import (
        OFFICIAL_COLUMNS,
        OFFICIAL_STAMP_COLUMNS,
    )

    assert OFFICIAL_COLUMNS == SOURCE_CONTRACT.ordered_columns
    assert OFFICIAL_STAMP_COLUMNS == SOURCE_CONTRACT.time_columns


def test_the_diagnostic_normalization_matches_the_source() -> None:
    from openalpha_kronos.model.normalization import CLIP_VALUE, EPSILON

    assert EPSILON == SOURCE_CONTRACT.normalization_epsilon
    assert CLIP_VALUE == float(SOURCE_CONTRACT.default_clip)


def test_nothing_imports_the_vendored_source() -> None:
    """It is reference material for conformance, never executable dependency."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, openalpha_kronos.model.source; "
                "print(any('vendor' in str(m) for m in sys.modules))"
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False"
