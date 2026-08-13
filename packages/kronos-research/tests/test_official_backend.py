"""The official backend, exercised with doubles.

No official asset is downloaded and no weight is loaded. Torch is not
installed in the base environment, so anything that needs it is asserted
structurally; everything that does not is executed.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import inspect
import sys
from pathlib import Path

import pytest
from openalpha_kronos.model import official as official_backend
from openalpha_kronos.model.official import (
    isolated_official_source,
    verify_asset_file,
)
from openalpha_research.failures import ResearchFailureError

ROOT = Path(__file__).resolve().parents[3]
VENDOR = ROOT / "vendor" / "kronos" / "67b630e6"


# ======================================================= identity verification


def test_a_matching_asset_verifies(tmp_path: Path) -> None:
    target = tmp_path / "config.json"
    payload = b'{"d_model": 256}'
    target.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    assert verify_asset_file(target, digest, label="model config") == digest


def test_a_drifted_asset_fails_closed(tmp_path: Path) -> None:
    target = tmp_path / "model.safetensors"
    target.write_bytes(b"not the pinned weights")
    with pytest.raises(ResearchFailureError) as excinfo:
        verify_asset_file(target, "0" * 64, label="model weights")
    assert excinfo.value.failures[0].code == "OFFICIAL_ASSET_HASH_MISMATCH"


def test_a_missing_asset_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ResearchFailureError) as excinfo:
        verify_asset_file(tmp_path / "absent.safetensors", "0" * 64, label="model weights")
    assert excinfo.value.failures[0].code == "OFFICIAL_ASSET_MISSING"


def test_assets_are_verified_before_any_weight_is_read() -> None:
    """Verification and loading are separate, and verification comes first."""
    verify = inspect.getsource(official_backend.verify_official_assets)
    for check in ("tokenizer config", "tokenizer weights", "model config", "model weights"):
        assert check in verify
    # The verifier loads nothing at all.
    assert "from_pretrained" not in verify

    runtime = inspect.getsource(official_backend.official_runtime)
    assert runtime.index("verify_official_assets(") < runtime.index("load_and_freeze_official(")
    assert runtime.index("verify_source_files(source_root)") < runtime.index(
        "load_and_freeze_official("
    )


# ============================================================ import isolation


def test_isolated_import_restores_path_and_official_aliases() -> None:
    """A real import of the verified source, then the aliases put back.

    The official kronos.py imports torch, which the base environment does not
    have, so the import is expected to fail here. What matters is that the
    failure still restores sys.path exactly and leaves none of the three
    official aliases behind.
    """
    path_before = list(sys.path)

    with contextlib.suppress(Exception), isolated_official_source(VENDOR):
        pass

    assert sys.path == path_before
    for alias in official_backend.OFFICIAL_ALIASES:
        assert alias not in sys.modules


def test_isolated_import_rejects_unverified_source(tmp_path: Path) -> None:
    fake = tmp_path / "model"
    fake.mkdir()
    (fake / "kronos.py").write_bytes(b"# not the pinned source\n")
    (fake / "module.py").write_bytes(b"# not the pinned source\n")
    with pytest.raises(ResearchFailureError) as excinfo, isolated_official_source(tmp_path):
        pass
    assert excinfo.value.failures[0].code.startswith("OFFICIAL_SOURCE_")


def test_isolation_snapshots_before_it_mutates() -> None:
    source = inspect.getsource(official_backend.isolated_official_source)
    assert source.index("path_snapshot = list(sys.path)") < source.index("sys.path.insert")
    assert source.index("alias_snapshot") < source.index('sys.modules["model.module"]')
    assert "finally:" in source


def test_isolation_never_purges_unrelated_modules() -> None:
    """The defect that produced the SystemError, asserted structurally."""
    source = inspect.getsource(official_backend.isolated_official_source)
    assert "set(sys.modules) - set(modules_snapshot)" not in source
    assert "for name, previous in alias_snapshot.items():" in source
    assert official_backend.OFFICIAL_ALIASES == ("model", "model.module", "model.kronos")


# ========================================================== freezing contract


def test_the_freeze_helper_evals_and_clears_every_gradient() -> None:
    source = inspect.getsource(official_backend._freeze)
    assert "module.eval()" in source
    assert "parameter.requires_grad = False" in source
    # Both are verified afterwards rather than assumed to have taken.
    assert "if module.training:" in source
    assert "if trainable != 0:" in source


def test_both_official_modules_are_frozen_and_counted() -> None:
    loader = inspect.getsource(official_backend.load_and_freeze_official)
    assert '_freeze(tokenizer, "tokenizer")' in loader
    assert '_freeze(model, "model")' in loader
    assert "tokenizer_total + model_total" in loader
    assert "tokenizer_trainable + model_trainable" in loader

    runtime = inspect.getsource(official_backend.official_runtime)
    assert "trainable_parameter_count=trainable" in runtime
    assert "total_parameter_count=total" in runtime


def test_generation_runs_under_inference_mode() -> None:
    source = inspect.getsource(official_backend.OfficialForecastModel.generate)
    assert "with torch.inference_mode():" in source
    guard = source.index("with torch.inference_mode():")
    assert source.index("self._tokenizer.encode(", guard) > guard
    assert source.index("decode_s1", guard) > guard


def test_the_codec_also_runs_under_inference_mode() -> None:
    for method in (
        official_backend.OfficialTokenizerCodec.encode,
        official_backend.OfficialTokenizerCodec.decode,
    ):
        assert "with torch.inference_mode():" in inspect.getsource(method)


# ================================================== six channels and stamps


def test_the_input_tensor_is_six_channels() -> None:
    source = inspect.getsource(official_backend._rows_to_tensor)
    assert "(1, len(rows), 6)" in source
    assert "state.normalize(rows)" in source


def test_the_stamp_tensor_is_five_features() -> None:
    source = inspect.getsource(official_backend._stamps_to_tensor)
    assert "(1, len(stamps), 5)" in source


def test_decode_asserts_the_six_channel_output_shape() -> None:
    source = inspect.getsource(official_backend.OfficialTokenizerCodec.decode)
    assert "(1, len(tokens), 6)" in source
    assert "state.invert(normalized)" in source


def test_the_codec_holds_no_normalization_state() -> None:
    """It is told the state on every call and remembers nothing."""
    init = inspect.getsource(official_backend.OfficialTokenizerCodec.__init__)
    assert "state" not in init
    for method in ("encode", "decode"):
        parameters = inspect.signature(
            getattr(official_backend.OfficialTokenizerCodec, method)
        ).parameters
        assert "state" in parameters
        assert parameters["state"].kind is inspect.Parameter.KEYWORD_ONLY


def test_the_model_holds_no_normalization_state() -> None:
    init = inspect.getsource(official_backend.OfficialForecastModel.__init__)
    assert "state" not in init
    parameters = inspect.signature(official_backend.OfficialForecastModel.generate).parameters
    assert parameters["state"].kind is inspect.Parameter.KEYWORD_ONLY


# ========================================================== failure surfaces


@pytest.mark.parametrize(
    "code",
    [
        "OFFICIAL_TENSOR_SHAPE_MISMATCH",
        "OFFICIAL_STAMP_SHAPE_MISMATCH",
        "OFFICIAL_ENCODE_SHAPE_MISMATCH",
        "OFFICIAL_DECODE_SHAPE_MISMATCH",
        "OFFICIAL_DECODE_SESSION_MISMATCH",
        "OFFICIAL_TOKEN_OUT_OF_VOCABULARY",
        "OFFICIAL_TOKEN_LENGTH_MISMATCH",
        "OFFICIAL_CONTEXT_WOULD_TRUNCATE",
        "OFFICIAL_ASSET_HASH_MISMATCH",
        "OFFICIAL_MODULE_NOT_FROZEN",
        "OFFICIAL_MODULE_IN_TRAINING_MODE",
        "OFFICIAL_SAMPLING_PROBABILITY_INVALID",
    ],
)
def test_every_declared_failure_is_reachable_in_source(code: str) -> None:
    assert code in inspect.getsource(official_backend)


def test_a_zero_probability_sample_is_refused() -> None:
    with pytest.raises(ResearchFailureError) as excinfo:
        official_backend._safe_log(0.0)
    assert excinfo.value.failures[0].code == "OFFICIAL_SAMPLING_PROBABILITY_INVALID"


def test_a_valid_probability_logs_normally() -> None:
    import math

    assert official_backend._safe_log(0.5) == pytest.approx(math.log(0.5))


# ================================================ sampling matches the source


def test_the_official_filtering_function_is_called_not_reimplemented() -> None:
    source = inspect.getsource(official_backend.OfficialForecastModel.generate)
    assert "self._official.top_k_top_p_filtering(" in source
    # The order the pinned source uses: temperature, filter, softmax, sample.
    draw = source[source.index("def draw(") : source.index("for step in range(steps)")]
    assert draw.index("logits / temperature") < draw.index("top_k_top_p_filtering")
    assert draw.index("top_k_top_p_filtering") < draw.index("F.softmax")
    assert draw.index("F.softmax") < draw.index("torch.multinomial")


def test_the_probability_comes_from_the_distribution_that_was_sampled() -> None:
    source = inspect.getsource(official_backend.OfficialForecastModel.generate)
    draw = source[source.index("def draw(") : source.index("for step in range(steps)")]
    # One softmax, used both to sample and to read the probability off.
    assert draw.count("F.softmax") == 1
    assert "probability = float(probabilities[0, index])" in draw


def test_the_fine_token_is_conditioned_on_the_sampled_coarse_token() -> None:
    source = inspect.getsource(official_backend.OfficialForecastModel.generate)
    assert "self._model.decode_s2(decoded_context, sampled_coarse)" in source
    assert source.index("coarse_index, coarse_probability = draw(") < source.index(
        "decode_s2(decoded_context, sampled_coarse)"
    )


def test_the_caller_seeds_because_the_source_does_not() -> None:
    source = inspect.getsource(official_backend.OfficialForecastModel.generate)
    assert "torch.manual_seed(seed)" in source


# ================================================== nothing trains, ever


def _call_graph(module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(inspect.getsource(module))):
        if isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Name):
                names.add(target.id)
            elif isinstance(target, ast.Attribute):
                names.add(target.attr)
    return names


@pytest.mark.parametrize(
    "forbidden",
    [
        "backward",
        "zero_grad",
        "Adam",
        "AdamW",
        "SGD",
        "optimizer",
        "train",
        "run_training",
        "select_checkpoint",
        "freeze_checkpoint",
        "save",
        "state_dict",
        "load_state_dict",
        "requires_grad_",
    ],
)
def test_no_training_call_is_reachable(forbidden: str) -> None:
    assert forbidden not in _call_graph(official_backend)


def test_the_beam_search_is_still_not_implemented() -> None:
    assert "beam" not in inspect.getsource(official_backend).lower()


def test_importing_the_backend_loads_neither_torch_nor_the_official_source() -> None:
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, openalpha_kronos.model.official as b; "
                "print('torch' in sys.modules, 'model.kronos' in sys.modules)"
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False False"


def test_the_ordinary_suite_downloads_no_official_asset() -> None:
    """Nothing in this module reaches the network."""
    source = inspect.getsource(official_backend)
    for network in ("requests", "urllib", "httpx", "hf_hub_download", "snapshot_download"):
        assert network not in source
    # from_pretrained is called against an already-resolved local directory.
    assert "from_pretrained(str(Path(tokenizer_directory)))" in source
    assert "from_pretrained(str(Path(model_directory)))" in source
