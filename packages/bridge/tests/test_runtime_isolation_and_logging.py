"""Operational-runtime corrections: targeted isolation, one shared runtime
lifetime, worker context ownership, and safe failure logging.

Every test executes real code. Nothing here touches a network, an official
asset, Torch, or any partition.
"""

from __future__ import annotations

import ast
import contextlib
import importlib.abc
import importlib.util
import inspect
import logging
import sys
import textwrap
import types
from pathlib import Path

import pytest
from openalpha_bridge.diagnostic import official_backend, safe_logging
from openalpha_bridge.diagnostic import worker as worker_module
from openalpha_bridge.diagnostic.official_backend import (
    OFFICIAL_ALIASES,
    isolated_official_source,
)
from openalpha_bridge.diagnostic.safe_logging import (
    STAGES,
    StageTracker,
    log_operational_failure,
    sanitized_frames,
)
from openalpha_bridge.errors import BridgeTransformError

ROOT = Path(__file__).resolve().parents[3]
VENDOR = ROOT / "vendor" / "kronos" / "67b630e6"
APP = ROOT / "cloud" / "modal" / "bridge_phase2_app.py"

SENTINEL_MODULE = "openalpha_isolation_probe_module"


@pytest.fixture(autouse=True)
def _clean_aliases():
    """No test may leak an official alias into another."""
    yield
    for alias in (*OFFICIAL_ALIASES, SENTINEL_MODULE):
        sys.modules.pop(alias, None)


class _SentinelLoader(importlib.abc.Loader):
    """Materialises a stub ``einops`` and, while doing so, an unrelated module.

    Torch is deliberately absent from the development environment, so the
    context can never reach its ``yield`` here and no test can put a module
    inside the block by hand. It does not need to: the vendored source imports
    einops before torch, so supplying einops causes two modules to be created
    *during* entry -- einops itself and the sentinel. Those stand in for the
    compiled Torch submodules a real load creates, which is precisely what the
    old teardown purged.
    """

    def create_module(self, spec: object) -> types.ModuleType:
        module = types.ModuleType("einops")
        module.rearrange = lambda *a, **k: None  # type: ignore[attr-defined]
        module.reduce = lambda *a, **k: None  # type: ignore[attr-defined]
        return module

    def exec_module(self, module: types.ModuleType) -> None:
        sys.modules[SENTINEL_MODULE] = types.ModuleType(SENTINEL_MODULE)


class _SentinelFinder:
    def find_spec(self, fullname: str, path: object = None, target: object = None):
        if fullname != "einops":
            return None
        return importlib.util.spec_from_loader(fullname, _SentinelLoader())


@contextlib.contextmanager
def _einops_available():
    finder = _SentinelFinder()
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)
        sys.modules.pop("einops", None)


def _enter_and_create_modules_inside():
    """Enter the context far enough that new modules appear inside it."""
    with _einops_available(), contextlib.suppress(Exception), isolated_official_source(VENDOR):
        pass  # pragma: no cover - unreachable without torch


# ============================================== A: targeted isolation


def test_an_unrelated_module_imported_inside_survives() -> None:
    """The defect that produced the SystemError.

    The context used to delete every module that appeared while it was open,
    which for a Torch model load meant evicting compiled submodules and leaving
    partially-initialised C extensions behind for anything that re-imported
    them.
    """
    assert SENTINEL_MODULE not in sys.modules
    assert "einops" not in sys.modules
    _enter_and_create_modules_inside()
    assert SENTINEL_MODULE in sys.modules, "an unrelated module was purged"


def test_all_three_aliases_are_removed_when_absent_before_entry() -> None:
    for alias in OFFICIAL_ALIASES:
        assert alias not in sys.modules
    with contextlib.suppress(Exception), isolated_official_source(VENDOR):
        pass
    for alias in OFFICIAL_ALIASES:
        assert alias not in sys.modules


def test_preexisting_aliases_are_restored_by_object_identity() -> None:
    originals = {name: types.ModuleType(name) for name in OFFICIAL_ALIASES}
    for name, module in originals.items():
        sys.modules[name] = module

    with contextlib.suppress(Exception), isolated_official_source(VENDOR):
        # Whatever the context put here, it is not the caller's object.
        pass

    for name, module in originals.items():
        assert sys.modules[name] is module, f"{name} was not restored by identity"


def test_restoration_happens_on_an_exceptional_exit() -> None:
    """Entry raises here, which reaches the same ``finally`` a body raise does.

    There is one ``try``/``finally`` in the context, so success, a failed entry
    and a raising body all restore through the same statements. The structural
    test below pins that single-teardown shape, since torch's absence means a
    body raise cannot be staged in this environment.
    """
    marker = types.ModuleType("model")
    sys.modules["model"] = marker
    path_before = list(sys.path)

    # Any entry failure is enough here; the point is that the finally ran.
    with pytest.raises(Exception), isolated_official_source(VENDOR):  # noqa: B017
        pass  # pragma: no cover - unreachable without torch

    assert sys.modules["model"] is marker
    assert sys.path == path_before


def test_restoration_is_unconditional_for_every_exit_path() -> None:
    """One try/finally, no early returns, so no exit path can skip teardown."""
    source = textwrap.dedent(inspect.getsource(isolated_official_source))
    function = ast.parse(source).body[0]
    assert isinstance(function, ast.FunctionDef)

    tries = [n for n in ast.walk(function) if isinstance(n, ast.Try) and n.finalbody]
    assert len(tries) == 1, "teardown must live in exactly one finally"

    # The yield is inside that try, so a raising body unwinds through it.
    yields = [n for n in ast.walk(tries[0]) if isinstance(n, ast.Yield)]
    assert yields, "the yield must sit inside the protected block"

    returns = [n for n in ast.walk(function) if isinstance(n, ast.Return)]
    assert not returns, "an early return could bypass the finally"


def test_sys_path_is_restored_on_every_exit() -> None:
    before = list(sys.path)

    with contextlib.suppress(Exception), isolated_official_source(VENDOR):
        pass  # pragma: no cover - unreachable without torch
    assert sys.path == before, "sys.path leaked after a failed entry"

    _enter_and_create_modules_inside()
    assert sys.path == before, "sys.path leaked after a deeper failed entry"

    # Not vacuous: entry really does mutate sys.path, and restores by assignment
    # rather than by removing what it thinks it added.
    source = inspect.getsource(isolated_official_source)
    assert "sys.path.insert(0, str(root))" in source
    assert "sys.path[:] = path_snapshot" in source


def test_no_unrelated_preexisting_module_is_changed() -> None:
    watched = ("json", "hashlib", "pathlib", "logging", "pydantic")
    before = {name: sys.modules.get(name) for name in watched}
    _enter_and_create_modules_inside()
    for name, module in before.items():
        assert sys.modules.get(name) is module, f"{name} was replaced or removed"


def test_the_context_only_snapshots_the_three_aliases() -> None:
    source = inspect.getsource(isolated_official_source)
    assert "set(sys.modules) - set(modules_snapshot)" not in source
    assert "alias_snapshot" in source
    assert "_ABSENT" in source


# ============================== B: one shared runtime, one lifetime


def test_official_runtime_is_the_only_lifecycle_owner() -> None:
    runtime = inspect.getsource(official_backend.official_runtime)
    assert runtime.count("isolated_official_source(") == 1
    # Loading, wrapper construction and the digest all happen inside it.
    entered = runtime.index("with isolated_official_source(")
    for inside in (
        "load_and_freeze_official(",
        "OfficialTokenizerCodec(",
        "OfficialForecastModel(",
        "parameter_digest(tokenizer, model)",
    ):
        assert runtime.index(inside) > entered


def test_the_loader_owns_no_context_of_its_own() -> None:
    loader = inspect.getsource(official_backend.load_and_freeze_official)
    assert "isolated_official_source" not in loader
    assert "official.KronosTokenizer.from_pretrained" in loader
    assert "official.Kronos.from_pretrained" in loader


def test_the_old_double_context_loader_is_gone() -> None:
    assert not hasattr(official_backend, "load_official_components")


def _modal_function(name: str) -> str:
    source = APP.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(source, node) or ""


def test_probe_and_diagnostic_share_one_loader_and_lifecycle() -> None:
    """Fails if either reverts to its own loader or its own import context."""
    probe = _modal_function("verify_frozen_inference_runtime")
    diagnostic = _modal_function("frozen_inference_diagnostic")
    for body in (probe, diagnostic):
        assert "official_runtime(" in body
        assert "load_official_components(" not in body
        assert "isolated_official_source(" not in body
        assert "OfficialTokenizerCodec(" not in body
        assert "OfficialForecastModel(" not in body


# ============================== E: one restricted snapshot file set


def test_both_functions_restrict_both_downloads_identically() -> None:
    probe = _modal_function("verify_frozen_inference_runtime")
    diagnostic = _modal_function("frozen_inference_diagnostic")
    for body in (probe, diagnostic):
        assert body.count("snapshot_download(") == 2
        assert body.count("allow_patterns=list(OFFICIAL_SNAPSHOT_ALLOW_PATTERNS)") == 2


def test_the_allow_pattern_set_is_a_single_shared_constant() -> None:
    from openalpha_bridge.diagnostic.spec import OFFICIAL_SNAPSHOT_ALLOW_PATTERNS

    assert OFFICIAL_SNAPSHOT_ALLOW_PATTERNS == ("config.json", "model.safetensors")
    source = APP.read_text(encoding="utf-8")
    # No second, hand-written list anywhere.
    assert '["config.json", "model.safetensors"]' not in source


def test_every_downloaded_file_is_hash_verified_before_loading() -> None:
    verify = inspect.getsource(official_backend.verify_official_assets)
    for label in ("tokenizer config", "tokenizer weights", "model config", "model weights"):
        assert label in verify
    runtime = inspect.getsource(official_backend.official_runtime)
    assert runtime.index("verify_official_assets(") < runtime.index("load_and_freeze_official(")


# ================================== D: safe operational logging


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


CREDENTIAL = "hunter2-super-secret-token"
URL = "https://provider.invalid/v1/bars?apikey=" + CREDENTIAL


def _raise_with_secrets() -> None:
    raise ConnectionResetError(f"auth failed for {URL} in /home/tommy/private/thing.py")


def test_safe_logging_records_class_stage_and_locations() -> None:
    logger = logging.getLogger("openalpha.test.safe")
    handler = _Capture()
    logger.addHandler(handler)
    try:
        with pytest.raises(ConnectionResetError) as excinfo:
            _raise_with_secrets()
        record = log_operational_failure(
            excinfo.value,
            run_id="canary_0badc0de",
            deployed_commit="a" * 40,
            stage="retrieve_series",
            logger=logger,
        )
    finally:
        logger.removeHandler(handler)

    message = "\n".join(handler.messages)
    assert "canary_0badc0de" in message
    assert "a" * 40 in message
    assert "stage=retrieve_series" in message
    assert "exception_class=ConnectionResetError" in message
    # A real location, file name only.
    assert "test_runtime_isolation_and_logging.py:_raise_with_secrets:" in message
    assert record.exception_class == "ConnectionResetError"
    assert record.frames


def test_safe_logging_leaks_no_secret_url_or_message() -> None:
    logger = logging.getLogger("openalpha.test.safe2")
    handler = _Capture()
    logger.addHandler(handler)
    try:
        with pytest.raises(ConnectionResetError) as excinfo:
            _raise_with_secrets()
        log_operational_failure(
            excinfo.value,
            run_id="canary_0badc0de",
            deployed_commit="a" * 40,
            stage="retrieve_series",
            logger=logger,
        )
    finally:
        logger.removeHandler(handler)

    message = "\n".join(handler.messages)
    for leaked in (CREDENTIAL, "provider.invalid", "apikey", "auth failed", "/home/tommy"):
        assert leaked not in message


def test_sanitized_frames_carry_no_path_or_source_text() -> None:
    with pytest.raises(ConnectionResetError) as excinfo:
        _raise_with_secrets()
    frames = sanitized_frames(excinfo.value.__traceback__)
    assert frames
    for frame in frames:
        assert frame.count(":") == 2
        assert "/" not in frame
        assert "\\" not in frame
        assert CREDENTIAL not in frame


def test_the_logger_never_formats_the_exception() -> None:
    source = inspect.getsource(safe_logging)
    assert ".exception(" not in source
    assert "str(error)" not in source
    assert "error.args" not in source


def test_the_stage_vocabulary_covers_the_pipeline() -> None:
    for stage in (
        "validate_invocation",
        "verify_existing_artifact",
        "enter_official_runtime",
        "construct_provider",
        "retrieve_series",
        "fit_normalization",
        "method_a",
        "method_b",
        "method_c",
        "method_d",
        "verify_parameters",
        "compute_decision",
    ):
        assert stage in STAGES


def test_the_stage_tracker_reports_the_latest_stage() -> None:
    tracker = StageTracker()
    assert tracker.stage == "validate_invocation"
    tracker.enter("method_c")
    assert tracker.stage == "method_c"


def test_the_worker_logs_only_unexpected_failures() -> None:
    source = inspect.getsource(worker_module.run_diagnostic_worker)
    # Typed failures already carry a message this code wrote.
    assert "except BridgeTransformError:" in source
    assert "log_operational_failure(" in source
    assert source.index("except BridgeTransformError:") < source.index("log_operational_failure(")


def test_a_typed_failure_needs_no_sanitising() -> None:
    assert issubclass(BridgeTransformError, Exception)
