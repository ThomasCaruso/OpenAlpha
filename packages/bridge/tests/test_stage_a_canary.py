"""Stage A canary tests. Fake official backend; no network, no real assets."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

import pytest
from openalpha_bridge.errors import BridgeTransformError
from openalpha_bridge.phase2.cache import FeatureCache
from openalpha_bridge.phase2.canary import (
    AMENDMENTS,
    CANARY_EVIDENCE_CLASS,
    CANARY_SUCCESS_CODE,
    CANARY_WINDOW,
    run_stage_a_canary,
)
from openalpha_bridge.phase2.identity import AMENDMENT_3_SHA256
from openalpha_bridge.phase2.kronos import (
    DeterministicFakeKronosBackend,
    KronosMode,
    ResolvedAssets,
)
from openalpha_bridge.phase2.provider import DeterministicFakeProvider

ROOT = Path(__file__).resolve().parents[3]
RESEARCH = ROOT / "research" / "bridge-v0"


class _PseudoOfficialBackend(DeterministicFakeKronosBackend):
    """Reports the official mode so the canary's gate can be exercised.

    Numerically it is still the deterministic fake. It exists only to test the
    canary's own logic; a real canary run uses OfficialKronosBackend.
    """

    observed_source_files: ClassVar[dict[str, str]] = {"model/kronos.py": "e" * 64}
    dependency_versions: ClassVar[dict[str, str]] = {"torch": "2.5.1"}

    @property
    def mode(self) -> KronosMode:
        return KronosMode.PINNED_OFFICIAL

    def resolve_assets(self) -> ResolvedAssets:
        return ResolvedAssets(
            kronos_mode=KronosMode.PINNED_OFFICIAL,
            repository="NeoQuasar/Kronos-Tokenizer-2k",
            revision="26966d0035065a0cae0ebad7af8ece35bc1fb51c",
            observed_config_sha256="0" * 64,
            observed_weights_sha256="1" * 64,
            frozen_parameter_sha256=self.frozen_parameter_sha256(),
            revisions_verified=True,
        )


class _CountingProvider(DeterministicFakeProvider):
    def __init__(self) -> None:
        super().__init__()
        self.requests: list[str] = []

    def fetch(self, request):
        self.requests.append(f"{request.symbol}:{request.start}:{request.end}")
        return super().fetch(request)


def _run(tmp_path: Path, **kwargs):
    provider = kwargs.pop("provider", None) or _CountingProvider()
    return provider, run_stage_a_canary(
        provider=provider,
        backend=kwargs.pop("backend", None) or _PseudoOfficialBackend(),
        cache=FeatureCache(tmp_path / "cache"),
        research_root=RESEARCH,
        source_commit="a" * 40,
        run_id="canary_0badc0de",
        now=datetime(2026, 8, 2, tzinfo=UTC),
        **kwargs,
    )


def test_the_amended_window_is_pinned_and_self_consistent() -> None:
    assert CANARY_WINDOW.symbol == "SPY"
    assert CANARY_WINDOW.sequence_id == "ecd7fd5797a9147a"
    assert CANARY_WINDOW.prefix_start == "2015-05-07"
    assert CANARY_WINDOW.target_end == "2017-05-17"
    spec = CANARY_WINDOW.to_spec()  # derives and cross-checks the sequence id
    assert spec.sequence_id == CANARY_WINDOW.sequence_id
    assert AMENDMENT_3_SHA256 in AMENDMENTS


def test_canary_passes_and_reports_the_locked_contract(tmp_path: Path) -> None:
    _, report = _run(tmp_path)
    assert report.outcome == CANARY_SUCCESS_CODE
    assert report.evidence_class == CANARY_EVIDENCE_CLASS
    assert report.authorizes_stage_b is False
    assert report.authorizes_real_run is False
    assert report.bipolar_latent_shape == (512, 20)
    assert report.frozen_hidden_shape == (512, 256)
    assert report.bridge_input_shape == (512, 269)
    assert report.bridge_input_dtype == "float32"
    assert 0 <= report.coarse_id_range[0] <= report.coarse_id_range[1] <= 1023
    assert report.deterministic_replay_matched
    assert report.scored_suffix_causality_holds
    assert report.cache_read_verified
    assert report.retrieved_candles == 512
    assert report.provider_request_count == 1
    assert report.wall_clock_seconds >= 0.0


def test_canary_makes_exactly_one_provider_request_for_spy_only(tmp_path: Path) -> None:
    """No validation, reconstruction-test, external, or unseen-symbol retrieval."""
    provider, _ = _run(tmp_path)
    assert len(provider.requests) == 1
    assert provider.requests[0].startswith("SPY:2015-05-07:2017-05-18")


def test_canary_refuses_a_non_official_backend(tmp_path: Path) -> None:
    with pytest.raises(BridgeTransformError) as excinfo:
        _run(tmp_path, backend=DeterministicFakeKronosBackend())
    assert excinfo.value.failures[0].code == "CANARY_REQUIRES_OFFICIAL_BACKEND"


def test_canary_report_carries_no_raw_candles(tmp_path: Path) -> None:
    _, report = _run(tmp_path)
    payload = report.model_dump_json()
    assert "open" not in payload.replace("openalpha", "").replace("_open", "")
    assert report.candle_data_sha256


def test_canary_shard_is_written_under_the_training_partition(tmp_path: Path) -> None:
    _, report = _run(tmp_path)
    assert report.shard_relative_path.startswith("train/")
    assert report.cache_schema_version == "openalpha.bridge.phase2.cache.v2"


def test_canary_module_has_no_path_into_later_stages() -> None:
    """The canary must be structurally incapable of continuing.

    Checked by import graph and call graph rather than substring, because a
    field named `authorizes_stage_b` legitimately mentions a later stage in
    order to deny it.
    """
    import ast

    path = (
        ROOT / "packages" / "bridge" / "src" / "openalpha_bridge" / "phase2" / "canary.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))

    forbidden_modules = {"pipeline", "runner", "training", "testgate", "gates", "metrics"}
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.rsplit(".", 1)[-1])
        if isinstance(node, ast.Import):
            imported.update(alias.name.rsplit(".", 1)[-1] for alias in node.names)
    leaked = sorted(forbidden_modules & imported)
    assert not leaked, f"the canary imports later-stage modules: {leaked}"

    forbidden_calls = {
        "run_training",
        "select_checkpoint",
        "open_test_partition",
        "open_cloud_test_partition",
        "evaluate_conclusion",
        "stage_b",
        "stage_c",
        "freeze_checkpoint",
    }
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                called.add(func.id)
            elif isinstance(func, ast.Attribute):
                called.add(func.attr)
    invoked = sorted(forbidden_calls & called)
    assert not invoked, f"the canary calls later-stage functions: {invoked}"


def test_canary_denies_authorization_explicitly() -> None:
    """The denial fields must exist and be immovable."""
    from openalpha_bridge.phase2.canary import CanaryReport

    fields = CanaryReport.model_fields
    assert "authorizes_stage_b" in fields
    assert "authorizes_real_run" in fields


# ------------------------------------------------ invocation identity (item 8)


@pytest.mark.parametrize(
    ("run_id", "code"),
    [
        ("canary_../escape", "CANARY_INVALID_RUN_ID"),
        ("canary_with space", "CANARY_INVALID_RUN_ID"),
        ("canary_/abs/path", "CANARY_INVALID_RUN_ID"),
        ("not_a_canary_id", "CANARY_INVALID_RUN_ID"),
        ("canary_XYZ", "CANARY_INVALID_RUN_ID"),
    ],
)
def test_run_id_must_match_the_strict_canary_format(
    tmp_path: Path, run_id: str, code: str
) -> None:
    with pytest.raises(BridgeTransformError) as excinfo:
        run_stage_a_canary(
            provider=_CountingProvider(),
            backend=_PseudoOfficialBackend(),
            cache=FeatureCache(tmp_path / "cache"),
            research_root=RESEARCH,
            source_commit="a" * 40,
            run_id=run_id,
        )
    assert excinfo.value.failures[0].code == code


@pytest.mark.parametrize(
    "commit", ["short", "A" * 40, "g" * 40, "a" * 39, "a" * 41], ids=lambda c: c[:6]
)
def test_source_commit_must_be_forty_lowercase_hex(tmp_path: Path, commit: str) -> None:
    with pytest.raises(BridgeTransformError) as excinfo:
        run_stage_a_canary(
            provider=_CountingProvider(),
            backend=_PseudoOfficialBackend(),
            cache=FeatureCache(tmp_path / "cache"),
            research_root=RESEARCH,
            source_commit=commit,
            run_id="canary_0badc0de",
        )
    assert excinfo.value.failures[0].code == "CANARY_INVALID_SOURCE_COMMIT"


def test_source_commit_must_match_the_deployed_image(tmp_path: Path) -> None:
    """The canary must run the code that was actually deployed."""
    with pytest.raises(BridgeTransformError) as excinfo:
        run_stage_a_canary(
            provider=_CountingProvider(),
            backend=_PseudoOfficialBackend(),
            cache=FeatureCache(tmp_path / "cache"),
            research_root=RESEARCH,
            source_commit="a" * 40,
            run_id="canary_0badc0de",
            deployed_commit="b" * 40,
        )
    assert excinfo.value.failures[0].code == "CANARY_SOURCE_COMMIT_MISMATCH"


# ------------------------------------------------------- causality (item 5)


def test_causality_is_checked_at_the_first_scored_position() -> None:
    """Perturbing only the final candle would be nearly vacuous."""
    source = (
        ROOT / "packages" / "bridge" / "src" / "openalpha_bridge" / "phase2" / "canary.py"
    ).read_text(encoding="utf-8")
    assert "pivot = CONTEXT_PREFIX_LENGTH" in source
    assert "frozen_hidden[:pivot]" in source
    assert "CANARY_CAUSALITY_PERTURBATION_INEFFECTIVE" in source


def test_an_ineffective_perturbation_is_reported_not_claimed(tmp_path: Path) -> None:
    """A backend that ignores values must not yield a passing causality claim."""

    class _ValueBlindBackend(_PseudoOfficialBackend):
        def encode_tokens(self, features):
            import numpy as _np

            length = features.shape[0]
            return (
                _np.zeros(length, dtype=_np.int64),
                _np.zeros(length, dtype=_np.int64),
            )

    with pytest.raises(BridgeTransformError) as excinfo:
        _run(tmp_path, backend=_ValueBlindBackend())
    assert excinfo.value.failures[0].code == "CANARY_CAUSALITY_PERTURBATION_INEFFECTIVE"


def test_the_fake_backend_is_content_sensitive_and_causal() -> None:
    """Otherwise every causality test in the suite would pass vacuously."""
    import numpy as _np
    from openalpha_bridge.phase2.kronos import DeterministicFakeKronosBackend as _B

    backend = _B()
    base = _np.zeros((32, 6), dtype=_np.float32)
    changed = base.copy()
    changed[10, 0] = 1.0

    c0, f0 = backend.encode_tokens(base)
    c1, _ = backend.encode_tokens(changed)
    assert _np.array_equal(c0[:10], c1[:10]), "earlier rows must not move"
    assert not _np.array_equal(c0[10:], c1[10:]), "later rows must move"
    # Deterministic for identical input.
    assert _np.array_equal(c0, backend.encode_tokens(base)[0])
    assert _np.array_equal(f0, backend.encode_tokens(base)[1])
