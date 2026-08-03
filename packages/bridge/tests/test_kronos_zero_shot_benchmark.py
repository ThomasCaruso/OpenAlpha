"""The zero-shot benchmark contract, and its separation from both completed studies.

No network, no official asset, no Torch, no market data. Every model and provider
here is a deterministic double, and the only bytes read from disk are the
preregistration documents and the repository's own source.

The numbered banner comments correspond to the twenty-five obligations this
study was required to prove.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import random
import subprocess
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from openalpha_bridge.base_study.spec import (
    BASE_ARTIFACT_ROOT,
    BASE_RUN_ID_PATTERN,
    BASE_SPECIFICATION_NAME,
    BASE_SPECIFICATION_SHA256,
    KRONOS_BASE_SPEC,
    KRONOS_BASE_TOKENIZER_SPEC,
)
from openalpha_bridge.cloud.objectstore import InMemoryObjectStore
from openalpha_bridge.diagnostic.backends import (
    GeneratedPath,
    ResolvedDiagnosticAssets,
    StepSampling,
    TokenPair,
)
from openalpha_bridge.diagnostic.normalization import ClippingReport, NormalizationState
from openalpha_bridge.diagnostic.official_input import OfficialRow, TimeStamp
from openalpha_bridge.diagnostic.spec import (
    OFFICIAL_SOURCE_FILES,
    verify_diagnostic_specifications,
)
from openalpha_bridge.errors import BridgeTransformError
from openalpha_bridge.phase2.identity import verify_locked_hashes
from openalpha_bridge.phase2.invocation import RUN_ID_PATTERN as MINI_RUN_ID_PATTERN
from openalpha_bridge.phase2.kronos import SOURCE_SPEC
from openalpha_bridge.phase2.provider import Candle, MarketSeries, ProviderMode, RetrievalRequest
from openalpha_bridge.zero_shot import aggregation as aggregation_module
from openalpha_bridge.zero_shot import baselines as baselines_module
from openalpha_bridge.zero_shot.aggregation import (
    BOOTSTRAP_ASSETS_PER_CLUSTER,
    BOOTSTRAP_AVAILABLE_BLOCK_STARTS,
    BOOTSTRAP_BASE_RESAMPLING_UNIT,
    BOOTSTRAP_BLOCK_LENGTH,
    BOOTSTRAP_BLOCKS_PER_RESAMPLE,
    BOOTSTRAP_CLUSTER_COUNT,
    BOOTSTRAP_METHOD,
    BOOTSTRAP_OBSERVATION_COUNT,
    BOOTSTRAP_TEMPORAL_RESAMPLING_UNIT,
    MovingBlockBootstrapInterval,
    OriginCluster,
    block_starts,
    extended_metrics,
    paired_origin_moving_block_bootstrap,
    summarize,
    undefined_moving_block_bootstrap,
)
from openalpha_bridge.zero_shot.aggregation import (
    ExtendedForecastMetrics as ExtendedForecastMetricsModel,
)
from openalpha_bridge.zero_shot.artifact import (
    ZERO_SHOT_FAILURE_CODE,
    ZERO_SHOT_OPERATIONAL_FAILURE_CODE,
    ZERO_SHOT_OPERATIONAL_FAILURE_MESSAGE,
    ZERO_SHOT_SUCCESS_CODE,
)
from openalpha_bridge.zero_shot.baselines import BASELINE_IDS, PRIMARY_BASELINE_ID, build_baselines
from openalpha_bridge.zero_shot.decision import (
    AssetSupport,
    ConfigurationEvidence,
    GenerationDirection,
    ZeroShotFinding,
    ZeroShotOutcome,
    decide,
)
from openalpha_bridge.zero_shot.inventory import FOREIGN_ROOTS, inventory_zero_shot_artifacts
from openalpha_bridge.zero_shot.invocation import ZeroShotInvocation
from openalpha_bridge.zero_shot.origins import (
    ORIGIN_INDEX_OFFSETS,
    resolve_origins,
    verify_origin_policy,
)
from openalpha_bridge.zero_shot.runner import verify_cross_asset_alignment
from openalpha_bridge.zero_shot.spec import (
    ASSET_PANEL,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CONTEXT_CANDLES,
    ENSEMBLE_SEEDS,
    HORIZON_CANDLES,
    PRIMARY_METRIC,
    REQUIRED_SESSIONS,
    SAMPLING_CONFIGURATIONS,
    ZERO_SHOT_ARTIFACT_ROOT,
    ZERO_SHOT_EXPERIMENT_ID,
    ZERO_SHOT_RUN_ID_PATTERN,
    ZERO_SHOT_SPECIFICATION_NAME,
    ZERO_SHOT_SPECIFICATION_SHA256,
    verify_pinned_pair,
    verify_zero_shot_specification,
    zero_shot_artifact_key,
)
from openalpha_bridge.zero_shot.worker import (
    ZeroShotWorkerResult,
    run_zero_shot_benchmark_worker,
)
from pydantic import ValidationError

REPO = Path(__file__).resolve().parents[3]
RESEARCH = REPO / "research" / "bridge-v0"
REPORT = REPO / "research" / "reports" / "kronos-structural-validity"
APP = REPO / "cloud" / "modal" / "bridge_phase2_app.py"
PACKAGE = Path(__file__).resolve().parents[1] / "src" / "openalpha_bridge" / "zero_shot"

COMMIT = "a" * 40
RUN_ID = "zsb_0123456789abcdef"
NOW = datetime(2026, 8, 3, tzinfo=UTC)


# ====== doubles ======================================================


def zero_shot_assets(**overrides: Any) -> ResolvedDiagnosticAssets:
    """The base pair identity, which this benchmark reuses."""
    values: dict[str, Any] = {
        "tokenizer_repository": KRONOS_BASE_TOKENIZER_SPEC.repository,
        "tokenizer_revision": KRONOS_BASE_TOKENIZER_SPEC.revision,
        "tokenizer_config_sha256": KRONOS_BASE_TOKENIZER_SPEC.config_sha256,
        "tokenizer_weights_sha256": KRONOS_BASE_TOKENIZER_SPEC.weights_sha256,
        "model_repository": KRONOS_BASE_SPEC.repository,
        "model_revision": KRONOS_BASE_SPEC.revision,
        "model_config_sha256": KRONOS_BASE_SPEC.config_sha256,
        "model_weights_sha256": KRONOS_BASE_SPEC.weights_sha256,
        "source_revision": SOURCE_SPEC.revision,
        "source_as_committed_sha256": {
            f.relative_path: f.as_committed_sha256 for f in OFFICIAL_SOURCE_FILES
        },
        "source_crlf_normalized_sha256": {
            f.relative_path: f.sealed_sha256_crlf_normalized for f in OFFICIAL_SOURCE_FILES
        },
        "parameter_sha256": "c" * 64,
        "trainable_parameter_count": 0,
        "total_parameter_count": 106_268_634,
    }
    values.update(overrides)
    return ResolvedDiagnosticAssets(**values)


def _sessions(count: int, *, start: date = date(2025, 1, 2)) -> list[date]:
    """Weekday sessions. A stand-in calendar, identical for every asset."""
    out: list[date] = []
    day = start
    while len(out) < count:
        if day.weekday() < 5:
            out.append(day)
        day += timedelta(days=1)
    return out


def _candles(symbol: str, count: int) -> tuple[Candle, ...]:
    """A deterministic, well-formed price path. Distinct per symbol."""
    offset = sum(symbol.encode()) % 17
    rows: list[Candle] = []
    level = 100.0 + offset
    for index, session in enumerate(_sessions(count)):
        level *= math.exp(0.0004 * math.sin((index + offset) / 6.0))
        close = level
        open_ = level * 0.999
        high = level * 1.004
        low = level * 0.996
        volume = 1_000_000.0 + index
        rows.append(
            Candle(
                session=session,
                open=open_,
                high=high,
                low=low,
                close=close,
                volume=volume,
                amount=volume * close,
            )
        )
    return tuple(rows)


class _PanelProvider:
    """One deterministic series per asset. Records every request it saw."""

    name = "deterministic_fake"
    mode = ProviderMode.FAKE

    def __init__(self, *, sessions: int = REQUIRED_SESSIONS + 20) -> None:
        self._sessions = sessions
        self.requests: list[str] = []

    def fetch(self, request: RetrievalRequest) -> MarketSeries:
        self.requests.append(f"{request.symbol}:{request.start}:{request.end}")
        return MarketSeries(
            symbol=request.symbol,
            interval="1d",
            provider=self.name,
            provider_mode=self.mode,
            client_version="fake-1",
            retrieval_timestamp=None,
            candles=_candles(request.symbol, self._sessions),
        )


class _Factory:
    def __init__(self, *, provider: _PanelProvider | None = None) -> None:
        self.calls = 0
        self.last = provider

    def __call__(self) -> _PanelProvider:
        self.calls += 1
        if self.last is None:
            self.last = _PanelProvider()
        return self.last


class _Model:
    """A frozen model that predicts from the context only.

    It is handed the context and never the target, so a test that asserts the
    absence of leakage is asserting something structural rather than a promise.
    """

    def __init__(self, *, per_step_return: float = 0.0005) -> None:
        self.per_step_return = per_step_return
        self.generate_calls = 0
        self.seeds_used: list[int] = []
        self.temperatures_used: list[float] = []
        self.states_seen: list[str] = []
        self.context_lengths: list[int] = []
        self.step_counts: list[int] = []
        self.rows_seen: list[tuple[OfficialRow, ...]] = []

    def generate(
        self,
        context: tuple[OfficialRow, ...],
        *,
        context_stamps: tuple[TimeStamp, ...],
        target_stamps: tuple[TimeStamp, ...],
        target_sessions: tuple,
        state: NormalizationState,
        steps: int,
        seed: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> GeneratedPath:
        self.generate_calls += 1
        self.seeds_used.append(seed)
        self.temperatures_used.append(temperature)
        self.states_seen.append(state.state_sha256)
        self.context_lengths.append(len(context))
        self.step_counts.append(steps)
        self.rows_seen.append(context)

        if len(context_stamps) != len(context):
            raise AssertionError("one stamp per context row is required")
        if len(target_stamps) != steps or len(target_sessions) != steps:
            raise AssertionError("one stamp and one session per predicted step is required")

        anchor = context[-1].close
        tokens: list[TokenPair] = []
        sampling: list[StepSampling] = []
        rows: list[OfficialRow] = []
        level = anchor
        for step in range(steps):
            digest = hashlib.sha256(f"{seed}:{step}".encode()).digest()
            wobble = (int.from_bytes(digest[:4], "big") / 2**32 - 0.5) * 0.004
            level *= math.exp(self.per_step_return + wobble)
            rows.append(
                OfficialRow(
                    session=target_sessions[step],
                    open=level * 0.999,
                    high=level * 1.004,
                    low=level * 0.996,
                    close=level,
                    volume=1_000_000.0,
                    amount=1_000_000.0 * level,
                )
            )
            tokens.append(
                TokenPair(
                    coarse=int.from_bytes(digest[4:6], "big") % 1024,
                    fine=int.from_bytes(digest[6:8], "big") % 1024,
                )
            )
            sampling.append(
                StepSampling(
                    coarse_log_probability=-0.30 - (step % 5) * 0.01,
                    fine_conditional_log_probability=-0.20 - (step % 3) * 0.01,
                )
            )
        return GeneratedPath(
            seed=seed,
            tokens=tuple(tokens),
            sampling=tuple(sampling),
            total_path_sampling_log_probability=math.fsum(
                s.pair_log_probability for s in sampling
            ),
            raw_decoded_suffix=tuple(rows),
        )


class _Codec:
    """Present so a caller cannot supply half a runtime. Never invoked."""

    def __init__(self) -> None:
        self.encode_calls = 0
        self.decode_calls = 0

    def encode(self, rows: Any, *, state: Any) -> Any:  # pragma: no cover - never called
        self.encode_calls += 1
        raise AssertionError("the benchmark decodes only through the official generate path")

    def decode(self, tokens: Any, *, state: Any, sessions: Any) -> Any:  # pragma: no cover
        self.decode_calls += 1
        raise AssertionError("the benchmark decodes only through the official generate path")


class _Runtime:
    def __init__(self, model: _Model, assets: ResolvedDiagnosticAssets) -> None:
        self.codec = _Codec()
        self.model = model
        self.assets = assets
        self.parameter_digest = lambda: "c" * 64


class _Resolver:
    """Stands in for official_runtime. Counts entries and exits."""

    def __init__(self, *, model: _Model | None = None, assets: Any = None) -> None:
        self.model = model or _Model()
        self.assets = assets if assets is not None else zero_shot_assets()
        self.calls = 0
        self.entered = 0
        self.exited = 0

    def __call__(self) -> _Resolver:
        self.calls += 1
        return self

    def __enter__(self) -> _Runtime:
        self.entered += 1
        return _Runtime(self.model, self.assets)

    def __exit__(self, *exc: object) -> bool:
        self.exited += 1
        return False


def _run(
    *,
    store: InMemoryObjectStore | None = None,
    resolver: _Resolver | None = None,
    factory: _Factory | None = None,
    run_id: str = RUN_ID,
    logger: Any = None,
) -> tuple[ZeroShotWorkerResult, InMemoryObjectStore, _Resolver, _Factory]:
    store = store if store is not None else InMemoryObjectStore()
    resolver = resolver if resolver is not None else _Resolver()
    factory = factory if factory is not None else _Factory()
    result = run_zero_shot_benchmark_worker(
        store=store,
        resolve_runtime=resolver,
        provider_factory=factory,
        research_root=RESEARCH,
        source_commit=COMMIT,
        deployed_commit=COMMIT,
        run_id=run_id,
        now=NOW,
        logger=logger,
    )
    return result, store, resolver, factory


def _referenced_names(root: Path) -> set[str]:
    """Every identifier the package mentions, by AST rather than by substring.

    A docstring saying "no projection" would false-positive a text search, so
    the check reads the syntax tree instead.
    """
    names: set[str] = set()
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.alias):
                names.add(node.name.rsplit(".", maxsplit=1)[-1])
                if node.asname:
                    names.add(node.asname)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.update(node.module.split("."))
    return names


ZERO_SHOT_NAMES = _referenced_names(PACKAGE)


# ====== 1: historical specifications remain byte-identical ===========


def test_the_historical_mini_and_base_specifications_are_byte_identical() -> None:
    base = RESEARCH / BASE_SPECIFICATION_NAME
    assert hashlib.sha256(base.read_bytes()).hexdigest() == BASE_SPECIFICATION_SHA256

    mini = verify_diagnostic_specifications(RESEARCH)
    assert mini == {
        "phase2-frozen-inference-diagnostic.yaml": (
            "c909e156a61ac5b5323de9e794aad20521839c0f43f491fb356c1968f730101b"
        ),
        "phase2-frozen-inference-diagnostic-v2.yaml": (
            "c39fff4541afcc948cd80fcc545398312897efe723deca26554143989e4a175e"
        ),
        "phase2-frozen-inference-diagnostic-v3.yaml": (
            "f10076b6676a72552b1c9c96720d0087c009fc939e4667509bfcfccf7929bcb6"
        ),
        "phase2-frozen-inference-diagnostic-v4.yaml": (
            "bd407722adfc3ebf92eb187828d42c2c9cfa57b2fdc0406121f27d39a5c44977"
        ),
    }

    sealed = verify_locked_hashes(RESEARCH)
    assert sealed == {
        "experiment.yaml": (
            "d52a9be733f4ec331e081346e64ab7415ac0f9510e494733746f975f114f47b2"
        ),
        "phase2-preregistration-amendment.yaml": (
            "4c60846e8cb791d922c29a5e704fe3a473740171dd2bbb7b3390796ceeeef3c8"
        ),
        "phase2-amendment-2-context-prefix.yaml": (
            "66f3c8171c2805bccf4b924125bd206edad27c957dfff40a0abf3864fbab10c1"
        ),
        "phase2-amendment-3-scale-features.yaml": (
            "d9484020a22df42c93edc19941374c628ea891f08c8bee8151b82264c00bb12b"
        ),
    }


def test_the_new_specification_verifies_and_is_not_a_historical_one() -> None:
    verified = verify_zero_shot_specification(RESEARCH)
    assert verified == {ZERO_SHOT_SPECIFICATION_NAME: ZERO_SHOT_SPECIFICATION_SHA256}

    path = RESEARCH / ZERO_SHOT_SPECIFICATION_NAME
    assert hashlib.sha256(path.read_bytes()).hexdigest() == ZERO_SHOT_SPECIFICATION_SHA256
    assert b"\r\n" not in path.read_bytes()

    historical = set(verify_diagnostic_specifications(RESEARCH)) | {BASE_SPECIFICATION_NAME}
    assert ZERO_SHOT_SPECIFICATION_NAME not in historical
    assert ZERO_SHOT_SPECIFICATION_SHA256 != BASE_SPECIFICATION_SHA256

    # The document states the corrected design explicitly, not just by hash.
    document = path.read_text(encoding="utf-8")
    for declaration in (
        "method: paired_percentile_moving_block_bootstrap_over_origin_clusters",
        "temporal_resampling_unit: four_consecutive_origin_clusters",
        "block_length: 4",
        "available_block_starts: 22",
        "blocks_drawn_per_resample: 7",
        "origins_resampled_independently: false",
        "wraparound: false",
        "computation: ceil(40 / 12) = 4",
        "selected_before_execution: true",
        "estimated_or_tuned_from_results: false",
        "interval_used: moving_block_bootstrap_only",
        "cross_asset_dependence_preserved: true",
        "temporal_dependence_preserved: true",
    ):
        assert declaration in document, declaration


# ====== 2: both completed artifact identities remain recorded ========


def test_the_completed_artifact_identities_are_recorded_correctly() -> None:
    manifest = json.loads((REPORT / "artifact-manifest.json").read_text(encoding="utf-8"))
    by_run = {entry["run_id"]: entry for entry in manifest["artifacts"]}

    mini = by_run["canary_0a92fde788bd685c"]
    assert mini["artifact_key"] == (
        "openalpha-compatibility/bridge-phase2/runs/canary_0a92fde788bd685c/"
        "frozen-inference-diagnostic/frozen_inference_diagnostic_terminal.json"
    )
    assert mini["artifact_sha256"] == (
        "8e3d8a333c11c1939c5d182db73012fc3e0fc01da42368eb23f71661aadab18c"
    )
    assert mini["primary_conclusion"] == "ROUNDTRIP_MATERIAL_INVALIDITY"

    base = by_run["base_03b08cbc706193d6"]
    assert base["artifact_key"] == (
        "openalpha-compatibility/kronos-base-diagnostic/runs/base_03b08cbc706193d6/"
        "kronos_base_diagnostic_terminal.json"
    )
    assert base["artifact_sha256"] == (
        "84ec1b197d4bf7a3b8f35a38d0feb8d5be086b61de23df0924d83ec273c731e6"
    )
    assert base["source"]["source_commit"] == "0e6193baabd747dc0b2610a0ab80c3320e32fdd0"
    assert base["operative_specification"]["sha256"] == BASE_SPECIFICATION_SHA256

    summary = json.loads((REPORT / "results-summary.json").read_text(encoding="utf-8"))
    assert summary["studies"]["kronos_mini"]["roundtrip_full_invalid_fraction"] == 0.24609375
    assert summary["studies"]["kronos_mini"]["method_b_close_return_mae"] == 0.007306554165097093
    assert summary["studies"]["kronos_mini"]["invalidity_error_spearman"] == 0.04015798783221637
    assert summary["studies"]["kronos_base"]["roundtrip_full_invalid_fraction"] == 0.123046875
    assert summary["studies"]["kronos_base"]["method_b_close_return_mae"] == 0.00920398038369002
    assert summary["studies"]["kronos_base"]["invalidity_error_spearman"] == 0.43092362328095035
    assert summary["studies"]["kronos_base"]["best_persistence_skill"] == -0.4481214885679963
    for study in summary["studies"].values():
        assert study["projection_primary_improvement"] == 0.0
        assert study["valid_rollout_count"] == 0
        assert study["persistence_close_return_mae"] == 0.0033651655739541575


def test_the_report_states_the_boundary_and_the_non_claims() -> None:
    report = (REPORT / "report.md").read_text(encoding="utf-8")
    assert "deterministic restoration of complete structural validity produced no" in report
    for claim in (
        "universal Kronos failure",
        "the paper's exact evaluation protocol",
        "task-specific fine-tuning",
        "absence of useful internal representations",
    ):
        assert claim in report
    for name in ("README.md", "report.md", "results-summary.json", "artifact-manifest.json",
                 "reproduction.md", "limitations.md"):
        assert (REPORT / name).is_file()


# ====== 3-4: namespaces and run identifiers cannot collide ===========


def test_the_benchmark_namespace_cannot_collide_with_either_completed_study() -> None:
    mini_root = "openalpha-compatibility/bridge-phase2"
    assert not ZERO_SHOT_ARTIFACT_ROOT.startswith(mini_root)
    assert not mini_root.startswith(ZERO_SHOT_ARTIFACT_ROOT)
    assert not ZERO_SHOT_ARTIFACT_ROOT.startswith(BASE_ARTIFACT_ROOT)
    assert not BASE_ARTIFACT_ROOT.startswith(ZERO_SHOT_ARTIFACT_ROOT)

    key = zero_shot_artifact_key(RUN_ID)
    assert key.startswith(f"{ZERO_SHOT_ARTIFACT_ROOT}/runs/{RUN_ID}/")
    assert mini_root not in key.removeprefix("openalpha-compatibility/")
    assert "kronos-base-diagnostic" not in key


@pytest.mark.parametrize(
    "run_id",
    ["canary_0a92fde788bd685c", "base_03b08cbc706193d6", "base_deadbeef", "canary_deadbeef"],
)
def test_a_foreign_run_identifier_is_refused_by_the_benchmark(run_id: str) -> None:
    with pytest.raises(BridgeTransformError) as excinfo:
        ZeroShotInvocation.validate_all(
            run_id=run_id, source_commit=COMMIT, deployed_commit=COMMIT
        )
    assert excinfo.value.failures[0].code in {
        "MINI_RUN_ID_REFUSED_BY_ZERO_SHOT_BENCHMARK",
        "BASE_RUN_ID_REFUSED_BY_ZERO_SHOT_BENCHMARK",
    }


def test_the_three_run_identifier_spaces_are_pairwise_disjoint() -> None:
    samples = ["canary_0a92fde788bd685c", "base_03b08cbc706193d6", "zsb_0123456789abcdef"]
    patterns = (MINI_RUN_ID_PATTERN, BASE_RUN_ID_PATTERN, ZERO_SHOT_RUN_ID_PATTERN)
    for sample in samples:
        matches = [pattern for pattern in patterns if pattern.fullmatch(sample)]
        assert len(matches) == 1, sample


@pytest.mark.parametrize(
    "run_id",
    ["zsb_", "zsb_0123", "zsb_ABCDEF12", "zsb_../escape", "zsb_0123456789abcdef0", " zsb_deadbeef"],
)
def test_a_malformed_benchmark_run_identifier_is_refused(run_id: str) -> None:
    if ZERO_SHOT_RUN_ID_PATTERN.fullmatch(run_id):
        pytest.skip("this sample is well formed")
    with pytest.raises(BridgeTransformError) as excinfo:
        ZeroShotInvocation.validate_all(
            run_id=run_id, source_commit=COMMIT, deployed_commit=COMMIT
        )
    assert excinfo.value.failures[0].code == "INVALID_ZERO_SHOT_RUN_ID"


# ====== 5-6: no structural projection, no validity filtering =========


def test_the_benchmark_cannot_invoke_structural_projection() -> None:
    forbidden = {
        "project_path",
        "ProjectionOutcome",
        "PROJECTION_METHOD",
        "run_method_c",
        "MethodCResult",
        "run_method_a",
        "run_method_b",
        "run_method_d",
    }
    assert not (ZERO_SHOT_NAMES & forbidden)

    text = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(PACKAGE.rglob("*.py"))
    )
    assert "TERMINAL_PROJECTION" not in text


def test_the_benchmark_cannot_filter_or_select_paths_by_structural_validity() -> None:
    result, _, resolver, _ = _run()
    payload = result.payload

    seeds = tuple(ENSEMBLE_SEEDS)
    for origin in payload["origins"]:
        for configuration in origin["configurations"]:
            paths = configuration["paths"]
            # Every seed present, in the declared order, whatever its validity.
            assert tuple(path["seed"] for path in paths) == seeds
            structural = configuration["structural"]
            assert structural["paths_filtered_by_validity"] == 0
            assert structural["paths_repaired"] == 0
            assert structural["used_in_any_decision_rule"] is False

    assert payload["forecasts_repaired"] is False
    assert payload["paths_filtered_by_structural_validity"] is False
    assert payload["structural_validity_used_in_decision"] is False
    assert payload["decision"]["structural_validity_used_in_any_rule"] is False
    assert resolver.entered == 1


def test_no_decision_rule_observes_a_structural_quantity() -> None:
    result, _, _, _ = _run()
    for evaluation in result.payload["decision"]["evaluations"]:
        observed = " ".join(evaluation["observed"].keys()).lower()
        for word in ("invalid", "valid", "structural", "candle", "projection", "repair"):
            assert word not in observed


# ====== 7-9: assets cannot be substituted, nothing loads locally =====


@pytest.mark.parametrize(
    ("override", "code"),
    [
        ({"model_repository": "NeoQuasar/Kronos-mini"}, "ZERO_SHOT_WRONG_MODEL_REPOSITORY"),
        (
            {"tokenizer_repository": "NeoQuasar/Kronos-Tokenizer-2k"},
            "ZERO_SHOT_WRONG_TOKENIZER_REPOSITORY",
        ),
        ({"model_revision": "f" * 40}, "ZERO_SHOT_WRONG_MODEL_REVISION"),
        ({"tokenizer_revision": "f" * 40}, "ZERO_SHOT_WRONG_TOKENIZER_REVISION"),
        ({"trainable_parameter_count": 5}, "ZERO_SHOT_PARAMETERS_NOT_FROZEN"),
    ],
)
def test_a_substituted_model_or_tokenizer_is_refused(override: dict, code: str) -> None:
    resolver = _Resolver(assets=zero_shot_assets(**override))
    result, store, _, _ = _run(resolver=resolver)
    assert result.outcome == ZERO_SHOT_FAILURE_CODE
    assert result.payload["failure_code"] == code
    assert len(store.list_keys("")) == 1


def test_the_pinned_pair_is_checked_against_literals_written_in_this_study() -> None:
    assert verify_pinned_pair() == {
        "model_repository": "NeoQuasar/Kronos-base",
        "model_revision": "2b554741eca47781b64468546e77fef3e85130e6",
        "tokenizer_repository": "NeoQuasar/Kronos-Tokenizer-base",
        "tokenizer_revision": "0e0117387f39004a9016484a186a908917e22426",
    }


def test_no_benchmark_module_can_download_an_asset() -> None:
    forbidden = {"snapshot_download", "hf_hub_download", "from_pretrained", "urlretrieve", "wget"}
    assert not (ZERO_SHOT_NAMES & forbidden)
    assert "huggingface_hub" not in ZERO_SHOT_NAMES


def test_every_snapshot_download_still_lives_in_modal_execution_code() -> None:
    """The benchmark reuses the base downloader rather than adding one.

    The volume already holds the two pinned repositories, so a second download
    function would mean a second copy of 409 MB and a second thing to keep
    honest.
    """
    source = APP.read_text(encoding="utf-8")
    tree = ast.parse(source)
    downloading: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for inner in ast.walk(node):
                if isinstance(inner, ast.Name) and inner.id == "snapshot_download":
                    downloading.add(node.name)

    # The pre-existing downloaders, plus the nested resolver inside the mini
    # diagnostic that calls the hub directly. This benchmark adds none of them.
    assert downloading == {
        "frozen_inference_diagnostic",
        "verify_frozen_inference_runtime",
        "_download_base_pair",
        "resolve_runtime",
    }
    for name in (
        "verify_zero_shot_benchmark_deployment",
        "verify_zero_shot_benchmark_runtime",
        "run_zero_shot_benchmark",
        "inventory_zero_shot_artifacts",
    ):
        assert name not in downloading

    # Every weight the benchmark reads comes through the base pair resolver,
    # against the volume that already holds them.
    benchmark = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "run_zero_shot_benchmark"
    )
    body = ast.get_source_segment(source, benchmark) or ""
    assert "_download_base_pair(cache_root)" in body
    assert "snapshot_download(" not in body


def test_no_model_weight_is_present_in_the_repository() -> None:
    suffixes = {".safetensors", ".bin", ".ckpt", ".pt", ".pth", ".onnx", ".gguf"}
    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.split("\0")
    offenders = [name for name in tracked if name and Path(name).suffix in suffixes]
    assert offenders == []

    for directory in ("packages", "cloud", "research"):
        for path in (REPO / directory).rglob("*"):
            assert path.suffix not in suffixes, path


# ====== 10-11: normalization is context-only, targets are unreachable =


def test_normalization_is_fitted_on_the_context_only() -> None:
    result, _, resolver, _ = _run()
    for origin in result.payload["origins"]:
        assert origin["normalization_fitted_candle_count"] == CONTEXT_CANDLES
        assert origin["context_clipping"]["rows"] == CONTEXT_CANDLES

    # Every generate call saw exactly forty context rows and no more.
    assert set(resolver.model.context_lengths) == {CONTEXT_CANDLES}
    # One distinct state per origin, and the model was told each one.
    states = {origin["normalization_state_sha256"] for origin in result.payload["origins"]}
    assert states.issubset(set(resolver.model.states_seen))


def test_the_model_never_receives_a_target_row() -> None:
    result, _, resolver, _ = _run()
    targets: set[tuple[str, float]] = set()
    for origin in result.payload["origins"]:
        for close in origin["target_closes"]:
            targets.add((origin["asset"], round(close, 9)))

    seen: set[float] = set()
    for context in resolver.model.rows_seen:
        assert len(context) == CONTEXT_CANDLES
        for row in context:
            seen.add(round(row.close, 9))

    # A context row of a later origin can legitimately equal an earlier target
    # row, because the windows are contiguous in time. What must never happen is
    # the model seeing a row from the target of the window it is predicting.
    for context, steps in zip(resolver.model.rows_seen, resolver.model.step_counts, strict=True):
        assert steps == HORIZON_CANDLES
        assert len(context) == CONTEXT_CANDLES
    assert seen  # the fixture really did feed rows


def test_a_wrongly_sized_context_is_refused() -> None:
    from openalpha_bridge.zero_shot.runner import _window_series

    rows = tuple(
        OfficialRow(
            session=session,
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=1.0,
            amount=100.0,
        )
        for session in _sessions(51)
    )
    with pytest.raises(BridgeTransformError) as excinfo:
        _window_series("SPY", rows)
    assert excinfo.value.failures[0].code == "OFFICIAL_ROW_COUNT_MISMATCH"


# ====== 12-13: origins are deterministic and uniformly applied =======


def test_the_origins_are_deterministic_and_preregistered() -> None:
    assert ORIGIN_INDEX_OFFSETS == (
        40, 52, 64, 76, 88, 100, 112, 124, 136, 148, 160, 172, 184, 196, 208,
        220, 232, 244, 256, 268, 280, 292, 304, 316, 328,
    )
    assert len(ORIGIN_INDEX_OFFSETS) == 25
    assert verify_origin_policy()["required_sessions"] == REQUIRED_SESSIONS == 340

    document = (RESEARCH / ZERO_SHOT_SPECIFICATION_NAME).read_text(encoding="utf-8")
    for offset in ORIGIN_INDEX_OFFSETS:
        assert f"\n    - {offset}\n" in document


def test_every_asset_receives_the_same_origin_selection_policy() -> None:
    sessions = tuple(_sessions(REQUIRED_SESSIONS))
    per_asset = {
        asset: resolve_origins(asset=asset, sessions=sessions) for asset in ASSET_PANEL
    }
    reference = [
        (s.ordinal, s.origin_index, s.context_start_index, s.target_end_exclusive)
        for s in per_asset[ASSET_PANEL[0]]
    ]
    for asset in ASSET_PANEL[1:]:
        assert [
            (s.ordinal, s.origin_index, s.context_start_index, s.target_end_exclusive)
            for s in per_asset[asset]
        ] == reference


def test_target_windows_are_disjoint_and_windows_are_complete() -> None:
    sessions = tuple(_sessions(REQUIRED_SESSIONS))
    selections = resolve_origins(asset="SPY", sessions=sessions)
    covered: list[int] = []
    for selection in selections:
        assert selection.context_length == CONTEXT_CANDLES
        assert selection.target_length == HORIZON_CANDLES
        covered.extend(range(selection.target_start_index, selection.target_end_exclusive))
    assert len(covered) == len(set(covered)) == 25 * HORIZON_CANDLES


def test_a_short_session_sequence_fails_closed_rather_than_reaching_backwards() -> None:
    resolver = _Resolver()
    factory = _Factory(provider=_PanelProvider(sessions=REQUIRED_SESSIONS - 1))
    result, store, _, _ = _run(resolver=resolver, factory=factory)
    assert result.outcome == ZERO_SHOT_FAILURE_CODE
    assert result.payload["failure_code"] == "ZERO_SHOT_INSUFFICIENT_SESSIONS"
    assert len(store.list_keys("")) == 1


# ====== 14-16: configuration, horizon and context are exact ==========


def test_the_temperature_configurations_are_exactly_zero_point_six_and_one() -> None:
    assert [c.label for c in SAMPLING_CONFIGURATIONS] == ["A", "B"]
    assert [c.temperature for c in SAMPLING_CONFIGURATIONS] == [0.6, 1.0]
    assert len(SAMPLING_CONFIGURATIONS) == 2
    for configuration in SAMPLING_CONFIGURATIONS:
        assert configuration.top_k == 0
        assert configuration.top_p == 0.9
        assert configuration.sample_count == 1


def test_a_third_temperature_is_refused() -> None:
    from openalpha_bridge.zero_shot.runner import _require_configurations
    from openalpha_bridge.zero_shot.spec import SamplingConfiguration

    extra = (*SAMPLING_CONFIGURATIONS, SamplingConfiguration(label="A", temperature=0.8))
    with pytest.raises(BridgeTransformError) as excinfo:
        _require_configurations(extra)
    assert excinfo.value.failures[0].code == "ZERO_SHOT_CONFIGURATIONS_NOT_PREREGISTERED"


def test_the_seed_set_is_preregistered_and_shared_by_every_cell() -> None:
    from openalpha_bridge.zero_shot.runner import _require_seeds

    assert ENSEMBLE_SEEDS == (
        20250102, 20250103, 20250104, 20250105, 20250106, 20250107, 20250108, 20250109,
    )
    assert len(ENSEMBLE_SEEDS) == 8
    with pytest.raises(BridgeTransformError) as excinfo:
        _require_seeds((1, 2, 3))
    assert excinfo.value.failures[0].code == "ZERO_SHOT_SEEDS_NOT_PREREGISTERED"


def test_the_context_is_forty_and_the_horizon_is_twelve() -> None:
    assert CONTEXT_CANDLES == 40
    assert HORIZON_CANDLES == 12
    result, _, resolver, _ = _run()
    assert set(resolver.model.step_counts) == {HORIZON_CANDLES}
    assert set(resolver.model.context_lengths) == {CONTEXT_CANDLES}
    assert result.payload["context_candles"] == 40
    assert result.payload["horizon_candles"] == 12
    assert result.payload["context_budget"]["truncation_occurs"] is False
    assert result.payload["context_budget"]["sum"] == 52
    for origin in result.payload["origins"]:
        assert len(origin["target_closes"]) == HORIZON_CANDLES
        for configuration in origin["configurations"]:
            assert len(configuration["ensemble_close_forecast"]) == HORIZON_CANDLES
            for path in configuration["paths"]:
                assert len(path["coarse_token_ids"]) == HORIZON_CANDLES


def test_every_asset_origin_and_seed_is_generated_exactly_once() -> None:
    result, _, resolver, _ = _run()
    expected = len(ASSET_PANEL) * len(ORIGIN_INDEX_OFFSETS) * len(SAMPLING_CONFIGURATIONS) * len(
        ENSEMBLE_SEEDS
    )
    assert resolver.model.generate_calls == expected == 1600
    assert result.payload["total_generations"] == expected
    assert result.payload["total_asset_origins"] == 100
    assert sorted(set(resolver.model.temperatures_used)) == [0.6, 1.0]


# ====== 17: baselines cannot access the target =======================


def test_the_baselines_cannot_access_target_values() -> None:
    context = tuple(
        OfficialRow(
            session=session,
            open=100.0 + index,
            high=101.0 + index,
            low=99.0 + index,
            close=100.0 + index,
            volume=1.0,
            amount=100.0,
        )
        for index, session in enumerate(_sessions(CONTEXT_CANDLES))
    )
    target_sessions = tuple(_sessions(HORIZON_CANDLES, start=date(2026, 1, 5)))

    built = build_baselines(context, target_sessions=target_sessions)
    assert set(built.as_mapping()) == set(BASELINE_IDS)

    # The builder takes only the context and the dates. There is no parameter
    # through which a target row could arrive.
    import inspect

    for name in BASELINE_IDS:
        signature = inspect.signature(getattr(baselines_module, name))
        assert list(signature.parameters) == ["context", "target_sessions"]

    persistence = built.zero_return_persistence
    assert {row.close for row in persistence} == {context[-1].close}
    # Zero-return persistence and last-close level are the same path.
    assert built.last_close_level == persistence
    # The drifting baselines actually move, so they are not silent duplicates.
    assert built.context_mean_return != persistence
    assert built.context_drift != persistence


def test_persistence_scores_exactly_zero_skill_against_itself() -> None:
    context = tuple(
        OfficialRow(
            session=session,
            open=100.0,
            high=100.0,
            low=100.0,
            close=100.0,
            volume=1.0,
            amount=100.0,
        )
        for session in _sessions(CONTEXT_CANDLES)
    )
    target = tuple(
        OfficialRow(
            session=session,
            open=101.0,
            high=101.0,
            low=101.0,
            close=101.0,
            volume=1.0,
            amount=101.0,
        )
        for session in _sessions(HORIZON_CANDLES, start=date(2026, 1, 5))
    )
    built = build_baselines(context, target_sessions=tuple(row.session for row in target))
    metrics = extended_metrics(built.zero_return_persistence, target, anchor_close=100.0)
    assert metrics.defined is True
    assert metrics.scored_steps == HORIZON_CANDLES
    assert metrics.close_return_mae is not None and metrics.close_return_mae > 0.0
    assert metrics.close_return_rmse is not None
    assert metrics.median_absolute_return_error is not None


def test_all_four_baselines_are_scored_at_every_origin() -> None:
    result, _, _, _ = _run()
    for origin in result.payload["origins"]:
        recorded = [entry["baseline_id"] for entry in origin["baselines"]]
        assert recorded == list(BASELINE_IDS)
        primary = [entry for entry in origin["baselines"] if entry["is_primary"]]
        assert len(primary) == 1
        assert primary[0]["baseline_id"] == PRIMARY_BASELINE_ID
        for entry in origin["baselines"]:
            assert entry["accessed_target_data"] is False


# ====== 18-19: duplicate invocation, immutable terminal object =======


def test_a_duplicate_invocation_loads_nothing_and_fetches_nothing() -> None:
    _, store, first, first_factory = _run()
    assert (first.calls, first.entered, first_factory.calls) == (1, 1, 1)

    second = _Resolver()
    second_factory = _Factory()
    result, _, _, _ = _run(store=store, resolver=second, factory=second_factory)

    assert result.already_existed is True
    assert (second.calls, second.entered) == (0, 0)
    assert second_factory.calls == 0
    assert second.model.generate_calls == 0
    assert len(store.list_keys("")) == 1


def test_one_immutable_terminal_object_is_written_under_this_study_alone() -> None:
    result, store, _, factory = _run()
    assert result.outcome == ZERO_SHOT_SUCCESS_CODE
    assert result.experiment_id == ZERO_SHOT_EXPERIMENT_ID
    assert result.study_type == "zero_shot_forecast_benchmark"
    assert result.artifact_key == zero_shot_artifact_key(RUN_ID)
    assert store.list_keys("") == (zero_shot_artifact_key(RUN_ID),)
    assert result.payload["schema_version"] == (
        "openalpha.bridge.zero_shot.kronos_zero_shot_benchmark.v1"
    )
    assert result.payload["specification_sha256"] == ZERO_SHOT_SPECIFICATION_SHA256
    assert result.payload["provider_request_count"] == len(ASSET_PANEL) == 4
    assert factory.last is not None
    assert [entry.split(":")[0] for entry in factory.last.requests] == list(ASSET_PANEL)

    stored = store.get(zero_shot_artifact_key(RUN_ID))
    assert hashlib.sha256(stored.body).hexdigest() == stored.metadata.content_sha256
    assert result.content_sha256 == stored.metadata.content_sha256

    # A second write to the same key is refused by the store, not overwritten.
    with pytest.raises(BridgeTransformError) as excinfo:
        store.put_immutable(zero_shot_artifact_key(RUN_ID), b"{}", stored.metadata)
    assert excinfo.value.failures[0].code == "OBJECT_ALREADY_EXISTS"


def test_the_inventory_reports_this_study_only() -> None:
    _, store, _, _ = _run()
    inventory = inventory_zero_shot_artifacts(store, deployed_commit=COMMIT, inspected_at=NOW)
    assert inventory.run_count == 1
    assert inventory.records[0].run_id == RUN_ID
    assert inventory.records[0].run_id_is_well_formed is True
    assert inventory.records[0].is_terminal_object is True
    assert inventory.read_only is True
    assert inventory.deletion_supported is False
    assert inventory.foreign_roots_inspected is False
    for root in FOREIGN_ROOTS:
        assert not inventory.artifact_root.startswith(root)


# ====== 20: failure publication preserves sanitized evidence =========


class _ExplodingResolver(_Resolver):
    def __enter__(self) -> _Runtime:
        self.entered += 1
        raise RuntimeError("provider token AKIA-SECRET-VALUE leaked into the message")


def test_an_operational_failure_is_preserved_without_exception_text(caplog) -> None:
    resolver = _ExplodingResolver()
    result, store, _, _ = _run(resolver=resolver)

    assert result.outcome == ZERO_SHOT_OPERATIONAL_FAILURE_CODE
    assert result.payload["failure_code"] == "ZERO_SHOT_OPERATIONAL_FAILURE"
    assert result.payload["exception_class"] == "RuntimeError"
    assert result.payload["message"] == ZERO_SHOT_OPERATIONAL_FAILURE_MESSAGE
    assert result.payload["failure_stage"] == "kronos_zero_shot_benchmark"
    assert result.payload["schema_version"] == (
        "openalpha.bridge.zero_shot.kronos_zero_shot_failure.v1"
    )

    body = store.get(zero_shot_artifact_key(RUN_ID)).body.decode("utf-8")
    assert "AKIA-SECRET-VALUE" not in body
    assert "AKIA-SECRET-VALUE" not in caplog.text
    assert len(store.list_keys("")) == 1


def test_a_typed_failure_records_its_code_and_stays_inside_this_namespace() -> None:
    resolver = _Resolver(assets=zero_shot_assets(model_repository="NeoQuasar/Kronos-mini"))
    result, store, _, _ = _run(resolver=resolver)
    assert result.outcome == ZERO_SHOT_FAILURE_CODE
    assert result.payload["experiment_id"] == ZERO_SHOT_EXPERIMENT_ID
    assert result.payload["specification_name"] == ZERO_SHOT_SPECIFICATION_NAME
    assert store.list_keys("") == (zero_shot_artifact_key(RUN_ID),)


# ====== 21-23: authorizations, training, partitions ==================


def test_no_benchmark_result_authorizes_anything() -> None:
    result, _, _, _ = _run()
    fields = (
        "authorizes_training",
        "authorizes_fine_tuning",
        "authorizes_representation_probe",
        "authorizes_stage_b",
        "authorizes_stage_c",
        "authorizes_test_opening",
        "authorizes_production_inference",
        "authorizes_trading_claims",
        "scientific_result_available",
    )
    for field in fields:
        assert getattr(result, field) is False
        assert result.payload[field] is False

    failure_resolver = _ExplodingResolver()
    failure, _, _, _ = _run(resolver=failure_resolver)
    for field in fields:
        assert failure.payload[field] is False


def test_training_and_optimizer_code_are_unreachable() -> None:
    forbidden = {
        "CloudRunner",
        "TorchTrainingBackend",
        "Phase2Config",
        "optimizer",
        "Optimizer",
        "AdamW",
        "Adam",
        "SGD",
        "lr_scheduler",
        "backward",
        "optimizer_step",
        "zero_grad",
        "requires_grad_",
        "state_dict",
        "load_state_dict",
        "loss",
        "train",
        "fit",
        "checkpoint",
        "Checkpoint",
        "save_checkpoint",
        "run_diagnostic_worker",
        "run_base_study_worker",
        "publish_base_artifact",
        "publish_diagnostic_artifact",
        "base_artifact_key",
        "diagnostic_artifact_key",
    }
    assert not (ZERO_SHOT_NAMES & forbidden)


def test_no_test_or_held_out_partition_is_opened() -> None:
    result, _, _, _ = _run()
    assert result.payload["held_out_partition_opened"] is False
    assert result.payload["training_performed"] is False
    assert result.payload["optimizer_constructed"] is False
    assert result.payload["market_backtest_performed"] is False
    assert result.payload["parameters_unmodified"] is True
    assert result.payload["assets_resolved"]["trainable_parameter_count"] == 0
    assert result.payload["parameter_sha256_before"] == result.payload["parameter_sha256_after"]


def test_the_claim_boundary_is_a_benchmark_boundary_not_a_diagnostic_one() -> None:
    result, _, _, _ = _run()
    assert result.payload["claim_boundary"] == (
        "DEVELOPMENT BENCHMARK - NOT HOLDOUT OR TRADING EVIDENCE"
    )
    assert result.payload["evidence_class"] == "development_compatibility_canary"


# ====== decision rules ===============================================


def _evidence(
    *,
    label: str,
    temperature: float,
    median: float | None,
    fraction: float | None,
    supporting: int,
    lower: float,
    upper: float,
) -> ConfigurationEvidence:
    return ConfigurationEvidence(
        label=label,
        temperature=temperature,
        origins_scored=25 * len(ASSET_PANEL),
        pooled_median_relative_skill=median,
        pooled_fraction_beating_persistence=fraction,
        assets=tuple(
            AssetSupport(
                asset=asset,
                origins_scored=25,
                median_relative_skill=0.1,
                fraction_beating_persistence=0.8,
                supports=index < supporting,
            )
            for index, asset in enumerate(ASSET_PANEL)
        ),
        bootstrap=MovingBlockBootstrapInterval(
            defined=True,
            cluster_count=BOOTSTRAP_CLUSTER_COUNT,
            assets_per_cluster=BOOTSTRAP_ASSETS_PER_CLUSTER,
            observation_count=BOOTSTRAP_OBSERVATION_COUNT,
            block_length=BOOTSTRAP_BLOCK_LENGTH,
            available_block_starts=BOOTSTRAP_AVAILABLE_BLOCK_STARTS,
            blocks_drawn_per_resample=BOOTSTRAP_BLOCKS_PER_RESAMPLE,
            clusters_retained_per_resample=BOOTSTRAP_CLUSTER_COUNT,
            resamples=BOOTSTRAP_RESAMPLES,
            confidence_level=0.95,
            seed=BOOTSTRAP_SEED,
            point_estimate=(lower + upper) / 2,
            lower=lower,
            upper=upper,
            excludes_zero=lower > 0.0 or upper < 0.0,
            excludes_zero_favorably=lower > 0.0,
        ),
    )


def _clusters(
    values: list[tuple[float, float, float, float]],
    *,
    assets: tuple[str, ...] = ASSET_PANEL,
    ordinals: list[int] | None = None,
) -> tuple[OriginCluster, ...]:
    """Build well-formed clusters, or deliberately malformed ones."""
    chosen = ordinals if ordinals is not None else list(range(len(values)))
    return tuple(
        OriginCluster(ordinal=ordinal, assets=assets, paired_differences=tuple(row))
        for ordinal, row in zip(chosen, values, strict=True)
    )


def _uniform_clusters(value: float = 0.001) -> tuple[OriginCluster, ...]:
    return _clusters([(value, value, value, value) for _ in range(BOOTSTRAP_CLUSTER_COUNT)])


def test_z1_requires_all_four_conditions() -> None:
    good: dict[str, float | int] = {
        "median": 0.1,
        "fraction": 0.7,
        "supporting": 3,
        "lower": 0.001,
        "upper": 0.01,
    }
    for broken, value in (
        ("median", 0.0),
        ("fraction", 0.59),
        ("supporting", 2),
        ("lower", -0.001),
    ):
        arguments = dict(good)
        arguments[broken] = value
        decision = decide(
            configurations=(_evidence(label="A", temperature=0.6, **arguments),)  # type: ignore[arg-type]
        )
        assert ZeroShotFinding.ZERO_SHOT_SKILL_OBSERVED not in decision.matched_findings

    decision = decide(
        configurations=(_evidence(label="A", temperature=0.6, **good),)  # type: ignore[arg-type]
    )
    assert ZeroShotFinding.ZERO_SHOT_SKILL_OBSERVED in decision.matched_findings
    assert decision.outcome is ZeroShotOutcome.PROCEED_TO_CONFIGURATION_CONFIRMATION


def test_z2_matches_when_exactly_one_configuration_is_favorable() -> None:
    decision = decide(
        configurations=(
            _evidence(
                label="A", temperature=0.6, median=0.1, fraction=0.7, supporting=3,
                lower=0.001, upper=0.01,
            ),
            _evidence(
                label="B", temperature=1.0, median=-0.1, fraction=0.2, supporting=0,
                lower=-0.01, upper=-0.001,
            ),
        )
    )
    assert ZeroShotFinding.ISOLATED_CONFIGURATION_EFFECT in decision.matched_findings
    assert decision.outcome is ZeroShotOutcome.PROCEED_TO_CONFIGURATION_CONFIRMATION


def test_z3_matches_when_the_effect_is_confined_to_one_asset() -> None:
    decision = decide(
        configurations=(
            _evidence(
                label="A", temperature=0.6, median=-0.1, fraction=0.3, supporting=1,
                lower=-0.01, upper=0.01,
            ),
        )
    )
    assert ZeroShotFinding.ASSET_SPECIFIC_EFFECT in decision.matched_findings
    assert ZeroShotFinding.ZERO_SHOT_SKILL_OBSERVED not in decision.matched_findings
    assert decision.outcome is ZeroShotOutcome.PROCEED_TO_CONFIGURATION_CONFIRMATION


def test_z4_stops_the_direction_but_still_permits_the_representation_probe() -> None:
    decision = decide(
        configurations=(
            _evidence(
                label="A", temperature=0.6, median=-0.4, fraction=0.1, supporting=0,
                lower=-0.02, upper=-0.005,
            ),
            _evidence(
                label="B", temperature=1.0, median=-0.5, fraction=0.1, supporting=0,
                lower=-0.03, upper=-0.006,
            ),
        )
    )
    assert ZeroShotFinding.NO_ZERO_SHOT_SKILL in decision.matched_findings
    assert decision.outcome is ZeroShotOutcome.PROCEED_TO_FROZEN_REPRESENTATION_PROBE
    assert decision.zero_shot_generation_direction is (
        GenerationDirection.STOP_KRONOS_ZERO_SHOT_DIRECTION
    )
    assert decision.frozen_representation_probe_permitted is True


def test_z5_takes_precedence_over_every_other_rule() -> None:
    decision = decide(
        configurations=(
            _evidence(
                label="A", temperature=0.6, median=0.5, fraction=0.9, supporting=4,
                lower=0.01, upper=0.02,
            ),
        ),
        limitations=("SPY supplied fewer than the required sessions",),
    )
    assert decision.outcome is ZeroShotOutcome.BENCHMARK_INCONCLUSIVE
    assert ZeroShotFinding.BENCHMARK_INCONCLUSIVE in decision.matched_findings
    assert ZeroShotFinding.ZERO_SHOT_SKILL_OBSERVED not in decision.matched_findings


def test_the_four_predeclared_outcomes_are_exactly_these() -> None:
    assert {outcome.value for outcome in ZeroShotOutcome} == {
        "PROCEED_TO_FROZEN_REPRESENTATION_PROBE",
        "PROCEED_TO_CONFIGURATION_CONFIRMATION",
        "STOP_KRONOS_ZERO_SHOT_DIRECTION",
        "BENCHMARK_INCONCLUSIVE",
    }


# ====== C1-C2: the resampling unit is the ordinal, and it is indivisible ====


def _record_draws(clusters: tuple[OriginCluster, ...], *, resamples: int) -> list[int]:
    """Capture the population size of every randrange draw the bootstrap makes."""
    calls: list[int] = []
    real_randrange = random.Random.randrange

    def recording(self: random.Random, *args: Any, **kwargs: Any) -> int:
        drawn = real_randrange(self, *args, **kwargs)
        calls.append(args[0] if args else -1)
        return drawn

    with mock.patch.object(random.Random, "randrange", recording):
        paired_origin_moving_block_bootstrap(
            clusters, seed=BOOTSTRAP_SEED, resamples=resamples, confidence_level=0.95
        )
    return calls


def test_the_bootstrap_draws_block_starts_not_rows_and_not_single_ordinals() -> None:
    """Observe the draws directly rather than inferring them from the output.

    A flat design would draw from 100 rows; the superseded cluster design drew
    25 times from 25 ordinals. This draws 7 times from 22 block starts, which is
    a signature neither of the other two can produce.
    """
    calls = _record_draws(_uniform_clusters(), resamples=10)

    assert set(calls) == {BOOTSTRAP_AVAILABLE_BLOCK_STARTS}
    assert BOOTSTRAP_AVAILABLE_BLOCK_STARTS == 22
    assert len(calls) == 10 * BOOTSTRAP_BLOCKS_PER_RESAMPLE
    # Explicitly not the two superseded signatures.
    assert BOOTSTRAP_OBSERVATION_COUNT not in set(calls)
    assert BOOTSTRAP_CLUSTER_COUNT not in set(calls)


def test_origins_are_not_sampled_independently() -> None:
    """Seven draws per resample, not twenty-five."""
    calls = _record_draws(_uniform_clusters(), resamples=100)
    assert len(calls) == 100 * BOOTSTRAP_BLOCKS_PER_RESAMPLE == 700
    assert len(calls) != 100 * BOOTSTRAP_CLUSTER_COUNT

    interval = paired_origin_moving_block_bootstrap(
        _uniform_clusters(), seed=BOOTSTRAP_SEED, resamples=10, confidence_level=0.95
    )
    assert interval.origins_resampled_independently is False
    assert interval.assets_resampled_within_cluster is False


def test_every_block_is_four_consecutive_ordinals_without_wraparound() -> None:
    starts = block_starts()
    assert starts == tuple(range(22))
    assert min(starts) == 0
    assert max(starts) == 21

    for start in starts:
        block = tuple(range(start, start + BOOTSTRAP_BLOCK_LENGTH))
        assert len(block) == 4
        assert list(block) == sorted(block), "origins keep chronological order"
        assert block[-1] - block[0] == 3, "the four ordinals are consecutive"
        # No wraparound: the last ordinal of any block never precedes its first.
        assert block[-1] <= BOOTSTRAP_CLUSTER_COUNT - 1
        assert 0 not in block[1:], "ordinal 24 never wraps round to ordinal 0"

    # A start of 22 would need ordinals 22..25, which do not exist.
    assert 22 not in starts
    assert BOOTSTRAP_CLUSTER_COUNT - 1 + 1 not in starts


def test_seven_blocks_are_drawn_and_twenty_five_positions_are_retained() -> None:
    assert BOOTSTRAP_BLOCKS_PER_RESAMPLE == 7
    assert BOOTSTRAP_BLOCKS_PER_RESAMPLE * BOOTSTRAP_BLOCK_LENGTH == 28
    assert BOOTSTRAP_CLUSTER_COUNT == 25

    interval = paired_origin_moving_block_bootstrap(
        _uniform_clusters(), seed=BOOTSTRAP_SEED, resamples=50, confidence_level=0.95
    )
    assert interval.blocks_drawn_per_resample == 7
    assert interval.clusters_retained_per_resample == 25
    assert interval.observation_count == 100

    calls = _record_draws(_uniform_clusters(), resamples=50)
    assert len(calls) == 50 * 7


def test_the_denominator_is_exactly_one_hundred_observations() -> None:
    """A constant panel must reproduce its own mean exactly.

    If the denominator or the retained-cluster count were wrong, a constant
    would not come back unchanged.
    """
    for value in (0.001, -0.0025, 0.0):
        interval = paired_origin_moving_block_bootstrap(
            _uniform_clusters(value), seed=BOOTSTRAP_SEED, resamples=200, confidence_level=0.95
        )
        assert interval.observation_count == 100
        assert interval.point_estimate == pytest.approx(value)
        assert interval.lower == pytest.approx(value)
        assert interval.upper == pytest.approx(value)


def test_the_point_estimate_equals_the_mean_of_the_twenty_five_cluster_means() -> None:
    rows = [
        (0.001 * index, 0.002 * index, -0.001 * index, 0.0005 * index)
        for index in range(BOOTSTRAP_CLUSTER_COUNT)
    ]
    clusters = _clusters(rows)
    interval = paired_origin_moving_block_bootstrap(
        clusters, seed=BOOTSTRAP_SEED, resamples=100, confidence_level=0.95
    )
    flat_mean = sum(v for row in rows for v in row) / BOOTSTRAP_OBSERVATION_COUNT
    cluster_means = [sum(row) / len(row) for row in rows]
    assert interval.point_estimate == pytest.approx(flat_mean)
    assert interval.point_estimate == pytest.approx(sum(cluster_means) / len(cluster_means))


@pytest.mark.parametrize("length", [1, 2, 3, 5, 6, 12, 25])
def test_any_block_length_other_than_four_is_rejected(length: int) -> None:
    with pytest.raises(BridgeTransformError) as excinfo:
        paired_origin_moving_block_bootstrap(
            _uniform_clusters(),
            block_length=length,
            seed=BOOTSTRAP_SEED,
            resamples=10,
            confidence_level=0.95,
        )
    assert excinfo.value.failures[0].code == "ZERO_SHOT_BOOTSTRAP_BLOCK_LENGTH_INVALID"
    assert "may not be estimated or tuned" in excinfo.value.failures[0].message


def test_the_block_length_is_derived_from_the_benchmark_geometry() -> None:
    from openalpha_bridge.zero_shot.origins import ORIGIN_STRIDE

    assert BOOTSTRAP_BLOCK_LENGTH == math.ceil(CONTEXT_CANDLES / ORIGIN_STRIDE) == 4
    # The overlap structure the block length exists to span.
    assert CONTEXT_CANDLES - 1 * ORIGIN_STRIDE == 28  # adjacent origins share 28
    assert CONTEXT_CANDLES - 2 * ORIGIN_STRIDE == 16  # two strides apart share 16
    assert CONTEXT_CANDLES - 3 * ORIGIN_STRIDE == 4   # three strides apart share 4
    assert CONTEXT_CANDLES - 4 * ORIGIN_STRIDE < 0    # four strides apart share none


def test_selecting_one_ordinal_always_selects_all_four_asset_differences() -> None:
    """A lone non-zero cluster can only ever contribute all four of its values."""
    rows = [(0.0, 0.0, 0.0, 0.0) for _ in range(BOOTSTRAP_CLUSTER_COUNT)]
    rows[7] = (1.0, 2.0, 3.0, 4.0)  # sums to 10.0
    interval = paired_origin_moving_block_bootstrap(
        _clusters(rows), seed=BOOTSTRAP_SEED, resamples=500, confidence_level=0.95
    )
    assert interval.defined is True
    assert interval.observation_count == BOOTSTRAP_OBSERVATION_COUNT

    # Each resampled mean must be k * 10.0 / 100 for an integer k: the cluster
    # enters whole or not at all. A partial contribution (say one asset's 1.0)
    # would produce a mean that is not a multiple of the cluster total.
    step = 10.0 / BOOTSTRAP_OBSERVATION_COUNT
    for value in (interval.lower, interval.upper, interval.point_estimate):
        assert value is not None
        multiples = value / step
        assert multiples == pytest.approx(round(multiples), abs=1e-9)
    assert interval.point_estimate == pytest.approx(10.0 / BOOTSTRAP_OBSERVATION_COUNT)


# ====== C3-C9: cluster structure is validated, never silently repaired =====


def test_exactly_twenty_five_clusters_are_required() -> None:
    for count in (24, 26, 0):
        rows = [(0.001, 0.001, 0.001, 0.001) for _ in range(count)]
        with pytest.raises(BridgeTransformError) as excinfo:
            paired_origin_moving_block_bootstrap(
                _clusters(rows), seed=BOOTSTRAP_SEED, resamples=10, confidence_level=0.95
            )
        assert excinfo.value.failures[0].code == "ZERO_SHOT_BOOTSTRAP_CLUSTER_COUNT_INVALID"
    assert BOOTSTRAP_CLUSTER_COUNT == 25


def test_every_cluster_must_carry_exactly_four_assets() -> None:
    clusters = list(_uniform_clusters())
    clusters[3] = OriginCluster(
        ordinal=3, assets=("SPY", "QQQ", "IWM"), paired_differences=(0.1, 0.1, 0.1)
    )
    with pytest.raises(BridgeTransformError) as excinfo:
        paired_origin_moving_block_bootstrap(
            tuple(clusters), seed=BOOTSTRAP_SEED, resamples=10, confidence_level=0.95
        )
    assert excinfo.value.failures[0].code == "ZERO_SHOT_BOOTSTRAP_CLUSTER_ASSET_COUNT_INVALID"
    assert BOOTSTRAP_ASSETS_PER_CLUSTER == 4


def test_the_cluster_asset_identities_must_be_the_preregistered_panel() -> None:
    assert ASSET_PANEL == ("SPY", "QQQ", "IWM", "DIA")
    clusters = list(_uniform_clusters())
    clusters[0] = OriginCluster(
        ordinal=0, assets=("SPY", "QQQ", "IWM", "VOO"), paired_differences=(0.1, 0.1, 0.1, 0.1)
    )
    with pytest.raises(BridgeTransformError) as excinfo:
        paired_origin_moving_block_bootstrap(
            tuple(clusters), seed=BOOTSTRAP_SEED, resamples=10, confidence_level=0.95
        )
    assert excinfo.value.failures[0].code == "ZERO_SHOT_BOOTSTRAP_ASSET_PANEL_MISMATCH"


def test_a_duplicated_asset_in_a_cluster_is_rejected() -> None:
    clusters = list(_uniform_clusters())
    clusters[5] = OriginCluster(
        ordinal=5, assets=("SPY", "SPY", "IWM", "DIA"), paired_differences=(0.1, 0.1, 0.1, 0.1)
    )
    with pytest.raises(BridgeTransformError) as excinfo:
        paired_origin_moving_block_bootstrap(
            tuple(clusters), seed=BOOTSTRAP_SEED, resamples=10, confidence_level=0.95
        )
    assert excinfo.value.failures[0].code == "ZERO_SHOT_BOOTSTRAP_DUPLICATE_ASSET"


def test_a_missing_paired_difference_is_rejected() -> None:
    clusters = list(_uniform_clusters())
    clusters[2] = OriginCluster(
        ordinal=2, assets=ASSET_PANEL, paired_differences=(0.1, 0.1, 0.1)
    )
    with pytest.raises(BridgeTransformError) as excinfo:
        paired_origin_moving_block_bootstrap(
            tuple(clusters), seed=BOOTSTRAP_SEED, resamples=10, confidence_level=0.95
        )
    assert excinfo.value.failures[0].code == "ZERO_SHOT_BOOTSTRAP_MISSING_DIFFERENCE"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_paired_difference_is_rejected_at_the_type_boundary(bad: float) -> None:
    """The cluster model refuses it before the bootstrap is ever called."""
    with pytest.raises(ValidationError):
        OriginCluster(ordinal=9, assets=ASSET_PANEL, paired_differences=(0.001, bad, 0.001, 0.001))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_difference_is_also_caught_if_the_model_is_bypassed(bad: float) -> None:
    """Defense in depth: the runtime guard stands even without model validation.

    ``model_construct`` skips validation the way a future refactor might. The
    bootstrap must still refuse rather than propagate a non-finite mean.
    """
    clusters = list(_uniform_clusters())
    clusters[9] = OriginCluster.model_construct(
        ordinal=9, assets=ASSET_PANEL, paired_differences=(0.001, bad, 0.001, 0.001)
    )
    with pytest.raises(BridgeTransformError) as excinfo:
        paired_origin_moving_block_bootstrap(
            tuple(clusters), seed=BOOTSTRAP_SEED, resamples=10, confidence_level=0.95
        )
    assert excinfo.value.failures[0].code == "ZERO_SHOT_BOOTSTRAP_NON_FINITE_DIFFERENCE"


def test_the_ordinals_must_be_exactly_zero_through_twenty_four() -> None:
    rows = [(0.001, 0.001, 0.001, 0.001) for _ in range(BOOTSTRAP_CLUSTER_COUNT)]
    for ordinals in (
        [*range(1, BOOTSTRAP_CLUSTER_COUNT + 1)],  # shifted
        [*range(BOOTSTRAP_CLUSTER_COUNT - 1), 0],  # duplicated / unsorted
    ):
        with pytest.raises(BridgeTransformError) as excinfo:
            paired_origin_moving_block_bootstrap(
                _clusters(rows, ordinals=ordinals),
                seed=BOOTSTRAP_SEED,
                resamples=10,
                confidence_level=0.95,
            )
        assert excinfo.value.failures[0].code == "ZERO_SHOT_BOOTSTRAP_ORDINALS_INVALID"


def test_an_undefined_difference_never_shrinks_the_bootstrap_sample() -> None:
    """The undefined path reports absence; it does not resample a smaller panel."""
    interval = undefined_moving_block_bootstrap(
        seed=BOOTSTRAP_SEED,
        resamples=BOOTSTRAP_RESAMPLES,
        confidence_level=0.95,
        reason="the paired difference is undefined at ordinal 4 for IWM",
    )
    assert interval.defined is False
    assert interval.cluster_count == 0
    assert interval.observation_count == 0
    assert interval.lower is None and interval.upper is None
    assert interval.excludes_zero_favorably is False
    assert interval.undefined_reason is not None


# ====== C10-C12: determinism, metadata, and the width property =============


def test_the_cluster_bootstrap_is_deterministic_under_the_preregistered_seed() -> None:
    rows = [
        (
            0.001 * ((index % 7) - 3),
            0.001 * ((index % 5) - 2),
            0.001 * ((index % 3) - 1),
            0.001 * ((index % 11) - 5),
        )
        for index in range(BOOTSTRAP_CLUSTER_COUNT)
    ]
    first = paired_origin_moving_block_bootstrap(
        _clusters(rows),
        seed=BOOTSTRAP_SEED,
        resamples=BOOTSTRAP_RESAMPLES,
        confidence_level=0.95,
    )
    second = paired_origin_moving_block_bootstrap(
        _clusters(rows),
        seed=BOOTSTRAP_SEED,
        resamples=BOOTSTRAP_RESAMPLES,
        confidence_level=0.95,
    )
    assert first == second
    assert BOOTSTRAP_SEED == 20260803
    assert first.seed == 20260803
    assert first.lower is not None and first.upper is not None
    assert first.lower <= first.upper

    different = paired_origin_moving_block_bootstrap(
        _clusters(rows), seed=BOOTSTRAP_SEED + 1, resamples=200, confidence_level=0.95
    )
    assert different.seed != first.seed


def test_the_interval_metadata_reports_the_corrected_design() -> None:
    interval = paired_origin_moving_block_bootstrap(
        _uniform_clusters(), seed=BOOTSTRAP_SEED, resamples=200, confidence_level=0.95
    )
    assert interval.method == "paired_percentile_moving_block_bootstrap_over_origin_clusters"
    assert interval.method == BOOTSTRAP_METHOD
    assert interval.base_resampling_unit == "origin_ordinal_cluster_all_assets"
    assert interval.base_resampling_unit == BOOTSTRAP_BASE_RESAMPLING_UNIT
    assert interval.temporal_resampling_unit == "four_consecutive_origin_clusters"
    assert interval.temporal_resampling_unit == BOOTSTRAP_TEMPORAL_RESAMPLING_UNIT
    assert interval.cluster_count == 25
    assert interval.assets_per_cluster == 4
    assert interval.observation_count == 100
    assert interval.block_length == 4
    assert interval.available_block_starts == 22
    assert interval.blocks_drawn_per_resample == 7
    assert interval.clusters_retained_per_resample == 25
    assert interval.resamples == 200
    assert interval.confidence_level == 0.95
    assert interval.seed == 20260803
    assert interval.assets_resampled_within_cluster is False
    assert interval.origins_resampled_independently is False
    assert interval.wraparound is False
    assert interval.block_length_selected_before_execution is True
    assert interval.block_length_tuned_from_results is False

    favorable = paired_origin_moving_block_bootstrap(
        _uniform_clusters(0.01), seed=BOOTSTRAP_SEED, resamples=200, confidence_level=0.95
    )
    assert favorable.excludes_zero_favorably is True
    unfavorable = paired_origin_moving_block_bootstrap(
        _uniform_clusters(-0.01), seed=BOOTSTRAP_SEED, resamples=200, confidence_level=0.95
    )
    assert unfavorable.excludes_zero is True
    assert unfavorable.excludes_zero_favorably is False


def _independent_row_interval(
    clusters: tuple[OriginCluster, ...], *, seed: int, resamples: int, confidence_level: float
) -> tuple[float, float]:
    """The discredited flat design, reimplemented HERE for comparison only.

    It lives in the test rather than the library on purpose: the library must
    not contain a function that could produce a decision-bearing flat interval.
    """
    rows = [value for cluster in clusters for value in cluster.paired_differences]
    generator = random.Random(seed)
    size = len(rows)
    means: list[float] = []
    for _ in range(resamples):
        means.append(sum(rows[generator.randrange(size)] for _ in range(size)) / size)
    means.sort()
    tail = (1.0 - confidence_level) / 2.0
    return means[math.floor(tail * (resamples - 1))], means[math.ceil((1.0 - tail) * (resamples - 1))]


def _independent_cluster_interval(
    clusters: tuple[OriginCluster, ...], *, seed: int, resamples: int, confidence_level: float
) -> tuple[float, float]:
    """The superseded independent-cluster design, reimplemented HERE only.

    It lives in the test rather than the library for the same reason the flat
    design does: the library must not contain a function that could produce a
    decision-bearing interval weaker than the moving-block one.
    """
    totals = [sum(cluster.paired_differences) for cluster in clusters]
    count = len(totals)
    observations = count * BOOTSTRAP_ASSETS_PER_CLUSTER
    generator = random.Random(seed)
    means: list[float] = []
    for _ in range(resamples):
        means.append(
            sum(totals[generator.randrange(count)] for _ in range(count)) / observations
        )
    means.sort()
    tail = (1.0 - confidence_level) / 2.0
    return (
        means[math.floor(tail * (resamples - 1))],
        means[math.ceil((1.0 - tail) * (resamples - 1))],
    )


def _serially_correlated_clusters(
    mean: float = 0.005, sd: float = 0.01
) -> tuple[OriginCluster, ...]:
    """Value held constant within each block of four consecutive origins.

    A deliberately clean stand-in for the real overlap structure: origins inside
    one block share information, origins in different blocks do not.
    """
    pattern = (-1.3, -0.8, -0.2, 0.2, 0.8, 1.3, 0.0)
    return _clusters(
        [
            (v, v, v, v)
            for v in (
                mean + sd * pattern[index // BOOTSTRAP_BLOCK_LENGTH]
                for index in range(BOOTSTRAP_CLUSTER_COUNT)
            )
        ]
    )


def test_serially_correlated_origins_give_a_wider_interval_than_independent_clusters() -> None:
    """The point of this correction, demonstrated numerically.

    When consecutive origins carry the same information -- which is what a
    40-session context on a 12-session stride guarantees -- sampling the 25
    clusters independently shrinks the interval by nearly half. The moving-block
    design refuses that free precision.
    """
    clusters = _serially_correlated_clusters()
    block = paired_origin_moving_block_bootstrap(
        clusters, seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES, confidence_level=0.95
    )
    indep_lower, indep_upper = _independent_cluster_interval(
        clusters, seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES, confidence_level=0.95
    )

    assert block.lower is not None and block.upper is not None
    block_width = block.upper - block.lower
    indep_width = indep_upper - indep_lower
    assert block_width >= indep_width
    assert block_width > indep_width * 1.5


def test_independent_clusters_would_have_falsely_satisfied_z1_here() -> None:
    """A case where the superseded design claims skill and this one refuses."""
    clusters = _serially_correlated_clusters(mean=0.005, sd=0.01)
    block = paired_origin_moving_block_bootstrap(
        clusters, seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES, confidence_level=0.95
    )
    indep_lower, _ = _independent_cluster_interval(
        clusters, seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES, confidence_level=0.95
    )

    assert block.point_estimate is not None and block.point_estimate > 0.0
    assert indep_lower > 0.0, "the independent-cluster design would have claimed skill"
    assert block.excludes_zero_favorably is False, "the moving-block design must not"


def test_perfectly_correlated_assets_give_a_wider_interval_than_row_resampling() -> None:
    """The whole point of the correction, demonstrated numerically.

    When the four ETFs move together -- which is the real situation being
    modelled -- treating them as four independent draws shrinks the interval by
    roughly a factor of two. The cluster design refuses that free precision.
    """
    rows = [
        (v, v, v, v)
        for v in (0.002 * ((index % 9) - 4) for index in range(BOOTSTRAP_CLUSTER_COUNT))
    ]
    clusters = _clusters(rows)

    clustered = paired_origin_moving_block_bootstrap(
        clusters, seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES, confidence_level=0.95
    )
    flat_lower, flat_upper = _independent_row_interval(
        clusters, seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES, confidence_level=0.95
    )

    assert clustered.lower is not None and clustered.upper is not None
    clustered_width = clustered.upper - clustered.lower
    flat_width = flat_upper - flat_lower
    assert clustered_width >= flat_width
    # Not merely equal: with perfect correlation the flat design understates
    # uncertainty substantially.
    assert clustered_width > flat_width * 1.5


def test_flat_row_resampling_would_have_falsely_satisfied_z1() -> None:
    """Why this matters: the corrected interval refuses a claim the flat one allows."""
    # The real dependence structure: four correlated assets AND serially
    # correlated origins. Flat row resampling ignores both and excludes zero;
    # the moving-block design respects both and does not.
    clusters = _serially_correlated_clusters()
    clustered = paired_origin_moving_block_bootstrap(
        clusters, seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES, confidence_level=0.95
    )
    flat_lower, _ = _independent_row_interval(
        clusters, seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES, confidence_level=0.95
    )
    assert clustered.point_estimate is not None and clustered.point_estimate > 0.0
    assert flat_lower > 0.0, "the flat design would have claimed favorable exclusion"
    assert clustered.excludes_zero_favorably is False, (
        "the clustered design must not claim it"
    )


# ====== C13-C14: only the clustered interval reaches the decision layer ====


def test_z1_reads_only_the_moving_block_interval() -> None:
    from openalpha_bridge.zero_shot.spec import ZERO_SHOT_THRESHOLDS

    evidence = _evidence(
        label="A", temperature=0.6, median=0.1, fraction=0.7, supporting=3,
        lower=0.001, upper=0.01,
    )
    conditions = evidence.z1_conditions(ZERO_SHOT_THRESHOLDS)
    assert "moving_block_bootstrap_excludes_zero_favorably" in conditions
    assert conditions["moving_block_bootstrap_excludes_zero_favorably"] is True
    assert not [key for key in conditions if key.startswith("bootstrap_")]
    assert "cluster_bootstrap_excludes_zero_favorably" not in conditions
    assert evidence.bootstrap.base_resampling_unit == "origin_ordinal_cluster_all_assets"
    assert evidence.bootstrap.temporal_resampling_unit == "four_consecutive_origin_clusters"
    assert evidence.bootstrap.block_length == 4


def test_no_weaker_interval_can_reach_the_decision_layer() -> None:
    """Structural, not conventional: neither superseded function exists.

    Both discredited designs -- the flat asset-origin bootstrap and the
    independent origin-cluster bootstrap -- are absent from the library, so
    "the decision layer cannot read one" is not something a reviewer has to
    take on trust.
    """
    for removed in (
        "paired_bootstrap",
        "BootstrapInterval",
        "paired_cluster_bootstrap",
        "ClusterBootstrapInterval",
        "undefined_cluster_bootstrap",
    ):
        assert not hasattr(aggregation_module, removed), removed
        assert removed not in ZERO_SHOT_NAMES, removed

    # ConfigurationEvidence accepts one interval type, and it is the moving-block one.
    annotation = ConfigurationEvidence.model_fields["bootstrap"].annotation
    assert annotation is MovingBlockBootstrapInterval

    with pytest.raises(ValidationError):
        ConfigurationEvidence(
            label="A",
            temperature=0.6,
            origins_scored=100,
            pooled_median_relative_skill=0.1,
            pooled_fraction_beating_persistence=0.7,
            assets=(),
            bootstrap={"defined": True, "sample_size": 100, "lower": 0.1},  # type: ignore[arg-type]
        )


def test_the_run_carries_the_moving_block_interval_into_the_decision() -> None:
    result, _, _, _ = _run()
    payload = result.payload
    assert payload["bootstrap_method"] == (
        "paired_percentile_moving_block_bootstrap_over_origin_clusters"
    )
    assert payload["bootstrap_base_resampling_unit"] == "origin_ordinal_cluster_all_assets"
    assert payload["bootstrap_temporal_resampling_unit"] == "four_consecutive_origin_clusters"
    assert payload["bootstrap_cluster_count"] == 25
    assert payload["bootstrap_assets_per_cluster"] == 4
    assert payload["bootstrap_observation_count"] == 100
    assert payload["bootstrap_block_length"] == 4
    assert payload["bootstrap_available_block_starts"] == 22
    assert payload["bootstrap_blocks_per_resample"] == 7
    assert payload["bootstrap_seed"] == 20260803
    assert payload["bootstrap_resamples"] == 2000

    for aggregate in payload["aggregates"]:
        assert "bootstrap" not in aggregate
        assert "cluster_bootstrap" not in aggregate
        interval = aggregate["moving_block_bootstrap"]
        assert interval["defined"] is True
        assert interval["cluster_count"] == 25
        assert interval["assets_per_cluster"] == 4
        assert interval["observation_count"] == 100
        assert interval["assets_resampled_within_cluster"] is False
        assert interval["origins_resampled_independently"] is False
        assert interval["wraparound"] is False
        assert interval["block_length"] == 4
        assert interval["blocks_drawn_per_resample"] == 7
        assert len(aggregate["origin_clusters"]) == 25
        for cluster in aggregate["origin_clusters"]:
            assert cluster["assets"] == list(ASSET_PANEL)
            assert len(cluster["paired_differences"]) == 4
        assert [c["ordinal"] for c in aggregate["origin_clusters"]] == list(range(25))

    for configuration in payload["decision"]["configurations"]:
        assert configuration["bootstrap"]["base_resampling_unit"] == (
            "origin_ordinal_cluster_all_assets"
        )
        assert configuration["bootstrap"]["temporal_resampling_unit"] == (
            "four_consecutive_origin_clusters"
        )
        assert configuration["bootstrap"]["cluster_count"] == 25
        assert configuration["bootstrap"]["block_length"] == 4


def test_the_recorded_clusters_reproduce_the_recorded_interval() -> None:
    """The artifact carries enough to recompute its own interval."""
    result, _, _, _ = _run()
    for aggregate in result.payload["aggregates"]:
        clusters = tuple(
            OriginCluster(
                ordinal=entry["ordinal"],
                assets=tuple(entry["assets"]),
                paired_differences=tuple(entry["paired_differences"]),
            )
            for entry in aggregate["origin_clusters"]
        )
        recomputed = paired_origin_moving_block_bootstrap(
            clusters,
            seed=BOOTSTRAP_SEED,
            resamples=BOOTSTRAP_RESAMPLES,
            confidence_level=0.95,
        )
        assert recomputed.lower == pytest.approx(aggregate["moving_block_bootstrap"]["lower"])
        assert recomputed.upper == pytest.approx(aggregate["moving_block_bootstrap"]["upper"])
        assert recomputed.point_estimate == pytest.approx(
            aggregate["moving_block_bootstrap"]["point_estimate"]
        )


# ====== C15-C17: cross-asset calendar alignment ===========================


def test_all_four_assets_share_target_and_context_dates_at_every_ordinal() -> None:
    result, _, _, _ = _run()
    alignment = result.payload["cross_asset_alignment"]
    assert alignment["ordinals_checked"] == 25
    assert alignment["assets_checked"] == list(ASSET_PANEL)
    assert alignment["target_dates_aligned"] is True
    assert alignment["context_dates_aligned"] is True
    assert alignment["proved_from_retrieved_sessions"] is True

    by_ordinal: dict[int, set[tuple[str, str, str, str]]] = {}
    for origin in result.payload["origins"]:
        by_ordinal.setdefault(origin["ordinal"], set()).add(
            (
                origin["target_first_session"],
                origin["target_last_session"],
                origin["context_first_session"],
                origin["context_last_session"],
            )
        )
    assert len(by_ordinal) == 25
    for ordinal, windows in by_ordinal.items():
        assert len(windows) == 1, f"ordinal {ordinal} spans more than one window"


def _origin_result(asset: str, ordinal: int, *, shift_target: int = 0, shift_context: int = 0):
    from openalpha_bridge.zero_shot.runner import OriginResult

    base = date(2025, 6, 2)
    return OriginResult(
        asset=asset,
        ordinal=ordinal,
        origin_index=40 + 12 * ordinal,
        context_first_session=base + timedelta(days=shift_context),
        context_last_session=base + timedelta(days=39 + shift_context),
        target_first_session=base + timedelta(days=40 + shift_target),
        target_last_session=base + timedelta(days=51 + shift_target),
        context_candles=CONTEXT_CANDLES,
        horizon_candles=HORIZON_CANDLES,
        anchor_close=100.0,
        target_closes=tuple(100.0 for _ in range(HORIZON_CANDLES)),
        normalization_state_sha256="d" * 64,
        normalization_fitted_candle_count=CONTEXT_CANDLES,
        context_clipping=ClippingReport(
            rows=CONTEXT_CANDLES,
            scalar_values=CONTEXT_CANDLES * 6,
            clipped_scalar_values=0,
            clipped_scalar_fraction=0.0,
            clipped_by_column={},
            clipped_fraction_by_column={},
            rows_with_any_clipped_channel=0,
            row_clipped_fraction=0.0,
            clipped_row_indices=(),
        ),
        persistence_metrics=ExtendedForecastMetricsModel(defined=True),
        persistence_step_absolute_return_errors=(),
        baselines=(),
        configurations=(),
    )


def test_a_mismatched_target_range_fails_with_the_typed_alignment_code() -> None:
    origins = tuple(
        _origin_result(asset, 0, shift_target=3 if asset == "IWM" else 0)
        for asset in ASSET_PANEL
    )
    with pytest.raises(BridgeTransformError) as excinfo:
        verify_cross_asset_alignment(origins)
    failure = excinfo.value.failures[0]
    assert failure.code == "ZERO_SHOT_CROSS_ASSET_ORIGIN_MISALIGNED"
    assert "ordinal 0" in failure.message
    assert "IWM" in failure.message
    assert failure.field == "ordinal_0"


def test_a_mismatched_context_range_is_also_rejected() -> None:
    origins = tuple(
        _origin_result(asset, 4, shift_context=2 if asset == "DIA" else 0)
        for asset in ASSET_PANEL
    )
    with pytest.raises(BridgeTransformError) as excinfo:
        verify_cross_asset_alignment(origins)
    failure = excinfo.value.failures[0]
    assert failure.code == "ZERO_SHOT_CROSS_ASSET_ORIGIN_MISALIGNED"
    assert "context window disagrees" in failure.message
    assert "DIA" in failure.message


def test_a_missing_asset_at_an_ordinal_fails_rather_than_shrinking_the_panel() -> None:
    origins = tuple(_origin_result(asset, 0) for asset in ASSET_PANEL[:3])
    with pytest.raises(BridgeTransformError) as excinfo:
        verify_cross_asset_alignment(origins)
    failure = excinfo.value.failures[0]
    assert failure.code == "ZERO_SHOT_CROSS_ASSET_ORIGIN_MISALIGNED"
    assert "DIA" in failure.message


def test_misalignment_publishes_a_typed_failure_and_no_interval() -> None:
    """Fail closed end to end: one artifact, typed code, no fallback interval."""

    class _MisalignedProvider(_PanelProvider):
        def fetch(self, request: RetrievalRequest) -> MarketSeries:
            series = super().fetch(request)
            if request.symbol != "DIA":
                return series
            # Drop DIA's first session so every one of its origins lands on a
            # different date than the other three.
            return series.model_copy(update={"candles": series.candles[1:]})

    factory = _Factory(provider=_MisalignedProvider())
    result, store, _, _ = _run(factory=factory)
    assert result.outcome == ZERO_SHOT_FAILURE_CODE
    assert result.payload["failure_code"] == "ZERO_SHOT_CROSS_ASSET_ORIGIN_MISALIGNED"
    assert store.list_keys("") == (zero_shot_artifact_key(RUN_ID),)
    assert "moving_block_bootstrap" not in result.payload
    assert result.payload["scientific_result_available"] is False


def test_every_declared_aggregation_level_is_present() -> None:
    result, _, _, _ = _run()
    assert len(result.payload["aggregates"]) == 2
    for aggregate in result.payload["aggregates"]:
        assert aggregate["origins_scored"] == 100
        assert [asset["asset"] for asset in aggregate["assets"]] == list(ASSET_PANEL)
        assert len(aggregate["steps"]) == HORIZON_CANDLES
        assert [step["step"] for step in aggregate["steps"]] == list(
            range(1, HORIZON_CANDLES + 1)
        )
        assert set(aggregate["secondary_metrics"]) == {
            "close_mae",
            "directional_accuracy",
            "close_return_rmse",
            "median_absolute_return_error",
        }
        for block in ("candidate_primary_error", "relative_skill", "paired_difference"):
            summary = aggregate[block]
            assert summary["count"] > 0
            for statistic in ("mean", "median", "standard_deviation"):
                assert summary[statistic] is not None
        assert aggregate["fraction_beating_persistence"] is not None
        assert aggregate["moving_block_bootstrap"]["seed"] == BOOTSTRAP_SEED
        assert aggregate["moving_block_bootstrap"]["resamples"] == BOOTSTRAP_RESAMPLES
    assert result.payload["primary_metric"] == PRIMARY_METRIC == "close_return_mae"


def test_summarize_reports_absence_rather_than_inventing_a_number() -> None:
    empty = summarize([])
    assert empty.count == 0
    assert empty.mean is None
    assert empty.undefined_reason is not None


def test_the_single_path_is_ensemble_member_zero() -> None:
    """The deterministic path is generated once and read twice, not regenerated."""
    result, _, resolver, _ = _run()
    per_cell = len(ENSEMBLE_SEEDS)
    assert resolver.model.generate_calls % per_cell == 0
    for origin in result.payload["origins"]:
        for configuration in origin["configurations"]:
            assert configuration["paths"][0]["seed"] == ENSEMBLE_SEEDS[0]
            assert configuration["single_path_metrics"]["close_return_mae"] == pytest.approx(
                configuration["paths"][0]["close_return_mae"]
            )


# ====== C18-C20: everything except the bootstrap structure is unchanged ====


def test_every_benchmark_constant_except_the_bootstrap_structure_is_unchanged() -> None:
    """The correction is narrow. This pins the blast radius.

    If any of these move, the change stopped being a statistical-design
    correction and became a redesign of the study.
    """
    from openalpha_bridge.zero_shot.origins import ORIGIN_STRIDE
    from openalpha_bridge.zero_shot.spec import (
        BOOTSTRAP_CONFIDENCE_LEVEL,
        MAXIMUM_CONTEXT,
        RETRIEVAL_END_EXCLUSIVE,
        RETRIEVAL_MAXIMUM_CANDLES,
        RETRIEVAL_START_INCLUSIVE,
        ZERO_SHOT_ARTIFACT_OBJECT_NAME,
        ZERO_SHOT_CACHE_VOLUME,
        ZERO_SHOT_THRESHOLDS,
    )

    # Panel, window and provider policy.
    assert ASSET_PANEL == ("SPY", "QQQ", "IWM", "DIA")
    assert RETRIEVAL_START_INCLUSIVE == "2025-01-02"
    assert RETRIEVAL_END_EXCLUSIVE == "2026-07-01"
    assert RETRIEVAL_MAXIMUM_CANDLES == 512
    assert REQUIRED_SESSIONS == 340

    # Geometry.
    assert CONTEXT_CANDLES == 40
    assert HORIZON_CANDLES == 12
    assert MAXIMUM_CONTEXT == 512
    assert len(ORIGIN_INDEX_OFFSETS) == 25
    assert ORIGIN_INDEX_OFFSETS[0] == 40
    assert ORIGIN_INDEX_OFFSETS[-1] == 328
    assert ORIGIN_STRIDE == 12

    # Configurations and seeds.
    assert [c.temperature for c in SAMPLING_CONFIGURATIONS] == [0.6, 1.0]
    assert ENSEMBLE_SEEDS == (
        20250102, 20250103, 20250104, 20250105, 20250106, 20250107, 20250108, 20250109,
    )

    # Candidate, baselines, metrics.
    assert BASELINE_IDS == (
        "zero_return_persistence",
        "last_close_level",
        "context_mean_return",
        "context_drift",
    )
    assert PRIMARY_BASELINE_ID == "zero_return_persistence"
    assert PRIMARY_METRIC == "close_return_mae"

    # Z1 thresholds are untouched by this correction.
    assert ZERO_SHOT_THRESHOLDS.minimum_median_relative_skill == 0.0
    assert ZERO_SHOT_THRESHOLDS.minimum_fraction_of_origins_beating_persistence == 0.60
    assert ZERO_SHOT_THRESHOLDS.minimum_supporting_assets == 3
    assert ZERO_SHOT_THRESHOLDS.total_assets == 4

    # Namespaces, identities and storage.
    assert ZERO_SHOT_EXPERIMENT_ID == "openalpha-kronos-zero-shot-benchmark-v1"
    assert ZERO_SHOT_ARTIFACT_ROOT == "openalpha-compatibility/kronos-zero-shot-benchmark"
    assert ZERO_SHOT_ARTIFACT_OBJECT_NAME == "kronos_zero_shot_benchmark_terminal.json"
    assert ZERO_SHOT_RUN_ID_PATTERN.pattern == r"^zsb_[0-9a-f]{8,32}$"
    assert ZERO_SHOT_CACHE_VOLUME == "openalpha-kronos-base-cache"

    # The parts of the bootstrap that were already correct.
    assert BOOTSTRAP_SEED == 20260803
    assert BOOTSTRAP_RESAMPLES == 2000
    assert BOOTSTRAP_CONFIDENCE_LEVEL == 0.95

    # And the model pair.
    assert verify_pinned_pair()["model_revision"] == "2b554741eca47781b64468546e77fef3e85130e6"


def test_the_completed_study_report_material_is_unchanged() -> None:
    """The correction must not touch the completed structural studies."""
    digests = {
        "artifact-manifest.json": (
            "666603480dd576cc1e6270a72cc853b25fe7477f715d59452433441c321ee11d"
        ),
        "results-summary.json": (
            "fabd86207ca37dd7d6ec1e04e6cca916813c5e324d9fa18f03d368b821d1f25a"
        ),
    }
    for name, expected in digests.items():
        observed = hashlib.sha256((REPORT / name).read_bytes()).hexdigest()
        assert observed == expected, f"{name} changed; the completed studies are immutable"


# ====== 24: the repository size guard still applies ==================


def test_the_new_files_are_source_sized() -> None:
    limit = 10 * 1024 * 1024
    for path in sorted(PACKAGE.rglob("*.py")):
        assert path.stat().st_size < limit, path
    for name in (ZERO_SHOT_SPECIFICATION_NAME, "kronos-frozen-representation-probe-design.md"):
        assert (RESEARCH / name).stat().st_size < limit


def test_the_modal_app_exposes_the_four_benchmark_functions() -> None:
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    for name in (
        "verify_zero_shot_benchmark_deployment",
        "verify_zero_shot_benchmark_runtime",
        "run_zero_shot_benchmark",
        "inventory_zero_shot_artifacts",
    ):
        assert name in functions

    # The completed studies' functions are untouched and still present.
    for name in (
        "verify_base_deployment",
        "verify_base_frozen_inference_runtime",
        "kronos_base_frozen_inference_diagnostic",
        "inventory_base_remote_cache",
    ):
        assert name in functions

    # No benchmark function is named as though it were a structural diagnostic.
    for name in functions:
        if "zero_shot" in name:
            assert "diagnostic" not in name


def test_the_benchmark_reuses_the_existing_volume_and_creates_no_new_one() -> None:
    text = APP.read_text(encoding="utf-8")
    assert text.count("modal.Volume.from_name(") == 2
    assert "openalpha-kronos-base-cache" in text
    from openalpha_bridge.zero_shot.spec import ZERO_SHOT_CACHE_VOLUME

    assert ZERO_SHOT_CACHE_VOLUME == "openalpha-kronos-base-cache"
