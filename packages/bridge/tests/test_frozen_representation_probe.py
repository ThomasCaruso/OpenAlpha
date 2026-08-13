"""The frozen-representation probe contract.

No network, no official asset, no Torch, no market data. The extractor is a
deterministic double producing a correctly shaped 832-dimensional tensor, and
every "test partition" here is synthetic.

The probe's real test partition does not exist yet, so the machinery that would
open it is exercised entirely against fixtures. What is tested is the gate, not
the data: that a premature or mismatched invocation refuses.
"""

from __future__ import annotations

import ast
import hashlib
import math
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from openalpha_bridge.base_study.spec import BASE_RUN_ID_PATTERN
from openalpha_bridge.cloud.objectstore import InMemoryObjectStore
from openalpha_bridge.diagnostic.normalization import fit_context_state
from openalpha_bridge.diagnostic.official_input import OfficialRow, official_stamp
from openalpha_bridge.errors import BridgeTransformError
from openalpha_bridge.phase2.invocation import RUN_ID_PATTERN as MINI_RUN_ID_PATTERN
from openalpha_bridge.phase2.provider import Candle, MarketSeries, ProviderMode, RetrievalRequest
from openalpha_bridge.representation_probe import controls
from openalpha_bridge.representation_probe.artifact import (
    PROBE_FAILURE_CODE,
    PROBE_FIT_CODE,
    load_verified_fit_artifact,
)
from openalpha_bridge.representation_probe.controls import (
    engineered_features,
    intercept_only,
    raw_ohlcv,
)
from openalpha_bridge.representation_probe.extraction import ExtractedRepresentation
from openalpha_bridge.representation_probe.features import (
    build_sample,
    cumulative_log_return,
    origin_offsets,
    resolve_windows,
    verify_cross_asset_alignment,
)
from openalpha_bridge.representation_probe.fit import (
    Standardizer,
    fit_logistic,
    fit_ridge,
)
from openalpha_bridge.representation_probe.invocation import ProbeInvocation
from openalpha_bridge.representation_probe.spec import (
    ASSET_PANEL,
    CONTEXT_CANDLES,
    EMBARGO_ORDINALS,
    ENGINEERED_FEATURE_NAMES,
    FEATURE_SET_DIMENSIONS,
    FEATURE_SET_IDS,
    HORIZON_CANDLES,
    LOGISTIC_C_GRID,
    MINIMUM_TEST_ORIGINS_PER_ASSET,
    MINIMUM_TEST_SAMPLES,
    MINIMUM_TEST_SESSIONS,
    PROBE_ARTIFACT_ROOT,
    PROBE_EXPERIMENT_ID,
    PROBE_RUN_ID_PATTERN,
    PROBE_SPECIFICATION_NAME,
    PROBE_SPECIFICATION_SHA256,
    REPRESENTATION_DIMENSION,
    RIDGE_ALPHA_GRID,
    SEALED_AT_UTC,
    SEALING_COMMIT,
    STRIDE,
    TEST_START_INCLUSIVE,
    TRAIN_ORDINALS,
    TRAIN_VALIDATION_ORIGIN_OFFSETS,
    VALIDATION_ORDINALS,
    verify_partition_geometry,
    verify_pinned_pair,
    verify_probe_specification,
)
from openalpha_bridge.representation_probe.test import (
    ControlComparison,
    OriginCluster,
    ProbeFinding,
    ProbeOutcome,
    decide,
    paired_moving_block_bootstrap,
    verify_test_eligibility,
)
from openalpha_bridge.representation_probe.worker import (
    run_probe_fit_worker,
    run_probe_test_worker,
)
from openalpha_bridge.zero_shot.spec import ZERO_SHOT_RUN_ID_PATTERN

REPO = Path(__file__).resolve().parents[3]
RESEARCH = REPO / "research" / "bridge-v0"
APP = REPO / "cloud" / "modal" / "bridge_phase2_app.py"
PACKAGE = Path(__file__).resolve().parents[1] / "src" / "openalpha_bridge" / "representation_probe"

COMMIT = "b" * 40
RUN_ID = "frp_0123456789abcdef"
NOW = datetime(2026, 8, 4, tzinfo=UTC)
TRAIN_VAL_SESSIONS = 340


# ====== doubles ======================================================


def _sessions(count: int, *, start: date) -> list[date]:
    out: list[date] = []
    day = start
    while len(out) < count:
        if day.weekday() < 5:
            out.append(day)
        day += timedelta(days=1)
    return out


def _rows(asset: str, count: int, *, start: date) -> tuple[OfficialRow, ...]:
    offset = sum(asset.encode()) % 13
    level = 100.0 + offset
    out: list[OfficialRow] = []
    for index, session in enumerate(_sessions(count, start=start)):
        level *= math.exp(0.0005 * math.sin((index + offset) / 7.0))
        out.append(
            OfficialRow(
                session=session,
                open=level * 0.999,
                high=level * 1.004,
                low=level * 0.996,
                close=level,
                volume=1_000_000.0 + index,
                amount=(1_000_000.0 + index) * level,
            )
        )
    return tuple(out)


def _candles(rows: tuple[OfficialRow, ...]) -> tuple[Candle, ...]:
    return tuple(
        Candle(
            session=r.session,
            open=r.open,
            high=r.high,
            low=r.low,
            close=r.close,
            volume=r.volume,
            amount=r.amount,
        )
        for r in rows
    )


class _Provider:
    """Serves the fit window, and optionally a synthetic test window."""

    name = "deterministic_fake"
    mode = ProviderMode.FAKE

    def __init__(self, *, sessions: int = TRAIN_VAL_SESSIONS + 5,
                 start: date = date(2025, 1, 2)) -> None:
        self._sessions = sessions
        self._start = start
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
            candles=_candles(_rows(request.symbol, self._sessions, start=self._start)),
        )


class _Extractor:
    """A deterministic stand-in producing a correctly shaped hidden state."""

    def __init__(self, *, dimension: int = REPRESENTATION_DIMENSION) -> None:
        self.dimension = dimension
        self.calls = 0
        self.context_lengths: list[int] = []
        self.states_seen: list[str] = []
        self.rows_seen: list[tuple[OfficialRow, ...]] = []

    def hidden_state(self, context, *, context_stamps, state) -> ExtractedRepresentation:
        self.calls += 1
        self.context_lengths.append(len(context))
        self.states_seen.append(state.state_sha256)
        self.rows_seen.append(context)
        if len(context_stamps) != len(context):
            raise AssertionError("one stamp per context row is required")
        seed = int(hashlib.sha256(state.state_sha256.encode()).hexdigest()[:8], 16)
        generator = np.random.default_rng(seed)
        # A weak but real signal so the estimators have something to fit.
        anchor = context[-1].close
        base = generator.normal(0.0, 1.0, self.dimension)
        base[0] = anchor / 100.0
        return ExtractedRepresentation(
            vector=tuple(float(v) for v in base),
            dimension=self.dimension,
            sequence_length=len(context),
            batch_size=1,
            encoded_token_count=len(context),
        )


class _Runtime:
    def __init__(self, extractor: _Extractor, digest: str = "d" * 64) -> None:
        self.extractor = extractor
        self._digest = digest
        self.assets = _assets()
        self.parameter_digest = lambda: self._digest


def _assets(**overrides: Any):
    from openalpha_bridge.diagnostic.backends import ResolvedDiagnosticAssets
    from openalpha_bridge.diagnostic.spec import OFFICIAL_SOURCE_FILES
    from openalpha_bridge.phase2.kronos import SOURCE_SPEC

    values: dict[str, Any] = {
        "tokenizer_repository": "NeoQuasar/Kronos-Tokenizer-base",
        "tokenizer_revision": "0e0117387f39004a9016484a186a908917e22426",
        "tokenizer_config_sha256": "2366e7ccfec76cbc19cf3c4c1b9c5d901be336ca1e83f2d2292c9bff381b77a2",
        "tokenizer_weights_sha256": "59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee",
        "model_repository": "NeoQuasar/Kronos-base",
        "model_revision": "2b554741eca47781b64468546e77fef3e85130e6",
        "model_config_sha256": "77ebc3038b647709b92be002f801d72e1a385f4c8c2c5aa1cc6cf21fcfe44eb2",
        "model_weights_sha256": "abff193acab6db1a0368e9773e75799d11403b6d054ee6d5f0a11aeabc5f4b83",
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


class _Resolver:
    def __init__(self, *, extractor: _Extractor | None = None, assets: Any = None,
                 digests: list[str] | None = None) -> None:
        self.extractor = extractor or _Extractor()
        self._assets = assets
        self._digests = digests
        self.calls = 0
        self.entered = 0

    def __call__(self) -> _Resolver:
        self.calls += 1
        return self

    def __enter__(self) -> _Runtime:
        self.entered += 1
        runtime = _Runtime(self.extractor)
        if self._assets is not None:
            runtime.assets = self._assets
        if self._digests is not None:
            sequence = iter(self._digests)
            runtime.parameter_digest = lambda: next(sequence)
        return runtime

    def __exit__(self, *exc: object) -> bool:
        return False


class _Factory:
    def __init__(self, provider: _Provider | None = None) -> None:
        self.calls = 0
        self.last = provider

    def __call__(self) -> _Provider:
        self.calls += 1
        if self.last is None:
            self.last = _Provider()
        return self.last


def _run_fit(*, store=None, resolver=None, factory=None, run_id: str = RUN_ID):
    store = store if store is not None else InMemoryObjectStore()
    resolver = resolver if resolver is not None else _Resolver()
    factory = factory if factory is not None else _Factory()
    result = run_probe_fit_worker(
        store=store,
        resolve_runtime=resolver,
        provider_factory=factory,
        research_root=RESEARCH,
        source_commit=COMMIT,
        deployed_commit=COMMIT,
        run_id=run_id,
        now=NOW,
    )
    return result, store, resolver, factory


def _context(count: int = CONTEXT_CANDLES) -> tuple[OfficialRow, ...]:
    return _rows("SPY", count, start=date(2025, 1, 2))


def _referenced_names(root: Path) -> set[str]:
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
    return names


PROBE_NAMES = _referenced_names(PACKAGE)


# ====== 1: split geometry ============================================


def test_the_split_geometry_tiles_the_origins_exactly_once() -> None:
    geometry = verify_partition_geometry()
    assert geometry == {
        "train_origins": 16,
        "embargo_origins": 4,
        "validation_origins": 5,
        "train_samples": 64,
        "validation_samples": 20,
        "stride": 12,
        "embargo_length": 4,
    }
    assert (*TRAIN_ORDINALS, *EMBARGO_ORDINALS, *VALIDATION_ORDINALS) == tuple(range(25))
    assert set(TRAIN_ORDINALS) & set(VALIDATION_ORDINALS) == set()
    assert TRAIN_VALIDATION_ORIGIN_OFFSETS[0] == CONTEXT_CANDLES
    assert TRAIN_VALIDATION_ORIGIN_OFFSETS[-1] == 328


def test_origins_are_non_overlapping_and_stride_equals_the_horizon() -> None:
    assert STRIDE == HORIZON_CANDLES == 12
    offsets = origin_offsets(TRAIN_VAL_SESSIONS)
    assert offsets == TRAIN_VALIDATION_ORIGIN_OFFSETS
    covered: list[int] = []
    for offset in offsets:
        covered.extend(range(offset, offset + HORIZON_CANDLES))
    assert len(covered) == len(set(covered)), "target rows must never be scored twice"


def test_windows_carry_exactly_forty_context_and_twelve_target_rows() -> None:
    windows = resolve_windows(asset="SPY", rows=_rows("SPY", TRAIN_VAL_SESSIONS,
                                                      start=date(2025, 1, 2)))
    assert len(windows) == 25
    for window in windows:
        assert len(window.context) == CONTEXT_CANDLES
        assert len(window.target) == HORIZON_CANDLES
        assert window.context[-1].session < window.target[0].session


def test_a_sequence_too_short_for_one_window_yields_no_origins() -> None:
    assert origin_offsets(CONTEXT_CANDLES + HORIZON_CANDLES - 1) == ()
    assert origin_offsets(CONTEXT_CANDLES + HORIZON_CANDLES) == (CONTEXT_CANDLES,)


# ====== 2: embargo behaviour =========================================


def _rows_of(window, kind: str) -> set[int]:
    if kind == "context":
        return set(range(window.origin_index - CONTEXT_CANDLES, window.origin_index))
    return set(range(window.origin_index, window.origin_index + HORIZON_CANDLES))


def test_no_validation_context_overlaps_a_training_target() -> None:
    """The direction that actually matters.

    Contexts look backwards, so a training origin's context can never reach a
    later validation target -- that separation is free and proves nothing. The
    real hazard is the reverse: a validation sample's 40-session context reaching
    back into rows that were training TARGETS, which would let the selection see
    outcomes it was fitted on.
    """
    windows = resolve_windows(asset="SPY", rows=_rows("SPY", TRAIN_VAL_SESSIONS,
                                                      start=date(2025, 1, 2)))
    training_targets: set[int] = set()
    for ordinal in TRAIN_ORDINALS:
        training_targets |= _rows_of(windows[ordinal], "target")

    for ordinal in VALIDATION_ORDINALS:
        context = _rows_of(windows[ordinal], "context")
        assert not (context & training_targets), (
            f"validation ordinal {ordinal} context reaches a training target"
        )


def test_without_the_embargo_the_validation_context_would_leak() -> None:
    """The embargo is load-bearing: removing it reintroduces the overlap."""
    windows = resolve_windows(asset="SPY", rows=_rows("SPY", TRAIN_VAL_SESSIONS,
                                                      start=date(2025, 1, 2)))
    # Suppose the four embargoed origins had been used for training instead.
    would_be_training_targets: set[int] = set()
    for ordinal in (*TRAIN_ORDINALS, *EMBARGO_ORDINALS):
        would_be_training_targets |= _rows_of(windows[ordinal], "target")

    first_validation_context = _rows_of(windows[VALIDATION_ORDINALS[0]], "context")
    assert first_validation_context & would_be_training_targets, (
        "if the embargo were removed the first validation context would overlap "
        "training targets; the four dropped origins are doing real work"
    )
    assert len(EMBARGO_ORDINALS) == math.ceil(CONTEXT_CANDLES / STRIDE) == 4


def test_no_training_target_is_also_a_validation_target() -> None:
    windows = resolve_windows(asset="SPY", rows=_rows("SPY", TRAIN_VAL_SESSIONS,
                                                      start=date(2025, 1, 2)))
    train = set().union(*(_rows_of(windows[o], "target") for o in TRAIN_ORDINALS))
    validation = set().union(*(_rows_of(windows[o], "target") for o in VALIDATION_ORDINALS))
    assert not (train & validation)


# ====== 3: representation dimensions =================================


def test_the_extractor_contract_requires_the_declared_dimension() -> None:
    assert REPRESENTATION_DIMENSION == 832
    assert FEATURE_SET_DIMENSIONS["kronos_hidden_state"] == 832
    extracted = _Extractor().hidden_state(
        _context(), context_stamps=tuple(official_stamp(r.session) for r in _context()),
        state=fit_context_state(_context()),
    )
    assert extracted.dimension == 832
    assert len(extracted.vector) == 832
    assert extracted.sequence_length == CONTEXT_CANDLES
    assert extracted.batch_size == 1


def test_a_wrongly_shaped_hidden_state_is_refused_by_the_official_extractor() -> None:
    """The real extractor validates rank, batch, sequence length and width."""
    from openalpha_bridge.representation_probe.extraction import OfficialHiddenStateExtractor

    class _FakeTensor:
        def __init__(self, shape): self.shape = shape
        def dim(self): return len(self.shape)

    class _BadModel:
        def __init__(self, shape): self._shape = shape
        def decode_s1(self, coarse, fine, stamp): return None, _FakeTensor(self._shape)

    class _Tok:
        def encode(self, x, half=True):
            class _Ids:
                shape = (1, CONTEXT_CANDLES)
            return (_Ids(), _Ids())

    pytest.importorskip("torch", reason="the official extractor needs torch")
    for shape, code in (
        ((1, CONTEXT_CANDLES, 512), "PROBE_HIDDEN_STATE_DIMENSION_MISMATCH"),
        ((2, CONTEXT_CANDLES, 832), "PROBE_HIDDEN_STATE_SHAPE_MISMATCH"),
        ((1, CONTEXT_CANDLES), "PROBE_HIDDEN_STATE_RANK_MISMATCH"),
    ):
        extractor = OfficialHiddenStateExtractor(
            model=_BadModel(shape), tokenizer=_Tok(), device="cpu"
        )
        rows = _context()
        with pytest.raises(BridgeTransformError) as excinfo:
            extractor.hidden_state(
                rows,
                context_stamps=tuple(official_stamp(r.session) for r in rows),
                state=fit_context_state(rows),
            )
        assert excinfo.value.failures[0].code == code


def test_every_control_produces_its_declared_dimension() -> None:
    context = _context()
    assert len(raw_ohlcv(context)) == FEATURE_SET_DIMENSIONS["raw_ohlcv"] == 240
    assert len(engineered_features(context)) == FEATURE_SET_DIMENSIONS["engineered_features"] == 12
    assert len(intercept_only(context)) == FEATURE_SET_DIMENSIONS["intercept_only"] == 0
    assert len(ENGINEERED_FEATURE_NAMES) == 12


def test_the_controls_cannot_receive_a_target_row() -> None:
    """Enforced by signature: there is no parameter through which one could arrive."""
    import inspect

    for builder in (raw_ohlcv, engineered_features, intercept_only):
        assert list(inspect.signature(builder).parameters) == ["context"]
    assert set(controls.CONTROL_BUILDERS) == {
        "raw_ohlcv", "engineered_features", "intercept_only"
    }


# ====== 4: estimator parity ==========================================


def test_the_harness_is_identical_across_feature_sets() -> None:
    """Same grids, same selection, same refit. Only the matrix differs."""
    result, _, _, _ = _run_fit()
    sets = result.payload["feature_sets"]
    assert set(sets) == set(FEATURE_SET_IDS)
    for name in FEATURE_SET_IDS:
        entry = sets[name]
        assert entry["dimension"] == FEATURE_SET_DIMENSIONS[name]
        assert set(entry["validation_metrics_by_alpha"]) == {repr(a) for a in RIDGE_ALPHA_GRID}
        assert set(entry["validation_directional_by_c"]) == {repr(c) for c in LOGISTIC_C_GRID}
        assert entry["selected_alpha"] in RIDGE_ALPHA_GRID
        assert entry["selected_inverse_regularization"] in LOGISTIC_C_GRID
        assert entry["refit_rows"] == 84  # 64 train + 20 validation


def test_ridge_is_the_exact_closed_form_solution() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(40, 5))
    y = x @ np.array([1.0, -2.0, 0.5, 0.0, 3.0]) + 0.25
    model = fit_ridge(x, y, alpha=1e-8)
    assert model.predict(x) == pytest.approx(y, abs=1e-6)

    strong = fit_ridge(x, y, alpha=1e9)
    # Heavy penalty shrinks coefficients to nothing and leaves the mean.
    assert max(abs(c) for c in strong.coefficients) < 1e-6
    assert strong.predict(x) == pytest.approx(np.full(40, y.mean()), abs=1e-6)


def test_ridge_with_no_features_predicts_the_training_mean() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    model = fit_ridge(np.zeros((4, 0)), y, alpha=1.0)
    assert model.dimension == 0
    assert model.predict(np.zeros((4, 0))) == pytest.approx(np.full(4, 2.5))


def test_logistic_separates_a_separable_problem_and_records_convergence() -> None:
    x = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    labels = np.array([-1.0, -1.0, 1.0, 1.0])
    model = fit_logistic(x, labels, inverse_regularization=1.0)
    assert model.iterations >= 1
    scores = model.decision_function(x)
    assert list(np.sign(scores)) == [-1.0, -1.0, 1.0, 1.0]


def test_the_estimators_are_deterministic() -> None:
    rng = np.random.default_rng(5)
    x, y = rng.normal(size=(30, 4)), rng.normal(size=30)
    labels = np.sign(y)
    assert fit_ridge(x, y, alpha=1.0) == fit_ridge(x, y, alpha=1.0)
    assert fit_logistic(x, labels, inverse_regularization=1.0) == fit_logistic(
        x, labels, inverse_regularization=1.0
    )


def test_the_published_standardizer_is_fitted_on_training_rows_only() -> None:
    """The refit changes coefficients, never the preprocessing state.

    The specification says features are standardized using training-partition
    statistics only, and that validation and test rows are transformed with
    those statistics and never refit. Refitting the standardizer on train plus
    validation would fold validation means and spreads into the state that later
    transforms the test partition -- a quiet path across a partition boundary.
    """
    result, _, _, _ = _run_fit()
    for name in FEATURE_SET_IDS:
        entry = result.payload["feature_sets"][name]
        preprocessing = entry["preprocessing"]
        # 64 training rows, not the 84 train-plus-validation refit rows.
        assert preprocessing["fitted_rows"] == 64, name
        assert entry["refit_rows"] == 84, name
        assert preprocessing["dimension"] == FEATURE_SET_DIMENSIONS[name]


def test_refitting_the_standardizer_would_change_the_published_state() -> None:
    """The distinction is real, not cosmetic: the two states differ."""
    rng = np.random.default_rng(3)
    train = rng.normal(0.0, 1.0, size=(64, 5))
    validation = rng.normal(4.0, 3.0, size=(20, 5))  # deliberately shifted
    train_only = Standardizer.fit(train)
    combined = Standardizer.fit(np.vstack([train, validation]))
    assert train_only.fitted_rows == 64
    assert combined.fitted_rows == 84
    assert train_only.mean != combined.mean
    assert train_only.scale != combined.scale


def test_standardization_uses_training_statistics_only() -> None:
    train = np.array([[0.0, 10.0], [2.0, 30.0]])
    later = np.array([[100.0, 100.0]])
    standardizer = Standardizer.fit(train)
    assert standardizer.transform(train).mean(axis=0) == pytest.approx([0.0, 0.0])
    # Later rows are transformed with the training statistics, never refit.
    expected = (later - np.asarray(standardizer.mean)) / np.asarray(standardizer.scale)
    assert standardizer.transform(later) == pytest.approx(expected)


def test_a_constant_column_does_not_divide_by_zero() -> None:
    standardizer = Standardizer.fit(np.array([[5.0, 1.0], [5.0, 2.0]]))
    transformed = standardizer.transform(np.array([[5.0, 1.5]]))
    assert bool(np.all(np.isfinite(transformed)))


# ====== 5: digest verification and artifact immutability =============


def test_the_sealed_specification_verifies() -> None:
    assert verify_probe_specification(RESEARCH) == {
        PROBE_SPECIFICATION_NAME: PROBE_SPECIFICATION_SHA256
    }
    path = RESEARCH / PROBE_SPECIFICATION_NAME
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PROBE_SPECIFICATION_SHA256
    assert b"\r\n" not in path.read_bytes()


def test_a_drifted_specification_fails_closed(tmp_path: Path) -> None:
    (tmp_path / PROBE_SPECIFICATION_NAME).write_bytes(b"experiment_id: tampered\n")
    with pytest.raises(BridgeTransformError) as excinfo:
        verify_probe_specification(tmp_path)
    assert excinfo.value.failures[0].code == "PROBE_SPECIFICATION_HASH_MISMATCH"


def test_the_pinned_pair_is_checked_against_literals() -> None:
    assert verify_pinned_pair()["model_revision"] == "2b554741eca47781b64468546e77fef3e85130e6"


def test_the_fit_artifact_is_immutable_and_verifies() -> None:
    result, store, _, _ = _run_fit()
    assert result.outcome == PROBE_FIT_CODE
    assert result.phase == "fit"
    assert result.already_existed is False
    key = result.artifact_key
    assert key.startswith(f"{PROBE_ARTIFACT_ROOT}/runs/{RUN_ID}/")

    stored = store.get(key)
    assert hashlib.sha256(stored.body).hexdigest() == stored.metadata.content_sha256
    assert result.content_sha256 == stored.metadata.content_sha256

    with pytest.raises(BridgeTransformError) as excinfo:
        store.put_immutable(key, b"{}", stored.metadata)
    assert excinfo.value.failures[0].code == "OBJECT_ALREADY_EXISTS"


def test_a_duplicate_fit_invocation_loads_nothing_and_fetches_nothing() -> None:
    _, store, first, first_factory = _run_fit()
    assert (first.calls, first.entered, first_factory.calls) == (1, 1, 1)

    second_resolver, second_factory = _Resolver(), _Factory()
    result, _, _, _ = _run_fit(store=store, resolver=second_resolver, factory=second_factory)
    assert result.already_existed is True
    assert (second_resolver.calls, second_resolver.entered) == (0, 0)
    assert second_factory.calls == 0
    assert second_resolver.extractor.calls == 0


# ====== 6: premature-test refusal ====================================


def test_the_test_phase_refuses_before_the_fit_artifact_exists() -> None:
    store = InMemoryObjectStore()
    invocation = ProbeInvocation.validate_all(
        run_id=RUN_ID, source_commit=COMMIT, deployed_commit=COMMIT
    )
    with pytest.raises(BridgeTransformError) as excinfo:
        load_verified_fit_artifact(store, invocation=invocation)
    assert excinfo.value.failures[0].code == "PROBE_FIT_ARTIFACT_MISSING"


def test_the_test_phase_refuses_a_fit_artifact_with_the_wrong_digest() -> None:
    result, store, _, _ = _run_fit()
    invocation = ProbeInvocation.validate_all(
        run_id=RUN_ID, source_commit=COMMIT, deployed_commit=COMMIT
    )
    load_verified_fit_artifact(store, invocation=invocation,
                               expected_digest=result.content_sha256)
    with pytest.raises(BridgeTransformError) as excinfo:
        load_verified_fit_artifact(store, invocation=invocation, expected_digest="f" * 64)
    assert excinfo.value.failures[0].code == "PROBE_FIT_ARTIFACT_DIGEST_MISMATCH"


@pytest.mark.parametrize(
    ("sessions", "origins"),
    [(MINIMUM_TEST_SESSIONS - 1, MINIMUM_TEST_ORIGINS_PER_ASSET),
     (MINIMUM_TEST_SESSIONS, MINIMUM_TEST_ORIGINS_PER_ASSET - 1)],
)
def test_the_test_partition_refuses_to_open_early(sessions: int, origins: int) -> None:
    boundary = date.fromisoformat(TEST_START_INCLUSIVE)
    with pytest.raises(BridgeTransformError) as excinfo:
        verify_test_eligibility(
            sessions_by_asset={a: sessions for a in ASSET_PANEL},
            first_session_by_asset={a: boundary for a in ASSET_PANEL},
            origins_by_asset={a: origins for a in ASSET_PANEL},
        )
    assert excinfo.value.failures[0].code == "PROBE_TEST_PARTITION_NOT_READY"
    assert "may not be reduced or waived" in excinfo.value.failures[0].message or (
        "complete non-overlapping origins" in excinfo.value.failures[0].message
    )


def test_test_data_before_the_sealed_boundary_is_refused() -> None:
    boundary = date.fromisoformat(TEST_START_INCLUSIVE)
    with pytest.raises(BridgeTransformError) as excinfo:
        verify_test_eligibility(
            sessions_by_asset={a: MINIMUM_TEST_SESSIONS for a in ASSET_PANEL},
            first_session_by_asset={a: boundary - timedelta(days=1) for a in ASSET_PANEL},
            origins_by_asset={a: MINIMUM_TEST_ORIGINS_PER_ASSET for a in ASSET_PANEL},
        )
    assert excinfo.value.failures[0].code == "PROBE_TEST_BOUNDARY_VIOLATED"


def test_an_eligible_panel_passes_the_gate() -> None:
    boundary = date.fromisoformat(TEST_START_INCLUSIVE)
    eligibility = verify_test_eligibility(
        sessions_by_asset={a: MINIMUM_TEST_SESSIONS for a in ASSET_PANEL},
        first_session_by_asset={a: boundary for a in ASSET_PANEL},
        origins_by_asset={a: MINIMUM_TEST_ORIGINS_PER_ASSET for a in ASSET_PANEL},
    )
    assert eligibility.eligible is True
    assert eligibility.total_test_samples == MINIMUM_TEST_SAMPLES == 64


def test_a_premature_real_test_run_publishes_a_typed_failure_and_reads_no_test_rows() -> None:
    """End to end: the gate fires and the run stops, without scoring anything."""
    result, store, _, _ = _run_fit()
    resolver = _Resolver()
    # The provider serves far too few post-boundary sessions.
    factory = _Factory(_Provider(sessions=60, start=date.fromisoformat(TEST_START_INCLUSIVE)))
    outcome = run_probe_test_worker(
        store=store,
        resolve_runtime=resolver,
        provider_factory=factory,
        research_root=RESEARCH,
        source_commit=COMMIT,
        deployed_commit=COMMIT,
        run_id=RUN_ID,
        expected_fit_digest=result.content_sha256,
        now=NOW,
    )
    assert outcome.outcome == PROBE_FAILURE_CODE
    assert outcome.payload["failure_code"] == "PROBE_TEST_PARTITION_NOT_READY"
    assert outcome.payload["phase"] == "test"
    assert outcome.payload["test_partition_opened"] is False
    # No scoring occurred: the extractor was never asked for a test row.
    assert resolver.extractor.calls == 0


# ====== 7: fit-artifact mismatch refusal =============================


def test_the_test_phase_refuses_a_fit_from_a_different_run() -> None:
    _, store, _, _ = _run_fit(run_id="frp_aaaaaaaaaaaaaaaa")
    other = ProbeInvocation.validate_all(
        run_id="frp_bbbbbbbbbbbbbbbb", source_commit=COMMIT, deployed_commit=COMMIT
    )
    with pytest.raises(BridgeTransformError) as excinfo:
        load_verified_fit_artifact(store, invocation=other)
    assert excinfo.value.failures[0].code == "PROBE_FIT_ARTIFACT_MISSING"


def test_the_test_phase_refuses_a_failed_fit_artifact() -> None:
    """A fit that did not succeed cannot be the basis of a test run."""
    resolver = _Resolver(assets=_assets(model_repository="NeoQuasar/Kronos-mini"))
    result, store, _, _ = _run_fit(resolver=resolver)
    assert result.outcome == PROBE_FAILURE_CODE

    invocation = ProbeInvocation.validate_all(
        run_id=RUN_ID, source_commit=COMMIT, deployed_commit=COMMIT
    )
    with pytest.raises(BridgeTransformError) as excinfo:
        load_verified_fit_artifact(store, invocation=invocation)
    assert excinfo.value.failures[0].code == "PROBE_FIT_ARTIFACT_NOT_SUCCESSFUL"


def test_the_fit_artifact_records_that_it_did_not_open_the_test_partition() -> None:
    result, _, _, _ = _run_fit()
    assert result.payload["test_partition_opened"] is False
    assert result.payload["training_performed"] is False
    assert result.payload["kronos_parameters_updated"] is False


# ====== identity, isolation and authorization ========================


@pytest.mark.parametrize(
    "run_id",
    ["canary_0a92fde788bd685c", "base_03b08cbc706193d6", "zsb_25e0256eefb2b07a"],
)
def test_a_foreign_run_identifier_is_refused(run_id: str) -> None:
    with pytest.raises(BridgeTransformError) as excinfo:
        ProbeInvocation.validate_all(
            run_id=run_id, source_commit=COMMIT, deployed_commit=COMMIT
        )
    assert "REFUSED_BY_REPRESENTATION_PROBE" in excinfo.value.failures[0].code


def test_the_four_run_identifier_spaces_are_pairwise_disjoint() -> None:
    patterns = (MINI_RUN_ID_PATTERN, BASE_RUN_ID_PATTERN, ZERO_SHOT_RUN_ID_PATTERN,
                PROBE_RUN_ID_PATTERN)
    for sample in ("canary_0a92fde788bd685c", "base_03b08cbc706193d6",
                   "zsb_25e0256eefb2b07a", "frp_0123456789abcdef"):
        assert sum(bool(p.fullmatch(sample)) for p in patterns) == 1, sample


def test_the_probe_namespace_cannot_collide_with_any_completed_study() -> None:
    for root in (
        "openalpha-compatibility/bridge-phase2",
        "openalpha-compatibility/kronos-base-diagnostic",
        "openalpha-compatibility/kronos-zero-shot-benchmark",
    ):
        assert not PROBE_ARTIFACT_ROOT.startswith(root)
        assert not root.startswith(PROBE_ARTIFACT_ROOT)


def test_no_probe_module_can_train_kronos_or_download_a_weight() -> None:
    forbidden = {
        "snapshot_download", "hf_hub_download", "from_pretrained",
        "backward", "zero_grad", "requires_grad_", "AdamW", "Adam", "SGD",
        "optimizer", "Optimizer", "lr_scheduler", "state_dict", "load_state_dict",
        "CloudRunner", "TorchTrainingBackend",
    }
    assert not (PROBE_NAMES & forbidden)


def test_the_probe_never_generates_or_repairs() -> None:
    """Generation-specific machinery must not leak into a representation study."""
    forbidden = {
        "generate", "GeneratedPath", "project_path", "ProjectionOutcome",
        "path_validity", "PROJECTION_METHOD", "run_method_a", "run_method_b",
        "run_method_c", "run_method_d", "top_k_top_p_filtering", "multinomial",
    }
    assert not (PROBE_NAMES & forbidden)


def test_no_probe_result_authorizes_anything() -> None:
    result, _, _, _ = _run_fit()
    for field in (
        "authorizes_training", "authorizes_fine_tuning", "authorizes_supervised_adaptation",
        "authorizes_production_inference", "authorizes_trading_claims",
        "scientific_result_available",
    ):
        assert getattr(result, field) is False
        assert result.payload[field] is False


def test_the_fit_never_reaches_the_test_window() -> None:
    result, _, resolver, _ = _run_fit()
    boundary = date.fromisoformat(TEST_START_INCLUSIVE)
    for retrieval in result.payload["retrievals"]:
        assert date.fromisoformat(retrieval["last_session_used"]) < boundary
    for context in resolver.extractor.rows_seen:
        assert max(row.session for row in context) < boundary


def test_normalization_is_refit_per_origin_from_context_rows_only() -> None:
    result, _, resolver, _ = _run_fit()
    assert set(resolver.extractor.context_lengths) == {CONTEXT_CANDLES}
    # One distinct normalization state per asset-origin.
    assert len(set(resolver.extractor.states_seen)) == resolver.extractor.calls
    assert result.payload["train_samples"] == 64
    assert result.payload["validation_samples"] == 20


def test_a_modified_parameter_digest_fails_the_fit() -> None:
    resolver = _Resolver(digests=["a" * 64, "b" * 64])
    result, _, _, _ = _run_fit(resolver=resolver)
    assert result.outcome == PROBE_FAILURE_CODE
    assert result.payload["failure_code"] == "PROBE_PARAMETERS_MODIFIED"


# ====== targets, bootstrap and decision ==============================


def test_the_target_is_the_cumulative_log_return_over_the_horizon() -> None:
    windows = resolve_windows(asset="SPY", rows=_rows("SPY", TRAIN_VAL_SESSIONS,
                                                      start=date(2025, 1, 2)))
    window = windows[0]
    expected = math.log(window.target[-1].close / window.context[-1].close)
    assert cumulative_log_return(window) == pytest.approx(expected)
    sample = build_sample(window)
    assert sample.target_return == pytest.approx(expected)
    assert sample.target_direction in (-1, 0, 1)
    assert sample.target_direction == (1 if expected > 0 else (-1 if expected < 0 else 0))


def test_cross_asset_alignment_is_required() -> None:
    good = {
        a: resolve_windows(asset=a, rows=_rows(a, TRAIN_VAL_SESSIONS, start=date(2025, 1, 2)))
        for a in ASSET_PANEL
    }
    assert verify_cross_asset_alignment(good)["ordinals_checked"] == 25

    shifted = dict(good)
    shifted["DIA"] = resolve_windows(
        asset="DIA", rows=_rows("DIA", TRAIN_VAL_SESSIONS, start=date(2025, 1, 5))
    )
    with pytest.raises(BridgeTransformError) as excinfo:
        verify_cross_asset_alignment(shifted)
    assert excinfo.value.failures[0].code == "PROBE_CROSS_ASSET_ORIGIN_MISALIGNED"


def _clusters(values: list[tuple[float, float, float, float]]) -> tuple[OriginCluster, ...]:
    return tuple(
        OriginCluster(ordinal=i, assets=ASSET_PANEL, paired_differences=v)
        for i, v in enumerate(values)
    )


def test_the_bootstrap_is_deterministic_and_clusters_whole_ordinals() -> None:
    values = [(0.001, 0.001, 0.001, 0.001)] * MINIMUM_TEST_ORIGINS_PER_ASSET
    first = paired_moving_block_bootstrap(_clusters(values))
    second = paired_moving_block_bootstrap(_clusters(values))
    assert first == second
    assert first.block_length == 4
    assert first.cluster_count == 16
    assert first.observation_count == 64
    assert first.assets_per_cluster == 4
    assert first.wraparound is False
    assert first.origins_resampled_independently is False
    assert first.seed == 20260803


def test_the_bootstrap_refuses_a_foreign_asset_panel() -> None:
    bad = list(_clusters([(0.1, 0.1, 0.1, 0.1)] * MINIMUM_TEST_ORIGINS_PER_ASSET))
    bad[0] = OriginCluster(
        ordinal=0, assets=("SPY", "QQQ", "IWM", "VOO"), paired_differences=(0.1, 0.1, 0.1, 0.1)
    )
    with pytest.raises(BridgeTransformError) as excinfo:
        paired_moving_block_bootstrap(tuple(bad))
    assert excinfo.value.failures[0].code == "PROBE_BOOTSTRAP_ASSET_PANEL_MISMATCH"


def _comparison(control: str, *, beats: bool) -> ControlComparison:
    values = [(0.01, 0.01, 0.01, 0.01)] * MINIMUM_TEST_ORIGINS_PER_ASSET
    interval = paired_moving_block_bootstrap(_clusters(values))
    return ControlComparison(
        control=control,
        candidate_primary_error=0.9 if beats else 1.0,
        control_primary_error=1.0,
        relative_improvement=0.10 if beats else 0.0,
        meets_margin=beats,
        interval=interval,
        interval_favorable=beats,
        beats_control=beats,
    )


def test_r1_requires_every_control_and_three_assets() -> None:
    all_beaten = tuple(
        _comparison(c, beats=True)
        for c in ("raw_ohlcv", "engineered_features", "intercept_only")
    )
    decision = decide(comparisons=all_beaten, supporting_assets=("SPY", "QQQ", "IWM"))
    assert ProbeFinding.REPRESENTATION_ADVANTAGE_OBSERVED in decision.matched_findings
    assert decision.outcome is ProbeOutcome.PROCEED_TO_CONFIRMATION_ON_FRESH_ORIGINS

    too_few_assets = decide(comparisons=all_beaten, supporting_assets=("SPY", "QQQ"))
    assert ProbeFinding.REPRESENTATION_ADVANTAGE_OBSERVED not in too_few_assets.matched_findings


def test_r2_matches_when_only_the_engineered_control_survives() -> None:
    decision = decide(
        comparisons=(
            _comparison("raw_ohlcv", beats=True),
            _comparison("intercept_only", beats=True),
            _comparison("engineered_features", beats=False),
        ),
        supporting_assets=("SPY", "QQQ", "IWM"),
    )
    assert ProbeFinding.ADVANTAGE_OVER_RAW_ONLY in decision.matched_findings
    assert decision.outcome is ProbeOutcome.PROCEED_TO_CONFIRMATION_ON_FRESH_ORIGINS


def test_r3_stops_the_direction() -> None:
    decision = decide(
        comparisons=tuple(
            _comparison(c, beats=False)
            for c in ("raw_ohlcv", "engineered_features", "intercept_only")
        ),
        supporting_assets=(),
    )
    assert ProbeFinding.NO_REPRESENTATION_ADVANTAGE in decision.matched_findings
    assert decision.outcome is ProbeOutcome.STOP_KRONOS_REPRESENTATION_DIRECTION
    assert decision.authorizes_fine_tuning is False
    assert "not that" in decision.power_qualification


def test_r0_takes_precedence_over_everything() -> None:
    decision = decide(
        comparisons=tuple(
            _comparison(c, beats=True)
            for c in ("raw_ohlcv", "engineered_features", "intercept_only")
        ),
        supporting_assets=ASSET_PANEL,
        limitations=("the test partition did not reach its minimum size",),
    )
    assert decision.outcome is ProbeOutcome.PROBE_INCONCLUSIVE
    assert ProbeFinding.REPRESENTATION_ADVANTAGE_OBSERVED not in decision.matched_findings


def test_structural_validity_never_reaches_a_decision() -> None:
    decision = decide(
        comparisons=(_comparison("raw_ohlcv", beats=False),), supporting_assets=()
    )
    assert decision.structural_validity_used_in_any_rule is False
    for evaluation in decision.evaluations:
        joined = " ".join(evaluation.observed).lower()
        for word in ("invalid", "structural", "candle", "projection", "repair"):
            assert word not in joined


# ====== modal wiring =================================================


def test_the_modal_app_exposes_the_five_probe_functions() -> None:
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    functions = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for name in (
        "verify_representation_probe_deployment",
        "verify_representation_probe_runtime",
        "fit_frozen_representation_probe",
        "test_frozen_representation_probe",
        "inventory_representation_probe_artifacts",
    ):
        assert name in functions
    # The completed studies' functions are untouched.
    for name in (
        "verify_base_deployment", "kronos_base_frozen_inference_diagnostic",
        "run_zero_shot_benchmark", "verify_zero_shot_benchmark_deployment",
    ):
        assert name in functions


def test_the_probe_adds_no_new_download_path_or_volume() -> None:
    source = APP.read_text(encoding="utf-8")
    tree = ast.parse(source)
    downloading: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for inner in ast.walk(node):
                if isinstance(inner, ast.Name) and inner.id == "snapshot_download":
                    downloading.add(node.name)
    assert downloading == {
        "frozen_inference_diagnostic", "verify_frozen_inference_runtime",
        "_download_base_pair", "resolve_runtime",
    }
    assert source.count("modal.Volume.from_name(") == 2

    probe = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_probe_runtime"
    )
    body = ast.get_source_segment(source, probe) or ""
    assert "_download_base_pair(cache_root)" in body
    assert "snapshot_download(" not in body


def test_the_probe_loader_uses_the_official_return_order() -> None:
    """load_and_freeze_official returns (tokenizer, model); swapping them is silent."""
    source = APP.read_text(encoding="utf-8")
    tree = ast.parse(source)
    probe = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_probe_runtime"
    )
    body = ast.get_source_segment(source, probe) or ""
    assert "tokenizer, model, (total, trainable) = load_and_freeze_official(" in body
    assert "OfficialHiddenStateExtractor(\n                        model=model, tokenizer=tokenizer" in body


def test_the_sealed_boundary_and_minimums_are_what_the_specification_says() -> None:
    document = (RESEARCH / PROBE_SPECIFICATION_NAME).read_text(encoding="utf-8")
    assert f"start_inclusive: '{TEST_START_INCLUSIVE}'" in document
    assert f"sessions_required_after_start: {MINIMUM_TEST_SESSIONS}" in document
    assert f"minimum_test_samples: {MINIMUM_TEST_SAMPLES}" in document
    assert f"sealed_by_commit: {SEALING_COMMIT}" in document
    assert f"sealed_at_utc: '{SEALED_AT_UTC}'" in document
    assert TEST_START_INCLUSIVE == "2026-08-05"
    assert MINIMUM_TEST_SESSIONS == 232
    assert PROBE_EXPERIMENT_ID == "openalpha-kronos-frozen-representation-probe-v1"
