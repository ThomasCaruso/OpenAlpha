from __future__ import annotations

import logging
import math
import random
from datetime import UTC, date, datetime
from unittest import mock

import pytest
from openalpha_research.calendars import CalendarName, sessions_in_half_open_range
from openalpha_research.failures import (
    FailureCategory,
    ResearchFailure,
    ResearchFailureError,
)
from openalpha_research.identity import canonical_json, canonical_sha256, file_sha256
from openalpha_research.objectstore import (
    InMemoryObjectStore,
    ObjectMetadata,
    S3CompatibleObjectStore,
    get_json,
    put_json,
)
from openalpha_research.protocols import verify_sha256_files
from openalpha_research.providers import (
    Candle,
    DeterministicFakeProvider,
    MarketSeries,
    ProviderMode,
    RetrievalRequest,
    validate_series,
)
from openalpha_research.redaction import REDACTED, SecretRedactor
from openalpha_research.resampling import (
    block_start_count,
    blocks_per_resample,
    moving_block_percentile_interval,
)
from openalpha_research.runtime import directory_bytes, gpu_snapshot, scoped_timer
from openalpha_research.safe_logging import log_operational_failure


def _series(*sessions: date) -> MarketSeries:
    candles = tuple(
        Candle(
            session=session,
            open=100.0 + index,
            high=102.0 + index,
            low=99.0 + index,
            close=101.0 + index,
            volume=1_000.0 + index,
            amount=100_500.0 + index,
        )
        for index, session in enumerate(sessions)
    )
    return MarketSeries(
        symbol="TEST",
        interval="1d",
        provider="fixture",
        provider_mode=ProviderMode.FAKE,
        client_version="fixture-1",
        retrieval_timestamp=None,
        candles=candles,
    )


def test_core_names_and_schemas_are_subject_neutral() -> None:
    failure = ResearchFailure(
        category=FailureCategory.INVALID_CONFIGURATION,
        code="LOCK_MISMATCH",
        message="sealed input changed",
    )
    assert failure.schema_version == "openalpha.research.failure.v1"
    assert "bridge" not in failure.model_dump_json().lower()
    assert "kronos" not in failure.model_dump_json().lower()


def test_failure_error_preserves_all_structured_failures() -> None:
    first = ResearchFailure(
        category=FailureCategory.INVALID_CONFIGURATION,
        code="FIRST",
        message="first failure",
    )
    second = ResearchFailure(
        category=FailureCategory.INVALID_CONFIGURATION,
        code="SECOND",
        message="second failure",
    )
    error = ResearchFailureError(first, second)
    assert error.failures == (first, second)
    assert "FIRST" in str(error)


def test_canonical_identity_is_order_independent_and_file_based(tmp_path) -> None:
    left = {"b": [2, 3], "a": 1}
    right = {"a": 1, "b": [2, 3]}
    expected = b'{"a":1,"b":[2,3]}'
    assert canonical_json(left) == expected
    assert canonical_json(right) == expected
    assert canonical_sha256(left) == canonical_sha256(right)

    target = tmp_path / "payload.json"
    target.write_bytes(expected)
    assert file_sha256(target) == canonical_sha256(left)


def test_canonical_identity_rejects_non_finite_numbers() -> None:
    with pytest.raises(ValueError):
        canonical_json({"value": math.nan})


def test_protocol_verification_uses_the_callers_mapping(tmp_path) -> None:
    target = tmp_path / "study.yaml"
    target.write_bytes(b"sealed\n")
    digest = file_sha256(target)
    assert verify_sha256_files(tmp_path, {"study.yaml": digest}) == {"study.yaml": digest}


def test_protocol_verification_fails_closed_on_missing_or_changed_files(tmp_path) -> None:
    with pytest.raises(ResearchFailureError) as missing:
        verify_sha256_files(tmp_path, {"missing.yaml": "0" * 64})
    assert missing.value.failures[0].code == "MISSING_PROTOCOL_FILE"

    target = tmp_path / "changed.yaml"
    target.write_bytes(b"changed\n")
    with pytest.raises(ResearchFailureError) as changed:
        verify_sha256_files(tmp_path, {"changed.yaml": "0" * 64})
    assert changed.value.failures[0].code == "PROTOCOL_HASH_MISMATCH"


def test_protocol_verification_confines_caller_paths_to_root(tmp_path) -> None:
    outside = tmp_path.parent / "outside-protocol.yaml"
    outside.write_bytes(b"outside\n")
    with pytest.raises(ResearchFailureError) as excinfo:
        verify_sha256_files(tmp_path, {str(outside): file_sha256(outside)})
    assert excinfo.value.failures[0].code == "PROTOCOL_PATH_OUTSIDE_ROOT"


def test_object_store_accepts_a_caller_owned_evidence_class() -> None:
    store = InMemoryObjectStore()
    metadata = put_json(
        store,
        "study/runs/run_1/result.json",
        {"result": 1},
        schema_version="study.result.v1",
        run_id="run_1",
        experiment_hash="0" * 64,
        evidence_class="development",
    )
    assert metadata.evidence_class == "development"
    assert get_json(store, "study/runs/run_1/result.json") == {"result": 1}
    assert store.list_keys("study/runs/") == ("study/runs/run_1/result.json",)


def test_object_store_enforces_immutable_writes_and_supports_overwrite() -> None:
    store = InMemoryObjectStore()
    put_json(
        store,
        "result.json",
        {"result": 1},
        schema_version="study.result.v1",
        run_id="run_1",
        experiment_hash="0" * 64,
        evidence_class="development",
    )
    with pytest.raises(ResearchFailureError) as excinfo:
        put_json(
            store,
            "result.json",
            {"result": 2},
            schema_version="study.result.v1",
            run_id="run_1",
            experiment_hash="0" * 64,
            evidence_class="development",
        )
    assert excinfo.value.failures[0].code == "OBJECT_ALREADY_EXISTS"

    put_json(
        store,
        "result.json",
        {"result": 2},
        schema_version="study.result.v1",
        run_id="run_1",
        experiment_hash="0" * 64,
        evidence_class="development",
        immutable=False,
    )
    assert get_json(store, "result.json") == {"result": 2}
    store.delete("result.json")
    assert not store.exists("result.json")


def test_s3_store_does_not_import_its_optional_dependency_on_construction() -> None:
    store = S3CompatibleObjectStore(bucket="research")
    assert store.bucket == "research"


def test_object_store_rejects_body_metadata_hash_mismatch_before_persistence() -> None:
    store = InMemoryObjectStore()
    metadata = ObjectMetadata(
        schema_version="study.result.v1",
        run_id="run_1",
        experiment_hash="0" * 64,
        content_sha256="0" * 64,
        created_at=datetime.now(UTC),
        evidence_class="development",
    )
    with pytest.raises(ResearchFailureError) as excinfo:
        store.put_immutable("result.json", b'{}', metadata)
    assert excinfo.value.failures[0].code == "OBJECT_HASH_MISMATCH"
    assert not store.exists("result.json")


def test_s3_store_rejects_objects_with_missing_provenance_metadata() -> None:
    class Body:
        def read(self) -> bytes:
            return b'{}'

    class MetadataFreeClient:
        def get_object(self, **_: object) -> dict[str, object]:
            return {"Body": Body(), "Metadata": {}}

    store = S3CompatibleObjectStore(bucket="research")
    store._client = MetadataFreeClient()
    with pytest.raises(ResearchFailureError) as excinfo:
        store.get("result.json")
    assert excinfo.value.failures[0].code == "OBJECT_METADATA_INVALID"


def test_provider_models_and_validation_are_subject_neutral() -> None:
    request = RetrievalRequest(
        symbol="TEST",
        start=date(2025, 1, 2),
        end=date(2025, 1, 4),
        maximum_candles=10,
    )
    assert request.interval == "1d"

    series = _series(date(2025, 1, 2), date(2025, 1, 3))
    assert validate_series(series) is None


def test_non_monotonic_market_sessions_fail_closed() -> None:
    series = _series(date(2025, 1, 3), date(2025, 1, 2))
    with pytest.raises(ResearchFailureError) as excinfo:
        validate_series(series)
    assert excinfo.value.failures[0].code == "NON_MONOTONIC_SESSIONS"


def test_market_series_rejects_invalid_price_bounds_and_calendar_drift() -> None:
    candle = Candle(
        session=date(2025, 1, 2),
        open=100.0,
        high=99.0,
        low=98.0,
        close=99.0,
        volume=1_000.0,
        amount=99_000.0,
    )
    series = _series(date(2025, 1, 2)).model_copy(update={"candles": (candle,)})
    with pytest.raises(ResearchFailureError) as invalid_high:
        validate_series(series)
    assert invalid_high.value.failures[0].code == "INVALID_HIGH"

    valid = _series(date(2025, 1, 2))
    with pytest.raises(ResearchFailureError) as calendar_mismatch:
        validate_series(valid, expected_sessions=(date(2025, 1, 3),))
    assert calendar_mismatch.value.failures[0].code == "CALENDAR_MISMATCH"


def test_deterministic_fake_provider_is_repeatable() -> None:
    request = RetrievalRequest(
        symbol="TEST",
        start=date(2025, 1, 2),
        end=date(2025, 1, 8),
        maximum_candles=10,
    )
    first = DeterministicFakeProvider(seed=11).fetch(request)
    second = DeterministicFakeProvider(seed=11).fetch(request)
    assert first == second
    assert first.provider_mode is ProviderMode.FAKE
    validate_series(first)


def test_deterministic_fake_provider_preserves_the_original_compatibility_vector() -> None:
    request = RetrievalRequest(
        symbol="TEST",
        start=date(2025, 1, 2),
        end=date(2025, 1, 8),
        maximum_candles=10,
    )
    series = DeterministicFakeProvider(seed=11).fetch(request)
    assert series.normalized_sha256 == (
        "3aafa56ae13a13ef101b0dc9a31664bfffbbb435848c07f6b61fa0d052c453c0"
    )
    assert series.candles[0] == Candle(
        session=date(2025, 1, 2),
        open=139.44023590029812,
        high=140.90620657804592,
        low=138.37564548342743,
        close=139.83042385125935,
        volume=1806155.8126559674,
        amount=252208216.47117415,
    )


def test_exchange_calendar_helpers_support_weekdays_and_continuous_days() -> None:
    assert sessions_in_half_open_range(
        date(2025, 1, 3), date(2025, 1, 7), calendar=CalendarName.XNYS
    ) == (date(2025, 1, 3), date(2025, 1, 6))
    assert sessions_in_half_open_range(
        date(2025, 1, 3), date(2025, 1, 7), calendar=CalendarName.CONTINUOUS
    ) == (
        date(2025, 1, 3),
        date(2025, 1, 4),
        date(2025, 1, 5),
        date(2025, 1, 6),
    )


def test_moving_block_arithmetic_is_generic_and_deterministic() -> None:
    totals = [0.0, 0.004, -0.002, 0.006, 0.002, -0.004, 0.008, 0.0]
    kwargs = {
        "observations": 16,
        "block_length": 2,
        "seed": 17,
        "resamples": 200,
        "confidence_level": 0.9,
    }
    first = moving_block_percentile_interval(totals, **kwargs)
    second = moving_block_percentile_interval(totals, **kwargs)
    assert first == second
    assert first.cluster_count == 8
    assert first.observation_count == 16
    assert first.available_block_starts == 7
    assert first.blocks_drawn_per_resample == 4
    assert first.point_estimate == pytest.approx(sum(totals) / 16)
    assert first.lower <= first.upper


@pytest.mark.parametrize("total", [0.004, -0.01, 0.0])
def test_constant_moving_block_panel_reproduces_its_mean(total: float) -> None:
    interval = moving_block_percentile_interval(
        [total] * 8,
        observations=16,
        block_length=2,
        seed=3,
        resamples=100,
        confidence_level=0.9,
    )
    expected = total / 2
    assert interval.point_estimate == pytest.approx(expected)
    assert interval.lower == pytest.approx(expected)
    assert interval.upper == pytest.approx(expected)


def test_moving_block_draws_block_starts_for_each_required_block() -> None:
    calls: list[int] = []
    original = random.Random.randrange

    def recording(generator: random.Random, *args: int) -> int:
        calls.append(args[0])
        return original(generator, *args)

    with mock.patch.object(random.Random, "randrange", recording):
        moving_block_percentile_interval(
            [0.001] * 8,
            observations=16,
            block_length=2,
            seed=3,
            resamples=10,
            confidence_level=0.9,
        )
    assert set(calls) == {7}
    assert len(calls) == 40


def test_moving_block_never_wraps_around() -> None:
    totals = [0.0] * 9
    totals[-1] = 1.0
    interval = moving_block_percentile_interval(
        totals,
        observations=18,
        block_length=4,
        seed=3,
        resamples=2_000,
        confidence_level=0.9,
    )
    assert interval.available_block_starts == 6
    assert interval.upper <= 2.0 / 18 + 1e-12


def test_moving_block_geometry_helpers_match_contiguous_arithmetic() -> None:
    assert block_start_count(8, 2) == 7
    assert block_start_count(9, 4) == 6
    assert blocks_per_resample(8, 2) == 4
    assert blocks_per_resample(9, 4) == 3


def test_moving_block_rejects_non_finite_totals() -> None:
    totals = [0.001] * 8
    totals[3] = math.nan
    with pytest.raises(ResearchFailureError) as excinfo:
        moving_block_percentile_interval(
            totals,
            observations=16,
            block_length=2,
            seed=3,
            resamples=10,
            confidence_level=0.9,
        )
    assert excinfo.value.failures[0].code == "RESAMPLING_NON_FINITE_TOTAL"


@pytest.mark.parametrize(
    ("override", "code"),
    [
        ({"block_length": 0}, "RESAMPLING_BLOCK_LENGTH_INVALID"),
        ({"block_length": 9}, "RESAMPLING_BLOCK_LENGTH_INVALID"),
        ({"resamples": 0}, "RESAMPLING_RESAMPLES_INVALID"),
        ({"confidence_level": 1.0}, "RESAMPLING_CONFIDENCE_INVALID"),
        ({"observations": 0}, "RESAMPLING_OBSERVATIONS_INVALID"),
    ],
)
def test_moving_block_core_validates_required_arithmetic(
    override: dict[str, int | float], code: str
) -> None:
    arguments: dict[str, int | float] = {
        "observations": 16,
        "block_length": 2,
        "seed": 3,
        "resamples": 10,
        "confidence_level": 0.9,
    }
    arguments.update(override)
    with pytest.raises(ResearchFailureError) as excinfo:
        moving_block_percentile_interval(
            [0.004] * 8,
            observations=int(arguments["observations"]),
            block_length=int(arguments["block_length"]),
            seed=int(arguments["seed"]),
            resamples=int(arguments["resamples"]),
            confidence_level=float(arguments["confidence_level"]),
        )
    assert excinfo.value.failures[0].code == code


def test_redaction_scrubs_registered_values_and_credential_shapes() -> None:
    redactor = SecretRedactor()
    redactor.register("registered-secret")
    scrubbed = redactor.scrub(
        {
            "known": "registered-secret",
            "header": "Bearer abcdefghijklmnop",
            "nested": ["token=abcdefghijkl"],
        }
    )
    assert scrubbed == {
        "known": REDACTED,
        "header": f"Bearer {REDACTED}",
        "nested": [f"token={REDACTED}"],
    }


def test_safe_logging_never_emits_exception_text_or_full_paths(caplog) -> None:
    logger = logging.getLogger("test.openalpha_research.safe")
    with caplog.at_level(logging.ERROR, logger=logger.name):
        try:
            raise RuntimeError("secret-token-and-local-path")
        except RuntimeError as error:
            record = log_operational_failure(
                error,
                run_id="run_1",
                deployed_commit="abc123",
                stage="compute",
                logger=logger,
            )
    assert record.exception_class == "RuntimeError"
    assert "secret-token-and-local-path" not in caplog.text
    assert "RuntimeError" in caplog.text
    assert all("/" not in frame and chr(92) not in frame for frame in record.frames)


def test_runtime_measurement_uses_observed_files_and_scope_time(tmp_path) -> None:
    (tmp_path / "first.bin").write_bytes(b"abc")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "second.bin").write_bytes(b"defgh")
    assert directory_bytes(tmp_path) == 8
    with scoped_timer() as timer:
        pass
    assert timer.seconds >= 0.0


def test_runtime_probe_is_safe_offline() -> None:
    assert gpu_snapshot().cuda_available in {True, False}
