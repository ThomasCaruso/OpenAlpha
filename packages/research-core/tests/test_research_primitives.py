from __future__ import annotations

import json
import logging
import math
import os
import random
import subprocess
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from unittest import mock

import pytest
from openalpha_research.calendars import (
    CalendarName,
    sessions_in_half_open_range,
    xnys_holidays,
)
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
from pydantic import ValidationError


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


class _S3Error(Exception):
    def __init__(self, message: str, *, response: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.response = response


class _FailingS3Client:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def get_object(self, **_: object) -> dict[str, object]:
        raise self.error

    def head_object(self, **_: object) -> dict[str, object]:
        raise self.error


def _not_found_error(signal: str) -> Exception:
    if signal == "http-status":
        return _S3Error(
            "hidden",
            response={"ResponseMetadata": {"HTTPStatusCode": 404}},
        )
    if signal == "exception-string":
        return _S3Error("404")
    return _S3Error("hidden", response={"Error": {"Code": signal}})


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


@pytest.mark.parametrize(
    "signal",
    ["NoSuchKey", "NotFound", "404", "http-status", "exception-string"],
)
def test_s3_get_maps_only_definite_not_found_signals(signal: str) -> None:
    store = S3CompatibleObjectStore(bucket="research")
    store._client = _FailingS3Client(_not_found_error(signal))
    with pytest.raises(ResearchFailureError) as excinfo:
        store.get("result.json")
    assert excinfo.value.failures[0].code == "OBJECT_NOT_FOUND"


@pytest.mark.parametrize(
    "signal",
    ["NoSuchKey", "NotFound", "404", "http-status", "exception-string"],
)
def test_s3_exists_returns_false_only_for_definite_not_found_signals(signal: str) -> None:
    store = S3CompatibleObjectStore(bucket="research")
    store._client = _FailingS3Client(_not_found_error(signal))
    assert store.exists("result.json") is False


@pytest.mark.parametrize("operation", ["get", "exists"])
def test_s3_read_failures_are_typed_without_copying_exception_text(operation: str) -> None:
    secret = "credential=abcdefghijklmnop"
    cause = _S3Error(
        secret,
        response={"Error": {"Code": "AccessDenied"}, "ResponseMetadata": {"HTTPStatusCode": 403}},
    )
    store = S3CompatibleObjectStore(bucket="research")
    store._client = _FailingS3Client(cause)

    with pytest.raises(ResearchFailureError) as excinfo:
        getattr(store, operation)("result.json")
    assert excinfo.value.failures[0].code == "OBJECT_STORE_REQUEST_FAILED"
    assert secret not in str(excinfo.value)
    assert excinfo.value.__cause__ is cause


def test_s3_listing_rejects_missing_pagination_tokens_without_looping() -> None:
    class MissingTokenClient:
        def __init__(self) -> None:
            self.calls = 0

        def list_objects_v2(self, **_: object) -> dict[str, object]:
            self.calls += 1
            if self.calls > 1:
                raise AssertionError("pagination looped after a missing token")
            return {"Contents": (), "IsTruncated": True}

    client = MissingTokenClient()
    store = S3CompatibleObjectStore(bucket="research")
    store._client = client
    with pytest.raises(ResearchFailureError) as excinfo:
        store.list_keys("runs/")
    assert excinfo.value.failures[0].code == "OBJECT_STORE_PAGINATION_INVALID"
    assert client.calls == 1


def test_s3_listing_rejects_repeated_pagination_tokens_without_looping() -> None:
    class RepeatedTokenClient:
        def __init__(self) -> None:
            self.calls = 0

        def list_objects_v2(self, **_: object) -> dict[str, object]:
            self.calls += 1
            if self.calls > 2:
                raise AssertionError("pagination looped after a repeated token")
            return {
                "Contents": ({"Key": f"runs/{self.calls}.json"},),
                "IsTruncated": True,
                "NextContinuationToken": "same-token",
            }

    client = RepeatedTokenClient()
    store = S3CompatibleObjectStore(bucket="research")
    store._client = client
    with pytest.raises(ResearchFailureError) as excinfo:
        store.list_keys("runs/")
    assert excinfo.value.failures[0].code == "OBJECT_STORE_PAGINATION_INVALID"
    assert client.calls == 2


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


@pytest.mark.parametrize("symbol", ["BAD|SYMBOL", "BAD\rSYMBOL", "BAD\nSYMBOL"])
def test_retrieval_request_rejects_hash_delimiter_collisions(symbol: str) -> None:
    with pytest.raises(ValidationError):
        RetrievalRequest(
            symbol=symbol,
            start=date(2025, 1, 2),
            end=date(2025, 1, 4),
            maximum_candles=10,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("symbol", "BAD|SYMBOL"),
        ("symbol", "BAD\rSYMBOL"),
        ("symbol", "BAD\nSYMBOL"),
        ("interval", "1|d"),
        ("interval", "1\rd"),
        ("interval", "1\nd"),
    ],
)
def test_market_series_rejects_hash_delimiter_collisions(field: str, value: str) -> None:
    payload = _series(date(2025, 1, 2)).model_dump(mode="python")
    payload[field] = value
    with pytest.raises(ValidationError):
        MarketSeries.model_validate(payload)


@pytest.mark.parametrize("symbol", ["BRK-B", "^GSPC", "BTC-USD", "EURUSD=X"])
def test_retrieval_request_preserves_ordinary_finance_symbols(symbol: str) -> None:
    request = RetrievalRequest(
        symbol=symbol,
        start=date(2025, 1, 2),
        end=date(2025, 1, 4),
        maximum_candles=10,
    )
    assert request.symbol == symbol


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


def test_xnys_calendar_includes_the_2001_emergency_closures() -> None:
    holidays = xnys_holidays(2001)
    emergency_closures = {
        date(2001, 9, 11),
        date(2001, 9, 12),
        date(2001, 9, 13),
        date(2001, 9, 14),
    }
    assert emergency_closures <= holidays
    assert sessions_in_half_open_range(date(2001, 9, 10), date(2001, 9, 18)) == (
        date(2001, 9, 10),
        date(2001, 9, 17),
    )


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (date(1999, 12, 31), date(2000, 1, 3)),
        (date(2030, 12, 31), date(2031, 1, 2)),
    ],
)
def test_xnys_calendar_fails_closed_outside_supported_bounds(start: date, end: date) -> None:
    with pytest.raises(ResearchFailureError) as excinfo:
        sessions_in_half_open_range(start, end)
    assert excinfo.value.failures[0].code == "XNYS_RANGE_UNSUPPORTED"


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


def test_redaction_replaces_entire_values_under_sensitive_structured_keys() -> None:
    redactor = SecretRedactor()
    payload = {
        "Api-Key": "short",
        "TOKEN": 7,
        "PassWord": None,
        "AuthoriZation": {"nested": "must not leak"},
        "safe": {
            "secret-key": ["must", "not", "leak"],
            "items": ({"api_key": "abcdefghijklmnop", "label": "visible"},),
        },
    }
    assert redactor.scrub(payload) == {
        "Api-Key": REDACTED,
        "TOKEN": REDACTED,
        "PassWord": REDACTED,
        "AuthoriZation": REDACTED,
        "safe": {
            "secret-key": REDACTED,
            "items": ({"api_key": REDACTED, "label": "visible"},),
        },
    }


def test_text_redaction_preserves_quotes_around_json_credential_values() -> None:
    redactor = SecretRedactor()
    source = '{"api_key": "abcdefghijklmnop", "safe": "visible"}'
    scrubbed = redactor.scrub_text(source)
    assert scrubbed == '{"api_key": "[REDACTED]", "safe": "visible"}'
    assert json.loads(scrubbed) == {"api_key": REDACTED, "safe": "visible"}


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


def _run_isolated_python(source: str, *, pythonpath: Path | None = None) -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "AWS_ACCESS_KEY_ID": "test-access-key",
            "AWS_SECRET_ACCESS_KEY": "test-secret-key",
            "AWS_EC2_METADATA_DISABLED": "true",
            "AWS_DEFAULT_REGION": "us-east-1",
        }
    )
    if pythonpath is not None:
        inherited = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            f"{pythonpath}{os.pathsep}{inherited}" if inherited else str(pythonpath)
        )
    completed = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_optional_integrations_are_not_loaded_by_import_or_construction() -> None:
    _run_isolated_python(
        """
import sys
from openalpha_research.calendars import CalendarName
from openalpha_research.objectstore import S3CompatibleObjectStore
from openalpha_research.providers import YahooDailyProvider

YahooDailyProvider()
S3CompatibleObjectStore(bucket="research")
assert CalendarName.XNYS.value == "XNYS"
for name in ("yfinance", "boto3", "exchange_calendars"):
    assert name not in sys.modules, name
"""
    )


@pytest.mark.parametrize(
    ("module", "source"),
    [
        (
            "exchange_calendars",
            """
from datetime import date
from openalpha_research.calendars import sessions_in_half_open_range
sessions_in_half_open_range(date(2025, 1, 2), date(2025, 1, 4))
""",
        ),
    ],
)
def test_optional_integrations_load_only_when_their_capability_is_used(
    module: str,
    source: str,
) -> None:
    _run_isolated_python(
        f"""
import sys
assert {module!r} not in sys.modules
{source}
assert {module!r} in sys.modules
"""
    )


@pytest.mark.parametrize(
    ("module", "stub", "capability"),
    [
        (
            "yfinance",
            '__version__ = "test-version"\n',
            """
from openalpha_research.providers import YahooDailyProvider
assert YahooDailyProvider().client_version == "test-version"
""",
        ),
        (
            "boto3",
            "def client(*args, **kwargs):\n    return object()\n",
            """
from openalpha_research.objectstore import S3CompatibleObjectStore
S3CompatibleObjectStore(bucket="research")._boto()
""",
        ),
    ],
)
def test_optional_extra_capabilities_resolve_modules_lazily_in_isolation(
    tmp_path: Path,
    module: str,
    stub: str,
    capability: str,
) -> None:
    (tmp_path / f"{module}.py").write_text(stub, encoding="utf-8")
    _run_isolated_python(
        f"""
import sys
assert {module!r} not in sys.modules
{capability}
assert {module!r} in sys.modules
""",
        pythonpath=tmp_path,
    )
