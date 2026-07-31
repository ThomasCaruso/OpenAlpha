from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from openalpha_bridge import (
    BridgeFinancialTransform,
    BridgeRepresentationConfig,
    ReconstructedSequence,
    VolumeMode,
    bridge_canonical_bytes,
    bridge_sha256,
)


def _optional_result() -> tuple[BridgeFinancialTransform, ReconstructedSequence]:
    transform = BridgeFinancialTransform(BridgeRepresentationConfig())
    result = transform.decode_features(
        transformed_features=np.array(
            [
                [0.0, 0.0, 0.0, 0.0, math.nan],
                [0.01, -0.01, 0.02, 0.03, 0.0],
            ],
            dtype=np.float64,
        ),
        initial_previous_close=100.0,
        volume_present=np.array([False, True]),
    )
    return transform, result


def test_equivalent_results_produce_identical_canonical_bytes_and_sha256() -> None:
    _, result = _optional_result()

    first = bridge_canonical_bytes(result)
    second = bridge_canonical_bytes(result)

    assert first == second
    assert bridge_sha256(result) == bridge_sha256(result)
    assert len(bridge_sha256(result)) == 64
    payload = json.loads(first)
    assert payload["schema_version"] == "openalpha.bridge.artifact.v1"
    assert payload["object_type"] == "reconstructed_sequence"
    assert payload["candles"]["dtype"] == "float32"
    assert payload["candles"]["endianness"] == "little"
    assert payload["candles"]["encoding"] == "ieee754-hex"
    assert payload["missing_volume"]["encoding"] == "explicit-presence-u8"
    assert "NaN" not in first.decode("utf-8")


def test_native_and_big_endian_arrays_have_the_same_canonical_tensor_hash() -> None:
    transform = BridgeFinancialTransform(
        BridgeRepresentationConfig(
            volume_mode=VolumeMode.VOLUME_REQUIRED,
            output_dtype="float64",
        )
    )
    native = transform.decode_features(
        transformed_features=np.zeros((2, 5), dtype=np.float64),
        initial_previous_close=100.0,
    )
    big_endian_features = np.asarray(native.transformed_features.values, dtype=">f8")
    big_endian = transform.decode_features(
        transformed_features=big_endian_features,
        initial_previous_close=100.0,
    )

    assert bridge_sha256(native) == bridge_sha256(big_endian)


def test_negative_zero_is_canonicalized_to_positive_zero() -> None:
    transform = BridgeFinancialTransform(
        BridgeRepresentationConfig(
            volume_mode=VolumeMode.VOLUME_REQUIRED,
            output_dtype="float64",
        )
    )
    positive = transform.decode_features(
        transformed_features=np.zeros((1, 5), dtype=np.float64),
        initial_previous_close=100.0,
    )
    negative_features = np.full((1, 5), -0.0, dtype=np.float64)
    negative = transform.decode_features(
        transformed_features=negative_features,
        initial_previous_close=100.0,
    )
    assert bridge_sha256(positive) == bridge_sha256(negative)


def test_missing_volume_and_present_zero_have_different_hashes() -> None:
    transform = BridgeFinancialTransform(BridgeRepresentationConfig())
    values = np.zeros((1, 5), dtype=np.float64)
    missing_values = values.copy()
    missing_values[0, 4] = math.nan
    missing = transform.decode_features(
        transformed_features=missing_values,
        initial_previous_close=100.0,
        volume_present=np.array([False]),
    )
    zero = transform.decode_features(
        transformed_features=values,
        initial_previous_close=100.0,
        volume_present=np.array([True]),
    )

    assert bridge_sha256(missing) != bridge_sha256(zero)


def test_configuration_validation_and_numerical_audits_are_serializable() -> None:
    transform, result = _optional_result()
    timestamps = [
        datetime(2026, 1, 1, 12, tzinfo=timezone(timedelta(hours=-7))),
        datetime(2026, 1, 2, 12, tzinfo=timezone(timedelta(hours=-7))),
    ]
    validation = transform.audit(result, timestamps=timestamps)

    assert bridge_canonical_bytes(transform.config) == transform.config.canonical_bytes()
    assert bridge_sha256(transform.config) == transform.config.canonical_sha256
    validation_payload = json.loads(bridge_canonical_bytes(validation))
    timestamp = validation_payload["violations"]
    assert timestamp == []
    assert validation_payload["schema_version"] == "openalpha.bridge.validation.v1"


def test_serialization_rejects_unsupported_or_nonfinite_objects() -> None:
    with pytest.raises(TypeError, match="unsupported Bridge serialization type"):
        bridge_canonical_bytes({"unscoped": "mapping"})

    transform = BridgeFinancialTransform(
        BridgeRepresentationConfig(
            volume_mode=VolumeMode.VOLUME_REQUIRED,
            output_dtype="float64",
        )
    )
    result = transform.decode_features(
        transformed_features=np.zeros((1, 5)),
        initial_previous_close=100.0,
    )
    result.candles.setflags(write=True)
    result.candles[0, 0] = math.nan
    result.candles.setflags(write=False)
    with pytest.raises(ValueError, match="nonfinite"):
        bridge_canonical_bytes(result)
