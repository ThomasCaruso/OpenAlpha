from __future__ import annotations

import math

import pytest
from openalpha_bridge import (
    BridgeRepresentationConfig,
    VolumeMode,
)
from pydantic import ValidationError


def test_default_configuration_is_the_immutable_phase_0_contract() -> None:
    config = BridgeRepresentationConfig()

    assert config.representation_version == "openalpha.bridge.financial.v1"
    assert config.gap_return_limit == math.log(4.0)
    assert config.body_return_limit == math.log(4.0)
    assert config.upper_wick_transform == "bounded_stable_softplus"
    assert config.lower_wick_transform == "bounded_stable_softplus"
    assert config.upper_wick_limit == math.log(4.0)
    assert config.lower_wick_limit == math.log(4.0)
    assert config.volume_mode is VolumeMode.VOLUME_OPTIONAL
    assert config.volume_transform == "bounded_stable_softplus_log1p"
    assert config.volume_limit == math.log1p(1.0e15)
    assert config.numerical_epsilon == 1.0e-14
    assert config.output_dtype == "float32"
    assert config.reference_dtype == "float64"
    assert config.overflow_policy == "raise"
    assert config.out_of_domain_policy == "raise"
    assert config.supported_feature_order == ("open", "high", "low", "close", "volume")
    assert config.minimum_price == 1.0e-12
    assert config.maximum_price == 1.0e12
    assert config.maximum_volume == 1.0e15
    assert config.log_price_guard == (-300.0, 300.0)
    assert config.maximum_decode_steps == 64

    with pytest.raises(ValidationError):
        config.representation_version = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError, match="updates bypass validation"):
        config.model_copy(update={"gap_return_limit": 2.0})


def test_equivalent_configurations_have_identical_canonical_bytes_and_hashes() -> None:
    first = BridgeRepresentationConfig()
    second = BridgeRepresentationConfig(
        representation_version="openalpha.bridge.financial.v1",
        volume_mode=VolumeMode.VOLUME_OPTIONAL,
    )

    assert first.canonical_bytes() == second.canonical_bytes()
    assert first.canonical_sha256 == second.canonical_sha256
    assert len(first.canonical_sha256) == 64
    assert b'"schema_version":"openalpha.bridge.config.v1"' in first.canonical_bytes()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gap_return_limit", 0.0),
        ("body_return_limit", -1.0),
        ("upper_wick_limit", 0.0),
        ("lower_wick_limit", -0.1),
        ("volume_limit", 0.0),
        ("numerical_epsilon", 0.0),
        ("minimum_price", 0.0),
        ("maximum_price", -1.0),
        ("maximum_volume", -1.0),
        ("maximum_decode_steps", 0),
    ],
)
def test_configuration_rejects_nonpositive_or_invalid_limits(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        BridgeRepresentationConfig(**{field: value})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("representation_version", "v2"),
        ("upper_wick_transform", "relu"),
        ("lower_wick_transform", "exp"),
        ("volume_transform", "identity"),
        ("output_dtype", "float16"),
        ("reference_dtype", "float32"),
        ("overflow_policy", "clip"),
        ("out_of_domain_policy", "clip"),
    ],
)
def test_configuration_rejects_unknown_versions_mappings_dtypes_and_policies(
    field: str,
    value: str,
) -> None:
    with pytest.raises(ValidationError):
        BridgeRepresentationConfig(**{field: value})  # type: ignore[arg-type]


def test_configuration_rejects_unsafe_exponential_guards_and_bounds() -> None:
    with pytest.raises(ValidationError, match="safe float64 exponential range"):
        BridgeRepresentationConfig(log_price_guard=(-300.0, 710.0))
    with pytest.raises(ValidationError, match="safe float64 exponential range"):
        BridgeRepresentationConfig(log_price_guard=(-750.0, 300.0))
    with pytest.raises(ValidationError, match="minimum_price must be below maximum_price"):
        BridgeRepresentationConfig(minimum_price=2.0, maximum_price=1.0)
    with pytest.raises(ValidationError, match="inside log_price_guard"):
        BridgeRepresentationConfig(minimum_price=1.0e-200)
    with pytest.raises(ValidationError, match="inside log_price_guard"):
        BridgeRepresentationConfig(maximum_price=1.0e200)


def test_configuration_rejects_contradictory_volume_feature_ordering() -> None:
    with pytest.raises(ValidationError, match="PRICE_ONLY.*four OHLC"):
        BridgeRepresentationConfig(volume_mode=VolumeMode.PRICE_ONLY)

    price_only = BridgeRepresentationConfig(
        volume_mode=VolumeMode.PRICE_ONLY,
        supported_feature_order=("open", "high", "low", "close"),
    )
    assert price_only.volume_mode is VolumeMode.PRICE_ONLY

    with pytest.raises(ValidationError, match="requires ordered OHLCV"):
        BridgeRepresentationConfig(
            volume_mode=VolumeMode.VOLUME_REQUIRED,
            supported_feature_order=("open", "high", "low", "close"),
        )


def test_configuration_rejects_nonfinite_values_and_extra_fields() -> None:
    with pytest.raises(ValidationError):
        BridgeRepresentationConfig(gap_return_limit=math.inf)
    with pytest.raises(ValidationError):
        BridgeRepresentationConfig(unknown=True)  # type: ignore[call-arg]
