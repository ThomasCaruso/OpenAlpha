"""Pinned identities of the official Kronos source and completed-study assets."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "KRONOS_BASE_SPEC",
    "KRONOS_BASE_TOKENIZER_SPEC",
    "KRONOS_MINI_SPEC",
    "OFFICIAL_SOURCE_FILES",
    "SOURCE_SPEC",
    "TOKENIZER_SPEC",
    "BaseForecastModelSpec",
    "ForecastModelSpec",
    "OfficialSourceFile",
    "PinnedAssetSpec",
    "PinnedSourceSpec",
]


class PinnedAssetSpec(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    repository: str
    revision: str = Field(min_length=40, max_length=40)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    weights_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class PinnedSourceSpec(BaseModel):
    """The pinned official source checkout, verified file by file."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    repository: str
    revision: str = Field(min_length=40, max_length=40)
    files: dict[str, str]


SOURCE_SPEC: Final[PinnedSourceSpec] = PinnedSourceSpec(
    repository="https://github.com/shiyu-coder/Kronos",
    revision="67b630e67f6a18c9e9be918d9b4337c960db1e9a",
    files={
        "model/kronos.py": "638a56e035856c600c9848b368be087cb706a61603a0790124968c95b8c69f3a",
        "model/module.py": "a07edbadc0e96804c8158c021bbc6063bb7cc43b34d7fc470d5c8ff2005a409f",
    },
)


TOKENIZER_SPEC: Final[PinnedAssetSpec] = PinnedAssetSpec(
    repository="NeoQuasar/Kronos-Tokenizer-2k",
    revision="26966d0035065a0cae0ebad7af8ece35bc1fb51c",
    config_sha256="0b30a443affb03e05a876a083857de9164f899feb7b4d261da02c485c9a3e3b6",
    weights_sha256="b97ec46b3b72160509e289183eaf7bdf5f0dac5bb9b49522f6d46638a99a8717",
)


class OfficialSourceFile(BaseModel):
    """Both recorded digests and byte identity for one pinned source file."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    relative_path: str
    sealed_sha256_crlf_normalized: str = Field(pattern=r"^[0-9a-f]{64}$")
    as_committed_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    as_committed_line_ending: Literal["LF", "CRLF"]
    as_committed_bytes: int = Field(gt=0)


OFFICIAL_SOURCE_FILES: Final[tuple[OfficialSourceFile, ...]] = (
    OfficialSourceFile(
        relative_path="model/kronos.py",
        sealed_sha256_crlf_normalized=(
            "638a56e035856c600c9848b368be087cb706a61603a0790124968c95b8c69f3a"
        ),
        as_committed_sha256="0a5f90282e2039c2de0771473419715c845def154896dbd0f5747837e6241032",
        as_committed_line_ending="LF",
        as_committed_bytes=30133,
    ),
    OfficialSourceFile(
        relative_path="model/module.py",
        sealed_sha256_crlf_normalized=(
            "a07edbadc0e96804c8158c021bbc6063bb7cc43b34d7fc470d5c8ff2005a409f"
        ),
        as_committed_sha256="a07edbadc0e96804c8158c021bbc6063bb7cc43b34d7fc470d5c8ff2005a409f",
        as_committed_line_ending="CRLF",
        as_committed_bytes=23426,
    ),
)


class ForecastModelSpec(BaseModel):
    """The pinned Kronos-mini forecasting model."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    name: Literal["Kronos-mini"] = "Kronos-mini"
    repository: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    weights_file: str
    weights_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    weights_size_bytes: int = Field(gt=0)
    coarse_vocabulary: int = Field(gt=0)
    fine_vocabulary: int = Field(gt=0)
    decoder_layers: int = Field(gt=0)
    model_dimension: int = Field(gt=0)


KRONOS_MINI_SPEC: Final[ForecastModelSpec] = ForecastModelSpec(
    repository="NeoQuasar/Kronos-mini",
    revision="f4e68697d9d5aed55cef5c96aabc3376bcad9f81",
    config_sha256="70daca2cb11e3a979dd6b8ac12ee08e2aace877acf28f5b8dfb4fe5609736201",
    weights_file="model.safetensors",
    weights_sha256="a7d5f37e2e9fbd9891f7d7d4f72574512dd1f704fee14223e0a8cd0fbf54197c",
    weights_size_bytes=16_440_776,
    coarse_vocabulary=1024,
    fine_vocabulary=1024,
    decoder_layers=4,
    model_dimension=256,
)


class BaseForecastModelSpec(BaseModel):
    """The pinned Kronos-base forecasting model and its paired tokenizer."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    model_family: Literal["kronos-base"] = "kronos-base"
    name: Literal["Kronos-base"] = "Kronos-base"
    repository: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_size_bytes: int = Field(gt=0)
    weights_file: str
    weights_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    weights_size_bytes: int = Field(gt=0)
    coarse_vocabulary: int = Field(gt=0)
    fine_vocabulary: int = Field(gt=0)
    layers: int = Field(gt=0)
    model_dimension: int = Field(gt=0)
    attention_heads: int = Field(gt=0)
    feedforward_dimension: int = Field(gt=0)
    learned_temporal_embedding: bool
    maximum_context: int = Field(gt=0)
    paired_tokenizer_repository: str


KRONOS_BASE_SPEC: Final[BaseForecastModelSpec] = BaseForecastModelSpec(
    repository="NeoQuasar/Kronos-base",
    revision="2b554741eca47781b64468546e77fef3e85130e6",
    config_sha256="77ebc3038b647709b92be002f801d72e1a385f4c8c2c5aa1cc6cf21fcfe44eb2",
    config_size_bytes=228,
    weights_file="model.safetensors",
    weights_sha256="abff193acab6db1a0368e9773e75799d11403b6d054ee6d5f0a11aeabc5f4b83",
    weights_size_bytes=409_264_008,
    coarse_vocabulary=1024,
    fine_vocabulary=1024,
    layers=12,
    model_dimension=832,
    attention_heads=16,
    feedforward_dimension=2048,
    learned_temporal_embedding=True,
    maximum_context=512,
    paired_tokenizer_repository="NeoQuasar/Kronos-Tokenizer-base",
)

KRONOS_BASE_TOKENIZER_SPEC: Final[PinnedAssetSpec] = PinnedAssetSpec(
    repository="NeoQuasar/Kronos-Tokenizer-base",
    revision="0e0117387f39004a9016484a186a908917e22426",
    config_sha256="2366e7ccfec76cbc19cf3c4c1b9c5d901be336ca1e83f2d2292c9bff381b77a2",
    weights_sha256="59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee",
)
