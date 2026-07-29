import json
from collections.abc import Callable, Mapping
from typing import Any

import yaml

from .migrations import migrate_mapping
from .models import ExperimentSpec

MAX_INPUT_BYTES = 1024 * 1024


def _decode_input(value: str | bytes) -> str:
    if isinstance(value, bytes):
        size = len(value)
        raw = value
    elif isinstance(value, str):
        raw = value.encode("utf-8")
        size = len(raw)
    else:
        raise TypeError("input must be str or bytes")
    if size > MAX_INPUT_BYTES:
        raise ValueError(f"input exceeds 1 MiB limit ({size} bytes received)")
    return raw.decode("utf-8")


def _load(
    value: str | bytes,
    parser: Callable[[str], Any],
) -> ExperimentSpec:
    parsed = parser(_decode_input(value))
    if not isinstance(parsed, Mapping):
        # This is a structurally invalid document, not an invalid Python argument type.
        raise ValueError("document root must be an object/mapping")  # noqa: TRY004
    return ExperimentSpec.model_validate(migrate_mapping(parsed))


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON numeric constant: {value}")


def load_json(value: str | bytes) -> ExperimentSpec:
    return _load(
        value,
        lambda text: json.loads(text, parse_constant=_reject_json_constant),
    )


def load_yaml(value: str | bytes) -> ExperimentSpec:
    return _load(value, yaml.safe_load)
