import json
from collections.abc import Callable, Mapping
from typing import Any

import yaml

from .migrations import migrate_mapping
from .models import ExperimentSpec

MAX_INPUT_BYTES = 1024 * 1024


def _mapping_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for key, value in pairs:
        if key in mapping:
            raise ValueError(f"document contains duplicate key {key!r}")
        mapping[key] = value
    return mapping


class _UniqueKeySafeLoader(yaml.SafeLoader):
    def construct_mapping(
        self,
        node: yaml.MappingNode,
        deep: bool = False,
    ) -> dict[Any, Any]:
        self.flatten_mapping(node)
        mapping: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ValueError(f"document contains duplicate key {key!r}")
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


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
        lambda text: json.loads(
            text,
            object_pairs_hook=_mapping_without_duplicates,
            parse_constant=_reject_json_constant,
        ),
    )


def load_yaml(value: str | bytes) -> ExperimentSpec:
    return _load(value, lambda text: yaml.load(text, Loader=_UniqueKeySafeLoader))
