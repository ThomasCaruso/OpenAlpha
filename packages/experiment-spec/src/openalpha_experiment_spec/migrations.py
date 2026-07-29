from collections.abc import Mapping
from copy import deepcopy
from typing import Any


def migrate_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    if "schema_version" not in value:
        raise ValueError("schema_version is required")
    version = value["schema_version"]
    if version != "1.0":
        raise ValueError(
            f"unsupported schema_version {version!r}; supported versions: 1.0"
        )
    return deepcopy(dict(value))
