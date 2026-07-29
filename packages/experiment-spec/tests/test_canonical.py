import json
import tomllib
from pathlib import Path
from typing import Any

import pytest
from openalpha_experiment_spec import (
    ExperimentSpec,
    canonical_bytes,
    experiment_id,
    load_json,
    load_yaml,
    migrate_mapping,
)
from pydantic import ValidationError

EXPECTED_EXPERIMENT_ID = "exp_8e72de5fcc485d3d512227f89f3d8e74e6f73bdfe6c90ddced158614d7ec6efb"


def _reverse_mappings(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _reverse_mappings(item) for key, item in reversed(value.items())}
    if isinstance(value, list):
        return [_reverse_mappings(item) for item in value]
    return value


def test_public_api_is_importable() -> None:
    assert all(
        callable(api)
        for api in (
            ExperimentSpec,
            canonical_bytes,
            experiment_id,
            load_json,
            load_yaml,
            migrate_mapping,
        )
    )


def test_root_workspace_sync_installs_experiment_spec(example_path: Path) -> None:
    workspace = tomllib.loads(
        (example_path.parents[2] / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert "openalpha-experiment-spec" in workspace["dependency-groups"]["dev"]
    assert workspace["tool"]["uv"]["sources"]["openalpha-experiment-spec"] == {
        "workspace": True
    }


def test_yaml_and_reordered_json_have_identical_canonical_identity(
    example_path: Path, valid_mapping: dict[str, object]
) -> None:
    yaml_spec = load_yaml(example_path.read_text(encoding="utf-8"))
    json_spec = load_json(json.dumps(_reverse_mappings(valid_mapping), indent=4))

    assert canonical_bytes(yaml_spec) == canonical_bytes(json_spec)
    assert experiment_id(yaml_spec) == EXPECTED_EXPERIMENT_ID
    assert experiment_id(json_spec) == EXPECTED_EXPERIMENT_ID


def test_canonical_bytes_are_sorted_compact_utf8(valid_mapping: dict[str, object]) -> None:
    metadata = valid_mapping["metadata"]
    assert isinstance(metadata, dict)
    metadata["name"] = "Café α SPY experiment"
    spec = ExperimentSpec.model_validate(valid_mapping)

    expected = json.dumps(
        spec.model_dump(mode="json", exclude_none=False),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    assert canonical_bytes(spec) == expected
    assert canonical_bytes(spec).startswith(b'{"benchmark":')
    assert b"Caf\xc3\xa9 \xce\xb1" in canonical_bytes(spec)
    assert b"\\u00e9" not in canonical_bytes(spec)


def test_hash_is_repeatable_and_changes_with_meaningful_input(
    valid_mapping: dict[str, object],
) -> None:
    first = ExperimentSpec.model_validate(valid_mapping)
    strategy = valid_mapping["strategy"]
    assert isinstance(strategy, dict)
    strategy["threshold_bps"] = 16
    changed = ExperimentSpec.model_validate(valid_mapping)

    assert experiment_id(first) == experiment_id(first)
    assert experiment_id(first) != experiment_id(changed)


def test_top_level_and_nested_values_are_deeply_immutable(
    valid_mapping: dict[str, object],
) -> None:
    spec = ExperimentSpec.model_validate(valid_mapping)

    with pytest.raises(ValidationError, match="frozen"):
        spec.random_seed = 7
    with pytest.raises(ValidationError, match="frozen"):
        spec.metadata.name = "A different experiment"
    with pytest.raises(ValidationError, match="frozen"):
        spec.models[0].parameters.temperature = 0.5  # type: ignore[union-attr]
    with pytest.raises(TypeError):
        spec.features[0] = "calendar"  # type: ignore[index]
    with pytest.raises(TypeError):
        spec.models[0] = spec.models[1]  # type: ignore[index]


def test_json_schema_is_formal_and_describes_discriminated_models() -> None:
    schema = ExperimentSpec.model_json_schema()
    required = {
        "schema_version",
        "metadata",
        "hypothesis",
        "data",
        "forecast",
        "features",
        "models",
        "training",
        "evaluation",
        "strategy",
        "execution",
        "benchmark",
        "statistics",
        "reporting",
        "random_seed",
    }

    def nodes(value: Any):
        if isinstance(value, dict):
            yield value
            for item in value.values():
                yield from nodes(item)
        elif isinstance(value, list):
            for item in value:
                yield from nodes(item)

    assert "$defs" in schema
    assert required <= set(schema["required"])
    assert any("oneOf" in node and "discriminator" in node for node in nodes(schema))
