import json
import tomllib
from importlib.resources import files
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
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


def test_installed_distribution_includes_py_typed_marker() -> None:
    marker = files("openalpha_experiment_spec").joinpath("py.typed")

    assert marker.is_file()


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


@pytest.mark.parametrize(
    "update",
    [
        {"features": ["calendar"]},
        {"random_seed": -1},
        {"random_seed": "7"},
    ],
)
def test_model_copy_rejects_unvalidated_top_level_updates(
    valid_mapping: dict[str, object],
    update: dict[str, object],
) -> None:
    spec = ExperimentSpec.model_validate(valid_mapping)

    with pytest.raises(
        TypeError,
        match="create and validate a new experiment",
    ):
        spec.model_copy(update=update)


def test_nested_model_copy_rejects_unvalidated_updates(
    valid_mapping: dict[str, object],
) -> None:
    spec = ExperimentSpec.model_validate(valid_mapping)

    with pytest.raises(
        TypeError,
        match="create and validate a new experiment",
    ):
        spec.models[0].model_copy(update={"parameters": {"temperature": float("inf")}})


def test_update_free_model_copy_preserves_immutable_identity(
    valid_mapping: dict[str, object],
) -> None:
    spec = ExperimentSpec.model_validate(valid_mapping)

    copied = spec.model_copy()

    assert copied is not spec
    assert isinstance(copied.features, tuple)
    assert canonical_bytes(copied) == canonical_bytes(spec)
    assert experiment_id(copied) == experiment_id(spec)


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


def test_json_schema_uses_array_collection_keywords() -> None:
    schema = ExperimentSpec.model_json_schema()
    definitions = schema["$defs"]
    bounded_arrays = [
        (definitions["Data"]["properties"]["assets"], 1, 1),
        (schema["properties"]["features"], 1, None),
        (schema["properties"]["models"], 1, None),
        (definitions["Reporting"]["properties"]["formats"], 1, None),
        (
            definitions["GradientBoostedTreeParameters"]["properties"]["lags"],
            1,
            None,
        ),
    ]

    for field_schema, minimum, maximum in bounded_arrays:
        assert field_schema["minItems"] == minimum
        assert "minLength" not in field_schema
        assert "maxLength" not in field_schema
        if maximum is None:
            assert "maxItems" not in field_schema
        else:
            assert field_schema["maxItems"] == maximum


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("features",), []),
        (("models",), []),
        (("reporting", "formats"), []),
        (("models", 5, "parameters", "lags"), []),
        (("data", "assets"), []),
        (("data", "assets"), ["SPY", "QQQ"]),
    ],
)
def test_draft_2020_12_schema_rejects_invalid_collection_sizes(
    valid_mapping: dict[str, Any],
    path: tuple[str | int, ...],
    value: object,
) -> None:
    target: Any = valid_mapping
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    validator = Draft202012Validator(ExperimentSpec.model_json_schema())

    with pytest.raises(JsonSchemaValidationError):
        validator.validate(valid_mapping)
