from copy import deepcopy

import pytest
from openalpha_experiment_spec import migrate_mapping


def test_current_version_returns_an_independent_deep_copy(
    valid_mapping: dict[str, object],
) -> None:
    original = deepcopy(valid_mapping)

    migrated = migrate_mapping(valid_mapping)
    metadata = migrated["metadata"]
    assert isinstance(metadata, dict)
    metadata["name"] = "Changed only in migrated copy"

    assert migrated is not valid_mapping
    assert valid_mapping == original
    assert valid_mapping["metadata"] != migrated["metadata"]


@pytest.mark.parametrize("version", [None, "0.9", "2.0", 1])
def test_missing_or_unsupported_versions_are_rejected(
    valid_mapping: dict[str, object], version: object
) -> None:
    if version is None:
        valid_mapping.pop("schema_version")
        expected = "schema_version is required"
    else:
        valid_mapping["schema_version"] = version
        expected = f"unsupported schema_version {version!r}; supported versions: 1.0"

    with pytest.raises(ValueError, match=expected.replace(".", r"\.")):
        migrate_mapping(valid_mapping)
