import sys
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

PACKAGE_SRC = Path(__file__).parents[1] / "src"
sys.path.insert(0, str(PACKAGE_SRC))

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_PATH = (
    REPOSITORY_ROOT
    / "examples"
    / "experiments"
    / "archive"
    / "spy-kronos-daily-v1.yaml"
)


@pytest.fixture
def example_path() -> Path:
    return EXAMPLE_PATH


@pytest.fixture
def repository_root() -> Path:
    return REPOSITORY_ROOT


@pytest.fixture
def valid_mapping(example_path: Path) -> dict[str, object]:
    loaded = yaml.safe_load(example_path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return deepcopy(loaded)
