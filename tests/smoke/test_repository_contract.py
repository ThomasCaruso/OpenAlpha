from pathlib import Path


def test_required_workspace_files_exist() -> None:
    root = Path(__file__).parents[2]
    for name in ("pyproject.toml", "package.json", "Makefile", ".python-version"):
        assert (root / name).is_file(), name
