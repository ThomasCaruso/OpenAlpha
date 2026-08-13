import ast
import sys
from pathlib import Path


def test_import_is_lightweight_and_names_the_subject() -> None:
    import openalpha_kronos

    assert openalpha_kronos.__name__ == "openalpha_kronos"
    assert "torch" not in sys.modules
    assert "huggingface_hub" not in sys.modules
    assert "yfinance" not in sys.modules


def test_model_and_evaluation_layers_do_not_import_studies() -> None:
    package = Path(__file__).resolve().parents[1] / "src" / "openalpha_kronos"
    violations: list[str] = []
    for layer in ("model", "evaluation"):
        for path in (package / layer).glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    if any(alias.name.startswith("openalpha_kronos.studies") for alias in node.names):
                        violations.append(f"{path.name}:{node.lineno}")
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if module.startswith("openalpha_kronos.studies") or (
                        node.level >= 2 and module.startswith("studies")
                    ):
                        violations.append(f"{path.name}:{node.lineno}")
    assert violations == []


def test_mini_spec_reexports_the_model_contract_vocabulary() -> None:
    from openalpha_kronos.model.contracts import OFFICIAL_TOKEN_VOCABULARY as model_value
    from openalpha_kronos.studies.structural_validity.mini.spec import (
        OFFICIAL_TOKEN_VOCABULARY as historical_value,
    )

    assert historical_value is model_value
