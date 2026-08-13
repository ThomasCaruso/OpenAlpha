import ast
import json
import subprocess
import sys
from pathlib import Path


def test_import_is_lightweight_and_names_the_subject() -> None:
    script = """
import json
import sys

heavy_modules = ("torch", "huggingface_hub", "yfinance")
before = {name: name in sys.modules for name in heavy_modules}
import openalpha_kronos
after = {name: name in sys.modules for name in heavy_modules}
print(json.dumps({"before": before, "after": after, "name": openalpha_kronos.__name__}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    observed = json.loads(completed.stdout)

    assert observed == {
        "before": {"torch": False, "huggingface_hub": False, "yfinance": False},
        "after": {"torch": False, "huggingface_hub": False, "yfinance": False},
        "name": "openalpha_kronos",
    }


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
