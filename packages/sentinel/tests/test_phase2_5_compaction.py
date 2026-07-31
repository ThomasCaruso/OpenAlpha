import importlib.util
from pathlib import Path
from types import ModuleType


def _load_cli() -> ModuleType:
    path = Path(__file__).resolve().parents[3] / "scripts" / "run_sentinel_phase2_5.py"
    spec = importlib.util.spec_from_file_location("phase2_5_compaction_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_validity_summary_keeps_counts_and_severity_not_repeated_rows() -> None:
    module = _load_cli()
    validity = {
        "path_id": "path-a",
        "valid": False,
        "candle_count": 5,
        "expected_candle_count": 5,
        "invalid_candle_count": 1,
        "earliest_invalid_horizon_step": 2,
        "violations": [
            {
                "code": "HIGH_BELOW_CLOSE",
                "normalized_severity": 0.002,
                "observed_gap": 1.25,
            }
        ],
    }

    summary = module._validity_summary(validity)

    assert summary["violation_count"] == 1
    assert summary["code_counts"] == {"HIGH_BELOW_CLOSE": 1}
    assert summary["max_normalized_severity"] == 0.002
    assert "violations" not in summary
