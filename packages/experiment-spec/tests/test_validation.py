from collections.abc import Callable
from copy import deepcopy
from typing import Any

import pytest
from openalpha_experiment_spec import ExperimentSpec, load_json, load_yaml
from pydantic import ValidationError


def _set_path(mapping: dict[str, Any], path: tuple[str | int, ...], value: Any) -> None:
    target: Any = mapping
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value


def _assert_error(
    mapping: dict[str, Any], location: tuple[str | int, ...], message: str
) -> None:
    with pytest.raises(ValidationError) as captured:
        ExperimentSpec.model_validate(mapping)
    matching = [error for error in captured.value.errors() if error["loc"] == location]
    assert matching, captured.value.errors()
    assert message in matching[0]["msg"]


def test_checked_in_mapping_is_valid(valid_mapping: dict[str, object]) -> None:
    spec = ExperimentSpec.model_validate(valid_mapping)

    assert spec.schema_version == "1.0"
    assert spec.data.assets == ("SPY",)
    assert len(spec.models) == 6


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        ("2025-01-02", "2025-01-02", "start must be before end"),
        ("2025-01-03", "2025-01-02", "start must be before end"),
        ("2024-06-30", "2025-01-02", "start must be strictly after 2024-06-30"),
    ],
)
def test_data_date_boundaries_are_actionable(
    valid_mapping: dict[str, Any], start: str, end: str, message: str
) -> None:
    valid_mapping["data"]["start"] = start
    valid_mapping["data"]["end"] = end

    _assert_error(valid_mapping, ("data", "start"), message)


@pytest.mark.parametrize("final_test_start", ["2024-07-01", "2026-07-01"])
def test_final_test_start_must_be_inside_data_period(
    valid_mapping: dict[str, Any], final_test_start: str
) -> None:
    valid_mapping["evaluation"]["final_test_start"] = final_test_start

    _assert_error(
        valid_mapping,
        ("evaluation", "final_test_start"),
        "final_test_start must be after data.start and on or before data.end",
    )


def test_final_test_start_may_equal_data_end(valid_mapping: dict[str, Any]) -> None:
    valid_mapping["evaluation"]["final_test_start"] = valid_mapping["data"]["end"]

    assert ExperimentSpec.model_validate(valid_mapping).evaluation.final_test_start.isoformat() == (
        "2026-06-30"
    )


def test_every_required_model_kind_must_appear_once(valid_mapping: dict[str, Any]) -> None:
    valid_mapping["models"].pop()

    _assert_error(
        valid_mapping,
        ("models",),
        "models must contain exactly one of each required kind",
    )


def test_duplicate_model_kind_is_rejected(valid_mapping: dict[str, Any]) -> None:
    duplicate = deepcopy(valid_mapping["models"][0])
    duplicate["name"] = "second-kronos"
    valid_mapping["models"][-1] = duplicate

    _assert_error(
        valid_mapping,
        ("models",),
        "models must contain exactly one of each required kind",
    )


def test_model_names_are_unique(valid_mapping: dict[str, Any]) -> None:
    valid_mapping["models"][1]["name"] = valid_mapping["models"][0]["name"]

    _assert_error(valid_mapping, ("models", 1, "name"), "model names must be unique")


@pytest.mark.parametrize(
    ("checkpoint", "tokenizer"),
    [
        ("NeoQuasar/Kronos-mini", "NeoQuasar/Kronos-Tokenizer-base"),
        ("NeoQuasar/Kronos-base", "NeoQuasar/Kronos-Tokenizer-2k"),
    ],
)
def test_kronos_tokenizer_must_match_checkpoint_family(
    valid_mapping: dict[str, Any], checkpoint: str, tokenizer: str
) -> None:
    parameters = valid_mapping["models"][0]["parameters"]
    parameters["checkpoint"] = checkpoint
    parameters["tokenizer"] = tokenizer

    _assert_error(
        valid_mapping,
        ("models", 0, "kronos", "parameters", "tokenizer"),
        "tokenizer must match checkpoint family",
    )


@pytest.mark.parametrize(
    ("path", "value", "location", "message"),
    [
        (
            ("forecast", "context_window"),
            300,
            ("evaluation", "initial_training_bars"),
            "at least forecast.context_window",
        ),
        (
            ("evaluation", "purge_bars"),
            3,
            ("evaluation", "purge_bars"),
            "at least forecast.horizon - 1",
        ),
        (
            ("statistics", "block_length"),
            4,
            ("statistics", "block_length"),
            "at least forecast.horizon",
        ),
        (
            ("training", "policy"),
            "zero_shot",
            ("training", "policy"),
            "trainable baselines require fit_per_origin",
        ),
    ],
)
def test_cross_section_training_and_evaluation_rules(
    valid_mapping: dict[str, Any],
    path: tuple[str, ...],
    value: object,
    location: tuple[str, ...],
    message: str,
) -> None:
    _set_path(valid_mapping, path, value)

    _assert_error(valid_mapping, location, message)


def test_execution_cost_must_be_nonzero(valid_mapping: dict[str, Any]) -> None:
    valid_mapping["execution"]["commission_bps"] = 0
    valid_mapping["execution"]["slippage_bps"] = 0

    _assert_error(
        valid_mapping,
        ("execution", "commission_bps"),
        "commission_bps + slippage_bps must be greater than zero",
    )


def test_benchmark_asset_must_match_data_asset(valid_mapping: dict[str, Any]) -> None:
    valid_mapping["benchmark"]["asset"] = "QQQ"

    _assert_error(
        valid_mapping, ("benchmark", "asset"), "benchmark asset must equal data.assets[0]"
    )


@pytest.mark.parametrize(
    ("path", "duplicate", "location", "message"),
    [
        (("features",), "close_lags", ("features", 1), "feature names must be unique"),
        (("metadata", "tags"), "spy", ("metadata", "tags", 1), "tags must be unique"),
        (
            ("reporting", "formats"),
            "html",
            ("reporting", "formats", 1),
            "reporting formats must be unique",
        ),
    ],
)
def test_tuple_values_are_unique(
    valid_mapping: dict[str, Any],
    path: tuple[str, ...],
    duplicate: str,
    location: tuple[str | int, ...],
    message: str,
) -> None:
    _set_path(valid_mapping, path, [duplicate, duplicate])

    _assert_error(valid_mapping, location, message)


@pytest.mark.parametrize(
    ("path", "value", "location"),
    [
        (("data", "calendar"), "NASDAQ", ("data", "calendar")),
        (("data", "assets"), ["spy"], ("data", "assets", 0)),
        (("data", "assets"), ["SPY", "QQQ"], ("data", "assets")),
        (("forecast", "context_window"), 19, ("forecast", "context_window")),
        (
            ("models", 0, "parameters", "temperature"),
            0,
            ("models", 0, "kronos", "parameters", "temperature"),
        ),
        (
            ("statistics", "confidence_level"),
            0.7,
            ("statistics", "confidence_level"),
        ),
        (("random_seed",), 2**32, ("random_seed",)),
        (("features",), [], ("features",)),
    ],
)
def test_invalid_enums_and_bounds_are_field_located(
    valid_mapping: dict[str, Any],
    path: tuple[str | int, ...],
    value: object,
    location: tuple[str | int, ...],
) -> None:
    _set_path(valid_mapping, path, value)

    with pytest.raises(ValidationError) as captured:
        ExperimentSpec.model_validate(valid_mapping)
    assert any(error["loc"] == location for error in captured.value.errors()), captured.value.errors()


@pytest.mark.parametrize(
    ("path", "location"),
    [
        (("unexpected",), ("unexpected",)),
        (("data", "unexpected"), ("data", "unexpected")),
        (
            ("models", 0, "parameters", "unexpected"),
            ("models", 0, "kronos", "parameters", "unexpected"),
        ),
    ],
)
def test_unknown_fields_fail_at_their_location(
    valid_mapping: dict[str, Any],
    path: tuple[str | int, ...],
    location: tuple[str | int, ...],
) -> None:
    _set_path(valid_mapping, path, "not allowed")

    with pytest.raises(ValidationError) as captured:
        ExperimentSpec.model_validate(valid_mapping)
    assert any(error["loc"] == location for error in captured.value.errors()), captured.value.errors()


@pytest.mark.parametrize(
    ("loader", "payload"),
    [
        (load_json, "[]"),
        (load_json, "null"),
        (load_yaml, "- one\n- two\n"),
        (load_yaml, "null\n"),
    ],
)
def test_loaders_reject_non_object_roots(
    loader: Callable[[str | bytes], ExperimentSpec], payload: str
) -> None:
    with pytest.raises(ValueError, match="root must be an object"):
        loader(payload)


@pytest.mark.parametrize("loader", [load_json, load_yaml])
def test_loaders_reject_inputs_over_one_mib(
    loader: Callable[[str | bytes], ExperimentSpec],
) -> None:
    payload = b" " * (1024 * 1024 + 1)

    with pytest.raises(ValueError, match="input exceeds 1 MiB limit"):
        loader(payload)


@pytest.mark.parametrize("loader", [load_json, load_yaml])
def test_loaders_migrate_before_validation(
    loader: Callable[[str | bytes], ExperimentSpec],
) -> None:
    payload = "{}" if loader is load_json else "{}\n"

    with pytest.raises(ValueError, match="schema_version is required"):
        loader(payload)
