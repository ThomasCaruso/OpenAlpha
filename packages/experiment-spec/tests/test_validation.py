import json
import math
from collections.abc import Callable
from copy import deepcopy
from datetime import date
from typing import Any

import pytest
import yaml
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
    ("path", "value", "location", "message"),
    [
        (
            ("forecast", "horizon"),
            "5",
            ("forecast", "horizon"),
            "Input should be a valid integer",
        ),
        (
            ("strategy", "threshold_bps"),
            "15.0",
            ("strategy", "threshold_bps"),
            "Input should be a valid number",
        ),
    ],
)
def test_numeric_strings_are_not_coerced(
    valid_mapping: dict[str, Any],
    path: tuple[str | int, ...],
    value: object,
    location: tuple[str | int, ...],
    message: str,
) -> None:
    _set_path(valid_mapping, path, value)

    _assert_error(valid_mapping, location, message)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_numbers_fail_during_validation(
    valid_mapping: dict[str, Any], value: float
) -> None:
    valid_mapping["execution"]["commission_bps"] = value

    _assert_error(
        valid_mapping,
        ("execution", "commission_bps"),
        "Input should be a finite number",
    )


def test_yaml_and_json_loaders_accept_exact_iso_date_strings(
    valid_mapping: dict[str, Any],
) -> None:
    json_spec = load_json(json.dumps(valid_mapping))
    yaml_spec = load_yaml(yaml.safe_dump(valid_mapping))

    assert json_spec.data.start == date(2024, 7, 1)
    assert json_spec.data.end == date(2026, 6, 30)
    assert json_spec.evaluation.final_test_start == date(2026, 1, 2)
    assert yaml_spec.data.start == json_spec.data.start
    assert yaml_spec.data.end == json_spec.data.end
    assert yaml_spec.evaluation.final_test_start == json_spec.evaluation.final_test_start


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
        (
            ("models", 3, "parameters", "window"),
            41,
            ("models", 3, "parameters", "window"),
            "moving_average window must not exceed forecast.context_window",
        ),
        (
            ("models", 5, "parameters", "lags"),
            [1, 41],
            ("models", 5, "parameters", "lags", 1),
            "tree lag must not exceed forecast.context_window",
        ),
        (
            ("statistics", "block_length"),
            253,
            ("statistics", "block_length"),
            "block_length must not exceed evaluation.initial_training_bars",
        ),
    ],
)
def test_cross_section_training_and_evaluation_rules(
    valid_mapping: dict[str, Any],
    path: tuple[str | int, ...],
    value: object,
    location: tuple[str | int, ...],
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
    ("path", "value", "location", "message"),
    [
        (("schema_version",), "2.0", ("schema_version",), "Input should be '1.0'"),
        (("data", "calendar"), "NASDAQ", ("data", "calendar"), "Input should be 'XNYS'"),
        (("data", "interval"), "1h", ("data", "interval"), "Input should be '1d'"),
        (
            ("data", "timezone"),
            "UTC",
            ("data", "timezone"),
            "Input should be 'America/New_York'",
        ),
        (("data", "adjusted"), False, ("data", "adjusted"), "Input should be True"),
        (
            ("data", "corporate_action_policy"),
            "raw",
            ("data", "corporate_action_policy"),
            "Input should be 'provider_adjusted'",
        ),
        (
            ("forecast", "target"),
            "open",
            ("forecast", "target"),
            "Input should be 'close' or 'log_return'",
        ),
        (("features", 0), "rsi", ("features", 0), "Input should be"),
        (
            ("models", 0, "kind"),
            "unsupported",
            ("models", 0),
            "does not match any of the expected tags",
        ),
        (
            ("models", 0, "fit_policy"),
            "fit_per_origin",
            ("models", 0, "kronos", "fit_policy"),
            "Input should be 'zero_shot'",
        ),
        (
            ("models", 0, "parameters", "checkpoint"),
            "NeoQuasar/Kronos-large",
            ("models", 0, "kronos", "parameters", "checkpoint"),
            "Input should be",
        ),
        (
            ("models", 0, "parameters", "tokenizer"),
            "NeoQuasar/Kronos-Tokenizer-large",
            ("models", 0, "kronos", "parameters", "tokenizer"),
            "Input should be",
        ),
        (
            ("models", 4, "parameters", "trend"),
            "linear",
            ("models", 4, "exponential_smoothing", "parameters", "trend"),
            "Input should be 'additive' or 'multiplicative'",
        ),
        (
            ("models", 4, "parameters", "seasonal"),
            "linear",
            ("models", 4, "exponential_smoothing", "parameters", "seasonal"),
            "Input should be 'additive' or 'multiplicative'",
        ),
        (
            ("training", "policy"),
            "sometimes",
            ("training", "policy"),
            "Input should be 'zero_shot' or 'fit_per_origin'",
        ),
        (
            ("evaluation", "protocol"),
            "holdout",
            ("evaluation", "protocol"),
            "Input should be",
        ),
        (
            ("strategy", "kind"),
            "long_short",
            ("strategy", "kind"),
            "Input should be 'long_cash_threshold'",
        ),
        (
            ("execution", "signal_time"),
            "open",
            ("execution", "signal_time"),
            "Input should be 'close'",
        ),
        (
            ("execution", "execution_time"),
            "same_close",
            ("execution", "execution_time"),
            "Input should be 'next_open'",
        ),
        (
            ("benchmark", "kind"),
            "cash",
            ("benchmark", "kind"),
            "Input should be 'buy_and_hold'",
        ),
        (
            ("statistics", "bootstrap_method"),
            "iid",
            ("statistics", "bootstrap_method"),
            "Input should be 'moving_block' or 'stationary'",
        ),
        (
            ("statistics", "multiple_testing"),
            "none",
            ("statistics", "multiple_testing"),
            "Input should be 'holm'",
        ),
        (
            ("reporting", "formats", 0),
            "json",
            ("reporting", "formats", 0),
            "Input should be 'html' or 'pdf'",
        ),
        (
            ("reporting", "include_lineage"),
            False,
            ("reporting", "include_lineage"),
            "Input should be True",
        ),
        (
            ("reporting", "include_limitations"),
            False,
            ("reporting", "include_limitations"),
            "Input should be True",
        ),
    ],
)
def test_declared_literals_are_enforced(
    valid_mapping: dict[str, Any],
    path: tuple[str | int, ...],
    value: object,
    location: tuple[str | int, ...],
    message: str,
) -> None:
    _set_path(valid_mapping, path, value)

    _assert_error(valid_mapping, location, message)


@pytest.mark.parametrize(
    ("path", "value", "location", "message"),
    [
        (("metadata", "name"), "ab", ("metadata", "name"), "at least 3 characters"),
        (("metadata", "name"), "x" * 121, ("metadata", "name"), "at most 120 characters"),
        (
            ("metadata", "description"),
            "too short",
            ("metadata", "description"),
            "at least 20 characters",
        ),
        (("metadata", "owner"), "", ("metadata", "owner"), "at least 1 character"),
        (("hypothesis",), "too short", ("hypothesis",), "at least 20 characters"),
        (("forecast", "context_window"), 19, ("forecast", "context_window"), "greater than or equal to 20"),
        (("forecast", "context_window"), 513, ("forecast", "context_window"), "less than or equal to 512"),
        (("forecast", "horizon"), 0, ("forecast", "horizon"), "greater than or equal to 1"),
        (("forecast", "horizon"), 65, ("forecast", "horizon"), "less than or equal to 64"),
        (
            ("models", 0, "parameters", "temperature"),
            0.0,
            ("models", 0, "kronos", "parameters", "temperature"),
            "greater than 0",
        ),
        (
            ("models", 0, "parameters", "temperature"),
            5.01,
            ("models", 0, "kronos", "parameters", "temperature"),
            "less than or equal to 5",
        ),
        (
            ("models", 0, "parameters", "top_p"),
            0.0,
            ("models", 0, "kronos", "parameters", "top_p"),
            "greater than 0",
        ),
        (
            ("models", 0, "parameters", "top_p"),
            1.01,
            ("models", 0, "kronos", "parameters", "top_p"),
            "less than or equal to 1",
        ),
        (
            ("models", 0, "parameters", "top_k"),
            -1,
            ("models", 0, "kronos", "parameters", "top_k"),
            "greater than or equal to 0",
        ),
        (
            ("models", 0, "parameters", "top_k"),
            1025,
            ("models", 0, "kronos", "parameters", "top_k"),
            "less than or equal to 1024",
        ),
        (
            ("models", 0, "parameters", "sample_count"),
            0,
            ("models", 0, "kronos", "parameters", "sample_count"),
            "greater than or equal to 1",
        ),
        (
            ("models", 0, "parameters", "sample_count"),
            101,
            ("models", 0, "kronos", "parameters", "sample_count"),
            "less than or equal to 100",
        ),
        (
            ("models", 3, "parameters", "window"),
            1,
            ("models", 3, "moving_average", "parameters", "window"),
            "greater than or equal to 2",
        ),
        (
            ("models", 3, "parameters", "window"),
            513,
            ("models", 3, "moving_average", "parameters", "window"),
            "less than or equal to 512",
        ),
        (
            ("models", 4, "parameters", "seasonal_periods"),
            1,
            ("models", 4, "exponential_smoothing", "parameters", "seasonal_periods"),
            "greater than or equal to 2",
        ),
        (
            ("models", 5, "parameters", "lags", 0),
            0,
            ("models", 5, "gradient_boosted_tree", "parameters", "lags", 0),
            "greater than or equal to 1",
        ),
        (
            ("models", 5, "parameters", "lags", 0),
            513,
            ("models", 5, "gradient_boosted_tree", "parameters", "lags", 0),
            "less than or equal to 512",
        ),
        (
            ("models", 5, "parameters", "estimators"),
            9,
            ("models", 5, "gradient_boosted_tree", "parameters", "estimators"),
            "greater than or equal to 10",
        ),
        (
            ("models", 5, "parameters", "estimators"),
            5001,
            ("models", 5, "gradient_boosted_tree", "parameters", "estimators"),
            "less than or equal to 5000",
        ),
        (
            ("models", 5, "parameters", "max_depth"),
            0,
            ("models", 5, "gradient_boosted_tree", "parameters", "max_depth"),
            "greater than or equal to 1",
        ),
        (
            ("models", 5, "parameters", "max_depth"),
            65,
            ("models", 5, "gradient_boosted_tree", "parameters", "max_depth"),
            "less than or equal to 64",
        ),
        (
            ("models", 5, "parameters", "learning_rate"),
            0.0,
            ("models", 5, "gradient_boosted_tree", "parameters", "learning_rate"),
            "greater than 0",
        ),
        (
            ("models", 5, "parameters", "learning_rate"),
            1.01,
            ("models", 5, "gradient_boosted_tree", "parameters", "learning_rate"),
            "less than or equal to 1",
        ),
        (("training", "retraining_cadence"), 0, ("training", "retraining_cadence"), "greater than or equal to 1"),
        (("training", "retraining_cadence"), 2521, ("training", "retraining_cadence"), "less than or equal to 2520"),
        (("evaluation", "initial_training_bars"), 39, ("evaluation", "initial_training_bars"), "greater than or equal to 40"),
        (("evaluation", "step_bars"), 0, ("evaluation", "step_bars"), "greater than or equal to 1"),
        (("evaluation", "purge_bars"), -1, ("evaluation", "purge_bars"), "greater than or equal to 0"),
        (("evaluation", "embargo_bars"), -1, ("evaluation", "embargo_bars"), "greater than or equal to 0"),
        (("strategy", "threshold_bps"), -0.1, ("strategy", "threshold_bps"), "greater than or equal to 0"),
        (("strategy", "neutral_zone_bps"), -0.1, ("strategy", "neutral_zone_bps"), "greater than or equal to 0"),
        (("strategy", "max_position_fraction"), 0.0, ("strategy", "max_position_fraction"), "greater than 0"),
        (("strategy", "max_position_fraction"), 1.01, ("strategy", "max_position_fraction"), "less than or equal to 1"),
        (("execution", "delay_bars"), 0, ("execution", "delay_bars"), "greater than or equal to 1"),
        (("execution", "delay_bars"), 21, ("execution", "delay_bars"), "less than or equal to 20"),
        (("execution", "commission_bps"), -0.1, ("execution", "commission_bps"), "greater than or equal to 0"),
        (("execution", "slippage_bps"), -0.1, ("execution", "slippage_bps"), "greater than or equal to 0"),
        (("statistics", "confidence_level"), 0.79, ("statistics", "confidence_level"), "greater than or equal to 0.8"),
        (("statistics", "confidence_level"), 1.0, ("statistics", "confidence_level"), "less than or equal to 0.999"),
        (("statistics", "bootstrap_samples"), 199, ("statistics", "bootstrap_samples"), "greater than or equal to 200"),
        (("statistics", "bootstrap_samples"), 1_000_001, ("statistics", "bootstrap_samples"), "less than or equal to 1000000"),
        (("statistics", "block_length"), 0, ("statistics", "block_length"), "greater than or equal to 1"),
        (("statistics", "block_length"), 513, ("statistics", "block_length"), "less than or equal to 512"),
        (("statistics", "random_seed"), -1, ("statistics", "random_seed"), "greater than or equal to 0"),
        (("statistics", "random_seed"), 2**32, ("statistics", "random_seed"), "less than or equal to 4294967295"),
        (("reporting", "formats"), [], ("reporting", "formats"), "at least 1 item"),
        (("random_seed",), -1, ("random_seed",), "greater than or equal to 0"),
        (("random_seed",), 2**32, ("random_seed",), "less than or equal to 4294967295"),
    ],
)
def test_declared_local_bounds_are_enforced(
    valid_mapping: dict[str, Any],
    path: tuple[str | int, ...],
    value: object,
    location: tuple[str | int, ...],
    message: str,
) -> None:
    _set_path(valid_mapping, path, value)

    _assert_error(valid_mapping, location, message)


def test_tree_max_depth_may_be_none(valid_mapping: dict[str, Any]) -> None:
    valid_mapping["models"][5]["parameters"]["max_depth"] = None

    model = ExperimentSpec.model_validate(valid_mapping).models[5]
    assert model.kind == "gradient_boosted_tree"
    assert model.parameters.max_depth is None


@pytest.mark.parametrize("formats", [["html"], ["pdf"], ["html", "pdf"]])
def test_reporting_accepts_unique_nonempty_supported_subsets(
    valid_mapping: dict[str, Any], formats: list[str]
) -> None:
    valid_mapping["reporting"]["formats"] = formats

    assert ExperimentSpec.model_validate(valid_mapping).reporting.formats == tuple(formats)


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
