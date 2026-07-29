from collections.abc import Mapping
from datetime import date
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
MetadataName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=120)]
LongText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=20, max_length=2000)]
UppercaseSymbol = Annotated[str, StringConstraints(pattern=r"^[A-Z][A-Z0-9.-]{0,9}$")]
FeatureName = Literal[
    "close_lags",
    "return_lags",
    "rolling_mean",
    "rolling_volatility",
    "calendar",
]


def _parse_wire_date(value: Any) -> Any:
    if isinstance(value, str):
        if len(value) != 10 or value[4] != "-" or value[7] != "-":
            return value
        try:
            return date.fromisoformat(value)
        except ValueError:
            return value
    return value


def _parse_wire_array(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(value)
    return value


WireDate = Annotated[date, BeforeValidator(_parse_wire_date)]

REQUIRED_MODEL_KINDS = {
    "kronos",
    "random_walk",
    "last_value",
    "moving_average",
    "exponential_smoothing",
    "gradient_boosted_tree",
}


def _raise_semantic(
    title: str, location: tuple[str | int, ...], message: str, value: Any
) -> None:
    raise ValidationError.from_exception_data(
        title,
        [
            {
                "type": PydanticCustomError(
                    "semantic_validation", "{message}", {"message": message}
                ),
                "loc": location,
                "input": value,
            }
        ],
    )


def _ensure_unique(
    title: str, values: tuple[str, ...], message: str
) -> tuple[str, ...]:
    seen: set[str] = set()
    for index, value in enumerate(values):
        if value in seen:
            _raise_semantic(title, (index,), message, value)
        seen.add(value)
    return values


class FrozenModel(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
    )

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        if update:
            raise TypeError(
                "model_copy updates bypass validation; create and validate a new experiment"
            )
        return super().model_copy(update=None, deep=deep)


class Metadata(FrozenModel):
    name: MetadataName
    description: LongText
    owner: NonEmptyString
    tags: Annotated[
        tuple[NonEmptyString, ...], BeforeValidator(_parse_wire_array)
    ] = ()

    @field_validator("tags")
    @classmethod
    def tags_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _ensure_unique(cls.__name__, value, "tags must be unique")


class Data(FrozenModel):
    provider: NonEmptyString
    assets: Annotated[
        tuple[UppercaseSymbol, ...],
        Field(min_length=1, max_length=1),
        BeforeValidator(_parse_wire_array),
    ]
    start: WireDate
    end: WireDate
    calendar: Literal["XNYS"]
    interval: Literal["1d"]
    timezone: Literal["America/New_York"]
    adjusted: Literal[True]
    corporate_action_policy: Literal["provider_adjusted"]

    @model_validator(mode="after")
    def valid_date_range(self) -> "Data":
        if self.start <= date(2024, 6, 30):
            _raise_semantic(
                self.__class__.__name__,
                ("start",),
                "start must be strictly after 2024-06-30 for the Phase 1 Kronos boundary",
                self.start,
            )
        if self.start >= self.end:
            _raise_semantic(
                self.__class__.__name__,
                ("start",),
                "start must be before end",
                self.start,
            )
        return self


class Forecast(FrozenModel):
    context_window: int = Field(ge=20, le=512)
    horizon: int = Field(ge=1, le=64)
    target: Literal["close", "log_return"]


class KronosParameters(FrozenModel):
    checkpoint: Literal["NeoQuasar/Kronos-mini", "NeoQuasar/Kronos-base"]
    tokenizer: Literal[
        "NeoQuasar/Kronos-Tokenizer-2k",
        "NeoQuasar/Kronos-Tokenizer-base",
    ]
    temperature: float = Field(gt=0, le=5)
    top_p: float = Field(gt=0, le=1)
    top_k: int = Field(ge=0, le=1024)
    sample_count: int = Field(ge=1, le=100)

    @model_validator(mode="after")
    def tokenizer_matches_checkpoint(self) -> "KronosParameters":
        expected = {
            "NeoQuasar/Kronos-mini": "NeoQuasar/Kronos-Tokenizer-2k",
            "NeoQuasar/Kronos-base": "NeoQuasar/Kronos-Tokenizer-base",
        }[self.checkpoint]
        if self.tokenizer != expected:
            _raise_semantic(
                self.__class__.__name__,
                ("tokenizer",),
                f"tokenizer must match checkpoint family; expected {expected}",
                self.tokenizer,
            )
        return self


class RandomWalkParameters(FrozenModel):
    drift: bool


class LastValueParameters(FrozenModel):
    pass


class MovingAverageParameters(FrozenModel):
    window: int = Field(ge=2, le=512)


class ExponentialSmoothingParameters(FrozenModel):
    trend: Literal["additive", "multiplicative"] | None
    seasonal: Literal["additive", "multiplicative"] | None
    seasonal_periods: int | None = Field(default=None, ge=2)


class GradientBoostedTreeParameters(FrozenModel):
    lags: Annotated[
        tuple[Annotated[int, Field(ge=1, le=512)], ...],
        Field(min_length=1),
        BeforeValidator(_parse_wire_array),
    ]
    estimators: int = Field(ge=10, le=5000)
    max_depth: Annotated[int, Field(ge=1, le=64)] | None
    learning_rate: float = Field(gt=0, le=1)

    @field_validator("lags")
    @classmethod
    def lags_are_unique(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if len(value) != len(set(value)):
            _raise_semantic(cls.__name__, (), "lags must be unique", value)
        return value


class KronosModel(FrozenModel):
    kind: Literal["kronos"]
    name: NonEmptyString
    fit_policy: Literal["zero_shot"]
    parameters: KronosParameters


class RandomWalkModel(FrozenModel):
    kind: Literal["random_walk"]
    name: NonEmptyString
    fit_policy: Literal["fit_per_origin"]
    parameters: RandomWalkParameters


class LastValueModel(FrozenModel):
    kind: Literal["last_value"]
    name: NonEmptyString
    fit_policy: Literal["fit_per_origin"]
    parameters: LastValueParameters


class MovingAverageModel(FrozenModel):
    kind: Literal["moving_average"]
    name: NonEmptyString
    fit_policy: Literal["fit_per_origin"]
    parameters: MovingAverageParameters


class ExponentialSmoothingModel(FrozenModel):
    kind: Literal["exponential_smoothing"]
    name: NonEmptyString
    fit_policy: Literal["fit_per_origin"]
    parameters: ExponentialSmoothingParameters


class GradientBoostedTreeModel(FrozenModel):
    kind: Literal["gradient_boosted_tree"]
    name: NonEmptyString
    fit_policy: Literal["fit_per_origin"]
    parameters: GradientBoostedTreeParameters


ModelDefinition = Annotated[
    KronosModel
    | RandomWalkModel
    | LastValueModel
    | MovingAverageModel
    | ExponentialSmoothingModel
    | GradientBoostedTreeModel,
    Field(discriminator="kind"),
]


class Training(FrozenModel):
    policy: Literal["zero_shot", "fit_per_origin"]
    retraining_cadence: int = Field(ge=1, le=2520)


class Evaluation(FrozenModel):
    protocol: Literal["rolling_origin", "expanding_window", "fixed_rolling"]
    initial_training_bars: int = Field(ge=40)
    step_bars: int = Field(ge=1)
    purge_bars: int = Field(ge=0)
    embargo_bars: int = Field(ge=0)
    final_test_start: WireDate


class Strategy(FrozenModel):
    kind: Literal["long_cash_threshold"]
    threshold_bps: float = Field(ge=0)
    neutral_zone_bps: float = Field(ge=0)
    max_position_fraction: float = Field(gt=0, le=1)


class Execution(FrozenModel):
    signal_time: Literal["close"]
    execution_time: Literal["next_open"]
    delay_bars: int = Field(ge=1, le=20)
    commission_bps: float = Field(ge=0)
    slippage_bps: float = Field(ge=0)


class Benchmark(FrozenModel):
    kind: Literal["buy_and_hold"]
    asset: UppercaseSymbol


class Statistics(FrozenModel):
    confidence_level: float = Field(ge=0.8, le=0.999)
    bootstrap_method: Literal["moving_block", "stationary"]
    bootstrap_samples: int = Field(ge=200, le=1_000_000)
    block_length: int = Field(ge=1, le=512)
    multiple_testing: Literal["holm"]
    random_seed: int = Field(ge=0, le=2**32 - 1)


class Reporting(FrozenModel):
    formats: Annotated[
        tuple[Literal["html", "pdf"], ...],
        Field(min_length=1),
        BeforeValidator(_parse_wire_array),
    ]
    include_lineage: Literal[True]
    include_limitations: Literal[True]

    @field_validator("formats")
    @classmethod
    def formats_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _ensure_unique(cls.__name__, value, "reporting formats must be unique")


class ExperimentSpec(FrozenModel):
    schema_version: Literal["1.0"]
    metadata: Metadata
    hypothesis: LongText
    data: Data
    forecast: Forecast
    features: Annotated[
        tuple[FeatureName, ...],
        Field(min_length=1),
        BeforeValidator(_parse_wire_array),
    ]
    models: Annotated[
        tuple[ModelDefinition, ...],
        Field(min_length=1),
        BeforeValidator(_parse_wire_array),
    ]
    training: Training
    evaluation: Evaluation
    strategy: Strategy
    execution: Execution
    benchmark: Benchmark
    statistics: Statistics
    reporting: Reporting
    random_seed: int = Field(ge=0, le=2**32 - 1)

    @field_validator("features")
    @classmethod
    def features_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _ensure_unique(cls.__name__, value, "feature names must be unique")

    @model_validator(mode="after")
    def semantic_contract(self) -> "ExperimentSpec":
        kinds = [model.kind for model in self.models]
        if set(kinds) != REQUIRED_MODEL_KINDS or len(kinds) != len(REQUIRED_MODEL_KINDS):
            _raise_semantic(
                self.__class__.__name__,
                ("models",),
                "models must contain exactly one of each required kind",
                kinds,
            )

        seen_names: set[str] = set()
        for index, model in enumerate(self.models):
            if model.name in seen_names:
                _raise_semantic(
                    self.__class__.__name__,
                    ("models", index, "name"),
                    "model names must be unique",
                    model.name,
                )
            seen_names.add(model.name)
            if (
                model.kind == "moving_average"
                and model.parameters.window > self.forecast.context_window
            ):
                _raise_semantic(
                    self.__class__.__name__,
                    ("models", index, "parameters", "window"),
                    "moving_average window must not exceed forecast.context_window",
                    model.parameters.window,
                )
            if model.kind == "gradient_boosted_tree":
                for lag_index, lag in enumerate(model.parameters.lags):
                    if lag > self.forecast.context_window:
                        _raise_semantic(
                            self.__class__.__name__,
                            ("models", index, "parameters", "lags", lag_index),
                            "tree lag must not exceed forecast.context_window",
                            lag,
                        )

        if self.training.policy != "fit_per_origin":
            _raise_semantic(
                self.__class__.__name__,
                ("training", "policy"),
                "trainable baselines require fit_per_origin",
                self.training.policy,
            )
        if self.evaluation.initial_training_bars < self.forecast.context_window:
            _raise_semantic(
                self.__class__.__name__,
                ("evaluation", "initial_training_bars"),
                "initial_training_bars must be at least forecast.context_window",
                self.evaluation.initial_training_bars,
            )
        if (
            self.forecast.horizon > 1
            and self.evaluation.purge_bars < self.forecast.horizon - 1
        ):
            _raise_semantic(
                self.__class__.__name__,
                ("evaluation", "purge_bars"),
                "purge_bars must be at least forecast.horizon - 1",
                self.evaluation.purge_bars,
            )
        if self.statistics.block_length < self.forecast.horizon:
            _raise_semantic(
                self.__class__.__name__,
                ("statistics", "block_length"),
                "block_length must be at least forecast.horizon",
                self.statistics.block_length,
            )
        if self.statistics.block_length > self.evaluation.initial_training_bars:
            _raise_semantic(
                self.__class__.__name__,
                ("statistics", "block_length"),
                "block_length must not exceed evaluation.initial_training_bars",
                self.statistics.block_length,
            )
        if not self.data.start < self.evaluation.final_test_start <= self.data.end:
            _raise_semantic(
                self.__class__.__name__,
                ("evaluation", "final_test_start"),
                "final_test_start must be after data.start and on or before data.end",
                self.evaluation.final_test_start,
            )
        if self.execution.commission_bps + self.execution.slippage_bps <= 0:
            _raise_semantic(
                self.__class__.__name__,
                ("execution", "commission_bps"),
                "commission_bps + slippage_bps must be greater than zero",
                self.execution.commission_bps,
            )
        if self.benchmark.asset != self.data.assets[0]:
            _raise_semantic(
                self.__class__.__name__,
                ("benchmark", "asset"),
                "benchmark asset must equal data.assets[0]",
                self.benchmark.asset,
            )
        return self
