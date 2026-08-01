"""Locked Bridge head, training backend protocol, and checkpoint handling.

The architecture is fixed at Linear(269, 64) -> SiLU -> Linear(64, 5) with
exactly 17,605 trainable parameters and is not configurable during Phase 2.

Torch is never imported by this module at import time. Orchestration is tested
against a numpy backend; the Torch backend resolves lazily on a GPU host.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from ..errors import BridgeFailure, BridgeTransformError, FailureCategory
from ..windowing import Partition
from .identity import canonical_sha256
from .kronos import BRIDGE_INPUT_DIMENSION
from .optional import require_module

__all__ = [
    "LOCKED_ARCHITECTURE",
    "LOCKED_HIDDEN_UNITS",
    "LOCKED_OUTPUT_UNITS",
    "LOCKED_PARAMETER_COUNT",
    "CheckpointRecord",
    "EpochRecord",
    "NumpyTrainingBackend",
    "TrainingBackend",
    "TrainingConfig",
    "TrainingHistory",
    "assert_locked_architecture",
    "select_checkpoint",
]

LOCKED_ARCHITECTURE = "Linear(269,64)-SiLU-Linear(64,5)"
LOCKED_HIDDEN_UNITS = 64
LOCKED_OUTPUT_UNITS = 5
LOCKED_PARAMETER_COUNT = 17_605


class Precision(StrEnum):
    FLOAT32 = "float32"


def _fail(code: str, message: str, *, field: str | None = None) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            field=field,
            message=message,
        )
    )


def expected_parameter_count(
    *,
    input_dimension: int = BRIDGE_INPUT_DIMENSION,
    hidden_units: int = LOCKED_HIDDEN_UNITS,
    output_units: int = LOCKED_OUTPUT_UNITS,
) -> int:
    first = input_dimension * hidden_units + hidden_units
    second = hidden_units * output_units + output_units
    return first + second


def assert_locked_architecture(
    *,
    input_dimension: int,
    hidden_units: int,
    output_units: int,
    activation: str,
    trainable_parameters: int,
    frozen_kronos_parameters_require_grad: bool,
    official_head_in_optimizer: bool,
) -> None:
    """Assert every locked architecture invariant at startup."""
    if (input_dimension, hidden_units, output_units) != (
        BRIDGE_INPUT_DIMENSION,
        LOCKED_HIDDEN_UNITS,
        LOCKED_OUTPUT_UNITS,
    ):
        raise _fail(
            "ARCHITECTURE_DIMENSION_MISMATCH",
            (
                f"expected ({BRIDGE_INPUT_DIMENSION}, {LOCKED_HIDDEN_UNITS}, "
                f"{LOCKED_OUTPUT_UNITS}), got "
                f"({input_dimension}, {hidden_units}, {output_units})"
            ),
        )
    if activation.lower() != "silu":
        raise _fail("ACTIVATION_MISMATCH", f"expected SiLU activation, got {activation}")
    if trainable_parameters != LOCKED_PARAMETER_COUNT:
        raise _fail(
            "PARAMETER_COUNT_MISMATCH",
            f"expected {LOCKED_PARAMETER_COUNT} trainable parameters, got {trainable_parameters}",
        )
    if frozen_kronos_parameters_require_grad:
        raise _fail(
            "KRONOS_PARAMETERS_NOT_FROZEN",
            "frozen Kronos parameters must not require gradients",
        )
    if official_head_in_optimizer:
        raise _fail(
            "OFFICIAL_HEAD_IN_OPTIMIZER",
            "the official six-output head must be excluded from Bridge optimization",
        )


class TrainingConfig(BaseModel):
    """The locked optimizer, schedule, and stopping policy."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    schema_version: Literal["openalpha.bridge.phase2.training_config.v1"] = (
        "openalpha.bridge.phase2.training_config.v1"
    )
    optimizer: Literal["AdamW"] = "AdamW"
    learning_rate: float = 0.001
    betas: tuple[float, float] = (0.9, 0.999)
    epsilon: float = 1.0e-8
    weight_decay: float = 0.0001
    gradient_norm_maximum: float = 1.0
    batch_size: int = 64
    maximum_epochs: int = 50
    early_stopping_patience: int = 5
    early_stopping_minimum_delta: float = 0.0001
    seed: int = 1729
    precision: Precision = Precision.FLOAT32
    scheduler: Literal["none"] = "none"

    def assert_locked(self) -> None:
        if self.learning_rate != 0.001 or self.scheduler != "none":
            raise _fail("LEARNING_RATE_POLICY_MISMATCH", "learning rate is locked at constant 0.001")
        if self.seed != 1729:
            raise _fail("SEED_MISMATCH", "the locked training seed is 1729")
        if self.precision is not Precision.FLOAT32:
            raise _fail("PRECISION_MISMATCH", "mixed precision is prohibited for Phase 2")


class EpochRecord(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    epoch: int = Field(ge=1)
    training_loss: float
    validation_total_loss: float
    optimizer_steps: int = Field(ge=0)


class TrainingHistory(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    schema_version: Literal["openalpha.bridge.phase2.history.v1"] = (
        "openalpha.bridge.phase2.history.v1"
    )
    epochs: tuple[EpochRecord, ...]
    stopped_early: bool
    frozen_parameter_sha256_before: str
    frozen_parameter_sha256_after: str

    def assert_frozen_weights_unchanged(self) -> None:
        if self.frozen_parameter_sha256_before != self.frozen_parameter_sha256_after:
            raise _fail(
                "FROZEN_WEIGHTS_CHANGED",
                "frozen Kronos parameters changed during Bridge training",
            )


class CheckpointRecord(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    schema_version: Literal["openalpha.bridge.phase2.checkpoint.v1"] = (
        "openalpha.bridge.phase2.checkpoint.v1"
    )
    selected_epoch: int = Field(ge=1)
    validation_total_loss: float
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_size_bytes: int = Field(ge=0)
    parameter_count: int
    architecture: str
    selection_partition: Literal["validation"] = "validation"

    def assert_locked_caps(self) -> None:
        if self.parameter_count != LOCKED_PARAMETER_COUNT:
            raise _fail(
                "PARAMETER_COUNT_MISMATCH",
                f"checkpoint holds {self.parameter_count} parameters",
            )
        if self.checkpoint_size_bytes > 26_214_400:
            raise _fail(
                "CHECKPOINT_SIZE_CAP_EXCEEDED",
                f"checkpoint is {self.checkpoint_size_bytes} bytes, above the 25 MiB cap",
            )


def select_checkpoint(history: TrainingHistory, *, parameters: dict[str, NDArray[np.float32]]) -> CheckpointRecord:
    """Lowest validation total loss; earliest epoch breaks ties.

    Selection reads validation only. No test quantity participates.
    """
    if not history.epochs:
        raise _fail("NO_EPOCHS_RECORDED", "cannot select a checkpoint from an empty history")

    best = min(history.epochs, key=lambda record: (record.validation_total_loss, record.epoch))
    payload = b"".join(
        np.ascontiguousarray(parameters[name]).tobytes() for name in sorted(parameters)
    )
    total = sum(int(value.size) for value in parameters.values())
    return CheckpointRecord(
        selected_epoch=best.epoch,
        validation_total_loss=best.validation_total_loss,
        checkpoint_sha256=hashlib.sha256(payload).hexdigest(),
        checkpoint_size_bytes=len(payload),
        parameter_count=total,
        architecture=LOCKED_ARCHITECTURE,
    )


class TrainingBackend(Protocol):
    """Numerical backend behind the locked training loop."""

    @property
    def name(self) -> str: ...

    def initialize(self, config: TrainingConfig) -> None: ...

    def parameter_count(self) -> int: ...

    def parameters(self) -> dict[str, NDArray[np.float32]]: ...

    def train_epoch(self, epoch: int) -> float: ...

    def evaluate_validation(self) -> float: ...


class NumpyTrainingBackend:
    """Deterministic numpy backend for orchestration tests.

    Exercises the real loop, selection, and history recording without Torch. It
    reports the locked parameter count so architecture assertions run, but it
    performs no gradient mathematics and is never used for real evidence.
    """

    name = "numpy_orchestration"

    def __init__(self, *, epochs: int = 6, seed: int = 1729) -> None:
        self._epochs = epochs
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._weights: dict[str, NDArray[np.float32]] = {}
        self._epoch = 0

    def initialize(self, config: TrainingConfig) -> None:
        config.assert_locked()
        rng = np.random.default_rng(config.seed)
        self._weights = {
            "layer1.weight": rng.standard_normal(
                (LOCKED_HIDDEN_UNITS, BRIDGE_INPUT_DIMENSION)
            ).astype(np.float32),
            "layer1.bias": np.zeros(LOCKED_HIDDEN_UNITS, dtype=np.float32),
            "layer2.weight": rng.standard_normal(
                (LOCKED_OUTPUT_UNITS, LOCKED_HIDDEN_UNITS)
            ).astype(np.float32),
            "layer2.bias": np.zeros(LOCKED_OUTPUT_UNITS, dtype=np.float32),
        }
        assert_locked_architecture(
            input_dimension=BRIDGE_INPUT_DIMENSION,
            hidden_units=LOCKED_HIDDEN_UNITS,
            output_units=LOCKED_OUTPUT_UNITS,
            activation="SiLU",
            trainable_parameters=self.parameter_count(),
            frozen_kronos_parameters_require_grad=False,
            official_head_in_optimizer=False,
        )

    def parameter_count(self) -> int:
        return sum(int(value.size) for value in self._weights.values())

    def parameters(self) -> dict[str, NDArray[np.float32]]:
        return dict(self._weights)

    def train_epoch(self, epoch: int) -> float:
        self._epoch = epoch
        # Monotone decreasing, deterministic: exercises early stopping.
        return float(1.0 / (1.0 + epoch))

    def evaluate_validation(self) -> float:
        # Improves then plateaus, so patience triggers deterministically.
        return float(1.0 / (1.0 + min(self._epoch, 3)))


class TorchTrainingBackend:
    """The real GPU backend. Resolves Torch lazily; never used locally."""

    name = "torch"

    def __init__(self, *, stage: str, device: str = "cuda") -> None:
        self._stage = stage
        self._device = device
        self._module = None

    def _torch(self):
        return require_module("torch", stage=self._stage)

    def initialize(self, config: TrainingConfig) -> None:
        config.assert_locked()
        torch = self._torch()
        torch.manual_seed(config.seed)
        model = torch.nn.Sequential(
            torch.nn.Linear(BRIDGE_INPUT_DIMENSION, LOCKED_HIDDEN_UNITS),
            torch.nn.SiLU(),
            torch.nn.Linear(LOCKED_HIDDEN_UNITS, LOCKED_OUTPUT_UNITS),
        ).to(self._device)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert_locked_architecture(
            input_dimension=BRIDGE_INPUT_DIMENSION,
            hidden_units=LOCKED_HIDDEN_UNITS,
            output_units=LOCKED_OUTPUT_UNITS,
            activation="SiLU",
            trainable_parameters=trainable,
            frozen_kronos_parameters_require_grad=False,
            official_head_in_optimizer=False,
        )
        self._module = model

    def parameter_count(self) -> int:
        if self._module is None:
            raise _fail("BACKEND_NOT_INITIALIZED", "call initialize before parameter_count")
        return sum(p.numel() for p in self._module.parameters() if p.requires_grad)

    def parameters(self) -> dict[str, NDArray[np.float32]]:
        if self._module is None:
            raise _fail("BACKEND_NOT_INITIALIZED", "call initialize before parameters")
        return {
            name: value.detach().cpu().numpy().astype(np.float32)
            for name, value in self._module.state_dict().items()
        }

    def train_epoch(self, epoch: int) -> float:
        raise _fail(
            "TRAINING_LOOP_REQUIRES_DATA",
            "the Torch backend trains only from a materialized feature cache on a GPU host",
        )

    def evaluate_validation(self) -> float:
        raise _fail(
            "TRAINING_LOOP_REQUIRES_DATA",
            "the Torch backend evaluates only from a materialized feature cache",
        )


def run_training(
    backend: TrainingBackend,
    config: TrainingConfig,
    *,
    frozen_sha256_before: str,
    frozen_sha256_after: str,
) -> TrainingHistory:
    """The locked training loop: early stopping on validation total loss."""
    config.assert_locked()
    backend.initialize(config)

    records: list[EpochRecord] = []
    best = float("inf")
    since_improvement = 0
    stopped_early = False

    for epoch in range(1, config.maximum_epochs + 1):
        training_loss = backend.train_epoch(epoch)
        validation_loss = backend.evaluate_validation()
        records.append(
            EpochRecord(
                epoch=epoch,
                training_loss=training_loss,
                validation_total_loss=validation_loss,
                optimizer_steps=epoch,
            )
        )
        if validation_loss < best - config.early_stopping_minimum_delta:
            best = validation_loss
            since_improvement = 0
        else:
            since_improvement += 1
            if since_improvement >= config.early_stopping_patience:
                stopped_early = True
                break

    history = TrainingHistory(
        epochs=tuple(records),
        stopped_early=stopped_early,
        frozen_parameter_sha256_before=frozen_sha256_before,
        frozen_parameter_sha256_after=frozen_sha256_after,
    )
    history.assert_frozen_weights_unchanged()
    return history


def preprocessing_state_sha256(
    *,
    scaler_median: float,
    scaler_scale: float,
    fitted_partitions: frozenset[Partition],
) -> str:
    """Hash of the training-only preprocessing state."""
    from ..windowing import assert_preprocessing_fit_partitions

    assert_preprocessing_fit_partitions(fitted_partitions)
    return canonical_sha256(
        {
            "scaler_median": scaler_median,
            "scaler_scale": scaler_scale,
            "fitted_partitions": sorted(p.value for p in fitted_partitions),
        }
    )
