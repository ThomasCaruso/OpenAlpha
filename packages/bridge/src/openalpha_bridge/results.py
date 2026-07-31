from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .models import FinancialFeatureTensor, readonly_array


@dataclass(frozen=True, slots=True)
class ReconstructedSequence:
    candles: NDArray[np.float32] | NDArray[np.float64]
    transformed_features: FinancialFeatureTensor
    volume_present: NDArray[np.bool_] | None
    single_sequence: bool
    output_dtype: str
    representation_version: str
    configuration_sha256: str
    numerical_warnings: tuple[str, ...] = ()
    projection_applied: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "candles", readonly_array(self.candles))
        if self.volume_present is not None:
            object.__setattr__(self, "volume_present", readonly_array(self.volume_present))
