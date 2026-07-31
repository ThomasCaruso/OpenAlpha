from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def readonly_array(values: NDArray[np.generic]) -> NDArray[np.generic]:
    result = np.array(values, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class FinancialFeatureTensor:
    """Transformed financial channels plus explicit volume-presence semantics."""

    values: NDArray[np.float32] | NDArray[np.float64]
    volume_present: NDArray[np.bool_] | None
    single_sequence: bool
    representation_version: str
    configuration_sha256: str
    numerical_warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", readonly_array(self.values))
        if self.volume_present is not None:
            object.__setattr__(self, "volume_present", readonly_array(self.volume_present))
