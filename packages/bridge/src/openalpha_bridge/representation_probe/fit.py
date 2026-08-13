"""Standardization, estimators, hyperparameter selection and the fit phase.

Both estimators are implemented here in closed form or by a fixed-iteration
deterministic solve rather than delegated to a library. A study that must
reproduce years from now should not depend on a third-party solver's default
tolerance, its iteration order, or its version. Ridge has an exact solution;
penalised logistic regression is fitted by damped IRLS with a declared iteration
cap and tolerance, and the achieved iteration count is published.

The harness is identical for every feature set. Only the input matrix differs --
that is what makes a comparison against the controls mean anything.
"""

from __future__ import annotations

import math
from typing import Any, Final, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..errors import BridgeFailure, BridgeTransformError, FailureCategory
from .spec import LOGISTIC_C_GRID, RIDGE_ALPHA_GRID

__all__ = [
    "FeatureSetFit",
    "LogisticFit",
    "RidgeFit",
    "Standardizer",
    "evaluate_regression",
    "fit_feature_set",
    "fit_logistic",
    "fit_ridge",
]

#: Declared so the solve is reproducible rather than "whatever converged".
LOGISTIC_MAX_ITERATIONS: Final[int] = 100
LOGISTIC_TOLERANCE: Final[float] = 1e-8
#: Guards a singular weighted Hessian when a class is nearly separable.
LOGISTIC_RIDGE_FLOOR: Final[float] = 1e-10


def _fail(code: str, message: str, *, field: str | None = None) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code=code,
            field=field,
            message=message,
        )
    )


class Standardizer(BaseModel):
    """Column means and spreads, fitted on the training partition only.

    Validation and test rows are transformed with these statistics and never
    refit. Zero-variance columns are divided by one rather than by zero, which
    maps a constant column to a constant zero rather than to a NaN.
    """

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    mean: tuple[float, ...]
    scale: tuple[float, ...]
    fitted_rows: int = Field(ge=0)
    dimension: int = Field(ge=0)

    @classmethod
    def fit(cls, matrix: np.ndarray) -> Standardizer:
        if matrix.ndim != 2:
            raise _fail("PROBE_STANDARDIZER_RANK", f"expected 2-D, got {matrix.ndim}-D")
        if matrix.shape[1] == 0:
            return cls(mean=(), scale=(), fitted_rows=int(matrix.shape[0]), dimension=0)
        mean = matrix.mean(axis=0)
        spread = matrix.std(axis=0, ddof=0)
        scale = np.where(spread > 0.0, spread, 1.0)
        return cls(
            mean=tuple(float(v) for v in mean),
            scale=tuple(float(v) for v in scale),
            fitted_rows=int(matrix.shape[0]),
            dimension=int(matrix.shape[1]),
        )

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        if self.dimension == 0:
            return np.zeros((matrix.shape[0], 0), dtype=np.float64)
        if matrix.shape[1] != self.dimension:
            raise _fail(
                "PROBE_STANDARDIZER_DIMENSION_MISMATCH",
                f"matrix has {matrix.shape[1]} columns, standardizer fitted {self.dimension}",
            )
        return (matrix - np.asarray(self.mean)) / np.asarray(self.scale)


class RidgeFit(BaseModel):
    """A fitted ridge model. Coefficients are published, so it reconstructs."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    alpha: float
    intercept: float
    coefficients: tuple[float, ...]
    dimension: int = Field(ge=0)
    fitted_rows: int = Field(ge=0)

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        if self.dimension == 0:
            return np.full(matrix.shape[0], self.intercept, dtype=np.float64)
        return matrix @ np.asarray(self.coefficients) + self.intercept


class LogisticFit(BaseModel):
    """A fitted L2 logistic model, with its convergence recorded."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    inverse_regularization: float
    intercept: float
    coefficients: tuple[float, ...]
    dimension: int = Field(ge=0)
    fitted_rows: int = Field(ge=0)
    iterations: int = Field(ge=0)
    converged: bool
    solver: Literal["damped_irls"] = "damped_irls"
    max_iterations: int = LOGISTIC_MAX_ITERATIONS
    tolerance: float = LOGISTIC_TOLERANCE

    def decision_function(self, matrix: np.ndarray) -> np.ndarray:
        if self.dimension == 0:
            return np.full(matrix.shape[0], self.intercept, dtype=np.float64)
        return matrix @ np.asarray(self.coefficients) + self.intercept

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(self.decision_function(matrix), -60.0, 60.0)))


def fit_ridge(matrix: np.ndarray, target: np.ndarray, *, alpha: float) -> RidgeFit:
    """Closed-form ridge with an unpenalised intercept.

    Solving ``(X'X + alpha I) w = X'y`` on centred data gives an exact solution
    with no tolerance and no iteration, so the result is bit-reproducible.
    """
    rows = int(matrix.shape[0])
    if rows == 0:
        raise _fail("PROBE_FIT_NO_ROWS", "cannot fit on zero rows")
    if matrix.shape[0] != target.shape[0]:
        raise _fail(
            "PROBE_FIT_SHAPE_MISMATCH",
            f"{matrix.shape[0]} feature rows against {target.shape[0]} targets",
        )
    dimension = int(matrix.shape[1])
    intercept = float(target.mean())
    if dimension == 0:
        return RidgeFit(
            alpha=alpha, intercept=intercept, coefficients=(), dimension=0, fitted_rows=rows
        )

    centre = matrix.mean(axis=0)
    centred = matrix - centre
    residual = target - intercept
    gram = centred.T @ centred + alpha * np.eye(dimension)
    weights = np.linalg.solve(gram, centred.T @ residual)
    # Fold the centring back into the intercept so predict() takes raw columns.
    return RidgeFit(
        alpha=alpha,
        intercept=float(intercept - float(centre @ weights)),
        coefficients=tuple(float(v) for v in weights),
        dimension=dimension,
        fitted_rows=rows,
    )


def fit_logistic(matrix: np.ndarray, labels: np.ndarray, *, inverse_regularization: float
                 ) -> LogisticFit:
    """L2 logistic regression by damped IRLS. Deterministic and capped.

    ``inverse_regularization`` is the usual C: the penalty is ``1 / C``. The
    intercept is unpenalised. Iteration count and convergence are published so a
    reader can see whether the solve actually settled.
    """
    rows = int(matrix.shape[0])
    if rows == 0:
        raise _fail("PROBE_FIT_NO_ROWS", "cannot fit on zero rows")
    if matrix.shape[0] != labels.shape[0]:
        raise _fail(
            "PROBE_FIT_SHAPE_MISMATCH",
            f"{matrix.shape[0]} feature rows against {labels.shape[0]} labels",
        )
    dimension = int(matrix.shape[1])
    binary = (labels > 0).astype(np.float64)

    design = np.hstack([np.ones((rows, 1)), matrix]) if dimension else np.ones((rows, 1))
    beta = np.zeros(design.shape[1], dtype=np.float64)
    # Unpenalised intercept, penalty 1/C on every real coefficient.
    penalty = np.full(design.shape[1], 1.0 / inverse_regularization, dtype=np.float64)
    penalty[0] = 0.0

    iterations, converged = 0, False
    for iterations in range(1, LOGISTIC_MAX_ITERATIONS + 1):
        eta = np.clip(design @ beta, -60.0, 60.0)
        probability = 1.0 / (1.0 + np.exp(-eta))
        weight = np.clip(probability * (1.0 - probability), LOGISTIC_RIDGE_FLOOR, None)
        gradient = design.T @ (binary - probability) - penalty * beta
        hessian = (design * weight[:, None]).T @ design + np.diag(penalty)
        # The floor plus the penalty keep this solvable even when a class is
        # nearly separable, which is likely at 832 features and 84 rows.
        step = np.linalg.solve(hessian + LOGISTIC_RIDGE_FLOOR * np.eye(hessian.shape[0]), gradient)
        beta = beta + step
        if float(np.max(np.abs(step))) < LOGISTIC_TOLERANCE:
            converged = True
            break

    return LogisticFit(
        inverse_regularization=inverse_regularization,
        intercept=float(beta[0]),
        coefficients=tuple(float(v) for v in beta[1:]),
        dimension=dimension,
        fitted_rows=rows,
        iterations=iterations,
        converged=converged,
    )


def evaluate_regression(predicted: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    """The primary metric and its regression secondaries."""
    if predicted.shape != actual.shape:
        raise _fail("PROBE_METRIC_SHAPE_MISMATCH", "predicted and actual shapes differ")
    if predicted.size == 0:
        raise _fail("PROBE_METRIC_NO_ROWS", "cannot score zero rows")
    errors = predicted - actual
    mae = float(np.mean(np.abs(errors)))
    rmse = float(math.sqrt(float(np.mean(errors**2))))
    total = float(np.sum((actual - float(np.mean(actual))) ** 2))
    residual = float(np.sum(errors**2))
    r2 = float("nan") if total == 0.0 else 1.0 - residual / total
    return {
        "horizon_return_mae": mae,
        "horizon_return_rmse": rmse,
        "out_of_sample_r2": r2,
    }


def directional_accuracy(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Sign agreement, excluding target ties which have no direction."""
    mask = labels != 0
    if not bool(np.any(mask)):
        return None
    predicted = np.where(scores > 0.0, 1, -1)[mask]
    return float(np.mean(predicted == np.sign(labels[mask])))


def directional_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Rank-based AUC (Mann-Whitney), with mid-ranks for ties."""
    positive = labels > 0
    negative = labels < 0
    n_pos, n_neg = int(np.sum(positive)), int(np.sum(negative))
    if n_pos == 0 or n_neg == 0:
        return None
    considered = scores[positive | negative]
    order = np.argsort(considered, kind="stable")
    ranks = np.empty(considered.shape[0], dtype=np.float64)
    sorted_scores = considered[order]
    index = 0
    while index < sorted_scores.size:
        stop = index
        while stop + 1 < sorted_scores.size and sorted_scores[stop + 1] == sorted_scores[index]:
            stop += 1
        ranks[order[index : stop + 1]] = (index + stop) / 2.0 + 1.0
        index = stop + 1
    positive_ranks = ranks[(labels[positive | negative] > 0)]
    return float((positive_ranks.sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


class FeatureSetFit(BaseModel):
    """Everything needed to reconstruct one feature set's fitted models."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    feature_set: str
    dimension: int = Field(ge=0)
    feature_names: tuple[str, ...]
    standardizer: Standardizer
    ridge_alpha_grid: tuple[float, ...] = RIDGE_ALPHA_GRID
    logistic_c_grid: tuple[float, ...] = LOGISTIC_C_GRID
    selected_alpha: float
    selected_inverse_regularization: float
    validation_metrics_by_alpha: dict[str, dict[str, float]]
    validation_directional_by_c: dict[str, float | None]
    selected_validation_metrics: dict[str, float]
    selected_validation_directional_accuracy: float | None = None
    selected_validation_directional_auc: float | None = None
    #: Refitted on train plus validation, exactly once, before test opens.
    ridge: RidgeFit
    logistic: LogisticFit
    train_rows: int = Field(ge=0)
    validation_rows: int = Field(ge=0)
    refit_rows: int = Field(ge=0)


def fit_feature_set(
    *,
    feature_set: str,
    names: tuple[str, ...],
    train_matrix: np.ndarray,
    train_target: np.ndarray,
    train_labels: np.ndarray,
    validation_matrix: np.ndarray,
    validation_target: np.ndarray,
    validation_labels: np.ndarray,
) -> FeatureSetFit:
    """Select on validation, then refit once on train plus validation.

    Identical for every feature set: same standardizer, same grids, same
    selection rule, same refit. Only the matrices differ.
    """
    standardizer = Standardizer.fit(train_matrix)
    train_scaled = standardizer.transform(train_matrix)
    validation_scaled = standardizer.transform(validation_matrix)

    # Ridge: select the alpha minimising validation primary metric.
    by_alpha: dict[str, dict[str, float]] = {}
    best_alpha, best_score = RIDGE_ALPHA_GRID[0], math.inf
    for alpha in RIDGE_ALPHA_GRID:
        model = fit_ridge(train_scaled, train_target, alpha=alpha)
        metrics = evaluate_regression(model.predict(validation_scaled), validation_target)
        by_alpha[repr(alpha)] = metrics
        score = metrics["horizon_return_mae"]
        # Strict improvement only, so ties keep the smaller alpha and the
        # selection is deterministic under grid order.
        if score < best_score:
            best_alpha, best_score = alpha, score

    # Logistic: select the C maximising validation directional accuracy.
    by_c: dict[str, float | None] = {}
    best_c, best_accuracy = LOGISTIC_C_GRID[0], -math.inf
    for inverse in LOGISTIC_C_GRID:
        model = fit_logistic(train_scaled, train_labels, inverse_regularization=inverse)
        accuracy = directional_accuracy(model.decision_function(validation_scaled),
                                        validation_labels)
        by_c[repr(inverse)] = accuracy
        if accuracy is not None and accuracy > best_accuracy:
            best_c, best_accuracy = inverse, accuracy

    # Refit the selected logistic on train alone purely to score validation for
    # the record. The published model is the train-plus-validation refit below.
    selected_logistic_on_train = fit_logistic(
        train_scaled, train_labels, inverse_regularization=best_c
    )
    validation_scores = selected_logistic_on_train.decision_function(validation_scaled)

    # Refit the coefficients once on train plus validation with the frozen
    # hyperparameters. The STANDARDIZER IS NOT REFIT: the specification says
    # features are standardized using training-partition statistics only, and
    # that validation and test rows are transformed with those statistics and
    # never refit. Refitting it here would fold validation-row means and spreads
    # into the preprocessing state that later transforms the test partition,
    # which is a quiet path for information to cross a partition boundary.
    combined_matrix = np.vstack([train_matrix, validation_matrix])
    combined_target = np.concatenate([train_target, validation_target])
    combined_labels = np.concatenate([train_labels, validation_labels])
    combined_scaled = standardizer.transform(combined_matrix)

    return FeatureSetFit(
        feature_set=feature_set,
        dimension=int(train_matrix.shape[1]),
        feature_names=names,
        # The train-only statistics, published so the test phase uses exactly
        # these and nothing derived from validation or test rows.
        standardizer=standardizer,
        selected_alpha=best_alpha,
        selected_inverse_regularization=best_c,
        validation_metrics_by_alpha=by_alpha,
        validation_directional_by_c=by_c,
        selected_validation_metrics=by_alpha[repr(best_alpha)],
        selected_validation_directional_accuracy=directional_accuracy(
            validation_scores, validation_labels
        ),
        selected_validation_directional_auc=directional_auc(validation_scores, validation_labels),
        ridge=fit_ridge(combined_scaled, combined_target, alpha=best_alpha),
        logistic=fit_logistic(
            combined_scaled, combined_labels, inverse_regularization=best_c
        ),
        train_rows=int(train_matrix.shape[0]),
        validation_rows=int(validation_matrix.shape[0]),
        refit_rows=int(combined_matrix.shape[0]),
    )


def matrix_from_rows(rows: list[tuple[float, ...]], *, dimension: int) -> np.ndarray:
    """Design matrix from feature tuples, with the width asserted."""
    if dimension == 0:
        return np.zeros((len(rows), 0), dtype=np.float64)
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != dimension:
        raise _fail(
            "PROBE_FEATURE_MATRIX_SHAPE_MISMATCH",
            f"expected {len(rows)}x{dimension}, got {matrix.shape}",
        )
    if not bool(np.all(np.isfinite(matrix))):
        raise _fail("PROBE_FEATURE_MATRIX_NON_FINITE", "a feature value is not finite")
    return matrix


def digest_of(value: Any) -> str:
    """Canonical digest of a fitted object, for artifact cross-checks."""
    import hashlib

    from ..phase2.identity import canonical_json

    return hashlib.sha256(canonical_json(value)).hexdigest()


__all__ += ["digest_of", "directional_accuracy", "directional_auc", "matrix_from_rows"]
