"""Small, interpretable dynamic extensions to a frozen impact surface."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


MODEL_FEATURES = {
    "bias_only": ("bias_buy", "bias_sell"),
    "lag_impact": ("bias_buy", "bias_sell", "lag_impulse"),
    "lag_activity": (
        "bias_buy",
        "bias_sell",
        "lag_impulse",
        "lag_impulse_x_previous_activity",
    ),
    "lag_activity_duration": (
        "bias_buy",
        "bias_sell",
        "lag_impulse",
        "lag_impulse_x_previous_activity",
        "lag_impulse_x_duration",
    ),
}


@dataclass(frozen=True)
class LaggedImpactFit:
    """Linear correction fitted after the nonlinear impact surface is frozen."""

    model: str
    feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    observations: int
    periods: int
    scaled_condition_number: float

    def coefficient(self, name: str) -> float:
        """Return a named coefficient, or zero when the model omits it."""

        if name not in self.feature_names:
            return 0.0
        return self.coefficients[self.feature_names.index(name)]


def lagged_design(
    model: str,
    *,
    current_side: np.ndarray,
    lag_impulse: np.ndarray,
    previous_log_activity: np.ndarray,
    log_duration: np.ndarray,
) -> np.ndarray:
    """Construct one of the nested lag-correction design matrices."""

    if model not in MODEL_FEATURES:
        raise ValueError(
            f"Unknown lag model {model!r}; expected one of {sorted(MODEL_FEATURES)}"
        )
    side = np.asarray(current_side)
    lag = np.asarray(lag_impulse, dtype=float)
    activity = np.asarray(previous_log_activity, dtype=float)
    duration = np.asarray(log_duration, dtype=float)
    if len({len(side), len(lag), len(activity), len(duration)}) != 1:
        raise ValueError("All lagged-design inputs must have the same length.")
    available = {
        "bias_buy": (side > 0).astype(float),
        "bias_sell": (side < 0).astype(float),
        "lag_impulse": lag,
        "lag_impulse_x_previous_activity": lag * activity,
        "lag_impulse_x_duration": lag * duration,
    }
    return np.column_stack([available[name] for name in MODEL_FEATURES[model]])


def fit_equal_period_ols(
    model: str,
    *,
    current_side: np.ndarray,
    lag_impulse: np.ndarray,
    previous_log_activity: np.ndarray,
    log_duration: np.ndarray,
    residual: np.ndarray,
    period: np.ndarray,
) -> LaggedImpactFit:
    """Fit a correction with each calendar period receiving equal total weight.

    The small normal equations are accumulated period by period, avoiding a
    second full-size design matrix for long event-bar histories. Columns are
    RMS-scaled before the pseudoinverse is taken, improving numerical stability
    without changing the reported coefficients or fitted values.
    """

    y = np.asarray(residual, dtype=float)
    periods = np.asarray(period)
    design = lagged_design(
        model,
        current_side=current_side,
        lag_impulse=lag_impulse,
        previous_log_activity=previous_log_activity,
        log_duration=log_duration,
    )
    if len(y) != len(periods) or len(y) != len(design):
        raise ValueError("Response, period and design arrays must be aligned.")
    finite = np.isfinite(y) & np.isfinite(design).all(axis=1)
    if finite.sum() <= design.shape[1]:
        raise ValueError("Too few finite observations for the requested model.")
    y = y[finite]
    design = design[finite]
    periods = periods[finite]
    unique_periods = np.unique(periods)
    gram = np.zeros((design.shape[1], design.shape[1]), dtype=float)
    moment = np.zeros(design.shape[1], dtype=float)
    for value in unique_periods:
        chosen = periods == value
        local_x = design[chosen]
        local_y = y[chosen]
        gram += (local_x.T @ local_x) / chosen.sum()
        moment += (local_x.T @ local_y) / chosen.sum()
    gram /= len(unique_periods)
    moment /= len(unique_periods)
    scales = np.sqrt(np.diag(gram))
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("Lagged design contains an empty or degenerate column.")
    scaled_gram = gram / np.outer(scales, scales)
    scaled_moment = moment / scales
    scaled_coefficients = np.linalg.pinv(scaled_gram, rcond=1e-10) @ scaled_moment
    coefficients = scaled_coefficients / scales
    return LaggedImpactFit(
        model=model,
        feature_names=MODEL_FEATURES[model],
        coefficients=tuple(float(value) for value in coefficients),
        observations=int(len(y)),
        periods=int(len(unique_periods)),
        scaled_condition_number=float(np.linalg.cond(scaled_gram)),
    )


def predict_lagged_correction(
    fit: LaggedImpactFit,
    *,
    current_side: np.ndarray,
    lag_impulse: np.ndarray,
    previous_log_activity: np.ndarray,
    log_duration: np.ndarray,
) -> np.ndarray:
    """Apply a fitted correction without changing the frozen base prediction."""

    design = lagged_design(
        fit.model,
        current_side=current_side,
        lag_impulse=lag_impulse,
        previous_log_activity=previous_log_activity,
        log_duration=log_duration,
    )
    return design @ np.asarray(fit.coefficients)

