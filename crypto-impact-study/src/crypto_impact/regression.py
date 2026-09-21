"""Optional power-law impact diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import statsmodels.api as sm


@dataclass(frozen=True)
class RegressionFilters:
    """User-specified filters applied before optional power-law fitting."""

    min_abs_signed_imbalance: float | None = None
    min_mean_impact: float | None = None
    min_absolute_imbalance_ratio: float | None = None
    min_obs_per_bin: int | None = None

    @property
    def min_signed_impact(self) -> float | None:
        """Backward-compatible alias."""

        return self.min_mean_impact


@dataclass(frozen=True)
class PowerLawFit:
    """Estimated parameters for ``impact = A * x**delta``."""

    amplitude: float
    delta: float
    intercept: float
    r_squared: float
    n_fit_bins: int
    standard_error_delta: float
    confidence_interval_delta_95: tuple[float, float]
    x_col: str = "mean_x"
    filters: dict[str, float | int | None] | None = None

    @property
    def n_obs(self) -> int:
        """Backward-compatible alias for the number of fitted bins."""

        return self.n_fit_bins


def apply_regression_filters(binned: pd.DataFrame, filters: RegressionFilters | None) -> pd.DataFrame:
    """Apply configured bin-level filters before fitting."""

    if filters is None:
        return binned.copy()
    data = binned.copy()
    if filters.min_abs_signed_imbalance is not None and "mean_abs_signed_imbalance" in data:
        data = data.loc[data["mean_abs_signed_imbalance"] >= filters.min_abs_signed_imbalance]
    if filters.min_mean_impact is not None:
        impact_col = "mean_impact" if "mean_impact" in data else "mean_signed_impact"
        data = data.loc[data[impact_col] >= filters.min_mean_impact]
    if filters.min_absolute_imbalance_ratio is not None and "mean_absolute_imbalance_ratio" in data:
        data = data.loc[data["mean_absolute_imbalance_ratio"] >= filters.min_absolute_imbalance_ratio]
    if filters.min_obs_per_bin is not None:
        data = data.loc[data["n_obs"] >= filters.min_obs_per_bin]
    return data.copy()


def fit_power_law(
    binned: pd.DataFrame,
    q_col: str = "mean_x",
    impact_col: str = "mean_impact",
    filters: RegressionFilters | None = None,
) -> PowerLawFit:
    """Fit a secondary power-law diagnostic by OLS in log-log space."""

    if impact_col not in binned.columns and impact_col == "mean_impact":
        impact_col = "mean_signed_impact"
    if q_col not in binned.columns and q_col == "mean_x" and "mean_abs_imbalance" in binned.columns:
        q_col = "mean_abs_imbalance"
    required = {q_col, impact_col}
    missing = required.difference(binned.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    filtered = apply_regression_filters(binned, filters)
    data = filtered.loc[(filtered[q_col] > 0) & (filtered[impact_col] > 0), [q_col, impact_col]]
    if len(data) < 2:
        raise ValueError("At least two positive binned observations are required.")

    x = np.log(data[q_col].astype(float).to_numpy())
    y = np.log(data[impact_col].astype(float).to_numpy())
    result = sm.OLS(y, sm.add_constant(x)).fit()

    intercept = float(result.params[0])
    delta = float(result.params[1])
    se_delta = float(result.bse[1]) if len(result.bse) > 1 else float("nan")
    ci_delta = result.conf_int(alpha=0.05)[1]
    return PowerLawFit(
        amplitude=float(np.exp(intercept)),
        delta=delta,
        intercept=intercept,
        r_squared=float(result.rsquared),
        n_fit_bins=int(len(data)),
        standard_error_delta=se_delta,
        confidence_interval_delta_95=(float(ci_delta[0]), float(ci_delta[1])),
        x_col=q_col,
        filters=asdict(filters) if filters is not None else asdict(RegressionFilters()),
    )


def fit_to_summary_row(
    fit: PowerLawFit,
    *,
    x_variable: str,
    sampling_method: str,
    window_length: str | None,
    window_step: str | None,
) -> dict[str, object]:
    """Convert a fit to the summary CSV schema."""

    return {
        "x_variable": x_variable,
        "sampling_method": sampling_method,
        "window_length": window_length,
        "window_step": window_step,
        "number_of_bins_used": fit.n_fit_bins,
        "amplitude_A": fit.amplitude,
        "exponent_delta": fit.delta,
        "intercept_log_A": fit.intercept,
        "r_squared": fit.r_squared,
        "standard_error_delta": fit.standard_error_delta,
        "confidence_interval_delta_95": f"[{fit.confidence_interval_delta_95[0]}, {fit.confidence_interval_delta_95[1]}]",
        "filter_settings": fit.filters,
    }


def predict_power_law(q_values: np.ndarray, fit: PowerLawFit) -> np.ndarray:
    """Predict impact for explanatory-variable values using a fitted power law."""

    return fit.amplitude * np.asarray(q_values, dtype=float) ** fit.delta


def compare_impact_models(
    binned: pd.DataFrame,
    *,
    x_col: str = "mean_x",
    impact_col: str = "mean_impact",
    standard_error_col: str = "standard_error",
    min_obs_per_bin: int | None = None,
    x_scale: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare square-root, free power, linear, and logarithmic curves.

    Models are fitted to positive binned conditional mean impact. Reported
    errors are evaluated in impact levels, making the alternatives directly
    comparable. This is a descriptive comparison rather than model selection.
    """

    required = {x_col, impact_col}
    missing = required.difference(binned.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    columns = [x_col, impact_col]
    if standard_error_col in binned:
        columns.append(standard_error_col)
    if "n_obs" in binned:
        columns.append("n_obs")
    data = binned.loc[:, columns].replace([np.inf, -np.inf], np.nan).dropna(
        subset=[x_col, impact_col]
    )
    data = data.loc[data[x_col] > 0].copy()
    if min_obs_per_bin is not None and "n_obs" in data:
        data = data.loc[data["n_obs"] >= min_obs_per_bin]
    if len(data) < 3:
        raise ValueError("At least three positive binned observations are required.")

    x = data[x_col].astype(float).to_numpy()
    y = data[impact_col].astype(float).to_numpy()
    if standard_error_col in data:
        sigma = data[standard_error_col].astype(float).to_numpy()
        valid_sigma = np.isfinite(sigma) & (sigma > 0)
        fallback = np.nanmedian(sigma[valid_sigma]) if valid_sigma.any() else 1.0
        sigma = np.where(valid_sigma, sigma, fallback)
    else:
        sigma = np.ones_like(y)
    weights = 1.0 / sigma**2

    def fixed_power_prediction(delta: float) -> tuple[float, np.ndarray]:
        basis = x**delta
        amplitude = float(np.sum(weights * basis * y) / np.sum(weights * basis**2))
        return amplitude, amplitude * basis

    sqrt_amplitude, sqrt_prediction = fixed_power_prediction(0.5)
    linear_amplitude, linear_prediction = fixed_power_prediction(1.0)

    def power_curve(values: np.ndarray, amplitude: float, delta: float) -> np.ndarray:
        return amplitude * values**delta

    free_params, _ = curve_fit(
        power_curve,
        x,
        y,
        p0=(sqrt_amplitude, 0.5),
        sigma=sigma,
        bounds=([0.0, 0.0], [np.inf, 2.0]),
        maxfev=20_000,
    )
    free_prediction = power_curve(x, *free_params)

    if x_scale is None: 
        x_scale = float(np.median(x))
    else:
        x_scale = float(x_scale)

    def log_curve(values: np.ndarray, amplitude: float, curvature: float) -> np.ndarray:
        return amplitude * np.log1p(curvature * values / x_scale)

    log_params, _ = curve_fit(
        log_curve,
        x,
        y,
        p0=(float(np.median(y)), 1.0),
        sigma=sigma,
        bounds=([0.0, 0.0], [np.inf, np.inf]),
        maxfev=20_000,
    )
    log_prediction = log_curve(x, *log_params)

    models = [
        ("square_root", sqrt_prediction, sqrt_amplitude, 0.5, np.nan),
        ("free_power", free_prediction, float(free_params[0]), float(free_params[1]), np.nan),
        ("linear", linear_prediction, linear_amplitude, 1.0, np.nan),
        ("logarithmic", log_prediction, float(log_params[0]), np.nan, float(log_params[1])),
    ]
    summary_rows = []
    prediction_rows = []
    for name, prediction, amplitude, delta, curvature in models:
        residual = y - prediction
        summary_rows.append(
            {
                "model": name,
                "amplitude": amplitude,
                "delta": delta,
                "curvature": curvature,
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "weighted_rmse": float(
                    np.sqrt(np.sum(weights * residual**2) / np.sum(weights))
                ),
                "mae": float(np.mean(np.abs(residual))),
                "bins_used": len(data),
            }
        )
        prediction_rows.extend(
            {"model": name, "x": x_value, "predicted_impact": y_hat}
            for x_value, y_hat in zip(x, prediction, strict=True)
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(prediction_rows)
