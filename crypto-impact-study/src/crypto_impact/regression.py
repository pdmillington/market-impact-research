"""Optional power-law impact diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass(frozen=True)
class RegressionFilters:
    """User-specified filters applied before optional power-law fitting."""

    min_abs_signed_imbalance: float | None = None
    min_mean_impact: float | None = None
    min_participation_ratio: float | None = None
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
    if filters.min_participation_ratio is not None and "mean_participation_ratio" in data:
        data = data.loc[data["mean_participation_ratio"] >= filters.min_participation_ratio]
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
