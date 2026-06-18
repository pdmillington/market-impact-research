"""Power-law impact regression."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass(frozen=True)
class PowerLawFit:
    """Estimated parameters for ``impact = A * Q**delta``."""

    amplitude: float
    delta: float
    intercept: float
    r_squared: float
    n_obs: int


def fit_power_law(
    binned: pd.DataFrame,
    q_col: str = "mean_abs_imbalance",
    impact_col: str = "mean_signed_impact",
) -> PowerLawFit:
    """Fit a power law by OLS regression in log-log space."""

    required = {q_col, impact_col}
    missing = required.difference(binned.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    data = binned.loc[(binned[q_col] > 0) & (binned[impact_col] > 0), [q_col, impact_col]]
    if len(data) < 2:
        raise ValueError("At least two positive binned observations are required.")

    x = np.log(data[q_col].astype(float).to_numpy())
    y = np.log(data[impact_col].astype(float).to_numpy())
    x_with_const = sm.add_constant(x)
    result = sm.OLS(y, x_with_const).fit()

    intercept = float(result.params[0])
    delta = float(result.params[1])
    return PowerLawFit(
        amplitude=float(np.exp(intercept)),
        delta=delta,
        intercept=intercept,
        r_squared=float(result.rsquared),
        n_obs=int(len(data)),
    )


def predict_power_law(q_values: np.ndarray, fit: PowerLawFit) -> np.ndarray:
    """Predict impact for imbalance values using a fitted power law."""

    return fit.amplitude * np.asarray(q_values, dtype=float) ** fit.delta
