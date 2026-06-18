"""Non-parametric impact-shape diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess


def compute_lowess_curve(
    observations: pd.DataFrame,
    x_col: str,
    impact_col: str = "signed_impact",
    frac: float = 0.3,
    max_points: int = 50_000,
    random_state: int = 20240618,
) -> pd.DataFrame:
    """Estimate a LOWESS curve in log-log space for positive observations."""

    data = observations.loc[(observations[x_col] > 0) & (observations[impact_col] > 0), [x_col, impact_col]].copy()
    if len(data) > max_points:
        data = data.sample(max_points, random_state=random_state)
    if len(data) < 2:
        return pd.DataFrame(columns=["x", "lowess_impact"])

    log_x = np.log(data[x_col].to_numpy(dtype=float))
    log_y = np.log(data[impact_col].to_numpy(dtype=float))
    fitted = lowess(log_y, log_x, frac=frac, return_sorted=True)
    return pd.DataFrame({"x": np.exp(fitted[:, 0]), "lowess_impact": np.exp(fitted[:, 1])})


def compute_local_loglog_slopes(
    binned: pd.DataFrame,
    x_col: str = "mean_x",
    impact_col: str = "mean_impact",
) -> pd.DataFrame:
    """Compute adjacent-bin local log-log slopes for positive bin averages."""

    data = binned.loc[(binned[x_col] > 0) & (binned[impact_col] > 0), [x_col, impact_col]].copy()
    data = data.sort_values(x_col).reset_index(drop=True)
    rows: list[dict[str, float | int]] = []
    for idx in range(len(data) - 1):
        x0 = float(data.loc[idx, x_col])
        x1 = float(data.loc[idx + 1, x_col])
        y0 = float(data.loc[idx, impact_col])
        y1 = float(data.loc[idx + 1, impact_col])
        denominator = np.log(x1) - np.log(x0)
        if denominator == 0:
            continue
        rows.append(
            {
                "bin_id_left": idx,
                "bin_id_right": idx + 1,
                "x_left": x0,
                "x_right": x1,
                "x_mid": float(np.sqrt(x0 * x1)),
                "impact_left": y0,
                "impact_right": y1,
                "local_slope": float((np.log(y1) - np.log(y0)) / denominator),
            }
        )
    return pd.DataFrame(rows)
