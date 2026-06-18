"""Logarithmic binning utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_log_bins(values: pd.Series, n_bins: int) -> np.ndarray:
    """Create logarithmically spaced bin edges for positive values."""

    positive = values[values > 0].astype(float)
    if positive.empty:
        raise ValueError("At least one positive value is required for log bins.")
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1.")

    min_value = positive.min()
    max_value = positive.max()
    if min_value == max_value:
        return np.array([min_value * 0.999, max_value * 1.001])
    return np.geomspace(min_value, max_value, n_bins + 1)


def bin_impact_observations(
    observations: pd.DataFrame,
    n_bins: int = 12,
    imbalance_col: str = "abs_imbalance",
    impact_col: str = "signed_impact",
) -> pd.DataFrame:
    """Bin impact observations by absolute imbalance."""

    required = {imbalance_col, impact_col}
    missing = required.difference(observations.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    data = observations.loc[observations[imbalance_col] > 0, [imbalance_col, impact_col]].copy()
    if data.empty:
        raise ValueError("No positive imbalance observations to bin.")

    bins = make_log_bins(data[imbalance_col], n_bins)
    data["bin"] = pd.cut(data[imbalance_col], bins=bins, include_lowest=True)
    binned = (
        data.dropna(subset=["bin"])
        .groupby("bin", observed=True)
        .agg(
            mean_abs_imbalance=(imbalance_col, "mean"),
            mean_signed_impact=(impact_col, "mean"),
            std_signed_impact=(impact_col, "std"),
            n_obs=(impact_col, "count"),
        )
        .reset_index(drop=True)
    )
    binned["standard_error"] = binned["std_signed_impact"] / np.sqrt(binned["n_obs"])
    binned.loc[binned["n_obs"] <= 1, "standard_error"] = np.nan
    return binned.drop(columns=["std_signed_impact"])
