"""Binning utilities for impact-shape analysis."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

BinMethod = Literal["log", "equal_count"]
PRIMARY_BIN_COLUMNS = ["bin_id", "mean_x", "median_x", "mean_impact", "median_impact", "stderr", "ci_lower", "ci_upper", "n_obs"]


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


def add_bin_ids(data: pd.DataFrame, x_col: str, n_bins: int, method: BinMethod = "equal_count") -> pd.DataFrame:
    """Return a copy of data with integer ``bin_id`` labels."""

    if n_bins < 1:
        raise ValueError("n_bins must be at least 1.")
    binned = data.copy()
    if method == "equal_count":
        binned["bin"] = pd.qcut(binned[x_col], q=n_bins, duplicates="drop")
    elif method == "log":
        binned["bin"] = pd.cut(binned[x_col], bins=make_log_bins(binned[x_col], n_bins), include_lowest=True)
    else:
        raise ValueError("method must be 'equal_count' or 'log'.")
    codes = binned["bin"].cat.codes
    binned["bin_id"] = codes.where(codes >= 0, np.nan)
    return binned.dropna(subset=["bin_id"])


def bin_impact_shape(
    observations: pd.DataFrame,
    x_col: str,
    n_bins: int = 12,
    method: BinMethod = "equal_count",
    impact_col: str = "signed_impact",
) -> pd.DataFrame:
    """Compute primary shape-summary bins for one explanatory variable."""

    required = {x_col, impact_col}
    missing = required.difference(observations.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    data = observations.loc[observations[x_col] > 0, [x_col, impact_col]].copy()
    if data.empty:
        raise ValueError("No positive observations to bin.")

    data = add_bin_ids(data, x_col=x_col, n_bins=n_bins, method=method)
    # Carry optional filter-support columns internally; callers can save the
    # primary schema by selecting PRIMARY_BIN_COLUMNS.
    optional_cols = [name for name in ["abs_signed_imbalance", "participation_ratio"] if name in observations.columns and name not in data.columns]
    if optional_cols:
        data = data.join(observations.loc[data.index, optional_cols])
    grouped = data.groupby("bin_id", observed=True)
    aggregations = {
        "mean_x": (x_col, "mean"),
        "median_x": (x_col, "median"),
        "mean_impact": (impact_col, "mean"),
        "median_impact": (impact_col, "median"),
        "std_impact": (impact_col, "std"),
        "n_obs": (impact_col, "count"),
    }
    if "abs_signed_imbalance" in data.columns:
        aggregations["mean_abs_signed_imbalance"] = ("abs_signed_imbalance", "mean")
    if "participation_ratio" in data.columns:
        aggregations["mean_participation_ratio"] = ("participation_ratio", "mean")
        aggregations["median_participation_ratio"] = ("participation_ratio", "median")
    result = grouped.agg(**aggregations).reset_index()
    result["bin_id"] = result["bin_id"].astype(int)
    result["stderr"] = result["std_impact"] / np.sqrt(result["n_obs"])
    result.loc[result["n_obs"] <= 1, "stderr"] = np.nan
    result["ci_lower"] = result["mean_impact"] - 1.96 * result["stderr"]
    result["ci_upper"] = result["mean_impact"] + 1.96 * result["stderr"]
    primary = ["bin_id", "mean_x", "median_x", "mean_impact", "median_impact", "stderr", "ci_lower", "ci_upper", "n_obs"]
    extras = [name for name in ["mean_abs_signed_imbalance", "mean_participation_ratio", "median_participation_ratio"] if name in result.columns]
    return result[primary + extras]


def _with_legacy_columns(summary: pd.DataFrame, x_col: str) -> pd.DataFrame:
    """Add legacy column names expected by older analysis code."""

    out = summary.copy()
    out["mean_signed_impact"] = out["mean_impact"]
    out["median_signed_impact"] = out["median_impact"]
    out["standard_error_signed_impact"] = out["stderr"]
    out["standard_error"] = out["stderr"]
    if x_col == "abs_signed_imbalance":
        out["mean_abs_signed_imbalance"] = out["mean_x"]
        out["mean_abs_imbalance"] = out["mean_x"]
    if x_col == "participation_ratio":
        out["mean_participation_ratio"] = out["mean_x"]
        out["median_participation_ratio"] = out["median_x"]
    return out


def bin_by_abs_signed_imbalance(observations: pd.DataFrame, n_bins: int = 12, method: BinMethod = "equal_count") -> pd.DataFrame:
    """Bin observations by absolute signed imbalance."""

    return _with_legacy_columns(bin_impact_shape(observations, "abs_signed_imbalance", n_bins, method), "abs_signed_imbalance")


def bin_by_participation_ratio(observations: pd.DataFrame, n_bins: int = 12, method: BinMethod = "equal_count") -> pd.DataFrame:
    """Bin observations by participation ratio."""

    return _with_legacy_columns(bin_impact_shape(observations, "participation_ratio", n_bins, method), "participation_ratio")


def bin_impact_observations(
    observations: pd.DataFrame,
    n_bins: int = 12,
    imbalance_col: str = "abs_imbalance",
    impact_col: str = "signed_impact",
    x_col: str | None = None,
    method: BinMethod = "equal_count",
    x_prefix: str | None = None,
) -> pd.DataFrame:
    """Backward-compatible wrapper around the primary shape binning function."""

    x_name = x_col or imbalance_col
    if x_name == "abs_imbalance" and "abs_imbalance" not in observations.columns:
        observations = observations.copy()
        observations["abs_imbalance"] = observations["abs_signed_imbalance"]
    summary = bin_impact_shape(observations, x_col=x_name, n_bins=n_bins, method=method, impact_col=impact_col)
    prefix = x_prefix or x_name
    out = _with_legacy_columns(summary, "abs_signed_imbalance" if prefix in {"abs_imbalance", "abs_signed_imbalance"} else prefix)
    if prefix == "abs_imbalance":
        out["mean_abs_imbalance"] = out["mean_x"]
    return out
