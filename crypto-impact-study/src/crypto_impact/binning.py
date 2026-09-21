"""Binning utilities for impact-shape analysis."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

BinMethod = Literal["log", "equal_count"]
PRIMARY_BIN_COLUMNS = [
    "bin_id",
    "mean_x",
    "median_x",
    "mean_impact",
    "median_impact",
    "impact_p10",
    "impact_p25",
    "impact_p75",
    "impact_p90",
    "n_obs",
]


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
    """Compute binned impact distribution summaries for one explanatory variable."""

    required = {x_col, impact_col}
    missing = required.difference(observations.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    data = observations.loc[observations[x_col] > 0, [x_col, impact_col]].copy()
    if data.empty:
        raise ValueError("No positive observations to bin.")

    data = add_bin_ids(data, x_col=x_col, n_bins=n_bins, method=method)
    optional_cols = [
        name
        for name in ["abs_signed_imbalance", "absolute_imbalance_ratio", "gross_volume"]
        if name in observations.columns and name not in data.columns
    ]
    if optional_cols:
        data = data.join(observations.loc[data.index, optional_cols])

    grouped = data.groupby("bin_id", observed=True)
    result = grouped.agg(
        mean_x=(x_col, "mean"),
        median_x=(x_col, "median"),
        mean_impact=(impact_col, "mean"),
        median_impact=(impact_col, "median"),
        impact_p10=(impact_col, lambda values: values.quantile(0.10)),
        impact_p25=(impact_col, lambda values: values.quantile(0.25)),
        impact_p75=(impact_col, lambda values: values.quantile(0.75)),
        impact_p90=(impact_col, lambda values: values.quantile(0.90)),
        n_obs=(impact_col, "count"),
    ).reset_index()
    result["bin_id"] = result["bin_id"].astype(int)

    if "abs_signed_imbalance" in data.columns:
        result["mean_abs_signed_imbalance"] = grouped["abs_signed_imbalance"].mean().to_numpy()
    if "absolute_imbalance_ratio" in data.columns:
        result["mean_absolute_imbalance_ratio"] = grouped["absolute_imbalance_ratio"].mean().to_numpy()
        result["median_absolute_imbalance_ratio"] = grouped["absolute_imbalance_ratio"].median().to_numpy()
    if "gross_volume" in data.columns:
        result["mean_gross_volume"] = grouped["gross_volume"].mean().to_numpy()
    return result


def bin_impact_curve_with_uncertainty(
    observations: pd.DataFrame,
    *,
    x_col: str,
    impact_col: str,
    n_bins: int = 12,
    method: BinMethod = "equal_count",
) -> pd.DataFrame:
    """Return the standard impact curve with within-bin uncertainty fields."""

    summary = bin_impact_shape(
        observations,
        x_col=x_col,
        impact_col=impact_col,
        n_bins=n_bins,
        method=method,
    )
    data = observations.loc[
        observations[x_col] > 0, [x_col, impact_col]
    ].replace([np.inf, -np.inf], np.nan).dropna()
    data = add_bin_ids(data, x_col=x_col, n_bins=n_bins, method=method)
    uncertainty = (
        data.groupby("bin_id", observed=True)[impact_col]
        .agg(impact_std="std", n_for_standard_error="count")
        .reset_index()
    )
    uncertainty["bin_id"] = uncertainty["bin_id"].astype(int)
    result = summary.merge(uncertainty, on="bin_id", how="left")
    result["standard_error"] = (
        result["impact_std"] / np.sqrt(result["n_for_standard_error"])
    )
    return result.drop(columns="n_for_standard_error")


def _with_legacy_columns(summary: pd.DataFrame, x_col: str) -> pd.DataFrame:
    """Add legacy column names expected by older analysis code."""

    out = summary.copy()
    out["mean_signed_impact"] = out["mean_impact"]
    out["median_signed_impact"] = out["median_impact"]
    if x_col == "abs_signed_imbalance":
        out["mean_abs_signed_imbalance"] = out["mean_x"]
        out["mean_abs_imbalance"] = out["mean_x"]
    if x_col == "absolute_imbalance_ratio":
        out["mean_absolute_imbalance_ratio"] = out["mean_x"]
        out["median_absolute_imbalance_ratio"] = out["median_x"]
    return out


def bin_by_abs_signed_imbalance(observations: pd.DataFrame, n_bins: int = 12, method: BinMethod = "equal_count") -> pd.DataFrame:
    """Bin observations by absolute signed imbalance."""

    return _with_legacy_columns(bin_impact_shape(observations, "abs_signed_imbalance", n_bins, method), "abs_signed_imbalance")


def bin_by_absolute_imbalance_ratio(observations: pd.DataFrame, n_bins: int = 12, method: BinMethod = "equal_count") -> pd.DataFrame:
    """Bin observations by absolute imbalance ratio."""

    return _with_legacy_columns(bin_impact_shape(observations, "absolute_imbalance_ratio", n_bins, method), "absolute_imbalance_ratio")


def imbalance_ratio_volume_heatmap(
    observations: pd.DataFrame,
    imbalance_ratio_bins: int = 20,
    volume_bins: int = 20,
) -> pd.DataFrame:
    """Compute mean signed impact in absolute-imbalance-ratio by gross-volume cells."""

    required = {"absolute_imbalance_ratio", "gross_volume", "signed_impact"}
    missing = required.difference(observations.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    data = observations.loc[(observations["absolute_imbalance_ratio"] > 0) & (observations["gross_volume"] > 0)].copy()
    if data.empty:
        raise ValueError("No positive absolute-imbalance-ratio/gross-volume observations to bin.")

    data["imbalance_ratio_bin"] = pd.qcut(data["absolute_imbalance_ratio"], q=imbalance_ratio_bins, duplicates="drop")
    data["gross_volume_bin"] = pd.qcut(data["gross_volume"], q=volume_bins, duplicates="drop")
    data = data.dropna(subset=["imbalance_ratio_bin", "gross_volume_bin"])
    data["imbalance_ratio_bin_id"] = data["imbalance_ratio_bin"].cat.codes.astype(int)
    data["gross_volume_bin_id"] = data["gross_volume_bin"].cat.codes.astype(int)
    grouped = data.groupby(["imbalance_ratio_bin_id", "gross_volume_bin_id"], observed=True)
    return grouped.agg(
        mean_absolute_imbalance_ratio=("absolute_imbalance_ratio", "mean"),
        median_absolute_imbalance_ratio=("absolute_imbalance_ratio", "median"),
        mean_gross_volume=("gross_volume", "mean"),
        median_gross_volume=("gross_volume", "median"),
        mean_signed_impact=("signed_impact", "mean"),
        median_signed_impact=("signed_impact", "median"),
        n_obs=("signed_impact", "count"),
    ).reset_index()


def bin_impact_by_shared_edges(
    observations: pd.DataFrame,
    *,
    regime_col: str,
    x_col: str,
    impact_col: str,
    n_bins: int = 20,
    min_obs_per_cell: int = 1,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Summarise regime curves using common pooled equal-count bin edges.

    Using one set of edges avoids comparing regime slopes over mechanically
    different x ranges. Cells below ``min_obs_per_cell`` are omitted.
    """

    required = {regime_col, x_col, impact_col}
    missing = required.difference(observations.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1.")
    if min_obs_per_cell < 1:
        raise ValueError("min_obs_per_cell must be at least 1.")

    data = observations.loc[:, [regime_col, x_col, impact_col]].copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data = data.loc[data[x_col] > 0]
    if data.empty:
        raise ValueError("No positive observations to bin.")

    edges = np.unique(
        data[x_col].quantile(np.linspace(0.0, 1.0, n_bins + 1)).to_numpy()
    )
    if len(edges) < 2:
        raise ValueError("At least two distinct x values are required.")
    edges[0] = np.nextafter(edges[0], -np.inf)
    edges[-1] = np.nextafter(edges[-1], np.inf)

    data["bin_id"] = pd.cut(
        data[x_col], bins=edges, labels=False, include_lowest=True
    )
    result = (
        data.groupby([regime_col, "bin_id"], observed=True)
        .agg(
            mean_x=(x_col, "mean"),
            median_x=(x_col, "median"),
            mean_impact=(impact_col, "mean"),
            median_impact=(impact_col, "median"),
            impact_std=(impact_col, "std"),
            n_obs=(impact_col, "count"),
        )
        .reset_index()
    )
    result["standard_error"] = result["impact_std"] / np.sqrt(result["n_obs"])
    result = result.loc[result["n_obs"] >= min_obs_per_cell].copy()
    result["bin_id"] = result["bin_id"].astype(int)
    return result, edges


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
