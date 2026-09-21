"""Impact-shape diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_impact.binning import bin_impact_curve_with_uncertainty, bin_impact_shape
from crypto_impact.regression import RegressionFilters, compare_impact_models, fit_power_law


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


def block_bootstrap_power_law(
    observations: pd.DataFrame,
    *,
    block_col: str,
    x_col: str,
    impact_col: str,
    n_bins: int = 30,
    n_bootstrap: int = 200,
    min_obs_per_bin: int = 100,
    random_state: int | None = 0,
) -> pd.DataFrame:
    """Bootstrap a binned power-law exponent by resampling whole blocks."""

    required = {block_col, x_col, impact_col}
    missing = required.difference(observations.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1.")

    data = observations.loc[:, [block_col, x_col, impact_col]].copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data = data.loc[data[x_col] > 0].reset_index(drop=True)
    block_indices = {
        block: group.index.to_numpy()
        for block, group in data.groupby(block_col, observed=True)
    }
    blocks = np.asarray(list(block_indices), dtype=object)
    if len(blocks) < 2:
        raise ValueError("At least two blocks are required for block bootstrap.")

    rng = np.random.default_rng(random_state)
    filters = RegressionFilters(min_obs_per_bin=min_obs_per_bin)
    rows: list[dict[str, float | int]] = []
    for sample_id in range(n_bootstrap):
        sampled_blocks = rng.choice(blocks, size=len(blocks), replace=True)
        sampled_indices = np.concatenate(
            [block_indices[block] for block in sampled_blocks]
        )
        sample = data.iloc[sampled_indices]
        binned = bin_impact_shape(
            sample,
            x_col=x_col,
            impact_col=impact_col,
            n_bins=n_bins,
            method="equal_count",
        )
        try:
            fit = fit_power_law(binned, filters=filters)
        except ValueError:
            continue
        rows.append(
            {
                "bootstrap_sample": sample_id,
                "delta": fit.delta,
                "amplitude": fit.amplitude,
                "bins_used": fit.n_fit_bins,
            }
        )
    if not rows:
        raise ValueError("No bootstrap sample produced a valid power-law fit.")
    return pd.DataFrame(rows)


def block_bootstrap_impact_models(
    observations: pd.DataFrame,
    *,
    block_col: str,
    x_col: str,
    impact_col: str,
    n_bins: int = 30,
    n_bootstrap: int = 200,
    min_obs_per_bin: int = 100,
    random_state: int | None = 0,
) -> pd.DataFrame:
    """Bootstrap the consistent weighted level-fit model comparison by block."""

    required = {block_col, x_col, impact_col}
    missing = required.difference(observations.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1.")

    data = observations.loc[:, [block_col, x_col, impact_col]].copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data = data.loc[data[x_col] > 0].reset_index(drop=True)
    block_indices = {
        block: group.index.to_numpy()
        for block, group in data.groupby(block_col, observed=True)
    }
    blocks = np.asarray(list(block_indices), dtype=object)
    if len(blocks) < 2:
        raise ValueError("At least two blocks are required for block bootstrap.")

    rng = np.random.default_rng(random_state)
    rows: list[dict[str, float | int]] = []
    for sample_id in range(n_bootstrap):
        sampled_blocks = rng.choice(blocks, size=len(blocks), replace=True)
        sampled_indices = np.concatenate(
            [block_indices[block] for block in sampled_blocks]
        )
        sample = data.iloc[sampled_indices]
        curve = bin_impact_curve_with_uncertainty(
            sample,
            x_col=x_col,
            impact_col=impact_col,
            n_bins=n_bins,
            method="equal_count",
        )
        try:
            comparison, _ = compare_impact_models(
                curve,
                min_obs_per_bin=min_obs_per_bin,
            )
        except (RuntimeError, ValueError):
            continue
        indexed = comparison.set_index("model")
        rows.append(
            {
                "bootstrap_sample": sample_id,
                "free_power_delta": indexed.loc["free_power", "delta"],
                "free_power_weighted_rmse": indexed.loc[
                    "free_power", "weighted_rmse"
                ],
                "logarithmic_weighted_rmse": indexed.loc[
                    "logarithmic", "weighted_rmse"
                ],
                "square_root_weighted_rmse": indexed.loc[
                    "square_root", "weighted_rmse"
                ],
                "log_minus_power_weighted_rmse": (
                    indexed.loc["logarithmic", "weighted_rmse"]
                    - indexed.loc["free_power", "weighted_rmse"]
                ),
                "bins_used": indexed.loc["free_power", "bins_used"],
            }
        )
    if not rows:
        raise ValueError("No bootstrap sample produced a valid model comparison.")
    return pd.DataFrame(rows)


def compare_impact_models_by_regime(
    regime_curves: pd.DataFrame,
    *,
    regime_col: str,
    regimes: list[str] | tuple[str, ...] | None = None,
    min_obs_per_bin: int = 100,
    require_common_bins: bool = True,
    x_scale: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply the weighted level-fit comparison to regime curves on common support."""

    required = {
        regime_col,
        "bin_id",
        "mean_x",
        "mean_impact",
        "standard_error",
        "n_obs",
    }
    missing = required.difference(regime_curves.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    regime_values = (
        list(regimes)
        if regimes is not None
        else list(regime_curves[regime_col].dropna().unique())
    )
    data = regime_curves.loc[
        regime_curves[regime_col].isin(regime_values)
    ].copy()
    if require_common_bins:
        regimes_per_bin = data.groupby("bin_id", observed=True)[
            regime_col
        ].nunique()
        common_bins = regimes_per_bin[
            regimes_per_bin == len(regime_values)
        ].index
        data = data.loc[data["bin_id"].isin(common_bins)].copy()

    summaries = []
    predictions = []
    for regime in regime_values:
        group = data.loc[data[regime_col] == regime]
        if group.empty:
            continue
        try:
            comparison, prediction = compare_impact_models(
                group,
                min_obs_per_bin=min_obs_per_bin,
                x_scale=x_scale,
            )
        except (RuntimeError, ValueError):
            continue
        comparison.insert(0, regime_col, regime)
        prediction.insert(0, regime_col, regime)
        summaries.append(comparison)
        predictions.append(prediction)
    if not summaries:
        raise ValueError("No regime produced a valid model comparison.")
    return (
        pd.concat(summaries, ignore_index=True),
        pd.concat(predictions, ignore_index=True),
        data,
    )
