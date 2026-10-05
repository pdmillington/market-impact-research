"""Patzelt-Bouchaud scale collapse, as used in the market-impact paper.

Ported unchanged in method from notebooks/PatzeltBouchaudScaling.ipynb so the
published analysis can be rerun on any block sample (the redo uses the shared
`timestamp_direction_v1` events; see scripts/build_pb_blocks.py).

- `add_trailing_volume_normalisation`: signed block volume divided by the
  preceding 24 h of block volume (current block excluded; a full elapsed
  window is required, measured from the start of the supplied sample).
- `independently_binned_signed_response`: equal-count bins on |x| separately for
  positive and negative x, with mean response and its standard error.
- `fit_pb_master_curve`: one shared shape R_N F(x / Q_N), with
  F(x) = x / (1 + |x|^alpha)^(beta/alpha) and free scales per N, fitted by
  robust (soft-L1), standard-error-weighted least squares from multiple starts.
- `scaling_slope`: Theil-Sen and OLS slopes of log scale on log N.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import linregress, theilslopes


def add_trailing_volume_normalisation(blocks: pd.DataFrame, window: str = "1D",
                                      require_full_window: bool = True) -> pd.DataFrame:
    result = blocks.sort_values("end_time_ms", kind="stable").copy()
    end_time = pd.to_datetime(result["end_time_ms"], unit="ms", utc=True)
    gross = pd.Series(result["gross_volume"].to_numpy(dtype=float), index=pd.DatetimeIndex(end_time))
    trailing = gross.rolling(window, closed="left", min_periods=1).sum()
    if require_full_window:
        full_at = end_time.min() + pd.Timedelta(window)
        trailing.loc[(end_time < full_at).to_numpy()] = np.nan
    result["trailing_volume"] = trailing.to_numpy()
    result["pb_signed_volume"] = result["signed_volume"] / result["trailing_volume"]
    return result.replace([np.inf, -np.inf], np.nan)


def independently_binned_signed_response(data: pd.DataFrame, variable: str, response: str = "log_return",
                                         n_bins: int = 60, min_bin_obs: int = 100) -> pd.DataFrame:
    work = data[[variable, response]].dropna()
    work = work.loc[work[variable] != 0].copy()
    work["side"] = np.where(work[variable] < 0, "negative", "positive")
    work["abs_condition"] = work[variable].abs()
    curves = []
    for side, half in work.groupby("side", observed=True):
        half = half.copy()
        half["bin"] = pd.qcut(half["abs_condition"], q=n_bins, labels=False, duplicates="drop")
        curve = half.groupby("bin", observed=True).agg(
            mean_q=(variable, "mean"), median_q=(variable, "median"),
            mean_response=(response, "mean"), response_std=(response, "std"),
            observations=(response, "size"),
        ).reset_index()
        curve.insert(0, "side", side)
        curves.append(curve)
    if not curves:
        raise ValueError(f"No positive or negative observations available for {variable!r}.")
    curve = pd.concat(curves, ignore_index=True)
    curve["standard_error"] = curve["response_std"] / np.sqrt(curve["observations"])
    curve = curve.loc[(curve["observations"] >= min_bin_obs) & np.isfinite(curve["standard_error"])
                      & (curve["standard_error"] > 0)].copy()
    return curve.sort_values("mean_q").reset_index(drop=True)


def pb_shape(x, alpha: float, beta: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x / np.power(1.0 + np.power(np.abs(x), alpha), beta / alpha)


def fit_pb_master_curve(curves: pd.DataFrame, n_values, n_starts: int = 12, random_state: int = 1207) -> dict:
    n_values = tuple(n_values)
    groups = {n: curves.loc[curves["n_events"] == n].copy() for n in n_values}
    missing = [n for n, group in groups.items() if group.empty]
    if missing:
        raise ValueError(f"No retained response bins for event counts {missing}.")
    q0 = np.array([groups[n]["mean_q"].abs().median() for n in n_values])
    r0 = np.array([max(groups[n]["mean_response"].abs().quantile(0.90), 1e-8) for n in n_values])
    initial = np.r_[np.log([1.2, 1.0]), np.log(q0), np.log(r0)]
    lower = np.r_[np.log([0.10, 0.01]), np.log(q0) - 5, np.log(r0) - 5]
    upper = np.r_[np.log([8.00, 5.00]), np.log(q0) + 5, np.log(r0) + 5]

    def residuals(params):
        alpha, beta = np.exp(params[:2])
        q_scales = np.exp(params[2:2 + len(n_values)])
        r_scales = np.exp(params[2 + len(n_values):])
        values = []
        for position, n in enumerate(n_values):
            group = groups[n]
            prediction = r_scales[position] * pb_shape(group["mean_q"] / q_scales[position], alpha, beta)
            sigma = group["standard_error"].to_numpy()
            positive = sigma[np.isfinite(sigma) & (sigma > 0)]
            sigma = np.clip(sigma, np.quantile(positive, 0.10), np.quantile(positive, 0.90))
            values.append((group["mean_response"].to_numpy() - prediction) / sigma)
        return np.concatenate(values)

    rng = np.random.default_rng(random_state)
    initial = np.clip(initial, lower + 1e-8, upper - 1e-8)
    starts = [initial] + [np.clip(initial + rng.normal(0, 0.20, len(initial)), lower + 1e-8, upper - 1e-8)
                          for _ in range(n_starts - 1)]
    best = None
    for start in starts:
        fit = least_squares(residuals, start, bounds=(lower, upper), loss="soft_l1", f_scale=1.0, max_nfev=50_000)
        objective = np.sum(2 * (np.sqrt(1 + fit.fun ** 2) - 1))
        if best is None or objective < best["objective"]:
            best = {"result": fit, "objective": objective}
    params = best["result"].x
    alpha, beta = np.exp(params[:2])
    scales = pd.DataFrame({"n_events": n_values, "fitted_q_scale": np.exp(params[2:2 + len(n_values)]),
                           "fitted_r_scale": np.exp(params[2 + len(n_values):])})
    return {"alpha": float(alpha), "beta": float(beta), "scales": scales, **best}


def scaling_slope(table: pd.DataFrame, value_column: str, label: str) -> dict:
    x = np.log(table["n_events"].to_numpy(dtype=float))
    y = np.log(table[value_column].to_numpy(dtype=float))
    robust = theilslopes(y, x, 0.95)
    ordinary = linregress(x, y)
    return {"quantity": label, "column": value_column, "robust_exponent": robust.slope,
            "robust_lower_95": robust.low_slope, "robust_upper_95": robust.high_slope,
            "ols_exponent": ordinary.slope, "ols_r_squared": ordinary.rvalue ** 2}


def collapse_diagnostics(curves: pd.DataFrame, fit: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    lookup = fit["scales"].set_index("n_events")
    collapsed = curves.copy()
    collapsed["q_scale"] = collapsed["n_events"].map(lookup["fitted_q_scale"])
    collapsed["r_scale"] = collapsed["n_events"].map(lookup["fitted_r_scale"])
    collapsed["scaled_q"] = collapsed["mean_q"] / collapsed["q_scale"]
    collapsed["scaled_response"] = collapsed["mean_response"] / collapsed["r_scale"]
    collapsed["master_prediction"] = pb_shape(collapsed["scaled_q"], fit["alpha"], fit["beta"])
    collapsed["collapse_residual"] = collapsed["scaled_response"] - collapsed["master_prediction"]
    summary = collapsed.groupby("n_events", observed=True).agg(
        collapse_rmse=("collapse_residual", lambda x: float(np.sqrt(np.mean(np.square(x))))),
        collapse_mae=("collapse_residual", lambda x: float(np.mean(np.abs(x)))),
        bins=("collapse_residual", "size"),
    ).reset_index()
    return collapsed, summary
