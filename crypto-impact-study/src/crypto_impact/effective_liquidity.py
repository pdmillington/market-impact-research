"""State-dependent effective-liquidity models (pooled log, state log, activity tail).

Moved from notebooks/StateDependentEffectiveLiquidity.ipynb so that the notebook and
the surface-versus-threshold comparison (scripts/surface_vs_threshold.py) fit the
same models with the same code. Behaviour is unchanged; settings the notebook held
as globals are arguments with the notebook's values as defaults.

Notation (as in the paper): u = x / x_med is normalised imbalance (x = |Q| / V_24h),
z = log(lambda / lambda_med) is log event intensity relative to its median, and
f = log of the relative flow rate relative to its median. The response is the
signed bar return over trailing 24 h realised volatility.

Models, for a response cell with mean u, z, f:
- pooled_log:    A log(1 + u / u0)
- state_log:     A(z) log(1 + u / u0(z)), A(z) = exp(a0 + a1 z), u0(z) = exp(b0 + b1 z)
- activity_tail: state_log with effective liquidity reduced above (z*, u*):
                 L = L0 exp{-gamma (z - z*)_+ h(u; u*)}, h = log[1 + (u - u*)_+ / u*]
- joint_tail:    activity_tail plus the same term in relative flow f.
- surface_volume / surface_intensity: the activity-adjusted P&B surface at a single
                 bar size, R F(q a^gamma / Q) with F(x) = x / (1 + x^alpha)^(beta/alpha),
                 q = |Q| / V and a = V / V_24h (volume share) or a = lambda / lambda_med
                 (event intensity). Fitted here with the same cells, weights and
                 least-squares loss as the models above, for a like-for-like comparison
                 (cells need mean_log_q and, for surface_volume, mean_log_a).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.optimize import least_squares

from .impact import add_trailing_impact_normalisation
from .regime_analysis import build_analysis_table


N_ACTIVITY_BINS = 5
N_IMBALANCE_BINS = 20
MIN_OBS_PER_CELL = 50
N_QUADRATURE_NODES = 32
N_STARTS = 16
TRAILING_WINDOW = "1D"

QUAD_NODES, QUAD_WEIGHTS = leggauss(N_QUADRATURE_NODES)
MODEL_ORDER = ["pooled_log", "state_log", "activity_tail", "joint_tail"]
PARAMETER_NAMES = {
    "pooled_log": ["log_A", "log_u0"],
    "state_log": ["log_A", "A_activity_slope", "log_u0", "u0_activity_slope"],
    "activity_tail": [
        "log_A", "A_activity_slope", "log_u0", "u0_activity_slope",
        "gamma_activity", "activity_threshold_z", "log_imbalance_threshold_u",
    ],
    "joint_tail": [
        "log_A", "A_activity_slope", "log_u0", "u0_activity_slope",
        "gamma_activity", "activity_threshold_z", "log_imbalance_threshold_u",
        "gamma_relative_flow", "relative_flow_threshold_f",
    ],
    "surface_volume": ["log_Q", "log_R", "log_alpha", "log_beta", "gamma"],
    "surface_intensity": ["log_Q", "log_R", "log_alpha", "log_beta", "gamma"],
    # Scales follow the bar's event count N_t (for time bars): Q = Q0 (N_t/N_med)^xi,
    # R = R0 (N_t/N_med)^psi, using the cell mean of log(N_t / N_med) (mean_log_n).
    "surface_volume_scaled": ["log_Q", "log_R", "log_alpha", "log_beta", "gamma", "xi", "psi"],
}
SURFACE_MODELS = ("surface_volume", "surface_intensity", "surface_volume_scaled")


# ---------------------------------------------------------------------------
# Bars to analysis sample and response cells.

def prepare_normalised_analysis(bars, trailing_window=TRAILING_WINDOW):
    analysis = build_analysis_table(bars, horizons=())
    analysis = add_trailing_impact_normalisation(
        analysis,
        window=trailing_window,
        min_periods=1,
        require_full_window=True,
        timestamp_col="end_time",
    ).replace([np.inf, -np.inf], np.nan)
    analysis["block_day"] = pd.to_datetime(analysis["end_time"], utc=True).dt.floor("D")
    analysis["event_intensity"] = analysis["trade_count"] / analysis["duration_seconds"]
    return analysis


def add_model_scales(data):
    result = data.copy()
    valid = result[[
        "normalised_imbalance", "normalised_signed_impact",
        "event_intensity", "flow_rate_ratio",
    ]].replace([np.inf, -np.inf], np.nan).dropna()
    valid = valid.loc[
        (valid["normalised_imbalance"] > 0)
        & (valid["event_intensity"] > 0)
        & (valid["flow_rate_ratio"] > 0)
    ]
    x_med = float(valid["normalised_imbalance"].median())
    intensity_med = float(valid["event_intensity"].median())
    relative_flow_med = float(valid["flow_rate_ratio"].median())
    result["imbalance_u"] = result["normalised_imbalance"] / x_med
    result["activity_z"] = np.log(result["event_intensity"] / intensity_med)
    result["relative_flow_f"] = np.log(result["flow_rate_ratio"] / relative_flow_med)
    result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=[
        "imbalance_u", "normalised_signed_impact", "activity_z",
        "relative_flow_f", "block_day",
    ])
    result = result.loc[result["imbalance_u"] > 0].copy()
    scales = {
        "x_med": x_med,
        "event_intensity_median": intensity_med,
        "relative_flow_ratio_median": relative_flow_med,
    }
    return result, scales


def build_response_surface(data, n_activity_bins=N_ACTIVITY_BINS, n_imbalance_bins=N_IMBALANCE_BINS,
                           min_obs_per_cell=MIN_OBS_PER_CELL, extra_columns=()):
    """Activity quintiles x imbalance bins (equal count within each quintile).

    `extra_columns` are averaged within each cell as mean_<column>, for models that
    need further cell coordinates (e.g. the activity surface's log q and log a).
    """

    columns = [
        "imbalance_u", "activity_z", "relative_flow_f",
        "normalised_signed_impact", "block_day", *extra_columns,
    ]
    sample = data.loc[:, columns].replace([np.inf, -np.inf], np.nan).dropna()
    sample = sample.loc[sample["imbalance_u"] > 0].copy()
    sample["activity_bin"] = pd.qcut(
        sample["activity_z"], q=n_activity_bins, labels=False, duplicates="drop",
    )
    if sample["activity_bin"].nunique() != n_activity_bins:
        raise ValueError("Insufficient distinct activity values for five quintiles.")

    frames = []
    assigned = []
    for activity_bin, group in sample.groupby("activity_bin", observed=True):
        group = group.copy()
        group["imbalance_bin"] = pd.qcut(
            group["imbalance_u"], q=n_imbalance_bins, labels=False, duplicates="drop",
        )
        if group["imbalance_bin"].nunique() != n_imbalance_bins:
            raise ValueError(f"Insufficient distinct imbalance values in activity bin {activity_bin}.")
        assigned.append(group)
        aggregations = dict(
            mean_u=("imbalance_u", "mean"),
            median_u=("imbalance_u", "median"),
            mean_z=("activity_z", "mean"),
            median_z=("activity_z", "median"),
            mean_f=("relative_flow_f", "mean"),
            median_f=("relative_flow_f", "median"),
            mean_impact=("normalised_signed_impact", "mean"),
            median_impact=("normalised_signed_impact", "median"),
            impact_std=("normalised_signed_impact", "std"),
            n_obs=("normalised_signed_impact", "count"),
            n_days=("block_day", "nunique"),
        )
        aggregations.update({f"mean_{column}": (column, "mean") for column in extra_columns})
        cell = group.groupby("imbalance_bin", observed=True).agg(**aggregations).reset_index()
        cell.insert(0, "activity_bin", int(activity_bin))
        frames.append(cell)
    surface = pd.concat(frames, ignore_index=True)
    surface["standard_error"] = surface["impact_std"] / np.sqrt(surface["n_obs"])
    surface = surface.loc[
        (surface["n_obs"] >= min_obs_per_cell)
        & np.isfinite(surface["standard_error"])
        & (surface["standard_error"] > 0)
    ].copy()
    surface["activity_bin"] = surface["activity_bin"].astype(int)
    surface["imbalance_bin"] = surface["imbalance_bin"].astype(int)
    assigned = pd.concat(assigned).sort_index()
    return surface.sort_values(["activity_bin", "imbalance_bin"]), assigned


# ---------------------------------------------------------------------------
# Models.

def central_parameters(params, z):
    A = np.exp(params[0] + params[1] * z)
    u0 = np.exp(params[2] + params[3] * z)
    return A, u0


def tail_response(u, z, f, params, include_relative_flow=False):
    u = np.asarray(u, dtype=float)
    z = np.asarray(z, dtype=float)
    f = np.asarray(f, dtype=float)
    A, u0 = central_parameters(params, z)
    gamma_z, z_star, log_u_star = params[4:7]
    u_star = np.exp(log_u_star)
    kappa = gamma_z * np.maximum(z - z_star, 0.0)
    if include_relative_flow:
        gamma_f, f_star = params[7:9]
        kappa = kappa + gamma_f * np.maximum(f - f_star, 0.0)

    central_end = np.minimum(u, u_star)
    central_response = A * np.log1p(central_end / u0)
    tail_width = np.maximum(u - u_star, 0.0)
    nodes = u_star + 0.5 * tail_width[:, None] * (QUAD_NODES[None, :] + 1.0)
    h = np.log(nodes / u_star)
    exponent = np.clip(kappa[:, None] * h, 0.0, 30.0)
    integrand = A[:, None] * np.exp(exponent) / (u0[:, None] + nodes)
    tail_response_value = 0.5 * tail_width * np.sum(QUAD_WEIGHTS[None, :] * integrand, axis=1)
    no_tail_adjustment = kappa <= 1e-12
    exact_log_tail = A * (np.log1p(u / u0) - np.log1p(central_end / u0))
    tail_response_value = np.where(no_tail_adjustment, exact_log_tail, tail_response_value)
    return central_response + tail_response_value


def surface_log_activity(model, data):
    """Cell mean log activity for the surface: log(V/V24) or log(lambda/lambda_med) (= z)."""

    column = "mean_z" if model == "surface_intensity" else "mean_log_a"
    return np.asarray(data[column], dtype=float)


def predict_model(model, params, data):
    if model in SURFACE_MODELS:
        log_Q, log_R, log_alpha, log_beta, gamma = params[:5]
        if model == "surface_volume_scaled":
            xi, psi = params[5:7]
            log_n = np.asarray(data["mean_log_n"], dtype=float)
            log_Q, log_R = log_Q + xi * log_n, log_R + psi * log_n
        alpha, beta = np.exp(log_alpha), np.exp(log_beta)
        x = np.exp(np.asarray(data["mean_log_q"], dtype=float) + gamma * surface_log_activity(model, data) - log_Q)
        return np.exp(log_R) * x / (1.0 + x**alpha) ** (beta / alpha)
    u = np.asarray(data["mean_u"], dtype=float)
    z = np.asarray(data["mean_z"], dtype=float)
    f = np.asarray(data["mean_f"], dtype=float)
    if model == "pooled_log":
        A, u0 = np.exp(params)
        return A * np.log1p(u / u0)
    if model == "state_log":
        A, u0 = central_parameters(params, z)
        return A * np.log1p(u / u0)
    if model == "activity_tail":
        return tail_response(u, z, f, params, include_relative_flow=False)
    if model == "joint_tail":
        return tail_response(u, z, f, params, include_relative_flow=True)
    raise ValueError(f"Unknown model: {model}")


def parameter_bounds(model, surface):
    if model in SURFACE_MODELS:
        log_x = surface["mean_log_q"].to_numpy() + 0.5 * surface_log_activity(model, surface)
        centre = float(np.median(log_x))
        lower = [centre - 8.0, -12.0, np.log(0.1), np.log(0.01), -0.5]
        upper = [centre + 8.0, 4.0, np.log(8.0), np.log(5.0), 1.8]
        if model == "surface_volume_scaled":
            lower, upper = lower + [-1.0, -1.0], upper + [2.0, 2.0]
        return np.array(lower), np.array(upper)
    z = surface["mean_z"].to_numpy()
    f = surface["mean_f"].to_numpy()
    u = surface["mean_u"].to_numpy()
    central_lower = [-12.0, -3.0, -7.0, -3.0]
    central_upper = [2.0, 3.0, 7.0, 3.0]
    if model == "pooled_log":
        return np.array([-12.0, -7.0]), np.array([2.0, 7.0])
    if model == "state_log":
        return np.array(central_lower), np.array(central_upper)
    tail_lower = [*central_lower, 0.0, np.quantile(z, 0.10), np.log(np.quantile(u, 0.40))]
    tail_upper = [*central_upper, 5.0, np.quantile(z, 0.95), np.log(np.quantile(u, 0.995))]
    if model == "activity_tail":
        return np.array(tail_lower), np.array(tail_upper)
    return (
        np.array([*tail_lower, 0.0, np.quantile(f, 0.10)]),
        np.array([*tail_upper, 5.0, np.quantile(f, 0.95)]),
    )


def initial_parameters(model, surface, previous_fits):
    if model in SURFACE_MODELS:
        log_x = surface["mean_log_q"].to_numpy() + 0.36 * surface_log_activity(model, surface)
        positive = surface.loc[surface["mean_impact"] > 0, "mean_impact"]
        R0 = float(positive.quantile(0.9)) if len(positive) else 0.1
        start = [float(np.median(log_x)), np.log(max(R0, 1e-6)), np.log(1.5), np.log(0.5), 0.36]
        return np.array(start + ([0.1, 0.5] if model == "surface_volume_scaled" else []))
    if model == "pooled_log":
        positive = surface.loc[surface["mean_impact"] > 0, "mean_impact"]
        A0 = float(positive.median()) if len(positive) else 0.03
        return np.log([max(A0, 1e-4), 1.0])
    if model == "state_log":
        pooled = previous_fits["pooled_log"]["params"]
        return np.array([pooled[0], 0.0, pooled[1], 0.0])
    if model == "activity_tail":
        central = previous_fits["state_log"]["params"]
        return np.array([
            *central, 0.15,
            surface["mean_z"].quantile(0.80),
            np.log(surface["mean_u"].quantile(0.90)),
        ])
    activity = previous_fits["activity_tail"]["params"]
    return np.array([*activity, 0.05, surface["mean_f"].quantile(0.80)])


def fit_one_model(model, surface, previous_fits, n_starts=N_STARTS, random_state=0, fixed_params=None):
    """Weighted least-squares fit of `model` to response cells.

    `fixed_params` maps parameter names (PARAMETER_NAMES[model]) to values held
    fixed; only the remaining parameters are estimated and counted. Used to fit the
    single-N surface with the master-curve shape (log_alpha, log_beta) taken from the
    common multi-N fit.
    """
    required = ["mean_u", "mean_z", "mean_f", "mean_impact", "standard_error"]
    if model in SURFACE_MODELS:
        required += ["mean_log_q"] + (["mean_log_a"] if model != "surface_intensity" else [])
        required += ["mean_log_n"] if model == "surface_volume_scaled" else []
    data = surface.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()
    lower, upper = parameter_bounds(model, data)
    initial = np.clip(initial_parameters(model, data, previous_fits), lower + 1e-8, upper - 1e-8)
    names = PARAMETER_NAMES[model]
    fixed_params = fixed_params or {}
    fixed_index = np.array([names.index(name) for name in fixed_params], dtype=int)
    free = np.setdiff1d(np.arange(len(names)), fixed_index)
    template = initial.copy()
    template[fixed_index] = [fixed_params[name] for name in fixed_params]

    def full(free_values):
        params = template.copy()
        params[free] = free_values
        return params
    sigma = data["standard_error"].to_numpy(dtype=float)
    fallback = np.median(sigma[np.isfinite(sigma) & (sigma > 0)])
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, fallback)
    y = data["mean_impact"].to_numpy(dtype=float)

    def residuals(free_values):
        return (y - predict_model(model, full(free_values), data)) / sigma

    lower_free, upper_free, initial_free = lower[free], upper[free], initial[free]
    rng = np.random.default_rng(random_state)
    starts = [initial_free]
    for _ in range(max(1, n_starts) - 1):
        draw = initial_free + rng.normal(0.0, 0.20, size=len(initial_free)) * (upper_free - lower_free)
        draw = np.clip(draw, lower_free + 1e-7, upper_free - 1e-7)
        starts.append(draw)

    best = None
    for start in starts:
        result = least_squares(
            residuals, start, bounds=(lower_free, upper_free),
            max_nfev=25_000, xtol=1e-10, ftol=1e-10, gtol=1e-10,
        )
        if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
            best = result
    params = full(best.x)
    prediction = predict_model(model, params, data)
    return {
        "model": model,
        "params": params,
        "free": free,
        "lower_bounds": lower,
        "upper_bounds": upper,
        "success": bool(best.success),
        "message": best.message,
        "cost": float(np.sum(best.fun**2)),
        "data": data,
        "prediction": prediction,
        "residual": y - prediction,
    }


def fit_model_sequence(surface, n_starts=N_STARTS, random_state=0, models=MODEL_ORDER):
    fits = {}
    for model_number, model in enumerate(models):
        fits[model] = fit_one_model(
            model, surface, fits, n_starts=n_starts, random_state=random_state + model_number,
        )
    return fits


# ---------------------------------------------------------------------------
# Fit statistics.

def weighted_rmse(residual, standard_error):
    weights = 1.0 / np.asarray(standard_error, dtype=float)**2
    residual = np.asarray(residual, dtype=float)
    return float(np.sqrt(np.sum(weights * residual**2) / np.sum(weights)))


def fit_summary_row(fit):
    data = fit["data"]
    residual = fit["residual"]
    tail = data["imbalance_bin"] == data["imbalance_bin"].max()
    high_tail = tail & (data["activity_bin"] == data["activity_bin"].max())
    free = fit.get("free", np.arange(len(fit["params"])))
    k = len(free)
    chi_square = fit["cost"]
    bound_tolerance = 1e-4 * (fit["upper_bounds"] - fit["lower_bounds"])
    at_bound = (
        (fit["params"] - fit["lower_bounds"] <= bound_tolerance)
        | (fit["upper_bounds"] - fit["params"] <= bound_tolerance)
    )
    at_bound &= np.isin(np.arange(len(fit["params"])), free)  # fixed parameters are not "at a bound"
    bound_names = np.asarray(PARAMETER_NAMES[fit["model"]])[at_bound]
    return {
        "model": fit["model"],
        "parameters": k,
        "cells": len(data),
        "weighted_rmse": weighted_rmse(residual, data["standard_error"]),
        "tail_wrmse": weighted_rmse(residual[tail], data.loc[tail, "standard_error"]),
        "high_activity_tail_wrmse": weighted_rmse(residual[high_tail], data.loc[high_tail, "standard_error"]),
        "unweighted_rmse": float(np.sqrt(np.mean(residual**2))),
        "weighted_chi_square": chi_square,
        "weighted_aic_up_to_constant": chi_square + 2 * k,
        "parameters_at_bound": int(at_bound.sum()),
        "bound_parameter_names": ";".join(bound_names),
        "converged": fit["success"],
    }


def transformed_parameters(model, params):
    values = dict(zip(PARAMETER_NAMES[model], params))
    if model in SURFACE_MODELS:
        output = {"Q_scale": np.exp(values["log_Q"]), "R_scale": np.exp(values["log_R"]),
                  "alpha": np.exp(values["log_alpha"]), "beta": np.exp(values["log_beta"]), "gamma": values["gamma"]}
        if model == "surface_volume_scaled":
            output.update({"xi": values["xi"], "psi": values["psi"]})
        return output
    output = {
        "A_at_median_activity": np.exp(values["log_A"]),
        "u0_at_median_activity": np.exp(values["log_u0"]),
    }
    if model != "pooled_log":
        output.update({
            "A_activity_slope": values["A_activity_slope"],
            "u0_activity_slope": values["u0_activity_slope"],
        })
    if model in {"activity_tail", "joint_tail"}:
        output.update({
            "gamma_activity": values["gamma_activity"],
            "activity_threshold_z": values["activity_threshold_z"],
            "activity_threshold_multiple": np.exp(values["activity_threshold_z"]),
            "imbalance_threshold_u": np.exp(values["log_imbalance_threshold_u"]),
        })
    if model == "joint_tail":
        output.update({
            "gamma_relative_flow": values["gamma_relative_flow"],
            "relative_flow_threshold_f": values["relative_flow_threshold_f"],
            "relative_flow_threshold_multiple": np.exp(values["relative_flow_threshold_f"]),
        })
    return output


# ---------------------------------------------------------------------------
# Uncertainty and validation.

def bootstrap_activity_tail(data, n_bootstrap=100, random_state=0, n_starts=N_STARTS, surface_kwargs=None):
    surface_kwargs = surface_kwargs or {}
    day_groups = {day: group for day, group in data.groupby("block_day", observed=True)}
    days = np.asarray(list(day_groups), dtype=object)
    if len(days) < 2:
        raise ValueError("At least two UTC days are required for the bootstrap.")
    rng = np.random.default_rng(random_state)
    rows = []
    for bootstrap_sample in range(n_bootstrap):
        sampled_days = rng.choice(days, size=len(days), replace=True)
        draw = pd.concat(
            [day_groups[day].assign(_bootstrap_day=draw_number)
             for draw_number, day in enumerate(sampled_days)],
            ignore_index=True,
        )
        try:
            surface, _ = build_response_surface(draw, **surface_kwargs)
            fits = {}
            for model_number, model in enumerate(["pooled_log", "state_log", "activity_tail"]):
                fits[model] = fit_one_model(
                    model, surface, fits,
                    n_starts=max(3, n_starts // 3),
                    random_state=random_state + bootstrap_sample * 10 + model_number,
                )
        except (RuntimeError, ValueError):
            continue
        central = fit_summary_row(fits["state_log"])
        tail = fit_summary_row(fits["activity_tail"])
        transformed = transformed_parameters("activity_tail", fits["activity_tail"]["params"])
        rows.append({
            "bootstrap_sample": bootstrap_sample,
            **transformed,
            "state_log_wrmse": central["weighted_rmse"],
            "activity_tail_wrmse": tail["weighted_rmse"],
            "wrmse_improvement_pct": 100 * (central["weighted_rmse"] - tail["weighted_rmse"]) / central["weighted_rmse"],
            "state_log_tail_wrmse": central["tail_wrmse"],
            "activity_tail_tail_wrmse": tail["tail_wrmse"],
            "tail_wrmse_improvement_pct": 100 * (central["tail_wrmse"] - tail["tail_wrmse"]) / central["tail_wrmse"],
        })
    return pd.DataFrame(rows)


def blocked_cross_validation(data, n_folds=5, random_state=0, n_starts=N_STARTS, surface_kwargs=None):
    surface_kwargs = surface_kwargs or {}
    days = np.asarray(sorted(data["block_day"].dropna().unique()))
    if len(days) < n_folds:
        raise ValueError("Fewer UTC days than requested cross-validation folds.")
    folds = np.array_split(days, n_folds)
    rows = []
    for fold_number, test_days in enumerate(folds):
        train = data.loc[~data["block_day"].isin(test_days)]
        test = data.loc[data["block_day"].isin(test_days)]
        train_surface, _ = build_response_surface(train, **surface_kwargs)
        test_surface, _ = build_response_surface(test, **surface_kwargs)
        fits = fit_model_sequence(
            train_surface, n_starts=max(4, n_starts // 2), random_state=random_state + fold_number * 100,
        )
        for model, fit in fits.items():
            prediction = predict_model(model, fit["params"], test_surface)
            residual = test_surface["mean_impact"].to_numpy() - prediction
            tail = test_surface["imbalance_bin"] == test_surface["imbalance_bin"].max()
            rows.append({
                "fold": fold_number,
                "test_start": pd.Timestamp(test_days[0]),
                "test_end": pd.Timestamp(test_days[-1]),
                "model": model,
                "test_cells": len(test_surface),
                "test_weighted_rmse": weighted_rmse(residual, test_surface["standard_error"]),
                "test_tail_weighted_rmse": weighted_rmse(residual[tail], test_surface.loc[tail, "standard_error"]),
            })
    return pd.DataFrame(rows)
