"""Interpretable impact-curve comparisons and conditional calibration checks."""

from __future__ import annotations

from typing import Any

import numpy as np


MODEL_ORDER = (
    "raw_log_affine",
    "binned_log_zero",
    "binned_power",
    "monotonic_bins",
)

TAIL_QUANTILE_EDGES = (0.0, 0.01, 0.05, 0.10, 0.20, 0.80, 0.90, 0.95, 0.99, 1.0)


def _finite_selected(x: np.ndarray, y: np.ndarray, selected: np.ndarray) -> np.ndarray:
    return np.asarray(selected, dtype=bool) & np.isfinite(x) & np.isfinite(y) & (x > 0)


def training_quantile_cuts(
    x: np.ndarray,
    selected: np.ndarray,
    *,
    bins: int = 20,
) -> np.ndarray:
    mask = np.asarray(selected, dtype=bool) & np.isfinite(x) & (x > 0)
    if mask.sum() < bins:
        raise ValueError("Insufficient observations for the requested impact bins.")
    cuts = np.quantile(x[mask], np.linspace(0, 1, bins + 1)[1:-1])
    if len(np.unique(cuts)) != len(cuts):
        raise ValueError("Impact-bin cut points are not unique.")
    return cuts


def frozen_tail_cuts(
    x: np.ndarray,
    selected: np.ndarray,
    *,
    quantile_edges: tuple[float, ...] = TAIL_QUANTILE_EDGES,
) -> np.ndarray:
    """Return training-frozen cuts with finer resolution in both tails."""

    if quantile_edges[0] != 0.0 or quantile_edges[-1] != 1.0:
        raise ValueError("Tail quantile edges must begin at zero and end at one.")
    if any(right <= left for left, right in zip(quantile_edges, quantile_edges[1:])):
        raise ValueError("Tail quantile edges must be strictly increasing.")
    mask = np.asarray(selected, dtype=bool) & np.isfinite(x) & (x > 0)
    if mask.sum() < len(quantile_edges) - 1:
        raise ValueError("Insufficient observations for tail diagnostics.")
    cuts = np.quantile(x[mask], quantile_edges[1:-1])
    if len(np.unique(cuts)) != len(cuts):
        raise ValueError("Tail cut points are not unique.")
    return cuts


def tail_summary_rows(
    x: np.ndarray,
    y: np.ndarray,
    aligned_return_bps: np.ndarray,
    selected: np.ndarray,
    cuts: np.ndarray,
    *,
    quantile_edges: tuple[float, ...] = TAIL_QUANTILE_EDGES,
) -> list[dict[str, float | int]]:
    """Summarise impact and aligned raw return in frozen tail bands."""

    mask = (
        _finite_selected(x, y, selected)
        & np.isfinite(np.asarray(aligned_return_bps, dtype=float))
    )
    labels = np.digitize(x[mask], cuts, right=True)
    returns = np.asarray(aligned_return_bps, dtype=float)[mask]
    rows: list[dict[str, float | int]] = []
    for label, (lower, upper) in enumerate(
        zip(quantile_edges, quantile_edges[1:])
    ):
        chosen = labels == label
        if not chosen.any():
            continue
        observed_y = y[mask][chosen]
        observed_x = x[mask][chosen]
        observed_return = returns[chosen]
        rows.append(
            {
                "quantile_low": float(lower),
                "quantile_high": float(upper),
                "observations": int(chosen.sum()),
                "minimum_normalised_imbalance": float(observed_x.min()),
                "mean_normalised_imbalance": float(observed_x.mean()),
                "median_normalised_imbalance": float(np.median(observed_x)),
                "maximum_normalised_imbalance": float(observed_x.max()),
                "mean_normalised_signed_impact": float(observed_y.mean()),
                "median_normalised_signed_impact": float(np.median(observed_y)),
                "mean_aligned_return_bps": float(observed_return.mean()),
                "median_aligned_return_bps": float(np.median(observed_return)),
                "return_standard_error_bps": float(
                    observed_return.std(ddof=1) / np.sqrt(chosen.sum())
                )
                if chosen.sum() > 1
                else np.nan,
            }
        )
    return rows


def fit_trimmed_affine_log(
    x: np.ndarray,
    y: np.ndarray,
    selected: np.ndarray,
    *,
    scale: float,
    upper_quantile: float,
) -> dict[str, Any]:
    """Fit an affine log curve after excluding a training upper-x tail.

    The full-sample scale is deliberately held fixed, so changes in intercept
    and slope isolate the influence of the excluded high-imbalance observations.
    """

    if not 0 < upper_quantile <= 1:
        raise ValueError("upper_quantile must lie in (0, 1].")
    mask = _finite_selected(x, y, selected)
    cutoff = float(np.quantile(x[mask], upper_quantile))
    fit_mask = mask & (x <= cutoff)
    z = np.log1p(x[fit_mask] / scale)
    design = np.column_stack((np.ones(fit_mask.sum()), z))
    intercept, slope = np.linalg.lstsq(design, y[fit_mask], rcond=None)[0]
    return {
        "model": "raw_log_affine",
        "intercept": float(intercept),
        "slope": float(slope),
        "scale": float(scale),
        "upper_quantile": float(upper_quantile),
        "upper_x_cut": cutoff,
        "observations": int(fit_mask.sum()),
    }


def _bin_means(
    x: np.ndarray,
    y: np.ndarray,
    selected: np.ndarray,
    cuts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mask = _finite_selected(x, y, selected)
    labels = np.digitize(x[mask], cuts, right=True)
    bins = len(cuts) + 1
    count = np.bincount(labels, minlength=bins).astype(int)
    sum_x = np.bincount(labels, weights=x[mask], minlength=bins)
    sum_y = np.bincount(labels, weights=y[mask], minlength=bins)
    mean_x = np.divide(sum_x, count, out=np.full(bins, np.nan), where=count > 0)
    mean_y = np.divide(sum_y, count, out=np.full(bins, np.nan), where=count > 0)
    sum_y2 = np.bincount(labels, weights=y[mask] ** 2, minlength=bins)
    variance = np.divide(sum_y2, count, out=np.full(bins, np.nan), where=count > 0) - mean_y**2
    standard_error = np.sqrt(np.maximum(variance, 0.0) / np.maximum(count, 1))
    return count, mean_x, mean_y, standard_error


def _pava(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted pool-adjacent-violators algorithm for increasing means."""

    blocks: list[list[float | int]] = []
    for index, (value, weight) in enumerate(zip(values, weights, strict=True)):
        blocks.append([index, index, float(weight), float(value)])
        while len(blocks) >= 2 and float(blocks[-2][3]) > float(blocks[-1][3]):
            right = blocks.pop()
            left = blocks.pop()
            total_weight = float(left[2]) + float(right[2])
            mean = (
                float(left[2]) * float(left[3])
                + float(right[2]) * float(right[3])
            ) / total_weight
            blocks.append([int(left[0]), int(right[1]), total_weight, mean])
    fitted = np.empty(len(values), dtype=float)
    for start, end, _, mean in blocks:
        fitted[int(start) : int(end) + 1] = float(mean)
    return fitted


def fit_curve_families(
    x: np.ndarray,
    y: np.ndarray,
    training_mask: np.ndarray,
    *,
    raw_intercept: float,
    raw_slope: float,
    raw_scale: float,
    bins: int = 20,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Fit diagnostic alternatives using frozen training quantile bins."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    selected = _finite_selected(x, y, training_mask)
    cuts = training_quantile_cuts(x, selected, bins=bins)
    count, mean_x, mean_y, _ = _bin_means(x, y, selected, cuts)
    valid = (count > 0) & np.isfinite(mean_x) & np.isfinite(mean_y)
    bx, by, bw = mean_x[valid], mean_y[valid], count[valid].astype(float)

    models: list[dict[str, Any]] = [
        {
            "model": "raw_log_affine",
            "intercept": float(raw_intercept),
            "slope": float(raw_slope),
            "scale": float(raw_scale),
        }
    ]

    # Match the academic diagnostic: a curve constrained through zero, fitted
    # to equal-count bin means. Curvature is selected on a dense log grid and
    # amplitude is solved analytically at every candidate.
    best: tuple[float, float, float] | None = None
    for curvature in np.logspace(-3, 3, 601):
        basis = np.log1p(curvature * bx / raw_scale)
        denominator = float(np.dot(basis, basis))
        if denominator <= 0:
            continue
        amplitude = max(0.0, float(np.dot(basis, by) / denominator))
        rmse = float(np.sqrt(np.mean((by - amplitude * basis) ** 2)))
        if best is None or rmse < best[0]:
            best = (rmse, amplitude, float(curvature))
    if best is None:
        raise ValueError("Could not fit the binned logarithmic curve.")
    models.append(
        {
            "model": "binned_log_zero",
            "amplitude": best[1],
            "curvature": best[2],
            "scale": float(raw_scale),
        }
    )

    positive = (bx > 0) & (by > 0)
    if positive.sum() < 3:
        raise ValueError("At least three positive bins are required for a power law.")
    log_x = np.log(bx[positive])
    log_y = np.log(by[positive])
    variance = float(np.var(log_x, ddof=1))
    exponent = float(np.cov(log_x, log_y, ddof=1)[0, 1] / variance)
    exponent = float(np.clip(exponent, 0.0, 2.0))
    amplitude = float(np.exp(np.mean(log_y - exponent * log_x)))
    models.append(
        {
            "model": "binned_power",
            "amplitude": amplitude,
            "exponent": exponent,
        }
    )

    monotonic_y = _pava(by, np.ones_like(bw))
    models.append(
        {
            "model": "monotonic_bins",
            "knot_x": bx,
            "knot_y": monotonic_y,
        }
    )
    return models, cuts


def predict_curve(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    name = model["model"]
    if name == "raw_log_affine":
        return model["intercept"] + model["slope"] * np.log1p(values / model["scale"])
    if name == "binned_log_zero":
        return model["amplitude"] * np.log1p(
            model["curvature"] * values / model["scale"]
        )
    if name == "binned_power":
        return model["amplitude"] * values ** model["exponent"]
    if name == "monotonic_bins":
        return np.interp(values, model["knot_x"], model["knot_y"])
    raise KeyError(f"Unknown curve model {name!r}.")


def calibration_rows(
    x: np.ndarray,
    y: np.ndarray,
    selected: np.ndarray,
    cuts: np.ndarray,
    models: list[dict[str, Any]],
) -> list[dict[str, float | int | str]]:
    """Summarise observed and fitted impact in frozen imbalance bins."""

    mask = _finite_selected(x, y, selected)
    labels = np.digitize(x[mask], cuts, right=True)
    bins = len(cuts) + 1
    rows: list[dict[str, float | int | str]] = []
    for model in models:
        prediction = predict_curve(model, x[mask])
        for label in range(bins):
            chosen = labels == label
            if not chosen.any():
                continue
            observed = y[mask][chosen]
            fitted = prediction[chosen]
            rows.append(
                {
                    "model": str(model["model"]),
                    "impact_bin": label + 1,
                    "observations": int(chosen.sum()),
                    "mean_normalised_imbalance": float(x[mask][chosen].mean()),
                    "mean_normalised_signed_impact": float(observed.mean()),
                    "standard_error": float(observed.std(ddof=1) / np.sqrt(chosen.sum()))
                    if chosen.sum() > 1
                    else np.nan,
                    "mean_predicted_impact": float(fitted.mean()),
                    "mean_residual": float((observed - fitted).mean()),
                }
            )
    return rows


def calibration_metrics(rows: list[dict[str, Any]]) -> list[dict[str, float | int | str]]:
    output = []
    for model in MODEL_ORDER:
        chosen = [row for row in rows if row["model"] == model]
        if not chosen:
            continue
        residual = np.array([row["mean_residual"] for row in chosen], dtype=float)
        weights = np.array([row["observations"] for row in chosen], dtype=float)
        output.append(
            {
                "model": model,
                "bins": len(chosen),
                "bin_rmse": float(np.sqrt(np.mean(residual**2))),
                "weighted_bin_rmse": float(
                    np.sqrt(np.sum(weights * residual**2) / np.sum(weights))
                ),
                "bin_mae": float(np.mean(np.abs(residual))),
                "maximum_absolute_bin_residual": float(np.max(np.abs(residual))),
                "mean_bin_bias": float(np.mean(residual)),
            }
        )
    return output


def serialisable_parameters(model: dict[str, Any]) -> dict[str, float | str]:
    return {
        "model": str(model["model"]),
        "intercept": float(model.get("intercept", np.nan)),
        "slope": float(model.get("slope", np.nan)),
        "amplitude": float(model.get("amplitude", np.nan)),
        "exponent": float(model.get("exponent", np.nan)),
        "curvature": float(model.get("curvature", np.nan)),
        "scale": float(model.get("scale", np.nan)),
    }
