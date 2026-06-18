"""Matplotlib plots for impact-shape diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crypto_impact.regression import PowerLawFit, predict_power_law


def plot_shape_overlay(
    observations: pd.DataFrame,
    equal_count_bins: pd.DataFrame,
    lowess_curve: pd.DataFrame,
    output_path: Path,
    raw_x_col: str,
    x_label: str,
    title: str,
    fit: PowerLawFit | None = None,
    raw_sample_size: int = 25_000,
    random_state: int = 20240618,
) -> Path:
    """Plot raw observations, equal-count bin averages, LOWESS, and optional fit."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw = observations.loc[(observations[raw_x_col] > 0) & (observations["signed_impact"] > 0), [raw_x_col, "signed_impact"]].copy()
    if len(raw) > raw_sample_size:
        raw = raw.sample(raw_sample_size, random_state=random_state)
    bins = equal_count_bins.loc[(equal_count_bins["mean_x"] > 0) & (equal_count_bins["mean_impact"] > 0)].copy()

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(raw[raw_x_col], raw["signed_impact"], s=7, alpha=0.08, color="#64748b", linewidths=0, label=f"Raw sampled n={len(raw):,}")
    ax.errorbar(
        bins["mean_x"],
        bins["mean_impact"],
        yerr=bins["stderr"],
        fmt="o-",
        color="#0f766e",
        ecolor="#0f766e",
        elinewidth=1,
        capsize=3,
        markersize=6,
        linewidth=1.8,
        label="Equal-count bin averages",
    )
    if not lowess_curve.empty:
        ax.plot(lowess_curve["x"], lowess_curve["lowess_impact"], color="#c2410c", linewidth=2.0, label="LOWESS")
    if fit is not None and len(bins) > 0:
        line_x = np.geomspace(bins["mean_x"].min(), bins["mean_x"].max(), 100)
        ax.plot(line_x, predict_power_law(line_x, fit), color="#111827", linewidth=1.5, linestyle="--", label=f"Power law delta={fit.delta:.3f}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Signed impact, positive observations only")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_local_slopes(slopes: pd.DataFrame, output_path: Path, x_label: str, title: str) -> Path:
    """Plot local log-log slope against the midpoint x value."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    if not slopes.empty:
        ax.plot(slopes["x_mid"], slopes["local_slope"], "o-", color="#7c3aed", linewidth=1.6, markersize=5)
        ax.axhline(0.5, color="#111827", linestyle="--", linewidth=1, alpha=0.7, label="Square-root benchmark")
    ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Adjacent-bin local slope")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


# Backward-compatible wrappers retained for older scripts.
def plot_overlay(*args, **kwargs):
    """Backward-compatible alias for the previous overlay function."""

    return plot_shape_overlay(*args, **kwargs)


def plot_impact_power_law(binned: pd.DataFrame, fit: PowerLawFit, output_path: Path, show_error_bars: bool = True) -> Path:
    """Save a legacy bin-only power-law plot."""

    bins = binned.rename(columns={"mean_signed_impact": "mean_impact", "standard_error": "stderr"}).copy()
    if "mean_x" not in bins and fit.x_col in bins:
        bins["mean_x"] = bins[fit.x_col]
    return plot_shape_overlay(
        observations=pd.DataFrame(columns=["x", "signed_impact"]),
        equal_count_bins=bins,
        lowess_curve=pd.DataFrame(columns=["x", "lowess_impact"]),
        output_path=output_path,
        raw_x_col="x",
        x_label=fit.x_col.replace("_", " "),
        title="Power-law diagnostic",
        fit=fit,
    )
