"""Matplotlib plots for impact estimates."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crypto_impact.regression import PowerLawFit, predict_power_law


def plot_impact_power_law(
    binned: pd.DataFrame,
    fit: PowerLawFit,
    output_path: Path,
    show_error_bars: bool = True,
) -> Path:
    """Save a log-log impact plot with fitted power-law curve."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    plot_data = binned.loc[(binned["mean_abs_imbalance"] > 0) & (binned["mean_signed_impact"] > 0)].copy()
    x = plot_data["mean_abs_imbalance"].to_numpy(dtype=float)
    y = plot_data["mean_signed_impact"].to_numpy(dtype=float)
    yerr = plot_data["standard_error"].to_numpy(dtype=float) if show_error_bars else None

    if show_error_bars:
        ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=3, label="Binned observations")
    else:
        ax.scatter(x, y, label="Binned observations")

    positive_x = x[x > 0]
    if len(positive_x) > 0:
        line_x = np.geomspace(positive_x.min(), positive_x.max(), 100)
        ax.plot(line_x, predict_power_law(line_x, fit), label=f"Fit: delta={fit.delta:.3f}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean absolute imbalance")
    ax.set_ylabel("Mean signed impact")
    ax.set_title("Empirical impact versus imbalance")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
