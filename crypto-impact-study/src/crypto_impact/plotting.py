"""Matplotlib plots for impact-shape diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crypto_impact.regression import PowerLawFit, predict_power_law


def plot_imbalance_ratio_percentile_bands(
    binned: pd.DataFrame,
    output_path: Path,
    fit: PowerLawFit | None = None,
) -> Path:
    """Plot median signed impact with percentile bands by absolute imbalance ratio."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = binned.loc[binned["median_x"] > 0].sort_values("median_x").copy()
    fig, ax = plt.subplots(figsize=(9, 6))
    x = data["median_x"].to_numpy(dtype=float)
    ax.fill_between(x, data["impact_p10"], data["impact_p90"], color="#94a3b8", alpha=0.25, label="10th-90th percentile")
    ax.fill_between(x, data["impact_p25"], data["impact_p75"], color="#0f766e", alpha=0.25, label="25th-75th percentile")
    ax.plot(x, data["median_impact"], "o-", color="#0f766e", linewidth=2.0, markersize=5, label="Median impact")
    ax.axhline(0, color="#111827", linewidth=1, alpha=0.5)
    if fit is not None and len(data) > 0:
        positive = data.loc[(data["mean_x"] > 0) & (data["mean_impact"] > 0)]
        if not positive.empty:
            line_x = np.geomspace(positive["mean_x"].min(), positive["mean_x"].max(), 100)
            ax.plot(line_x, predict_power_law(line_x, fit), color="#7c2d12", linestyle=":", linewidth=1.8, label=f"Power law delta={fit.delta:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("Absolute imbalance ratio |Q_T| / V_T")
    ax.set_ylabel("Signed impact")
    ax.set_title("Impact distribution by absolute imbalance ratio")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_imbalance_ratio_volume_heatmap(heatmap: pd.DataFrame, output_path: Path) -> Path:
    """Plot mean signed impact over absolute-imbalance-ratio and gross-volume bins."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = heatmap.copy()
    pivot = data.pivot(index="gross_volume_bin_id", columns="imbalance_ratio_bin_id", values="mean_signed_impact")
    x_labels = data.groupby("imbalance_ratio_bin_id")["median_absolute_imbalance_ratio"].median().sort_index()
    y_labels = data.groupby("gross_volume_bin_id")["median_gross_volume"].median().sort_index()

    fig, ax = plt.subplots(figsize=(10, 7))
    max_abs = np.nanmax(np.abs(pivot.to_numpy(dtype=float))) if not pivot.empty else 1.0
    image = ax.imshow(
        pivot.to_numpy(dtype=float),
        origin="lower",
        aspect="auto",
        cmap="RdBu_r",
        vmin=-max_abs,
        vmax=max_abs,
    )
    ax.set_xlabel("Absolute imbalance ratio bin median")
    ax.set_ylabel("Gross volume bin median")
    ax.set_title("Mean signed impact by absolute imbalance ratio and gross volume")

    x_positions = np.arange(len(x_labels))
    y_positions = np.arange(len(y_labels))
    x_step = max(len(x_positions) // 8, 1)
    y_step = max(len(y_positions) // 8, 1)
    ax.set_xticks(x_positions[::x_step], [f"{value:.3g}" for value in x_labels.iloc[::x_step]], rotation=45, ha="right")
    ax.set_yticks(y_positions[::y_step], [f"{value:.3g}" for value in y_labels.iloc[::y_step]])
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Mean signed impact")
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
