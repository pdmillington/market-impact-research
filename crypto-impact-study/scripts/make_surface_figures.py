"""Figures and tables for the activity-adjusted P&B surface (thesis chapter).

Reads the outputs of run_activity_surface.py and diagnose_activity_surface.py
(reports/redo/activity_surface_2021_2026/) and recomputes the full-sample cells
(same cuts and response) for the collapse figure.

Writes to reports/redo/paper/:
- figures/26_activity_surface_collapse.pdf   P&B coordinate vs activity-adjusted coordinate
- figures/27_activity_surface_gamma.pdf      gamma by bar size and by calendar year
- figures/28_activity_surface_oos.pdf        out-of-sample weighted RMSE by test year
- figures/29_activity_surface_contours.pdf   response over (activity, imbalance): data vs fitted contours
- tables/activity_surface_*.csv              the data behind each figure
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/crypto-impact-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "crypto-impact-study" / "src"), str(REPO / "crypto-impact-study" / "scripts"),
                str(REPO / "shared-methodology" / "src")]

import run_activity_surface as run  # noqa: E402
from crypto_impact.pb_collapse import pb_shape  # noqa: E402
from crypto_impact.pb_surface import aggregate_cells  # noqa: E402

SOURCE = REPO / "crypto-impact-study" / "reports" / "redo" / "activity_surface_2021_2026"
OUT = REPO / "crypto-impact-study" / "reports" / "redo" / "paper"
# Categorical slots 1-3 of the dataviz reference palette (pre-validated); secondary
# encoding (hatching / markers) is added so colour never carries identity alone.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED = "#1f1f1f", "#6b6b6b"
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"axes.edgecolor": "#bdbdbd", "grid.color": "#e6e6e6", "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "font.size": 10})


def full_sample_cells() -> pd.DataFrame:
    """The redo's full-sample cells (same cuts and response as run_activity_surface.py)."""

    stats = []
    for n in run.N_VALUES:
        arrays = run.load_arrays(REPO / "shared-data", n, "2020-12", "2021-01", "2026-08")
        stats.append(run.stats_for(arrays, n, None, "y_vol"))
    return aggregate_cells(pd.concat(stats, ignore_index=True))


def collapse_figure(cells: pd.DataFrame, fits: pd.DataFrame, scales: pd.DataFrame, figures: Path, tables: Path) -> None:
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=False)
    activity_colours = plt.cm.Blues(np.linspace(0.35, 1.0, int(cells["ia"].max()) + 1))
    for ax, model, title in ((axes[0], "pb_gamma_1", r"P&B coordinate $z\,a/\mathcal{Q}_N$ ($\gamma=1$)"),
                             (axes[1], "activity_surface", None)):
        f = fits.loc[fits.model == model].iloc[0]
        sc = scales.loc[scales.model == model].set_index("n_events")
        q = cells["n_events"].map(sc["q_scale"]).to_numpy()
        r = cells["n_events"].map(sc["r_scale"]).to_numpy()
        x = np.exp(cells["mean_logz"] + f.gamma * cells["mean_loga"]).to_numpy() / q
        y = cells["side"].to_numpy() * cells["mean_y"].to_numpy() / r
        keep = (y > 0) & (x > 0)
        ax.scatter(x[keep], y[keep], s=9, c=activity_colours[cells["ia"].to_numpy()[keep]], edgecolors="none", alpha=0.85)
        grid = np.geomspace(x[keep].min(), x[keep].max(), 400)
        ax.plot(grid, pb_shape(grid, f.alpha, f.beta), color=INK, lw=2.0,
                label=rf"master curve ($\alpha$={f.alpha:.2f}, $\beta$={f.beta:.2f})")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(title or rf"Activity-adjusted coordinate $z\,a^{{\gamma}}/\mathcal{{Q}}_N$ ($\gamma$={f.gamma:.2f})")
        ax.set_ylabel(r"Folded response $s\,\bar y/\mathcal{R}_N$  ($y=r/\sigma_{24h}$)")
        ax.set_title(f"{'P&B (γ = 1)' if model == 'pb_gamma_1' else 'Activity surface'}: weighted RMSE {f.weighted_rmse:.1f}",
                     color=INK, fontsize=11)
        ax.legend(loc="upper left", fontsize=9, frameon=True)
        for i in np.flatnonzero(keep):
            rows.append({"model": model, "n_events": int(cells["n_events"].iat[i]), "side": int(cells["side"].iat[i]),
                         "imbalance_bin": int(cells["iz"].iat[i]), "activity_bin": int(cells["ia"].iat[i]),
                         "observations": int(cells["count"].iat[i]), "scaled_coordinate": x[i], "folded_scaled_response": y[i]})
    sm = plt.cm.ScalarMappable(cmap=matplotlib.colors.ListedColormap(activity_colours),
                               norm=matplotlib.colors.Normalize(1, len(activity_colours)))
    bar = fig.colorbar(sm, ax=axes, shrink=0.85, pad=0.02)
    bar.set_label("Relative-activity decile ($a=V/V_{24h}$), low → high")
    fig.savefig(figures / "26_activity_surface_collapse.pdf", bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(tables / "activity_surface_collapse_cells.csv", index=False)


def gamma_figure(fits: pd.DataFrame, figures: Path, tables: Path) -> None:
    boot = pd.read_csv(SOURCE / "gamma_bootstrap.csv")
    common = fits.loc[fits.model == "activity_surface", "gamma"].iat[0]
    lo, hi = np.quantile(boot["gamma"], [0.025, 0.975])
    by_n = fits.loc[fits.model == "independent_N_activity_surface", ["n_events", "gamma"]].astype({"n_events": int})
    by_year = pd.read_csv(SOURCE / "gamma_by_year.csv")
    by_year = by_year.loc[by_year.model == "activity_surface", ["sample", "gamma"]].rename(columns={"sample": "year"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, frame, xcol, xlabel in ((axes[0], by_n, "n_events", "Events per bar, $N$"),
                                    (axes[1], by_year, "year", "Calendar year")):
        xs = frame[xcol].astype(float).to_numpy() if xcol == "n_events" else np.arange(len(frame))
        ax.axhspan(lo, hi, color=BLUE, alpha=0.12, lw=0, label=f"common γ 95% bootstrap [{lo:.2f}, {hi:.2f}]")
        ax.axhline(common, color=BLUE, lw=1.5, ls="--", label=f"common γ = {common:.2f}")
        ax.plot(xs, frame["gamma"], color=INK, lw=2, marker="o", ms=7,
                markerfacecolor="white", markeredgewidth=2, label="γ fitted separately")
        if xcol == "n_events":
            ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels([f"{int(v):,}" for v in xs], rotation=0, fontsize=8)
        else:
            ax.set_xticks(xs); ax.set_xticklabels(frame["year"].astype(str))
        ax.set_xlabel(xlabel); ax.set_ylabel("Activity exponent γ"); ax.set_ylim(0, 0.8)
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_title("γ by aggregation scale (2021–Aug 2026)", fontsize=11)
    axes[1].set_title("γ by calendar year (all N; 2026 to August)", fontsize=11)
    fig.tight_layout()
    fig.savefig(figures / "27_activity_surface_gamma.pdf", bbox_inches="tight")
    plt.close(fig)
    by_n.to_csv(tables / "activity_surface_gamma_by_n.csv", index=False)
    by_year.to_csv(tables / "activity_surface_gamma_by_year.csv", index=False)
    pd.DataFrame([{"common_gamma": common, "bootstrap_2_5": lo, "bootstrap_97_5": hi,
                   "replicates": len(boot)}]).to_csv(tables / "activity_surface_gamma_bootstrap_summary.csv", index=False)


def oos_figure(figures: Path, tables: Path) -> None:
    oos = pd.read_csv(SOURCE / "oos.csv")
    common = oos.loc[~oos.model.str.startswith("independent")].pivot(index="test_year", columns="model", values="test_weighted_rmse")
    independent = (oos.loc[oos.model == "independent_N_activity_surface"].groupby("test_year")["test_weighted_rmse"]
                   .apply(lambda s: float(np.sqrt(np.mean(s ** 2)))))
    years = common.index.to_numpy()
    width = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.4))
    specs = (("activity_surface", "Activity surface (common γ)", BLUE, ""),
             ("pb_gamma_1", "P&B (γ = 1)", ORANGE, "//"),
             ("imbalance_ratio_gamma_0", "Imbalance ratio (γ = 0)", AQUA, ".."))
    for k, (column, label, colour, hatch) in enumerate(specs):
        ax.bar(np.arange(len(years)) + (k - 1) * width, common[column], width=width - 0.02, color=colour,
               hatch=hatch, edgecolor="white", linewidth=0.8, label=label)
    ax.plot(np.arange(len(years)) - width, independent.loc[years], ls="none", marker="D", ms=7, color=INK,
            markerfacecolor="white", markeredgewidth=1.8, label="Separate surface per N (benchmark)")
    ax.set_xticks(np.arange(len(years)))
    ax.set_xticklabels([f"{y}" + (" (Jan–Aug)" if y == 2026 else "") for y in years])
    ax.set_ylabel("Out-of-sample weighted RMSE (lower is better)")
    ax.set_xlabel("Test year (training: 2021 to the preceding year)")
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(l) for l in [s[1] for s in specs] + ["Separate surface per N (benchmark)"]]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=8, loc="upper left", ncol=2)
    ax.set_ylim(0, max(common.max().max(), independent.max()) * 1.25)
    fig.tight_layout()
    fig.savefig(figures / "28_activity_surface_oos.pdf", bbox_inches="tight")
    plt.close(fig)
    table = common.join(independent.rename("independent_N_activity_surface"))
    table.to_csv(tables / "activity_surface_oos_weighted_rmse.csv")


CONTOUR_N = (250, 2000, 16000)


def contour_figure(cells: pd.DataFrame, fits: pd.DataFrame, scales: pd.DataFrame, figures: Path, tables: Path) -> None:
    """Response surface over (log a, log z) for three N: data as filled contours (log scale), fits as lines.

    The colour is the folded cell mean (sides averaged) divided by the surface's R_N, so the
    panels share one scale. Lines are contours of the fitted response on the same scale and at
    the same levels: solid = activity surface (slope -gamma in these axes), dashed = P&B (gamma = 1).
    """

    surface = fits.loc[fits.model == "activity_surface"].iloc[0]
    pb = fits.loc[fits.model == "pb_gamma_1"].iloc[0]
    sc = {m: scales.loc[scales.model == m].set_index("n_events") for m in ("activity_surface", "pb_gamma_1")}
    levels = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 4.0])
    fill_levels = np.geomspace(0.05, 8.0, 25)
    norm = matplotlib.colors.LogNorm(fill_levels[0], fill_levels[-1])
    ramp = matplotlib.colors.ListedColormap(plt.cm.Blues(np.linspace(0.05, 0.75, 256)))
    ln10 = np.log(10)
    fig, axes = plt.subplots(1, len(CONTOUR_N), figsize=(4.6 * len(CONTOUR_N), 4.4), sharey=False)
    rows = []
    for ax, n in zip(axes, CONTOUR_N):
        sub = cells.loc[cells["n_events"] == n].assign(fy=lambda d: d["side"] * d["mean_y"])
        fc = sub.groupby(["iz", "ia"])[["mean_logz", "mean_loga", "fy"]].mean().reset_index()
        r_surface = sc["activity_surface"].loc[n, "r_scale"]
        fc["scaled"] = fc["fy"] / r_surface
        # The cells form a complete (z bin x a bin) grid, so the data are drawn on that curvilinear
        # grid (no interpolation outside the cells' hull).
        grid = {c: fc.pivot(index="iz", columns="ia", values=c).to_numpy() for c in ("mean_loga", "mean_logz", "scaled")}
        fill = ax.contourf(grid["mean_loga"] / ln10, grid["mean_logz"] / ln10,
                           np.clip(grid["scaled"], fill_levels[0], fill_levels[-1]),
                           levels=fill_levels, cmap=ramp, norm=norm)
        A, Z = grid["mean_loga"], grid["mean_logz"]   # fitted contours on the same cells
        for f, model, style, colour in ((surface, "activity_surface", "-", INK), (pb, "pb_gamma_1", "--", ORANGE)):
            q, r = sc[model].loc[n, "q_scale"], sc[model].loc[n, "r_scale"]
            pred = r * pb_shape(np.exp(Z + f.gamma * A) / q, f.alpha, f.beta) / r_surface
            ax.contour(A / ln10, Z / ln10, pred, levels=levels, colors=colour, linestyles=style, linewidths=1.4)
        ax.set_title(f"N = {n:,} events", fontsize=11, color=INK)
        ax.set_xlabel(r"$\log_{10} a$  ($a=V/V_{24h}$)")
        ax.grid(False)
        for row in fc.itertuples():
            rows.append({"n_events": n, "imbalance_bin": int(row.iz), "activity_bin": int(row.ia),
                         "log10_z": row.mean_logz / ln10, "log10_a": row.mean_loga / ln10,
                         "folded_mean_response": row.fy, "scaled_by_surface_r_n": row.scaled})
    axes[0].set_ylabel(r"$\log_{10} z$  ($z=|Q|/V$)")
    handles = [matplotlib.lines.Line2D([], [], color=INK, lw=1.4, label=f"Activity surface, γ = {surface.gamma:.2f}"),
               matplotlib.lines.Line2D([], [], color=ORANGE, lw=1.4, ls="--", label="P&B, γ = 1")]
    axes[0].legend(handles=handles, loc="lower left", fontsize=8, frameon=True)
    bar = fig.colorbar(fill, ax=axes, shrink=0.85, pad=0.02, ticks=levels)
    bar.ax.set_yticklabels([f"{v:g}" for v in levels])
    bar.set_label(r"Folded mean response $s\,\bar y/\mathcal{R}_N$ (contour levels marked)")
    fig.savefig(figures / "29_activity_surface_contours.pdf", bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(tables / "activity_surface_contour_cells.csv", index=False)
    pd.DataFrame({"contour_level_scaled_response": levels}).to_csv(tables / "activity_surface_contour_levels.csv", index=False)


def main() -> None:
    figures, tables = OUT / "figures", OUT / "tables"
    figures.mkdir(parents=True, exist_ok=True); tables.mkdir(parents=True, exist_ok=True)
    fits = pd.read_csv(SOURCE / "fits.csv")
    scales = pd.read_csv(SOURCE / "scales.csv")
    common = fits.loc[~fits.model.str.startswith("independent")]
    common[["model", "gamma", "alpha", "beta", "weighted_rmse", "cells"]].to_csv(tables / "activity_surface_model_comparison.csv", index=False)
    pd.read_csv(SOURCE / "identification_by_n.csv").to_csv(tables / "activity_surface_identification_by_n.csv", index=False)
    pd.read_csv(SOURCE / "scaling_exponents_identified.csv").to_csv(tables / "activity_surface_scaling_exponents.csv", index=False)
    pd.read_csv(SOURCE / "displacement.csv").to_csv(tables / "activity_surface_displacement.csv", index=False)
    pd.read_csv(SOURCE / "robustness.csv").to_csv(tables / "activity_surface_robustness.csv", index=False)
    gamma_figure(fits, figures, tables)
    oos_figure(figures, tables)
    cells = full_sample_cells()
    collapse_figure(cells, fits, scales, figures, tables)
    contour_figure(cells, fits, scales, figures, tables)
    print(json.dumps(sorted(p.name for p in figures.glob("2[6-9]_*.pdf"))))


if __name__ == "__main__":
    main()
