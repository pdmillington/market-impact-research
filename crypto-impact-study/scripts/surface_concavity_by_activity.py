"""How much of the pooled impact concavity comes from mixing activity levels? (revision log A3c-2, A3d)

Hypothesis: the pooled impact curve is concave in x = |Q| / V_24h partly because it
mixes activity levels. Within a fixed level of relative activity nu = V / V_24h (column
`a`) the surface coordinate x_eta / Q_N = q nu^eta / Q_N moves with q = |Q| / V only. If the exponent of the
volatility-normalised response in q within activity deciles is clearly higher than the
pooled exponent in x, part of the pooled concavity comes from mixing. (delta is a
single-power summary of the normalised variables, not a test of proportionality.)

At N = 1,000, 4,000 and 16,000 (pb_blocks_v1, 2021-01..2026-08, volatility-normalised
signed response y_t = sign(Q_t) r_t / sigma_24h, as in section 3 of the paper):
- relative-activity deciles of a; within each, 20 equal-count bins of q;
- a free power s y = c q^delta fitted to the bin means (geometric-mean q, inverse-
  variance weights), with a UTC-day block bootstrap (200 replicates) for delta;
- the same fit pooled over all activity levels against x = |Q| / V_24h, for contrast;
- the share of surface cells beyond the master-curve bend (x > 1 in surface units),
  from diagnose_activity_surface.py.

Outputs: reports/redo/surface_concavity/{delta_by_decile.csv, bins.csv} and
reports/redo/paper/figures/30_surface_concavity_by_activity.pdf
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/crypto-impact-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares


REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "crypto-impact-study" / "src"), str(REPO / "crypto-impact-study" / "scripts"),
                str(REPO / "shared-methodology" / "src")]

import run_activity_surface as run  # noqa: E402

OUT = REPO / "crypto-impact-study" / "reports" / "redo" / "surface_concavity"
FIGURES = REPO / "crypto-impact-study" / "reports" / "redo" / "paper" / "figures"
IDENTIFICATION = REPO / "crypto-impact-study" / "reports" / "redo" / "activity_surface_n250_2021_2026" / "identification_by_n.csv"
N_SHOWN = (1000, 4000, 16000)
N_DECILES, N_QBINS, N_BOOT = 10, 20, 200
INK, MUTED = "#1f1f1f", "#6b6b6b"


def day_bin_stats(group: np.ndarray, x: np.ndarray, y: np.ndarray, day: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"group": group, "day": day, "y": y, "y2": y * y, "logx": np.log(x)})
    return frame.groupby(["group", "day"], observed=True).agg(
        count=("y", "size"), sum_y=("y", "sum"), sum_y2=("y2", "sum"), sum_logx=("logx", "sum")).reset_index()


def bin_means(stats: pd.DataFrame, weights: pd.Series | None = None) -> pd.DataFrame:
    data = stats
    if weights is not None:
        data = stats.loc[stats["day"].isin(weights.index)].copy()
        factor = data["day"].map(weights).to_numpy()
        for column in ("count", "sum_y", "sum_y2", "sum_logx"):
            data[column] = data[column] * factor
    g = data.groupby("group", observed=True)[["count", "sum_y", "sum_y2", "sum_logx"]].sum()
    mean = g["sum_y"] / g["count"]
    variance = (g["sum_y2"] - g["count"] * mean**2) / (g["count"] - 1)
    return pd.DataFrame({"x": np.exp(g["sum_logx"] / g["count"]), "y": mean,
                         "se": np.sqrt(variance.clip(lower=0) / g["count"]), "count": g["count"]})


def fit_power(bins: pd.DataFrame) -> tuple[float, float]:
    """WLS fit of y = c x^delta to bin means; returns (c, delta)."""

    x, y, se = bins["x"].to_numpy(), bins["y"].to_numpy(), bins["se"].to_numpy()
    scale = np.exp(np.mean(np.log(x)))

    def residual(p):
        return (y - np.exp(p[0]) * (x / scale) ** p[1]) / se

    start = [np.log(max(np.median(y[y > 0]) if (y > 0).any() else 1e-3, 1e-6)), 0.8]
    result = least_squares(residual, start, bounds=([-30, 0.0], [10, 3.0]))
    return float(np.exp(result.x[0]) * scale ** (-result.x[1])), float(result.x[1])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261015)
    delta_rows, bin_rows = [], []
    for n in N_SHOWN:
        arr = run.load_arrays(REPO / "shared-data", n, "2020-12", "2021-01", "2026-08")
        ok = np.isfinite(arr["y_vol"]) & (arr["z"] > 0) & np.isfinite(arr["a"]) & (arr["side"] != 0)
        q, a, x = arr["z"][ok], arr["a"][ok], arr["pb_x"][ok]   # q = |Q|/V; x = |Q|/V24
        y, day = (arr["side"] * arr["y_vol"])[ok], arr["day"][ok]
        decile = pd.qcut(a, N_DECILES, labels=False)
        groups = {}
        for d in range(N_DECILES):
            in_d = decile == d
            qbin = pd.qcut(q[in_d], N_QBINS, labels=False, duplicates="drop")
            groups[d] = (in_d, qbin)
        pooled_bin = pd.qcut(x, N_QBINS, labels=False, duplicates="drop")
        specs = {f"decile_{d + 1}": (q[g[0]], y[g[0]], day[g[0]], g[1]) for d, g in groups.items()}
        specs["pooled_x"] = (x, y, day, pooled_bin)
        for label, (xv, yv, dv, bins_id) in specs.items():
            stats = day_bin_stats(bins_id, xv, yv, dv)
            bins = bin_means(stats)
            c, delta = fit_power(bins)
            days = np.asarray(sorted(stats["day"].unique()))
            boot = []
            for _ in range(N_BOOT):
                draw = pd.Series(rng.choice(days, len(days), replace=True)).value_counts()
                boot.append(fit_power(bin_means(stats, draw))[1])
            decile_a = np.nan if label == "pooled_x" else float(np.median(a[groups[int(label.split("_")[1]) - 1][0]]))
            delta_rows.append({"n_events": n, "group": label, "median_a": decile_a, "delta": delta, "c": c,
                               "delta_p025": np.quantile(boot, 0.025), "delta_p975": np.quantile(boot, 0.975),
                               "bins": len(bins), "blocks": int(bins["count"].sum())})
            bins = bins.reset_index().rename(columns={"group": "bin"})
            bin_rows.append(bins.assign(n_events=n, group=label, fitted=c * bins["x"] ** delta))
        print(n, "done", flush=True)
    deltas = pd.DataFrame(delta_rows)
    ident = pd.read_csv(IDENTIFICATION)
    ident = ident.loc[ident["model"] == "activity_surface", ["n_events", "share_beyond_bend", "x_over_q_p90"]]
    deltas = deltas.merge(ident, on="n_events", how="left")
    deltas.to_csv(OUT / "delta_by_decile.csv", index=False)
    bins = pd.concat(bin_rows, ignore_index=True)
    bins.to_csv(OUT / "bins.csv", index=False)
    print(deltas.round(3).to_string(index=False))
    figure(deltas, bins)


def figure(deltas: pd.DataFrame, bins: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, len(N_SHOWN), figsize=(4.6 * len(N_SHOWN), 8.2),
                             gridspec_kw={"height_ratios": [1.5, 1.0]})
    colours = plt.cm.Blues(np.linspace(0.30, 1.0, N_DECILES))
    for col, n in enumerate(N_SHOWN):
        ax = axes[0, col]
        for d in range(N_DECILES):
            b = bins.loc[(bins["n_events"] == n) & (bins["group"] == f"decile_{d + 1}")].sort_values("x")
            keep = b["y"] > 0
            ax.plot(b.loc[keep, "x"], b.loc[keep, "y"], "o", ms=3, color=colours[d], alpha=0.85)
            ax.plot(b["x"], b["fitted"], "-", lw=1.1, color=colours[d])
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"N = {n:,} events", fontsize=11, color=INK)
        ax.set_xlabel(r"Imbalance ratio $q=|Q|/V$")
        ax.grid(color="#e6e6e6", lw=0.7)
        dax = axes[1, col]
        dsub = deltas.loc[(deltas["n_events"] == n) & deltas["group"].str.startswith("decile")]
        idx = np.arange(1, len(dsub) + 1)
        dax.errorbar(idx, dsub["delta"], yerr=[dsub["delta"] - dsub["delta_p025"], dsub["delta_p975"] - dsub["delta"]],
                     fmt="o", color=INK, ms=5, capsize=2, label=r"within decile: $\delta$ in $q$ (95% day-block CI)")
        pooled = deltas.loc[(deltas["n_events"] == n) & (deltas["group"] == "pooled_x"), "delta"].iat[0]
        dax.axhline(pooled, color="#eb6834", ls="--", lw=1.4, label=rf"pooled: $\delta$ in $x=|Q|/V_{{24h}}$ = {pooled:.2f}")
        dax.axhline(1.0, color=MUTED, lw=0.9, ls=":", label=r"proportional ($\delta = 1$)")
        dax.set_xticks(idx); dax.set_xlabel(r"Relative-activity decile ($\nu=V/V_{24h}$), low $\rightarrow$ high")
        dax.set_ylim(0.3, 1.3); dax.grid(color="#e6e6e6", lw=0.7)
        if col == 0:
            dax.set_ylabel(r"Fitted exponent $\delta$")
            ax.set_ylabel(r"Mean response $\bar y$  ($y_t=\mathrm{sign}(Q_t)\,r_t/\sigma^-_{24h,t}$)")
            dax.legend(fontsize=7.5, loc="lower left", frameon=True)
    fig.tight_layout()
    fig.savefig(FIGURES / "30_surface_concavity_by_activity.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
