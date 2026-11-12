"""Section 5.2 P&B activity-residual table rebuilt on the volatility-normalised response.

The paper's table tab:pb-activity-residuals (notebooks/PatzeltBouchaudScaling.ipynb,
reports/redo/paper/tables/pb_activity_residuals.csv) uses the raw block log return:
the master curve is fitted to raw binned responses and residuals are scaled by R_N.
The surface's displacement table (activity_surface_displacement.csv) uses
y_t = sign(Q_t) r_t / sigma_{24h,t}, so the two are not directly comparable. This
script rebuilds the section 5.2 table under both responses with everything else
held fixed, and adds the surface on the same footing.

Variants (all on 2021-01..2026-08, 2020-12 used only to warm up the 24 h windows):
- pb_binned_raw:      notebook method, raw log return (reproduction check);
- pb_binned_y:        notebook method, y_t (the like-for-like rebuild);
- pb_cells_y:         P&B (eta = 1) from the headline surface run's cell fit, y_t;
- surface_y:          headline activity surface (eta free), y_t.
"Notebook method": 60 equal-count bins of x = Q/V_24h per side, min 100 per bin,
robust SE-weighted master-curve fit across N = 100..16,000 (crypto_impact.pb_collapse).

Each variant gives the table's columns for N = 250, 1,000, 4,000, 16,000, with
activity measured by event intensity N/T (the paper's state variable) and, as a
second measure, by the volume share nu = V/V_24h (the surface's activity variable).
Standard errors: the table's descriptive std/sqrt(n), and UTC-day clustered SEs
for the two contrasts.

Section 4.2 quantities for y_t (added 2026-11-12), from the same master-curve fits:
- scaling_exponents.csv: xi and psi, Theil-Sen (95% interval) and OLS slopes of the
  fitted log scales on log N, as in section 4.2 (crypto_impact.pb_collapse.scaling_slope);
- collapse_diagnostics.csv: collapse RMSE, MAE and R^2 per N and pooled, on the
  rescaled binned responses (crypto_impact.pb_collapse.collapse_diagnostics);
- binned_curves.csv / collapsed_curves.csv: the binned and rescaled curves;
- figure reports/redo/paper/figures/22c_pb_collapse_normalised_response.pdf
  (unscaled and collapsed y_t curves, in the layout of figure 22).
Both responses are reported; the raw rows are the like-for-like reference for 4.2.

Outputs: reports/redo/pb_activity_residuals_normalised/{residuals.csv,
master_fits.csv, pb_activity_residuals_y.tex, scaling_exponents.csv,
collapse_diagnostics.csv, binned_curves.csv, collapsed_curves.csv}.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "crypto-impact-study" / "src"))
sys.path.insert(0, str(REPO / "crypto-impact-study" / "scripts"))
sys.path.insert(0, str(REPO / "shared-methodology" / "src"))

from build_pb_blocks import N_VALUES  # noqa: E402
from crypto_impact.pb_collapse import (  # noqa: E402
    collapse_diagnostics, fit_pb_master_curve, independently_binned_signed_response, pb_shape, scaling_slope,
)
from run_activity_surface import load_arrays  # noqa: E402

TABLE_N = (250, 1000, 4000, 16000)
HEADLINE = REPO / "crypto-impact-study" / "reports" / "redo" / "activity_surface_n250_2021_2026"
OUT = REPO / "crypto-impact-study" / "reports" / "redo" / "pb_activity_residuals_normalised"
FIGURE = REPO / "crypto-impact-study" / "reports" / "redo" / "paper" / "figures" / "22c_pb_collapse_normalised_response.pdf"


def collapse_figure(curves: pd.DataFrame, collapsed: pd.DataFrame, fit: dict) -> None:
    """Figure 22's layout for y_t: unscaled binned curves, and the collapse onto the master curve."""

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    colours = plt.cm.viridis(np.linspace(0.05, 0.95, len(N_VALUES)))
    for colour, n in zip(colours, N_VALUES):
        raw = curves.loc[curves["n_events"] == n]
        col = collapsed.loc[collapsed["n_events"] == n]
        axes[0].plot(raw["mean_q"], raw["mean_response"], marker="o", ms=2.5, lw=1, color=colour, label=f"N={n:,}")
        axes[1].plot(col["scaled_q"], col["scaled_response"], marker="o", ms=2.5, lw=1, color=colour)
    positive = np.geomspace(max(collapsed["scaled_q"].abs().min(), 1e-8), collapsed["scaled_q"].abs().max(), 300)
    grid = np.r_[-positive[::-1], 0.0, positive]
    axes[1].plot(grid, pb_shape(grid, fit["alpha"], fit["beta"]), color="black", lw=2.2,
                 label=rf"Shared P&B curve ($\alpha$={fit['alpha']:.3f}, $\beta$={fit['beta']:.3f})")
    axes[0].set_xscale("symlog", linthresh=max(curves["mean_q"].abs().quantile(0.10), 1e-12))
    axes[1].set_xscale("symlog", linthresh=max(collapsed["scaled_q"].abs().quantile(0.10), 1e-12))
    axes[0].set_xlabel("Signed imbalance / trailing 24-hour volume")
    axes[0].set_ylabel(r"Mean $y_t = r_t/\sigma_{24h,t}$")
    axes[0].set_title(r"Unscaled response, $y_t$")
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].set_xlabel(r"$Q/\mathcal{Q}_N$")
    axes[1].set_ylabel(r"$\bar y/\mathcal{I}^\star_N$")
    axes[1].set_title(r"Scale-collapsed response, $y_t$")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.axhline(0, color="0.35", lw=0.8)
        ax.axvline(0, color="0.35", lw=0.8)
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, bbox_inches="tight")
    plt.close(fig)


def frame_for(arrays: dict) -> pd.DataFrame:
    ok = (arrays["side"] != 0) & np.isfinite(arrays["pb_x"]) & np.isfinite(arrays["intensity"]) \
        & np.isfinite(arrays["y_vol"]) & np.isfinite(arrays["a"])
    return pd.DataFrame({
        "pb_signed": arrays["side"][ok] * arrays["pb_x"][ok], "abs_x": arrays["pb_x"][ok],
        "z": arrays["z"][ok], "nu": arrays["a"][ok], "side": arrays["side"][ok],
        "log_return": arrays["y"][ok], "y_vol": arrays["y_vol"][ok],
        "intensity": arrays["intensity"][ok], "day": arrays["day"][ok],
    })


def cluster_se(resid: np.ndarray, design: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """OLS coefficients and day-clustered standard errors (CR1)."""

    xtx_inv = np.linalg.inv(design.T @ design)
    beta = xtx_inv @ design.T @ resid
    u = resid - design @ beta
    codes, inverse = np.unique(clusters, return_inverse=True)
    scores = np.zeros((len(codes), design.shape[1]))
    np.add.at(scores, inverse, design * u[:, None])
    g, n, k = len(codes), len(resid), design.shape[1]
    meat = scores.T @ scores * (g / (g - 1)) * ((n - 1) / (n - k))
    return beta, np.sqrt(np.diag(xtx_inv @ meat @ xtx_inv))


def residual_row(scaled: np.ndarray, activity: np.ndarray, abs_x: np.ndarray, day: np.ndarray) -> dict:
    act = pd.qcut(activity, 5, labels=False, duplicates="drop")
    imb = pd.qcut(abs_x, 5, labels=False, duplicates="drop")
    high, extreme = act == act.max(), imb == imb.max()

    def mean_se(v):
        return v.mean(), v.std(ddof=1) / np.sqrt(len(v))

    o_m, o_se = mean_se(scaled[~high])
    h_m, h_se = mean_se(scaled[high])
    cells = {(h, e): mean_se(scaled[(high == h) & (extreme == e)]) for h in (True, False) for e in (True, False)}
    one = np.ones(len(scaled))
    _, se_hmo = cluster_se(scaled, np.column_stack([one, high]).astype(float), day)
    _, se_int = cluster_se(scaled, np.column_stack([one, high, extreme, high & extreme]).astype(float), day)
    return {
        "ordinary_activity": o_m, "ordinary_activity_se": o_se,
        "high_activity": h_m, "high_activity_se": h_se,
        "all_states": scaled.mean(),
        "high_minus_ordinary": h_m - o_m, "high_minus_ordinary_se": np.hypot(h_se, o_se),
        "high_minus_ordinary_se_day": se_hmo[1],
        "activity_imbalance_interaction": (cells[(True, True)][0] - cells[(True, False)][0])
        - (cells[(False, True)][0] - cells[(False, False)][0]),
        "activity_imbalance_interaction_se": np.sqrt(sum(se ** 2 for _, se in cells.values())),
        "activity_imbalance_interaction_se_day": se_int[3],
        "high_extreme": cells[(True, True)][0],
        "blocks": len(scaled), "high_activity_blocks": int(high.sum()),
    }


def headline_model(name: str) -> tuple[dict, pd.DataFrame]:
    fits = pd.read_csv(HEADLINE / "fits.csv").set_index("model")
    scales = pd.read_csv(HEADLINE / "scales.csv")
    row = fits.loc[name]
    return ({"alpha": row["alpha"], "beta": row["beta"], "eta": row["gamma"]},
            scales.loc[scales["model"] == name].set_index("n_events"))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    root = REPO / "shared-data"
    frames, curves = {}, {"log_return": [], "y_vol": []}
    for n in N_VALUES:
        f = frame_for(load_arrays(root, n, "2020-12", "2021-01", "2026-08"))
        for response in curves:
            c = independently_binned_signed_response(f, "pb_signed", response=response)
            c.insert(0, "n_events", n)
            curves[response].append(c)
        if n in TABLE_N:
            frames[n] = f
        print(f"N={n}: {len(f):,} blocks", flush=True)

    # Notebook-method master curves, raw and normalised response.
    pb_fits = {r: fit_pb_master_curve(pd.concat(c, ignore_index=True), N_VALUES) for r, c in curves.items()}
    master = []
    for r, fit in pb_fits.items():
        for _, s in fit["scales"].iterrows():
            master.append({"response": r, "alpha": fit["alpha"], "beta": fit["beta"],
                           "n_events": int(s["n_events"]), "q_scale": s["fitted_q_scale"], "r_scale": s["fitted_r_scale"]})
    pd.DataFrame(master).to_csv(OUT / "master_fits.csv", index=False)
    print({r: (round(f["alpha"], 4), round(f["beta"], 4)) for r, f in pb_fits.items()}, flush=True)

    # Section 4.2 quantities for each response: xi, psi and collapse quality.
    exponents, diagnostics, binned, collapsed_all = [], [], [], []
    for r, fit in pb_fits.items():
        response_curves = pd.concat(curves[r], ignore_index=True)
        collapsed, summary = collapse_diagnostics(response_curves, fit)
        for column, label in (("fitted_q_scale", "xi: fitted response width"),
                              ("fitted_r_scale", "psi: fitted response height")):
            exponents.append({"response": r, **scaling_slope(fit["scales"], column, label)})
        diagnostics.append(summary.assign(response=r))
        binned.append(response_curves.assign(response=r))
        collapsed_all.append(collapsed.assign(response=r))
        if r == "y_vol":
            collapse_figure(response_curves, collapsed, fit)
    pd.DataFrame(exponents).to_csv(OUT / "scaling_exponents.csv", index=False)
    pd.concat(diagnostics, ignore_index=True).to_csv(OUT / "collapse_diagnostics.csv", index=False)
    pd.concat(binned, ignore_index=True).to_csv(OUT / "binned_curves.csv", index=False)
    pd.concat(collapsed_all, ignore_index=True).to_csv(OUT / "collapsed_curves.csv", index=False)
    print(pd.DataFrame(exponents)[["response", "quantity", "robust_exponent", "robust_lower_95", "robust_upper_95",
                                   "ols_exponent", "ols_r_squared"]].round(4).to_string(), flush=True)
    print(pd.concat(diagnostics).round(4).to_string(), flush=True)

    variants = {}
    for r, label in (("log_return", "pb_binned_raw"), ("y_vol", "pb_binned_y")):
        fit = pb_fits[r]
        sc = fit["scales"].set_index("n_events")
        variants[label] = (r, lambda f, n, fit=fit, sc=sc: (
            sc.loc[n, "fitted_r_scale"] * pb_shape(f["abs_x"] / sc.loc[n, "fitted_q_scale"], fit["alpha"], fit["beta"]),
            sc.loc[n, "fitted_r_scale"]))
    for name, label in (("pb_gamma_1", "pb_cells_y"), ("activity_surface", "surface_y")):
        p, sc = headline_model(name)
        variants[label] = ("y_vol", lambda f, n, p=p, sc=sc: (
            sc.loc[n, "r_scale"] * pb_shape(f["z"] * f["nu"] ** p["eta"] / sc.loc[n, "q_scale"], p["alpha"], p["beta"]),
            sc.loc[n, "r_scale"]))

    rows = []
    for label, (response, predict) in variants.items():
        for n in TABLE_N:
            f = frames[n]
            pred, r_scale = predict(f, n)
            scaled = ((np.sign(f["pb_signed"]) * f[response] - pred) / r_scale).to_numpy()
            for measure, col in (("event_intensity", "intensity"), ("volume_share", "nu")):
                rows.append({"variant": label, "response": response, "n_events": n, "activity_measure": measure,
                             **residual_row(scaled, f[col].to_numpy(), f["abs_x"].to_numpy(), f["day"].to_numpy())})
        print(f"{label} done", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "residuals.csv", index=False)

    # LaTeX in the paper table's layout: notebook method on y_t, event-intensity quintiles.
    t = table.loc[(table["variant"] == "pb_binned_y") & (table["activity_measure"] == "event_intensity")]

    def num(v, d=4):
        return f"\\({v:.{d}f}\\)"

    lines = []
    for _, r in t.iterrows():
        lines.append(f"{int(r['n_events']):,} & {num(r['ordinary_activity'])} \\(({r['ordinary_activity_se']:.4f})\\) & "
                     f"{num(r['high_activity'])} \\(({r['high_activity_se']:.4f})\\) & {num(r['all_states'])} & "
                     f"{num(r['high_minus_ordinary'])} \\(({r['high_minus_ordinary_se_day']:.4f})\\) & "
                     f"{num(r['activity_imbalance_interaction'])} \\(({r['activity_imbalance_interaction_se_day']:.4f})\\) \\\\")
    (OUT / "pb_activity_residuals_y.tex").write_text("\n".join(lines) + "\n")
    with pd.option_context("display.width", 200):
        print(table.loc[table["activity_measure"] == "event_intensity",
                        ["variant", "n_events", "ordinary_activity", "high_activity", "all_states",
                         "high_minus_ordinary", "high_minus_ordinary_se_day",
                         "activity_imbalance_interaction", "activity_imbalance_interaction_se_day"]].round(4).to_string())
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
