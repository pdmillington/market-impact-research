"""Power-law scales inside the surface fit (revision log A3g): additional work, for the paper.

The headline surface (activity_surface_n250_2021_2026) has shared alpha, beta, eta and
free scales Q_N, I*_N for each N: 3 + 2 x 7 = 17 parameters; xi and psi are slopes
computed after the fit. Here the scales are restricted inside the fit,

    Q_N = Q_1 N^xi,   I*_N = I*_1 N^psi,

(crypto_impact.pb_surface.fit_activity_surface(scale_law=True)), giving 7 parameters
(eta, alpha, beta, Q_1, xi, I*_1, psi), with 6-parameter benchmarks at eta = 1 (P&B) and
eta = 0 (imbalance ratio). The headline fits and descriptions are not changed.

Same data, response, cells and loss as the headline: pb_blocks_v1, N = 250..16,000,
2021-01..2026-08, y_t = sign(Q_t) r_t / sigma_24h, 20 x 10 equal-count (q, nu) cells per N
and side, full-sample cuts, clipped standard-error weights, soft-L1 loss.

1. Full-sample fits: free-scale surface (reproduces the headline), the 7-parameter
   surface and the two 6-parameter benchmarks; standardised RMSE on the same cells.
2. Month-block bootstrap (200) of eta, xi, psi for the 7-parameter surface.
3. Out of sample by year, as run_activity_surface.py: train 2021..Y-1 (cuts from the
   training months), test Y in 2023..2026, standardised RMSE on the test cells.
4. Rolling: train on the 12 months before each test quarter, test the next 3 months
   (2022Q1..2026Q3; the last quarter is July-August), against an expanding window from
   2021-01; 7-parameter and free-scale surfaces. Training cells use the adaptive grid of
   E-008 amendment 1 (n_nu = clamp(floor(sqrt(n / 200)), 2, 10), n_q = 2 n_nu per side;
   an N with fewer than 8 cells is dropped from that fit). Training blocks end before
   the test quarter. Scoring is block by block (3-month tests are too small for cells):
   MSE of y_t, and of y_t clipped at the training 0.1% / 99.9% quantiles; per month m,
   D_m = mean over N of the relative MSE reduction against the comparison model;
   month-block bootstrap (2,000) of the mean of D_m.
   The E-008 result (a trailing 12-month window does not help the free-scale surface)
   was known when this design was chosen.

Outputs: reports/redo/surface_power_law/{fits,scales,bootstrap,oos_by_year,rolling_fits,
rolling_monthly_scores}.csv, rolling_summary.json; and
reports/redo/paper/figures/32_surface_power_law_scales.pdf
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
from crypto_impact.pb_surface import (aggregate_cells, fit_activity_surface, grid_cuts,  # noqa: E402
                                      monthly_cell_stats, weighted_rmse)

OUT = REPO / "crypto-impact-study" / "reports" / "redo" / "surface_power_law"
FIGURES = REPO / "crypto-impact-study" / "reports" / "redo" / "paper" / "figures"
HEADLINE = REPO / "crypto-impact-study" / "reports" / "redo" / "activity_surface_n250_2021_2026"
N_SET = (250, 500, 1000, 2000, 4000, 8000, 16000)
START, END = "2021-01", "2026-08"          # nothing from 2026-09 onward (P-006)
MODELS = {  # name: (fixed eta or None, power-law scales, parameter count)
    "surface_free": (None, False, 3 + 2 * len(N_SET)),
    "surface_power": (None, True, 7),
    "pb_power": (1.0, True, 6),
    "ratio_power": (0.0, True, 6),
}
TEST_YEARS = (2023, 2024, 2025, 2026)
ROLLING_MODELS = ("surface_power", "surface_free")
WINDOWS = {"trailing_12": 12, "expanding": None}
N_BOOT, N_BOOT_ROLL, SEED = 200, 2_000, 20261031
MIN_CELLS_PER_N = 8
INK, BLUE, ORANGE, MUTED = "#1f1f1f", "#2a78d6", "#eb6834", "#6b6b6b"


def month_start_day(m: int) -> int:
    return int((np.datetime64(f"{m // 12}-{m % 12 + 1:02d}-01", "D") - np.datetime64("1970-01-01", "D")).astype(int))


def label(m: int) -> str:
    return f"{m // 12}-{m % 12 + 1:02d}"


FIRST, LAST = run.month_index(START), run.month_index(END)
QUARTERS = list(range(run.month_index("2022-01"), LAST + 1, 3))       # first month of each test quarter


def test_months(q: int) -> list[int]:
    return [m for m in range(q, q + 3) if m <= LAST]


def training_mask(arr: dict, window: str, q: int) -> np.ndarray:
    first = FIRST if WINDOWS[window] is None else max(FIRST, q - WINDOWS[window])
    return (arr["month"] >= first) & (arr["end_day"] < month_start_day(q)) & np.isfinite(arr["y_vol"])


def grid_size(n_blocks: int) -> tuple[int, int]:
    n_nu = int(min(10, max(2, np.floor(np.sqrt(n_blocks / 200)))))
    return 2 * n_nu, n_nu


def fit(model: str, cells: pd.DataFrame, n_starts: int = 8, initial=None):
    gamma, law, _ = MODELS[model]
    return fit_activity_surface(cells, gamma=gamma, scale_law=law, n_starts=n_starts, initial=initial)


def fit_record(model: str, f, cells: pd.DataFrame, extra: dict) -> dict:
    law = f.scale_law or {}
    return {**extra, "model": model, "parameters": MODELS[model][2], "gamma": f.gamma, "alpha": f.alpha, "beta": f.beta,
            "q1": law.get("q1", np.nan), "xi": law.get("xi", np.nan), "r1": law.get("r1", np.nan),
            "psi": law.get("psi", np.nan), "standardised_rmse": weighted_rmse(f, cells), "cells": len(cells),
            "objective": f.objective, "converged": f.converged}


def load(n: int) -> dict:
    arr = run.load_arrays(REPO / "shared-data", n, "2020-12", START, END)
    if arr["month"].max() > LAST:
        raise RuntimeError("data after 2026-08 loaded (P-006)")
    return arr


def pass_one() -> tuple[pd.DataFrame, dict, dict]:
    """Cell statistics for the full sample, the by-year folds and the rolling training windows."""

    full, folds = [], {year: [] for year in TEST_YEARS}
    rolling = {(w, q): [] for w in WINDOWS for q in QUARTERS}
    for n in N_SET:
        arr = load(n)
        full.append(run.stats_for(arr, n, None, "y_vol"))
        months = sorted(set(arr["month"].tolist()))
        for year in TEST_YEARS:
            folds[year].append(run.stats_for(arr, n, {m for m in months if m // 12 < year}, "y_vol"))
        for (window, q) in rolling:
            train = training_mask(arr, window, q)
            cuts = {}
            for s in (-1, 1):
                usable = train & (arr["side"] == s) & (arr["z"] > 0) & np.isfinite(arr["a"]) & (arr["a"] > 0)
                n_q, n_nu = grid_size(int(usable.sum()))
                cuts[s] = grid_cuts(arr["z"][usable], arr["a"][usable], n_q, n_nu)
            frame = monthly_cell_stats(np.zeros(int(train.sum()), dtype=np.int32), arr["side"][train], arr["z"][train],
                                       arr["a"][train], arr["y_vol"][train], cuts)
            frame.insert(0, "n_events", n)
            rolling[(window, q)].append(frame)
        print(f"pass 1: N = {n:,}", flush=True)
        del arr
    return pd.concat(full, ignore_index=True), {y: pd.concat(v, ignore_index=True) for y, v in folds.items()}, rolling


def full_sample(stats: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    cells = aggregate_cells(stats)
    fits = {m: fit(m, cells) for m in MODELS}
    rows = [fit_record(m, f, cells, {"sample": f"{START}..{END}"}) for m, f in fits.items()]
    scales = pd.concat([f.scales.assign(model=m) for m, f in fits.items()], ignore_index=True)
    headline = pd.read_csv(HEADLINE / "fits.csv").set_index("model").loc["activity_surface"]
    print(f"free-scale surface eta {fits['surface_free'].gamma:.4f} (headline {headline['gamma']:.4f}); "
          f"7-parameter eta {fits['surface_power'].gamma:.4f}", flush=True)
    return fits, pd.DataFrame(rows), scales


def bootstrap(stats: pd.DataFrame, start_fit) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    months = np.asarray(sorted(stats["month"].unique()))
    rows = []
    for k in range(N_BOOT):
        draw = pd.Series(rng.choice(months, len(months), replace=True)).value_counts()
        f = fit("surface_power", aggregate_cells(stats, weights=draw), n_starts=1, initial=start_fit)
        rows.append({"replicate": k, "gamma": f.gamma, "alpha": f.alpha, "beta": f.beta,
                     "xi": f.scale_law["xi"], "psi": f.scale_law["psi"]})
    return pd.DataFrame(rows)


def oos_by_year(folds: dict) -> pd.DataFrame:
    rows = []
    for year, stats in folds.items():
        months = sorted(stats["month"].unique())
        train = aggregate_cells(stats, months=[m for m in months if m // 12 < year])
        test = aggregate_cells(stats, months=[m for m in months if m // 12 == year])
        for model in MODELS:
            f = fit(model, train, n_starts=6)
            rows.append({"test_year": year, "model": model, "parameters": MODELS[model][2], "gamma": f.gamma,
                         "test_standardised_rmse": weighted_rmse(f, test), "test_cells": len(test)})
        print(f"OOS {year} done", flush=True)
    return pd.DataFrame(rows)


def rolling_fits(rolling: dict) -> tuple[dict, pd.DataFrame]:
    fits, rows = {}, []
    for window in WINDOWS:
        for model in ROLLING_MODELS:
            previous = None
            for q in QUARTERS:
                cells = aggregate_cells(pd.concat(rolling[(window, q)], ignore_index=True))
                counts = cells.groupby("n_events").size()
                kept = sorted(counts.index[counts >= MIN_CELLS_PER_N])
                cells = cells.loc[cells["n_events"].isin(kept)].reset_index(drop=True)
                warm = previous is not None and (MODELS[model][1] or sorted(previous.scales["n_events"]) == kept)
                f = fit(model, cells, n_starts=2, initial=previous) if warm else fit(model, cells)
                fits[(window, model, q)] = previous = f
                rows.append({**fit_record(model, f, cells, {"window": window, "test_quarter": label(q)}),
                             "n_included": ",".join(str(n) for n in kept), "warm_start": warm})
            print(f"rolling fits: {window} {model}", flush=True)
    return fits, pd.DataFrame(rows)


def rolling_scores(fits: dict) -> pd.DataFrame:
    rows = []
    for n in N_SET:
        arr = load(n)
        for (window, model, q), f in fits.items():
            sc = f.scales.set_index("n_events")
            if n not in sc.index:
                continue
            train = training_mask(arr, window, q)
            lo, hi = np.quantile(arr["y_vol"][train], [0.001, 0.999])
            for m in test_months(q):
                test = (arr["month"] == m) & np.isfinite(arr["y_vol"]) & np.isfinite(arr["a"]) & (arr["a"] > 0)
                q_, a, side, y = arr["z"][test], arr["a"][test], arr["side"][test], arr["y_vol"][test]
                pred = side * sc.loc[n, "r_scale"] * pb_shape(q_ * a ** f.gamma / sc.loc[n, "q_scale"], f.alpha, f.beta)
                pred = np.where(q_ > 0, pred, 0.0)
                rows.append({"window": window, "model": model, "test_quarter": label(q), "test_month": label(m),
                             "n_events": n, "blocks": int(test.sum()), "sse": float(((y - pred) ** 2).sum()),
                             "sse_clip": float(((np.clip(y, lo, hi) - pred) ** 2).sum()), "sum_y2": float((y ** 2).sum())})
        print(f"rolling scores: N = {n:,}", flush=True)
        del arr
    return pd.DataFrame(rows)


def compare(scores: pd.DataFrame, cand: tuple[str, str], base: tuple[str, str], rng) -> dict:
    out = {}
    key = ["test_month", "n_events"]
    c = scores.loc[(scores.window == cand[0]) & (scores.model == cand[1])].set_index(key)
    b = scores.loc[(scores.window == base[0]) & (scores.model == base[1])].set_index(key)
    common = c.index.intersection(b.index)
    c, b = c.loc[common], b.loc[common]
    for kind, col in (("raw", "sse"), ("clipped", "sse_clip")):
        rel = (b[col] - c[col]) / b[col]
        d_m = rel.groupby(level="test_month").mean()
        boot = [rng.choice(d_m.to_numpy(), len(d_m), replace=True).mean() for _ in range(N_BOOT_ROLL)]
        out[kind] = {"months": int(len(d_m)), "month_n_pairs": int(len(rel)), "mean_D": float(d_m.mean()),
                     "ci_2_5": float(np.quantile(boot, 0.025)), "ci_97_5": float(np.quantile(boot, 0.975)),
                     "share_months_positive": float((d_m > 0).mean()),
                     "per_n": {int(n): float(v) for n, v in rel.groupby(level="n_events").mean().items()},
                     "per_year": {y: float(v) for y, v in d_m.groupby(d_m.index.str[:4]).mean().items()}}
    return out


COMPARISONS = {
    "7-parameter: trailing 12 vs expanding": (("trailing_12", "surface_power"), ("expanding", "surface_power")),
    "free scales: trailing 12 vs expanding": (("trailing_12", "surface_free"), ("expanding", "surface_free")),
    "trailing 12: 7-parameter vs free scales": (("trailing_12", "surface_power"), ("trailing_12", "surface_free")),
    "expanding: 7-parameter vs free scales": (("expanding", "surface_power"), ("expanding", "surface_free")),
}


def scales_figure(fits: dict) -> None:
    """Free per-N scales (headline-style fit) against the power laws fitted inside the 7-parameter surface."""

    free, power = fits["surface_free"], fits["surface_power"]
    n = np.asarray(N_SET, float)
    grid = np.geomspace(n.min() * 0.8, n.max() * 1.25, 100)
    law = power.scale_law
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    specs = ((axes[0], "q_scale", "q1", "xi", r"Imbalance scale $\mathcal{Q}_N$"),
             (axes[1], "r_scale", "r1", "psi", r"Response scale $\mathcal{I}^\star_N$"))
    for ax, col, c1, exponent, title in specs:
        sc = free.scales.set_index("n_events")[col]
        ax.plot(sc.index, sc.to_numpy(), "o", ms=7, mfc="white", mec=INK, mew=1.8,
                label=rf"free per $N$ (17 parameters, $\eta$ = {free.gamma:.2f})")
        ax.plot(grid, law[c1] * grid ** law[exponent], color=BLUE, lw=2,
                label=rf"power law inside the fit (7 parameters, $\eta$ = {power.gamma:.2f}): "
                      rf"$N^{{{law[exponent]:.3f}}}$")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks(n); ax.set_xticklabels([f"{int(v):,}" for v in n], fontsize=8); ax.minorticks_off()
        ax.set_xlabel(r"Events per bar, $N$"); ax.set_title(title, fontsize=11)
        ax.grid(color="#e6e6e6", lw=0.7); ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "32_surface_power_law_scales.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stats, folds, rolling = pass_one()
    fits, fit_rows, scales = full_sample(stats)
    fit_rows.to_csv(OUT / "fits.csv", index=False)
    scales.to_csv(OUT / "scales.csv", index=False)
    scales_figure(fits)
    boot = bootstrap(stats, fits["surface_power"])
    boot.to_csv(OUT / "bootstrap.csv", index=False)
    oos = oos_by_year(folds)
    oos.to_csv(OUT / "oos_by_year.csv", index=False)
    rfits, rfit_rows = rolling_fits(rolling)
    rfit_rows.to_csv(OUT / "rolling_fits.csv", index=False)
    scores = rolling_scores(rfits)
    scores.to_csv(OUT / "rolling_monthly_scores.csv", index=False)

    rng = np.random.default_rng(SEED + 1)
    r2 = (scores.groupby(["window", "model", "n_events"])[["sse", "sum_y2"]].sum()
          .assign(oos_r2=lambda d: 1 - d["sse"] / d["sum_y2"])["oos_r2"])
    summary = {"design": "train 12 months (or expanding from 2021-01) before each test quarter; test 3 months; "
                         "block-level MSE; D_m = mean over N of the relative MSE reduction vs the comparison model",
               "test_quarters": [label(q) for q in QUARTERS],
               "comparisons": {name: compare(scores, c, b, rng) for name, (c, b) in COMPARISONS.items()},
               "oos_r2_vs_zero": {f"{w}|{m}|{n}": float(v) for (w, m, n), v in r2.items()},
               "coverage": rfit_rows.groupby(["window", "model"])["n_included"].agg(
                   lambda v: {k: int(c) for k, c in v.value_counts().items()}).reset_index().to_dict("records")}
    (OUT / "rolling_summary.json").write_text(json.dumps(summary, indent=2))
    pd.set_option("display.width", 200)
    print(fit_rows[["model", "parameters", "gamma", "alpha", "beta", "xi", "psi", "standardised_rmse"]].round(4).to_string(index=False))
    print("eta 95% month-block:", np.quantile(boot["gamma"], [0.025, 0.975]).round(4))
    print(oos.pivot(index="test_year", columns="model", values="test_standardised_rmse").round(3).to_string())
    for name, res in summary["comparisons"].items():
        print(name, {k: round(v, 4) for k, v in res["raw"].items() if isinstance(v, float)},
              "| clipped", {k: round(v, 4) for k, v in res["clipped"].items() if isinstance(v, float)})


if __name__ == "__main__":
    main()
