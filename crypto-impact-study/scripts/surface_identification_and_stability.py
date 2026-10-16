"""Identified-N refit and stability of gamma (revision log A3c-3, A3c-4, A9, A10).

Same data, cells and estimator as run_activity_surface.py (pb_blocks_v1, 2021-01 to
2026-08, volatility-normalised response, 20 x 10 equal-count (q, a) cells per N and
side, full-sample cuts), but the cell statistics are kept by UTC day so that
uncertainty can come from a day-block bootstrap:

1. Identified-N refit: the common surface (and the P&B and gamma = 0 benchmarks) on
   N = 250..16,000, i.e. without N = 100, where only R_N / Q_N is identified. Compared
   with the all-N fit (which this script reproduces as a check).
2. Day-block bootstrap (200 replicates) of the common gamma for both N sets.
3. Gamma by calendar year on the headline set N = 250..16,000, each with a day-block
   bootstrap within that year (reproduces activity_surface_n250_2021_2026/gamma_by_year.csv).
4. Gamma by N (separate surface per N), each with a day-block bootstrap.

Outputs: reports/redo/surface_stability/{identified_n_fits,gamma_bootstrap,
gamma_by_year,gamma_by_n}.csv
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "crypto-impact-study" / "src"), str(REPO / "crypto-impact-study" / "scripts"),
                str(REPO / "shared-methodology" / "src")]

import run_activity_surface as run  # noqa: E402
from crypto_impact.pb_collapse import scaling_slope  # noqa: E402
from crypto_impact.pb_surface import aggregate_cells, fit_activity_surface, grid_cuts, monthly_cell_stats  # noqa: E402

OUT = REPO / "crypto-impact-study" / "reports" / "redo" / "surface_stability"
N_ALL = run.N_VALUES
N_IDENTIFIED = tuple(n for n in N_ALL if n >= 250)
N_BOOT = 200
YEARS = range(2021, 2027)


def day_stats(n: int) -> pd.DataFrame:
    """run.stats_for with the UTC day in place of the month (column keeps the name 'month')."""

    arr = run.load_arrays(REPO / "shared-data", n, "2020-12", "2021-01", "2026-08")
    cuts = {s: grid_cuts(arr["z"][(arr["side"] == s) & (arr["z"] > 0)],
                         arr["a"][(arr["side"] == s) & np.isfinite(arr["a"])], run.N_Z, run.N_A) for s in (-1, 1)}
    stats = monthly_cell_stats(arr["day"], arr["side"], arr["z"], arr["a"], arr["y_vol"], cuts)
    stats.insert(0, "n_events", n)
    return stats


def bootstrap_gamma(stats: pd.DataFrame, days: np.ndarray, initial, rng, n_boot: int = N_BOOT) -> np.ndarray:
    gammas = []
    for _ in range(n_boot):
        draw = pd.Series(rng.choice(days, len(days), replace=True)).value_counts()
        fit = fit_activity_surface(aggregate_cells(stats, weights=draw), n_starts=1, initial=initial)
        gammas.append(fit.gamma)
    return np.asarray(gammas)


def identified_exponents(fit, cells) -> dict:
    """xi and psi on the N with at least 10% of cells beyond the bend (as diagnose_activity_surface.py)."""

    lookup = fit.scales.set_index("n_events")
    x = np.exp(cells["mean_logz"] + fit.gamma * cells["mean_loga"]).to_numpy() / cells["n_events"].map(lookup["q_scale"]).to_numpy()
    share = pd.Series(x > 1).groupby(cells["n_events"].to_numpy()).mean()
    identified = share.index[share >= 0.10].tolist()
    sub = lookup.loc[identified].reset_index().rename(columns={"q_scale": "fitted_q_scale", "r_scale": "fitted_r_scale"})
    if len(sub) < 3:
        return {"identified_n": identified, "xi_identified": np.nan, "psi_identified": np.nan}
    return {"identified_n": identified,
            "xi_identified": scaling_slope(sub, "fitted_q_scale", "xi")["robust_exponent"],
            "psi_identified": scaling_slope(sub, "fitted_r_scale", "psi")["robust_exponent"]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261015)
    stats = pd.concat([day_stats(n) for n in N_ALL], ignore_index=True)
    days = np.asarray(sorted(stats["month"].unique()))
    day_year = pd.Series(pd.to_datetime(days, unit="D").year, index=days)
    print(f"{len(stats):,} day-cell rows over {len(days):,} days", flush=True)

    fit_rows, boot_rows, common = [], [], {}
    for label, n_set in (("all_N", N_ALL), ("N_250_16000", N_IDENTIFIED)):
        sub = stats.loc[stats["n_events"].isin(n_set)]
        cells = aggregate_cells(sub)
        fits = run.fit_all(cells)
        common[label] = (sub, fits["activity_surface"])
        for name, fit in fits.items():
            row = run.fit_row(name, fit, cells, {"n_set": label})
            fit_rows.append({**{k: row[k] for k in ("n_set", "model", "gamma", "alpha", "beta", "weighted_rmse", "cells", "xi", "psi")},
                             **identified_exponents(fit, cells), "parameters": 2 * len(n_set) + (3 if name == "activity_surface" else 2)})
        gammas = bootstrap_gamma(sub, days, fits["activity_surface"], rng)
        boot_rows.append({"n_set": label, "gamma": fits["activity_surface"].gamma, "gamma_p025": np.quantile(gammas, 0.025),
                          "gamma_p975": np.quantile(gammas, 0.975), "replicates": len(gammas)})
        print(label, "done", fit_rows[-1]["gamma"], boot_rows[-1], flush=True)
    pd.DataFrame(fit_rows).to_csv(OUT / "identified_n_fits.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(OUT / "gamma_bootstrap.csv", index=False)

    all_stats, _ = common["all_N"]
    head_stats, head_fit = common["N_250_16000"]
    year_rows = []
    for year in YEARS:
        year_days = day_year.index[day_year == year].to_numpy()
        cells = aggregate_cells(head_stats, months=year_days)
        fit = fit_activity_surface(cells, n_starts=6, initial=head_fit)
        year_stats = head_stats.loc[head_stats["month"].isin(year_days)]
        gammas = bootstrap_gamma(year_stats, year_days, fit, rng)
        year_rows.append({"year": year, "n_set": "N_250_16000", "days": len(year_days), "gamma": fit.gamma, "gamma_p025": np.quantile(gammas, 0.025),
                          "gamma_p975": np.quantile(gammas, 0.975), "replicates": len(gammas)})
        print(year_rows[-1], flush=True)
    pd.DataFrame(year_rows).to_csv(OUT / "gamma_by_year.csv", index=False)

    n_rows = []
    for n in N_ALL:
        sub = all_stats.loc[all_stats["n_events"] == n]
        fit = fit_activity_surface(aggregate_cells(sub), n_starts=6)
        gammas = bootstrap_gamma(sub, days, fit, rng)
        n_rows.append({"n_events": n, "gamma": fit.gamma, "gamma_p025": np.quantile(gammas, 0.025),
                       "gamma_p975": np.quantile(gammas, 0.975), "alpha": fit.alpha, "beta": fit.beta, "replicates": len(gammas)})
        print(n_rows[-1], flush=True)
    pd.DataFrame(n_rows).to_csv(OUT / "gamma_by_n.csv", index=False)


if __name__ == "__main__":
    main()
