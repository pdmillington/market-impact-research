"""Identification diagnostics for the activity-surface fits (post hoc, same cells).

When almost all cells lie in the linear part of the master curve (x*/Q_N << 1),
only R_N / Q_N is identified and Q_N, R_N individually are not. This script
recomputes the full-sample cells (same cuts and response as run_activity_surface.py)
and reports, for each N and model:

- share of cells beyond the bend (x*/Q_N > 1) and the 90th percentile of x*/Q_N;
- standard-error-weighted RMSE (the comparison metric; scaled RMSE is not used);
- the identified slope R_N / Q_N;
- xi and psi restricted to N with at least 10% of cells beyond the bend.
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
from crypto_impact.pb_surface import _clipped_sigma, aggregate_cells  # noqa: E402

OUT = REPO / "crypto-impact-study" / "reports" / "redo" / "activity_surface_2021_2026"
BEND_SHARE_MIN = 0.10


def main() -> None:
    fits = pd.read_csv(OUT / "fits.csv")
    scales = pd.read_csv(OUT / "scales.csv")
    stats = []
    for n in run.N_VALUES:
        arrays = run.load_arrays(REPO / "shared-data", n, "2020-12", "2021-01", "2026-08")
        stats.append(run.stats_for(arrays, n, None, "y_vol"))
    cells = aggregate_cells(pd.concat(stats, ignore_index=True))
    sigma = _clipped_sigma(cells)
    rows, exps = [], []
    for model in ("pb_gamma_1", "imbalance_ratio_gamma_0", "activity_surface"):
        f = fits.loc[fits.model == model].iloc[0]
        sc = scales.loc[scales.model == model].set_index("n_events")
        q = cells["n_events"].map(sc["q_scale"]).to_numpy()
        r = cells["n_events"].map(sc["r_scale"]).to_numpy()
        x = np.exp(cells["mean_logz"] + f.gamma * cells["mean_loga"]).to_numpy() / q
        pred = cells["side"].to_numpy() * r * x / np.power(1 + np.abs(x) ** f.alpha, f.beta / f.alpha)
        wres = (cells["mean_y"].to_numpy() - pred) / sigma
        for n in run.N_VALUES:
            m = (cells["n_events"] == n).to_numpy()
            rows.append({"model": model, "n_events": n, "cells": int(m.sum()),
                         "share_beyond_bend": float((x[m] > 1).mean()), "x_over_q_p90": float(np.quantile(x[m], 0.9)),
                         "weighted_rmse": float(np.sqrt(np.mean(wres[m] ** 2))), "slope_r_over_q": float(sc.loc[n, "r_scale"] / sc.loc[n, "q_scale"])})
        table = pd.DataFrame([r for r in rows if r["model"] == model])
        identified = table.loc[table["share_beyond_bend"] >= BEND_SHARE_MIN, "n_events"].tolist()
        sub = sc.loc[identified].reset_index().rename(columns={"q_scale": "fitted_q_scale", "r_scale": "fitted_r_scale"})
        if len(sub) >= 3:
            exps.append({"model": model, "identified_n": identified,
                         "xi": scaling_slope(sub, "fitted_q_scale", "xi")["robust_exponent"],
                         "psi": scaling_slope(sub, "fitted_r_scale", "psi")["robust_exponent"]})
        else:
            exps.append({"model": model, "identified_n": identified, "xi": np.nan, "psi": np.nan})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "identification_by_n.csv", index=False)
    pd.DataFrame(exps).to_csv(OUT / "scaling_exponents_identified.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.pivot(index="n_events", columns="model", values="weighted_rmse").round(2).to_string())
    print(out.pivot(index="n_events", columns="model", values="share_beyond_bend").round(3).to_string())
    print(pd.DataFrame(exps).to_string(index=False))


if __name__ == "__main__":
    main()
