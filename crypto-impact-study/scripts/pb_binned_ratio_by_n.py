"""Binned ratio of the y_t and raw P&B response curves, by N (check on psi_y vs psi_raw).

The section 4.2 method bins the blocks at each N into 60 equal-count bins per side of
x = Q / V_24h. The bins depend only on x, and the y_t rerun uses the same blocks for both
responses, so bin b at N holds the same blocks for the raw log return and for
y_t = r_t / sigma_24h,t. Per bin the ratio

    rho_{N,b} = mean(y_t in b) / mean(r_t in b)

measures how much the normaliser rescales the measured response, with no curve fitted.
If the fitted height ratio R_N(y) / R_N(raw) ~ N^0.131 (sigma24_by_n.csv) came from the
data rather than the joint fit, the median of rho over bins should rise with N at the same
rate.

Per N the ratio is summarised by its median over bins: all bins, inner bins (the half of
each side's bins with the smallest |x|, bin index < 30) and outer bins (the larger-|x| half).
The inner/outer split is by bin rank, so it does not depend on any fitted scale.
Slopes of log median rho on log N use Theil-Sen (95%) and OLS, as in pb_collapse.scaling_slope.

Input: reports/redo/pb_activity_residuals_normalised/binned_curves.csv (written by
pb_activity_residuals_normalised.py). Output in the same folder: binned_ratio.csv (per bin),
binned_ratio_by_n.csv and binned_ratio_slopes.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress, theilslopes

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "crypto-impact-study" / "reports" / "redo" / "pb_activity_residuals_normalised"
INNER_BINS = 30  # bins 0..29 of the 60 per side


def main() -> None:
    curves = pd.read_csv(OUT / "binned_curves.csv")
    keys = ["n_events", "side", "bin"]
    raw = curves.loc[curves["response"] == "log_return"].set_index(keys)
    y = curves.loc[curves["response"] == "y_vol"].set_index(keys)
    # Same blocks in each bin: identical bin sets, counts and mean x for both responses.
    assert raw.index.sort_values().equals(y.index.sort_values())
    y = y.reindex(raw.index)
    assert (raw["observations"] == y["observations"]).all()
    assert np.allclose(raw["mean_q"], y["mean_q"], rtol=0, atol=0)

    bins = raw[["mean_q", "observations"]].assign(
        mean_raw=raw["mean_response"], mean_y=y["mean_response"],
        rho=y["mean_response"] / raw["mean_response"]).reset_index()
    bins["region"] = np.where(bins["bin"] < INNER_BINS, "inner", "outer")
    bins.to_csv(OUT / "binned_ratio.csv", index=False)

    rows = []
    for n, group in bins.groupby("n_events"):
        row = {"n_events": n, "bins": len(group), "negative_rho_bins": int((group["rho"] <= 0).sum())}
        for region, part in (("all", group), ("inner", group.loc[group["region"] == "inner"]),
                             ("outer", group.loc[group["region"] == "outer"])):
            row[f"rho_median_{region}"] = float(part["rho"].median())
        rows.append(row)
    by_n = pd.DataFrame(rows)
    sigma = pd.read_csv(OUT / "sigma24_by_n.csv").set_index("n_events")
    by_n["height_ratio_y_over_raw"] = sigma["height_ratio_y_over_raw"].reindex(by_n["n_events"]).to_numpy()
    by_n["inverse_mean_sigma24"] = 1 / sigma["sigma24_inverse_mean"].reindex(by_n["n_events"]).to_numpy()
    by_n.to_csv(OUT / "binned_ratio_by_n.csv", index=False)

    log_n = np.log(by_n["n_events"].to_numpy(float))
    slopes = []
    for column in ("rho_median_all", "rho_median_inner", "rho_median_outer", "inverse_mean_sigma24",
                   "height_ratio_y_over_raw"):
        values = by_n[column].to_numpy(float)
        robust, ols = theilslopes(np.log(values), log_n, 0.95), linregress(log_n, np.log(values))
        slopes.append({"quantity": column, "theil_sen": robust.slope, "lower_95": robust.low_slope,
                       "upper_95": robust.high_slope, "ols": ols.slope, "ols_r_squared": ols.rvalue ** 2})
    slopes = pd.DataFrame(slopes)
    slopes.to_csv(OUT / "binned_ratio_slopes.csv", index=False)
    with pd.option_context("display.width", 220):
        print(by_n.round(3).to_string())
        print(slopes.round(4).to_string())


if __name__ == "__main__":
    main()
