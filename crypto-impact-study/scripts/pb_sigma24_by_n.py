"""Does the trailing 24 h volatility sigma_24h fall with N? (check on the y_t P&B exponents)

On y_t = r_t / sigma_24h,t the P&B height exponent is psi = 0.710, against 0.575 on raw
returns (pb_activity_residuals_normalised.py). If psi_y - psi_raw ~ 0.135 comes from the
normaliser, the typical sigma_24h measured from N-event blocks should scale as about
N^-0.135. This script measures sigma_24h (the same definition as the surface and the y_t
rerun: square root of the sum of squared block log returns over the preceding 24 h,
current block excluded) by N and compares its slope on log N with:
- the slope of the fitted height ratio R_N(y) / R_N(raw) (from master_fits.csv);
- the Hurst prediction: daily realised variance from N-blocks ~ N^(2 H_r - 1).

Per N: median, mean and harmonic-type mean (1 / mean(1/sigma)) of sigma_24h over blocks,
and the mean daily realised variance (sum of squared block returns per UTC day, days
with complete coverage only, the "signature" of realised variance across N).

Output: reports/redo/pb_activity_residuals_normalised/sigma24_by_n.csv and
sigma24_slopes.csv.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import linregress, theilslopes

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "crypto-impact-study" / "src"))
sys.path.insert(0, str(REPO / "crypto-impact-study" / "scripts"))
sys.path.insert(0, str(REPO / "shared-methodology" / "src"))

from build_pb_blocks import N_VALUES  # noqa: E402
from run_activity_surface import load_arrays  # noqa: E402

OUT = REPO / "crypto-impact-study" / "reports" / "redo" / "pb_activity_residuals_normalised"


def main() -> None:
    rows = []
    for n in N_VALUES:
        d = load_arrays(REPO / "shared-data", n, "2020-12", "2021-01", "2026-08")
        ok = np.isfinite(d["y"]) & np.isfinite(d["y_vol"]) & (d["y_vol"] != 0)
        sigma = d["y"][ok] / d["y_vol"][ok]
        # Daily realised variance from this N's blocks, by block start day. Gap-spanning blocks
        # are already excluded, so days touching a gap are slightly understated at every N alike.
        daily_rv = pd.Series(d["y"] ** 2).groupby(d["day"]).sum()
        rows.append({"n_events": n, "blocks": int(ok.sum()),
                     "sigma24_median": float(np.median(sigma)), "sigma24_mean": float(np.mean(sigma)),
                     "sigma24_inverse_mean": float(1 / np.mean(1 / sigma)),
                     "daily_rv_mean": float(daily_rv.mean()), "daily_rv_median": float(daily_rv.median()),
                     "days": int(len(daily_rv))})
        print(rows[-1], flush=True)
        del d
    table = pd.DataFrame(rows)
    master = pd.read_csv(OUT / "master_fits.csv").pivot(index="n_events", columns="response", values="r_scale")
    table["height_ratio_y_over_raw"] = (master["y_vol"] / master["log_return"]).reindex(table["n_events"]).to_numpy()
    table.to_csv(OUT / "sigma24_by_n.csv", index=False)

    log_n = np.log(table["n_events"].to_numpy(float))
    slopes = []
    for column in ("sigma24_median", "sigma24_mean", "sigma24_inverse_mean", "daily_rv_mean",
                   "daily_rv_median", "height_ratio_y_over_raw"):
        y = np.log(table[column].to_numpy(float))
        robust, ols = theilslopes(y, log_n, 0.95), linregress(log_n, y)
        slopes.append({"quantity": column, "theil_sen": robust.slope, "lower_95": robust.low_slope,
                       "upper_95": robust.high_slope, "ols": ols.slope, "ols_r_squared": ols.rvalue ** 2})
    slopes = pd.DataFrame(slopes)
    slopes.to_csv(OUT / "sigma24_slopes.csv", index=False)
    with pd.option_context("display.width", 200):
        print(table.round(6).to_string())
        print(slopes.round(4).to_string())
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
