"""Why high-nu blocks sit below the raw P&B curve: a split by the preceding day's volume.

Follow-up to pb_activity_residuals_normalised.py. On raw returns, blocks in the top
quintile of nu = V/V_24h sit well below the P&B master curve (scaled residual about
-0.42 to -0.48), the opposite of the event-intensity result. The proposed reading:
high nu often comes from a small V_24h (a quiet preceding day), which inflates P&B's
coordinate |Q|/V_24h and so its prediction, while raw returns stay small because
volatility is low. This script tests that reading.

Relative V_24h = V_24h at the block / mean of the daily-average V_24h over the 30
preceding UTC days (at least 20 days required; this drops the first weeks of 2021).
Relative sigma_24h is defined the same way. Using a 30-day baseline keeps the split
from simply separating calendar periods with different BTC volumes.

Cells: nu quintile x relative-V_24h tercile (both over the whole sample per N). For
each cell: share of blocks, median relative V_24h and sigma_24h, mean P&B scaled
residual on the raw response and on y_t (notebook-method master curves saved by
pb_activity_residuals_normalised.py), and the raw residual's two parts (mean signed
return / R_N and mean prediction / R_N).

Output: reports/redo/pb_activity_residuals_normalised/nu_v24_split.csv
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "crypto-impact-study" / "src"))
sys.path.insert(0, str(REPO / "crypto-impact-study" / "scripts"))
sys.path.insert(0, str(REPO / "shared-methodology" / "src"))

from crypto_impact.pb_collapse import pb_shape  # noqa: E402
from run_activity_surface import load_arrays  # noqa: E402

TABLE_N = (250, 1000, 4000, 16000)
OUT = REPO / "crypto-impact-study" / "reports" / "redo" / "pb_activity_residuals_normalised"


def relative_to_baseline(values: np.ndarray, day: np.ndarray, days: int = 30, min_days: int = 20) -> np.ndarray:
    """values / mean of daily averages over the `days` preceding UTC days."""

    daily = pd.Series(values).groupby(day).mean()
    daily = daily.reindex(np.arange(daily.index.min(), daily.index.max() + 1))
    baseline = daily.rolling(days, min_periods=min_days).mean().shift(1)
    return values / baseline.reindex(day).to_numpy()


def main() -> None:
    master = pd.read_csv(OUT / "master_fits.csv")
    rows = []
    for n in TABLE_N:
        d = load_arrays(REPO / "shared-data", n, "2020-12", "2021-01", "2026-08")
        v24 = d["gross"] / d["a"]
        sigma = d["y"] / d["y_vol"]
        rel_v24 = relative_to_baseline(v24, d["day"])
        rel_sigma = relative_to_baseline(sigma, d["day"])
        ok = (d["side"] != 0) & np.isfinite(d["pb_x"]) & np.isfinite(d["a"]) & np.isfinite(d["y_vol"]) \
            & np.isfinite(rel_v24) & np.isfinite(rel_sigma)
        f = pd.DataFrame({"abs_x": d["pb_x"][ok], "side": d["side"][ok], "nu": d["a"][ok],
                          "log_return": d["y"][ok], "y_vol": d["y_vol"][ok],
                          "rel_v24": rel_v24[ok], "rel_sigma": rel_sigma[ok]})
        del d
        f["nu_quintile"] = pd.qcut(f["nu"], 5, labels=False) + 1
        f["v24_tercile"] = pd.qcut(f["rel_v24"], 3, labels=["low (quiet day)", "middle", "high"])
        for response in ("log_return", "y_vol"):
            m = master.loc[(master["response"] == response) & (master["n_events"] == n)].iloc[0]
            pred = m["r_scale"] * pb_shape(f["abs_x"] / m["q_scale"], m["alpha"], m["beta"])
            f[f"{response}_signed_scaled"] = f["side"] * f[response] / m["r_scale"]
            f[f"{response}_pred_scaled"] = pred / m["r_scale"]
            f[f"{response}_resid"] = f[f"{response}_signed_scaled"] - f[f"{response}_pred_scaled"]
        for keys, g in f.groupby(["nu_quintile", "v24_tercile"], observed=True):
            rows.append({"n_events": n, "nu_quintile": keys[0], "v24_tercile": keys[1], "blocks": len(g),
                         "share_of_nu_quintile": len(g) / (f["nu_quintile"] == keys[0]).sum(),
                         "median_rel_v24": g["rel_v24"].median(), "median_rel_sigma": g["rel_sigma"].median(),
                         "raw_resid": g["log_return_resid"].mean(),
                         "raw_signed_return": g["log_return_signed_scaled"].mean(),
                         "raw_prediction": g["log_return_pred_scaled"].mean(),
                         "y_resid": g["y_vol_resid"].mean()})
        # Quintile totals, so the cells can be read against the published contrast.
        for q, g in f.groupby("nu_quintile"):
            rows.append({"n_events": n, "nu_quintile": q, "v24_tercile": "all", "blocks": len(g),
                         "share_of_nu_quintile": 1.0, "median_rel_v24": g["rel_v24"].median(),
                         "median_rel_sigma": g["rel_sigma"].median(), "raw_resid": g["log_return_resid"].mean(),
                         "raw_signed_return": g["log_return_signed_scaled"].mean(),
                         "raw_prediction": g["log_return_pred_scaled"].mean(), "y_resid": g["y_vol_resid"].mean()})
        print(f"N={n}: {len(f):,} blocks", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "nu_v24_split.csv", index=False)
    with pd.option_context("display.width", 220):
        print(table.round(3).to_string())
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
