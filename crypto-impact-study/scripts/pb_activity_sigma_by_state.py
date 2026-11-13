"""Trailing volatility by activity state (why the section 5.2 displacement shrinks on y_t).

The section 5.2 table splits blocks into the highest event-intensity quintile (high activity)
and the other four (ordinary). On raw returns high-activity blocks lie well above the P&B
master curve; on y_t = r_t / sigma_24h,t most of that gap disappears
(pb_activity_residuals_normalised.py). This script measures sigma_24h in each activity state
on the same blocks and quintiles, so the text can say whether high-activity blocks fall in
high-volatility periods.

Per N in (250, 1000, 4000, 16000): median and mean sigma_24h in ordinary and high activity,
and the high / ordinary ratios. Output:
reports/redo/pb_activity_residuals_normalised/sigma24_by_activity.csv.
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

from pb_activity_residuals_normalised import OUT, TABLE_N, frame_for  # noqa: E402
from run_activity_surface import load_arrays  # noqa: E402


def main() -> None:
    rows = []
    for n in TABLE_N:
        f = frame_for(load_arrays(REPO / "shared-data", n, "2020-12", "2021-01", "2026-08"))
        sigma = (f["log_return"] / f["y_vol"]).to_numpy()
        quintile = pd.qcut(f["intensity"], 5, labels=False, duplicates="drop").to_numpy()
        high = quintile == quintile.max()
        row = {"n_events": n}
        for label, mask in (("ordinary", ~high), ("high", high)):
            row[f"sigma24_median_{label}"] = float(np.median(sigma[mask]))
            row[f"sigma24_mean_{label}"] = float(np.mean(sigma[mask]))
        row["median_ratio_high_over_ordinary"] = row["sigma24_median_high"] / row["sigma24_median_ordinary"]
        row["mean_ratio_high_over_ordinary"] = row["sigma24_mean_high"] / row["sigma24_mean_ordinary"]
        rows.append(row)
        print(row, flush=True)
        del f
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "sigma24_by_activity.csv", index=False)
    print(table.round(4).to_string())


if __name__ == "__main__":
    main()
