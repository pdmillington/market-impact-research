"""Does the event-price rule or the bar-return convention matter? (revision log E3)

Each `timestamp_direction_v1` event can be priced at its last fill (current rule),
its first fill (the paper's original inputs) or its volume-weighted price. Bar
returns can be measured within the bar (first event to last event, current rule)
or continuously (previous bar's end price to this bar's end price, so no event's
move is lost between bars). Four combinations are compared:

    within_last      ln(P_last[end] / P_last[start])          current rule
    continuous_last  ln(P_last[end] / P_last[end of previous bar])
    continuous_vwap  ln(P_vwap[end] / P_vwap[end of previous bar])
    within_first     ln(P_first[end] / P_first[start])        the paper's original inputs

For each rule, on 2021-01..2026-08 (2020-12 as the 24 h warm-up) and all eight N:
- correlation of bar returns with the current rule at N = 100, 1,000, 4,000;
- the activity surface (volatility-normalised response; P&B gamma = 1, gamma = 0,
  free gamma): gamma, alpha, beta, weighted RMSE;
- the paper's P&B collapse on raw returns: alpha, beta, xi, psi.

Blocks come from the shared builder (crypto_impact.bars) and match pb_blocks_v1.

Output: reports/redo/event_price_rule/{return_correlations,surface_fits,pb_collapse}.csv
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "crypto-impact-study" / "src"), str(REPO / "crypto-impact-study" / "scripts"),
                str(REPO / "shared-methodology" / "src")]

import run_activity_surface as surface  # noqa: E402
from build_pb_blocks import N_VALUES  # noqa: E402
from crypto_impact.bars import iter_event_bars, lake_event_sources  # noqa: E402
from crypto_impact.pb_collapse import (  # noqa: E402
    add_trailing_volume_normalisation, fit_pb_master_curve, independently_binned_signed_response, scaling_slope,
)
from crypto_impact.pb_surface import aggregate_cells  # noqa: E402
from crypto_market_data.months import iter_months  # noqa: E402

OUT = REPO / "crypto-impact-study" / "reports" / "redo" / "event_price_rule"
RULES = ("within_last", "continuous_last", "continuous_vwap", "within_first")
CORRELATION_N = (100, 1000, 4000)
WARM_START, START, END = "2020-12", "2021-01", "2026-08"


def build_blocks() -> dict[int, pd.DataFrame]:
    parts = {n: [] for n in N_VALUES}
    sources = lake_event_sources(REPO / "shared-data", "BTCUSDT", iter_months(WARM_START, END), extra_prices=True)
    for month, completed in iter_event_bars(sources, N_VALUES, drop_data_gaps=False):
        for n, bars in completed.items():
            parts[n].append(pd.DataFrame({
                "start_time_ms": bars.index.as_unit("ms").asi8,
                "end_time_ms": bars["end_time"].dt.as_unit("ms").astype("int64").to_numpy(),
                "duration_seconds": bars["duration_seconds"].to_numpy(),
                "gross_volume": bars["gross_volume"].to_numpy(),
                "signed_volume": bars["signed_imbalance"].to_numpy(),
                "sign_sum": bars["sign_sum"].to_numpy(),
                "spans_data_gap": bars["spans_data_gap"].to_numpy(),
                "start_last": bars["start_price"].to_numpy(), "end_last": bars["end_price"].to_numpy(),
                "start_first": bars["start_price_first"].to_numpy(), "end_first": bars["end_price_first"].to_numpy(),
                "end_vwap": bars["end_price_vwap"].to_numpy(),
            }))
        print(month, flush=True)
    blocks = {}
    for n in N_VALUES:
        b = pd.concat(parts[n], ignore_index=True).sort_values("end_time_ms", kind="stable").reset_index(drop=True)
        b["within_last"] = np.log(b["end_last"] / b["start_last"])
        b["continuous_last"] = np.log(b["end_last"] / b["end_last"].shift(1))
        b["continuous_vwap"] = np.log(b["end_vwap"] / b["end_vwap"].shift(1))
        b["within_first"] = np.log(b["end_first"] / b["start_first"])
        # A continuous return across a data gap or the very first bar is undefined.
        undefined = b["spans_data_gap"].shift(1, fill_value=True)
        b.loc[undefined, ["continuous_last", "continuous_vwap"]] = np.nan
        blocks[n] = b
    return blocks


def in_sample(b: pd.DataFrame) -> pd.Series:
    return (b["start_time_ms"] >= pd.Timestamp(START + "-01", tz="UTC").value // 10**6) & ~b["spans_data_gap"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    blocks = build_blocks()

    correlations = []
    for n in CORRELATION_N:
        b = blocks[n].loc[in_sample(blocks[n]), list(RULES)].dropna()
        for rule in RULES:
            correlations.append({"n_events": n, "rule": rule, "bars": len(b),
                                 "corr_with_within_last": b[rule].corr(b["within_last"]),
                                 "std_bps": 1e4 * b[rule].std(),
                                 "std_ratio_to_within_last": b[rule].std() / b["within_last"].std()})
    pd.DataFrame(correlations).to_csv(OUT / "return_correlations.csv", index=False)
    print(pd.DataFrame(correlations).round(4).to_string(index=False), flush=True)

    surface_rows, pb_rows = [], []
    for rule in RULES:
        # Activity surface, volatility-normalised response (as run_activity_surface.py).
        stats = []
        for n in N_VALUES:
            b = blocks[n].assign(log_return=blocks[n][rule]).dropna(subset=["log_return"])
            stats.append(surface.stats_for(surface.arrays_from_blocks(b, n, START), n, None, "y_vol"))
        cells = aggregate_cells(pd.concat(stats, ignore_index=True))
        for name, fit in surface.fit_all(cells).items():
            row = surface.fit_row(name, fit, cells, {"rule": rule})
            surface_rows.append({k: row[k] for k in ("rule", "model", "gamma", "alpha", "beta", "weighted_rmse", "cells")})

        # The paper's P&B collapse on raw returns (as run_pb_collapse.py).
        curves = []
        for n in N_VALUES:
            b = blocks[n].assign(log_return=blocks[n][rule])
            b = add_trailing_volume_normalisation(b.loc[b["start_time_ms"] >= pd.Timestamp(START + "-01", tz="UTC").value // 10**6])
            b = b.loc[~b["spans_data_gap"]].dropna(subset=["log_return"])
            curve = independently_binned_signed_response(b, "pb_signed_volume", n_bins=60)
            curve.insert(0, "n_events", n)
            curves.append(curve)
        fit = fit_pb_master_curve(pd.concat(curves, ignore_index=True), N_VALUES, n_starts=12)
        pb_rows.append({"rule": rule, "alpha": fit["alpha"], "beta": fit["beta"],
                        "xi": scaling_slope(fit["scales"], "fitted_q_scale", "xi")["robust_exponent"],
                        "psi": scaling_slope(fit["scales"], "fitted_r_scale", "psi")["robust_exponent"]})
        print(rule, surface_rows[-1], pb_rows[-1], flush=True)

    pd.DataFrame(surface_rows).to_csv(OUT / "surface_fits.csv", index=False)
    pd.DataFrame(pb_rows).to_csv(OUT / "pb_collapse.csv", index=False)
    print(pd.DataFrame(surface_rows).round(4).to_string(index=False))
    print(pd.DataFrame(pb_rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
