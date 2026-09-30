"""E-003: passive interest from trades (absorption, resiliency, asymmetry), BTC 2022.

Hypothesis generation only (HYPOTHESIS_LEDGER.md, E-003). Reuses the E-002
event handling: mid proxy, targets and day-level aggregation.

- A1 absorption ratio: quantity executed in the current run of same-side,
  single-level events at an unchanged price, divided by the typical depth per
  level on that side (trailing 30-minute geometric mean of quantity / levels over
  same-side sweeps).
- A2 resiliency state: trailing 30-minute mean of the fraction of each past
  sweep's mid-proxy impact that reverted within 5 s (outcome known 5 s later).
- A3 absorption asymmetry: over the trailing 5 minutes, (sell volume absorbed at
  an unchanged bid - buy volume absorbed at an unchanged ask) / total volume.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import polars as pl


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "alpha-search" / "scripts"))
sys.path.insert(0, str(REPO / "shared-methodology" / "src"))

from crypto_market_data.months import iter_months  # noqa: E402
from explore_seconds_signals import (  # noqa: E402
    DAY_MS, HIT_BINS, LEVEL_BINS, TAUS, aggregate, forward_fill, load_events,
    month_start_ms, tercile, trailing_mean_at,
)
from run_residual_decomposition import spearman  # noqa: E402


def trailing_sum_at(times: np.ndarray, sample_times: np.ndarray, sample_values: np.ndarray,
                    window_ms: int) -> np.ndarray:
    """Sum of sample values with sample time in [t - window, t)."""

    cumulative = np.concatenate([[0.0], np.cumsum(sample_values)])
    upper = np.searchsorted(sample_times, times, side="left")
    lower = np.searchsorted(sample_times, times - window_ms, side="left")
    return cumulative[upper] - cumulative[lower]


def process_month(root: Path, month: str, cuts: dict | None) -> tuple[list, dict]:
    events = load_events(root, month)
    t = events["timestamp_ms"].to_numpy()
    s = events["aggressor_sign"].to_numpy().astype(np.float64)
    qty = events["quantity"].to_numpy()
    vwap = events["vwap"].to_numpy()
    first = events["first_price"].to_numpy()
    last = events["last_price"].to_numpy()
    levels = events["price_level_count"].to_numpy()
    del events

    ask = forward_fill(last, s > 0)
    bid = forward_fill(last, s < 0)
    mid = 0.5 * (ask + bid)
    pre_mid = np.concatenate([[np.nan], mid[:-1]])
    responses, markouts, pre_responses, mid_5 = {}, {}, {}, None
    for tau in TAUS:
        future = mid[np.searchsorted(t, t + tau * 1_000, side="right") - 1]
        if tau == 5:
            mid_5 = future
        responses[tau] = s * 1e4 * np.log(future / mid)
        markouts[tau] = -s * 1e4 * np.log(future / vwap)
        pre_responses[tau] = 1e4 * np.log(future / pre_mid)

    # A1: absorption ratio for runs of same-side single-level events at an unchanged price.
    same = np.zeros(len(t), bool)
    same[1:] = (s[1:] == s[:-1]) & (levels[1:] == 1) & (levels[:-1] == 1) & (first[1:] == last[:-1])
    run_id = np.cumsum(~same)
    run_start = np.flatnonzero(~same)
    hits = np.arange(len(t)) - run_start[run_id - 1] + 1
    cumulative_qty = np.cumsum(qty)
    run_qty = cumulative_qty - np.concatenate([[0.0], cumulative_qty])[run_start[run_id - 1]]
    sweep = levels >= 2
    depth_by_side = {}
    for side in (1.0, -1.0):
        chosen = sweep & (s == side)
        depth_by_side[side] = np.exp(trailing_mean_at(t, t[chosen], np.log(qty[chosen] / levels[chosen]),
                                                      1_800_000))
    typical_depth = np.where(s > 0, depth_by_side[1.0], depth_by_side[-1.0])
    absorption = run_qty / typical_depth

    # A2: resiliency state from past sweeps' 5-second reversion fraction.
    impact = s * 1e4 * np.log(mid / pre_mid)
    reverted = s * 1e4 * np.log(mid / mid_5)
    usable = sweep & (impact > 0) & np.isfinite(reverted)
    reversion = np.clip(reverted[usable] / impact[usable], -2.0, 2.0)
    resiliency = trailing_mean_at(t, t[usable] + 5_000, reversion, 1_800_000)

    # A3: absorption asymmetry over the trailing 5 minutes.
    prior_ask = np.concatenate([[np.nan], ask[:-1]])
    prior_bid = np.concatenate([[np.nan], bid[:-1]])
    absorbed_buy = (s > 0) & (levels == 1) & (first == prior_ask)
    absorbed_sell = (s < 0) & (levels == 1) & (first == prior_bid)
    window = 300_000
    total = trailing_sum_at(t, t, qty, window)
    net_passive = (trailing_sum_at(t, t[absorbed_sell], qty[absorbed_sell], window)
                   - trailing_sum_at(t, t[absorbed_buy], qty[absorbed_buy], window))
    asymmetry = net_passive / np.where(total > 0, total, np.nan)
    aggressive = (trailing_sum_at(t, t[s > 0], qty[s > 0], window)
                  - trailing_sum_at(t, t[s < 0], qty[s < 0], window)) / np.where(total > 0, total, np.nan)

    in_month = t >= month_start_ms(month)
    if cuts is None:
        # Frozen from January 2022, the first explored month.
        cuts = {
            "absorption": np.nanquantile(absorption[in_month & (levels == 1)], [1 / 3, 2 / 3]),
            "resiliency": np.nanquantile(resiliency[in_month], [1 / 3, 2 / 3]),
            "asymmetry": np.nanquantile(asymmetry[in_month], np.linspace(0, 1, 11)[1:-1]),
        }
    absorption_state = tercile(absorption, cuts["absorption"])
    resiliency_state = tercile(resiliency, cuts["resiliency"])
    asymmetry_decile = np.digitize(asymmetry, cuts["asymmetry"]).astype(np.int8) + 1
    asymmetry_decile[~np.isfinite(asymmetry)] = 0
    day = (t // DAY_MS).astype(np.int64)
    hit_class = np.digitize(hits, HIT_BINS).astype(np.int8)
    level_class = np.digitize(levels, LEVEL_BINS).astype(np.int8)

    rows: list = []
    for tau in TAUS:
        tau_key = np.full(len(t), tau, dtype=np.int16)
        values = {"response": responses[tau], "markout": markouts[tau]}
        aggregate(rows, "A1_hits_absorption", day, {"tau": tau_key, "hit_class": hit_class,
                                                     "state": absorption_state}, values,
                  in_month & (levels == 1))
        aggregate(rows, "A2_sweeps_resiliency", day, {"tau": tau_key, "level_class": level_class,
                                                       "state": resiliency_state}, values,
                  in_month & sweep)
        aggregate(rows, "A3_asymmetry", day, {"tau": tau_key, "decile": asymmetry_decile},
                  {"future_move": pre_responses[tau]}, in_month & (asymmetry_decile > 0))

    rng = np.random.default_rng(int(month.replace("-", "")) + 3)
    candidates = np.flatnonzero(in_month)
    sample = np.sort(rng.choice(candidates, size=min(2_000_000, len(candidates)), replace=False))
    ic_rows = []
    for tau in TAUS:
        signals = {
            # Continuation after a single-level hit, against its absorption ratio (expected negative).
            "A1_absorption_vs_continuation": (absorption, responses[tau], levels == 1),
            # Sweep continuation against the resiliency state (expected negative).
            "A2_resiliency_vs_sweep_continuation": (resiliency, responses[tau], sweep),
            # Future move against passive-buying asymmetry (expected positive).
            "A3_asymmetry_vs_future_move": (asymmetry, pre_responses[tau], np.ones(len(t), bool)),
            # Benchmark: aggressive flow imbalance over the same 5 minutes.
            "A3_benchmark_aggressive_flow": (aggressive, pre_responses[tau], np.ones(len(t), bool)),
        }
        for name, (x, y, mask) in signals.items():
            chosen = sample[mask[sample]]
            xs, ys = x[chosen], y[chosen]
            ok = np.isfinite(xs) & np.isfinite(ys)
            ic_rows.append({"month": month, "tau": tau, "signal": name, "spearman_ic": spearman(xs[ok], ys[ok])})
    correlation = spearman(*(lambda ok: (asymmetry[sample][ok], aggressive[sample][ok]))(
        np.isfinite(asymmetry[sample]) & np.isfinite(aggressive[sample])))
    ic_rows.append({"month": month, "tau": 0, "signal": "corr_asymmetry_aggressive_flow",
                    "spearman_ic": correlation})
    rows.append(pl.DataFrame(ic_rows).with_columns(pl.lit("IC").alias("table")))
    return rows, cuts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-data-root", type=Path, default=REPO / "shared-data")
    parser.add_argument("--start-month", default="2022-01")
    parser.add_argument("--end-month", default="2022-12")
    parser.add_argument("--output-dir", type=Path, default=REPO / "alpha-search" / "reports" / "absorption_e003")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cuts, tables = None, {}
    for month in iter_months(args.start_month, args.end_month):
        rows, cuts = process_month(args.shared_data_root, month, cuts)
        for frame in rows:
            tables.setdefault(frame["table"][0], []).append(frame.drop("table"))
        print(f"{month} done", flush=True)
    for name, frames in tables.items():
        pl.concat(frames, how="diagonal").write_parquet(args.output_dir / f"{name}.parquet")
    (args.output_dir / "state_cuts.txt").write_text(
        "\n".join(f"{k}: {np.round(v, 4).tolist()}" for k, v in cuts.items()) + "\n")
    print(args.output_dir)


if __name__ == "__main__":
    main()
