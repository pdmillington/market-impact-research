"""Validate the trade-based mid proxy against bookTicker quotes (measurement only).

Runs on a month inside the order-book exploration window (HYPOTHESIS_LEDGER.md
sample register). For every reconstructed event it compares the E-002 mid proxy
with the true quote mid, re-measures continuation, maker markout and rank IC by
sweep class with both mids, and measures how the E-003 resiliency state relates
to activity and volatility.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import polars as pl


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "alpha-search" / "scripts"))
sys.path.insert(0, str(REPO / "shared-methodology" / "src"))

from crypto_market_data.book_ticker import canonical_month  # noqa: E402
from explore_seconds_signals import (  # noqa: E402
    DAY_MS, LEVEL_BINS, LEVEL_LABELS, TAUS, forward_fill, load_events, trailing_count, trailing_mean_at,
)
from run_residual_decomposition import spearman  # noqa: E402


def resiliency_state(t, s, levels, mid, pre_mid, mid_5):
    impact = s * 1e4 * np.log(mid / pre_mid)
    reverted = s * 1e4 * np.log(mid / mid_5)
    usable = (levels >= 2) & (impact > 0) & np.isfinite(reverted)
    reversion = np.clip(reverted[usable] / impact[usable], -2.0, 2.0)
    return trailing_mean_at(t, t[usable] + 5_000, reversion, 1_800_000)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-data-root", type=Path, default=REPO / "shared-data")
    parser.add_argument("--month", default="2023-05")
    parser.add_argument("--output-dir", type=Path, default=REPO / "alpha-search" / "reports" / "mid_proxy_validation")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    quotes = (pl.scan_parquet(canonical_month(args.shared_data_root, "BTCUSDT", args.month))
              .select("transaction_time", "update_id", "best_bid_price", "best_ask_price")
              .sort("transaction_time", "update_id").collect())
    q_time = quotes["transaction_time"].to_numpy()
    q_mid = 0.5 * (quotes["best_bid_price"].to_numpy() + quotes["best_ask_price"].to_numpy())
    q_spread = 1e4 * np.log(quotes["best_ask_price"].to_numpy() / quotes["best_bid_price"].to_numpy())
    del quotes

    def true_mid(times: np.ndarray) -> np.ndarray:
        index = np.searchsorted(q_time, times, side="right") - 1
        output = np.full(len(times), np.nan)
        ok = index >= 0
        output[ok] = q_mid[index[ok]]
        return output

    events = load_events(args.shared_data_root, args.month)
    t = events["timestamp_ms"].to_numpy()
    s = events["aggressor_sign"].to_numpy().astype(np.float64)
    vwap = events["vwap"].to_numpy()
    last = events["last_price"].to_numpy()
    levels = events["price_level_count"].to_numpy()
    del events

    ask = forward_fill(last, s > 0)
    bid = forward_fill(last, s < 0)
    ask_time = forward_fill(t.astype(np.float64), s > 0)
    bid_time = forward_fill(t.astype(np.float64), s < 0)
    proxy = 0.5 * (ask + bid)
    proxy_pre = np.concatenate([[np.nan], proxy[:-1]])
    true_post = true_mid(t)
    true_pre = true_mid(t - 1)
    stale_age = t - np.fmin(ask_time, bid_time)

    # Only events at least one hour after quote coverage begins, within the month.
    start = max(int(q_time[0]) + 3_600_000, int(t[t >= q_time[0]].min()))
    chosen = (t >= start) & (t <= q_time[-1] - 60_000) & np.isfinite(true_post) & np.isfinite(proxy)

    # 1. Proxy error at the event and its dependence on staleness.
    error = 1e4 * np.log(proxy / true_post)
    age_bins = np.digitize(stale_age, [10, 100, 1_000])
    age_labels = ["<10 ms", "10-100 ms", "100-1000 ms", ">1 s"]
    error_rows = []
    for label_index, label in enumerate(age_labels):
        e = error[chosen & (age_bins == label_index)]
        error_rows.append({"stale_side_age": label, "share": float(len(e) / chosen.sum()),
                           "mean_error_bps": float(e.mean()), "median_abs_error_bps": float(np.median(np.abs(e))),
                           "p90_abs_error_bps": float(np.quantile(np.abs(e), 0.9))})
    e = error[chosen]
    error_rows.append({"stale_side_age": "all", "share": 1.0, "mean_error_bps": float(e.mean()),
                       "median_abs_error_bps": float(np.median(np.abs(e))),
                       "p90_abs_error_bps": float(np.quantile(np.abs(e), 0.9))})
    pl.DataFrame(error_rows).write_csv(out / "proxy_error.csv")

    # 2. E-002 quantities with both mids, by sweep class.
    level_class = np.digitize(levels, LEVEL_BINS)
    comparison, ic_rows, true_future = [], [], {}
    rng = np.random.default_rng(7)
    sample = np.sort(rng.choice(np.flatnonzero(chosen), size=min(2_000_000, int(chosen.sum())), replace=False))
    for tau in TAUS:
        future_index = np.searchsorted(t, t + tau * 1_000, side="right") - 1
        proxy_future = proxy[future_index]
        true_future[tau] = true_mid(t + tau * 1_000)
        measures = {
            "proxy": (s * 1e4 * np.log(proxy_future / proxy), -s * 1e4 * np.log(proxy_future / vwap)),
            "true": (s * 1e4 * np.log(true_future[tau] / true_post), -s * 1e4 * np.log(true_future[tau] / vwap)),
        }
        for level in range(len(LEVEL_LABELS)):
            mask = chosen & (level_class == level)
            row = {"tau": tau, "levels": LEVEL_LABELS[level], "events": int(mask.sum())}
            for name, (response, markout) in measures.items():
                row[f"{name}_continuation_bps"] = float(np.nanmean(response[mask]))
                row[f"{name}_markout_bps"] = float(np.nanmean(markout[mask]))
            comparison.append(row)
        for name, (response, _) in measures.items():
            x, y = (s * levels)[sample], (response * s)[sample]
            ok = np.isfinite(x) & np.isfinite(y)
            ic_rows.append({"tau": tau, "mid": name, "ic_signed_levels": spearman(x[ok], y[ok])})
        pr, tr = measures["proxy"][0][sample], measures["true"][0][sample]
        ok = np.isfinite(pr) & np.isfinite(tr)
        ic_rows.append({"tau": tau, "mid": "proxy_vs_true_response_rank_corr",
                        "ic_signed_levels": spearman(pr[ok], tr[ok])})
    pl.DataFrame(comparison).write_csv(out / "continuation_markout_comparison.csv")
    pl.DataFrame(ic_rows).write_csv(out / "ic_comparison.csv")

    # 3. Resiliency (proxy and true mid) versus activity and volatility.
    proxy_5 = proxy[np.searchsorted(t, t + 5_000, side="right") - 1]
    resiliency_proxy = resiliency_state(t, s, levels, proxy, proxy_pre, proxy_5)
    resiliency_true = resiliency_state(t, s, levels, true_post, true_pre, true_future[5])
    rate_24h = trailing_count(t, np.ones(len(t), bool), DAY_MS) / 86_400.0
    activity = trailing_count(t, np.ones(len(t), bool), 300_000) / 300.0 / np.maximum(rate_24h, 1e-9)
    minute_index = np.flatnonzero(np.diff(t // 60_000, append=t[-1] // 60_000 + 1) != 0)
    minute_r2 = np.nan_to_num(np.concatenate([[np.nan], np.diff(np.log(proxy[minute_index])) * 1e4]) ** 2)
    rv_1h = trailing_mean_at(t, t[minute_index], minute_r2, 3_600_000)
    rv_24h = trailing_mean_at(t, t[minute_index], minute_r2, DAY_MS)
    volatility = np.sqrt(rv_1h / np.where(rv_24h > 0, rv_24h, np.nan))
    pairs = {
        "resiliency_proxy vs resiliency_true": (resiliency_proxy, resiliency_true),
        "resiliency_proxy vs activity": (resiliency_proxy, activity),
        "resiliency_true vs activity": (resiliency_true, activity),
        "resiliency_proxy vs volatility": (resiliency_proxy, volatility),
        "resiliency_true vs volatility": (resiliency_true, volatility),
        "activity vs volatility": (activity, volatility),
    }
    correlation_rows = []
    for name, (a, b) in pairs.items():
        x, y = a[sample], b[sample]
        ok = np.isfinite(x) & np.isfinite(y)
        correlation_rows.append({"pair": name, "spearman": spearman(x[ok], y[ok])})
    pl.DataFrame(correlation_rows).write_csv(out / "state_correlations.csv")

    summary = {"month": args.month, "events_evaluated": int(chosen.sum()),
               "first_event_ms": int(start), "quote_updates": int(len(q_time)),
               "median_quoted_spread_bps": float(np.median(q_spread)),
               "share_one_tick_or_less": float(np.mean(q_spread <= np.median(q_spread) * 1.01))}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
