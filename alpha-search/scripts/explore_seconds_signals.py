"""E-002: seconds-scale signal exploration on BTCUSDT events, 2022 only.

Hypothesis generation only (see HYPOTHESIS_LEDGER.md, E-002). For every
reconstructed event the script computes a mid proxy (mean of the latest buyer-
and seller-initiated trade prices), forward mid changes over tau seconds, the
markout earned by the liquidity providers the event hit, and features for the
sweep, burst, level-depletion and spread families together with three market
states. Results are stored as day-level sums per bin so standard errors can be
clustered by day.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import polars as pl


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "shared-methodology" / "src"))
sys.path.insert(0, str(REPO / "alpha-search" / "scripts"))

from crypto_market_data.layout import DataLayout  # noqa: E402
from crypto_market_data.months import iter_months  # noqa: E402
from run_residual_decomposition import spearman  # noqa: E402


TAUS = (1, 5, 15, 60)
FRESH_MS = 1_000
DAY_MS = 86_400_000
LEVEL_BINS = np.array([2, 3, 5, 10])            # classes: 1, 2, 3-4, 5-9, 10+
LEVEL_LABELS = ["1", "2", "3-4", "5-9", "10+"]
NOTIONAL_BINS = np.array([1e3, 1e4, 1e5, 1e6])  # USD: <1k, 1k-10k, 10k-100k, 100k-1m, >1m
NOTIONAL_LABELS = ["<1k", "1k-10k", "10k-100k", "100k-1m", ">1m"]
HIT_BINS = np.array([2, 3, 6, 11])               # run length: 1, 2, 3-5, 6-10, 11+
HIT_LABELS = ["1", "2", "3-5", "6-10", "11+"]


def month_start_ms(month: str) -> int:
    return int(datetime.strptime(f"{month}-01", "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1_000)


def previous_month(month: str) -> str:
    year, number = map(int, month.split("-"))
    return f"{year - 1}-12" if number == 1 else f"{year}-{number - 1:02d}"


def load_events(root: Path, month: str, symbol: str = "BTCUSDT") -> pl.DataFrame:
    layout = DataLayout(root)
    start = month_start_ms(month)
    columns = ["timestamp_ms", "aggressor_sign", "quantity", "vwap", "first_price", "last_price",
               "price_level_count", "first_trade_id"]
    previous = (pl.scan_parquet(layout.event_month(symbol, previous_month(month)))
                .select(columns).filter(pl.col("timestamp_ms") >= start - DAY_MS))
    current = pl.scan_parquet(layout.event_month(symbol, month)).select(columns)
    return pl.concat([previous, current]).sort("timestamp_ms", "first_trade_id").collect()


def forward_fill(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    index = np.where(valid, np.arange(len(values)), 0)
    np.maximum.accumulate(index, out=index)
    output = values[index]
    output[: np.argmax(valid)] = np.nan
    return output


def trailing_count(times: np.ndarray, mask: np.ndarray, window_ms: int) -> np.ndarray:
    """Number of masked events in [t - window, t), excluding the current event."""

    cumulative = np.concatenate([[0], np.cumsum(mask)])
    upper = np.arange(len(times))
    lower = np.searchsorted(times, times - window_ms, side="left")
    return cumulative[upper] - cumulative[lower]


def trailing_mean_at(times: np.ndarray, sample_times: np.ndarray, sample_values: np.ndarray,
                     window_ms: int) -> np.ndarray:
    """Mean of sample values with sample time in [t - window, t)."""

    cumulative = np.concatenate([[0.0], np.cumsum(sample_values)])
    upper = np.searchsorted(sample_times, times, side="left")
    lower = np.searchsorted(sample_times, times - window_ms, side="left")
    count = upper - lower
    return np.where(count > 0, (cumulative[upper] - cumulative[lower]) / np.maximum(count, 1), np.nan)


def tercile(values: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    labels = np.digitize(values, cuts).astype(np.int8) + 1
    labels[~np.isfinite(values)] = 0
    return labels


def aggregate(frame_rows: list, table: str, day: np.ndarray, keys: dict[str, np.ndarray],
              values: dict[str, np.ndarray], mask: np.ndarray) -> None:
    """Append day-level sums (and counts) of each value by key combination."""

    selected = mask.copy()
    for v in values.values():
        selected &= np.isfinite(v)
    if not selected.any():
        return
    columns = {"day": day[selected], **{k: v[selected] for k, v in keys.items()}}
    frame = pl.DataFrame(columns)
    for name, v in values.items():
        frame = frame.with_columns(pl.Series(name, v[selected]))
    grouped = frame.group_by(["day", *keys]).agg(
        pl.len().alias("n"), *[pl.col(name).sum().alias(f"sum_{name}") for name in values]
    )
    frame_rows.append(grouped.with_columns(pl.lit(table).alias("table")))


def process_month(root: Path, month: str, cuts: dict[str, np.ndarray] | None) -> tuple[list, dict]:
    events = load_events(root, month)
    t = events["timestamp_ms"].to_numpy()
    s = events["aggressor_sign"].to_numpy().astype(np.float64)
    qty = events["quantity"].to_numpy()
    vwap = events["vwap"].to_numpy()
    first = events["first_price"].to_numpy()
    last = events["last_price"].to_numpy()
    levels = events["price_level_count"].to_numpy()
    del events

    # Mid proxy after each event: mean of the latest buyer- and seller-initiated prices.
    ask = forward_fill(last, s > 0)
    bid = forward_fill(last, s < 0)
    ask_time = forward_fill(t.astype(np.float64), s > 0)
    bid_time = forward_fill(t.astype(np.float64), s < 0)
    mid = 0.5 * (ask + bid)
    pre_mid = np.concatenate([[np.nan], mid[:-1]])
    notional = qty * vwap
    sweep_bps = s * 1e4 * np.log(last / first)
    # A side of the proxy is stale when its last trade is older than FRESH_MS; a stale
    # side can make later proxy moves look like continuation. Fresh = both sides recent.
    fresh_now = (t - np.maximum(0, np.fmin(ask_time, bid_time)) <= FRESH_MS) & np.isfinite(ask_time + bid_time)

    responses, markouts, pre_responses, fresh = {}, {}, {}, {}
    for tau in TAUS:
        future_index = np.searchsorted(t, t + tau * 1_000, side="right") - 1
        future = mid[future_index]
        future_time = t + tau * 1_000
        fresh[tau] = fresh_now & (future_time - np.fmin(ask_time, bid_time)[future_index] <= FRESH_MS)
        responses[tau] = s * 1e4 * np.log(future / mid)          # after the event, aligned with aggressor
        markouts[tau] = -s * 1e4 * np.log(future / vwap)         # profit to the makers who were hit
        pre_responses[tau] = 1e4 * np.log(future / pre_mid)      # from just before the event (unsigned)

    # Market states.
    rate_24h = trailing_count(t, np.ones(len(t), bool), DAY_MS) / 86_400.0
    activity = trailing_count(t, np.ones(len(t), bool), 300_000) / 300.0 / np.maximum(rate_24h, 1e-9)
    minute_index = np.flatnonzero(np.diff(t // 60_000, append=t[-1] // 60_000 + 1) != 0)
    minute_t = t[minute_index]
    minute_r2 = np.concatenate([[np.nan], np.diff(np.log(mid[minute_index])) * 1e4]) ** 2
    minute_r2 = np.nan_to_num(minute_r2)
    rv_1h = trailing_mean_at(t, minute_t, minute_r2, 3_600_000)
    rv_24h = trailing_mean_at(t, minute_t, minute_r2, DAY_MS)
    volatility = np.sqrt(rv_1h / np.where(rv_24h > 0, rv_24h, np.nan))
    sweep_mask = (levels >= 2) & (sweep_bps > 0)
    log_depth = np.log(notional[sweep_mask] / sweep_bps[sweep_mask])
    depth_30m = trailing_mean_at(t, t[sweep_mask], log_depth, 1_800_000)
    depth_24h = trailing_mean_at(t, t[sweep_mask], log_depth, DAY_MS)
    liquidity = depth_30m - depth_24h                            # >0: thicker book than usual

    # Burst intensity over the preceding 5 s (current event excluded).
    buys = trailing_count(t, s > 0, 5_000)
    sells = trailing_count(t, s < 0, 5_000)
    burst = (buys - sells) / np.maximum(rate_24h * 5.0, 1e-9)

    # Level depletion: run of consecutive same-side single-level events at an unchanged price.
    same = np.zeros(len(t), bool)
    same[1:] = (s[1:] == s[:-1]) & (levels[1:] == 1) & (levels[:-1] == 1) & (first[1:] == last[:-1])
    run_id = np.cumsum(~same)
    run_start = np.flatnonzero(~same)
    hits = np.arange(len(t)) - run_start[run_id - 1] + 1

    # Flip-gap spread proxy over the preceding 30 s relative to 24 h.
    flip = np.zeros(len(t), bool)
    flip[1:] = (s[1:] != s[:-1]) & (t[1:] - t[:-1] <= 250)
    gap = np.zeros(len(t))
    gap[1:] = np.minimum(np.abs(1e4 * np.log(first[1:] / last[:-1])), 100.0)
    spread_30s = trailing_mean_at(t, t[flip], gap[flip], 30_000)
    spread_24h = trailing_mean_at(t, t[flip], gap[flip], DAY_MS)
    spread_state = spread_30s / np.where(spread_24h > 0, spread_24h, np.nan)

    in_month = t >= month_start_ms(month)
    state_values = {"activity": activity, "liquidity": liquidity, "volatility": volatility,
                    "spread": spread_state}
    if cuts is None:
        # State tercile cuts come from January 2022 (the first explored month) and are then frozen.
        cuts = {name: np.nanquantile(v[in_month], [1 / 3, 2 / 3]) for name, v in state_values.items()}
        cuts["burst"] = np.nanquantile(burst[in_month], np.linspace(0, 1, 11)[1:-1])
    states = {name: tercile(v, cuts[name]) for name, v in state_values.items()}
    day = (t // DAY_MS).astype(np.int64)
    level_class = np.digitize(levels, LEVEL_BINS).astype(np.int8)
    notional_class = np.digitize(notional, NOTIONAL_BINS).astype(np.int8)
    hit_class = np.digitize(hits, HIT_BINS).astype(np.int8)
    burst_decile = np.digitize(burst, cuts["burst"]).astype(np.int8) + 1

    rows: list = []
    for tau in TAUS:
        tau_key = np.full(len(t), tau, dtype=np.int16)
        fresh_key = fresh[tau].astype(np.int8)
        values = {"response": responses[tau], "markout": markouts[tau]}
        # S1: sweeps by price levels and notional (size control), split by proxy freshness.
        aggregate(rows, "S1_levels_notional", day, {"tau": tau_key, "level_class": level_class,
                                                     "notional_class": notional_class, "fresh": fresh_key},
                  values, in_month)
        # S1 conditioned on states, for deep sweeps (5+ levels).
        for name in ("liquidity", "activity", "volatility"):
            aggregate(rows, f"S1_state_{name}", day, {"tau": tau_key, "level_class": level_class,
                                                      "state": states[name]}, values, in_month & (levels >= 2))
        # S2: burst deciles; response from just before the event, signed by burst direction.
        burst_sign = np.sign(burst)
        aggregate(rows, "S2_burst", day, {"tau": tau_key, "burst_decile": burst_decile},
                  {"burst_response": burst_sign * pre_responses[tau]}, in_month & (burst_sign != 0))
        aggregate(rows, "S2_burst_activity", day, {"tau": tau_key, "burst_decile": burst_decile,
                                                   "state": states["activity"]},
                  {"burst_response": burst_sign * pre_responses[tau]}, in_month & (burst_sign != 0))
        # S3: level depletion, single-level events only.
        aggregate(rows, "S3_level_hits", day, {"tau": tau_key, "hit_class": hit_class,
                                               "state": states["liquidity"]}, values,
                  in_month & (levels == 1))
        # S4: spread state vs response magnitude and maker markout for all events.
        aggregate(rows, "S4_spread", day, {"tau": tau_key, "state": states["spread"]},
                  {"abs_response": np.abs(responses[tau]), "markout": markouts[tau]}, in_month)

    # Monthly rank ICs for continuous signals (subsample of 2 million events for speed).
    rng = np.random.default_rng(int(month.replace("-", "")))
    candidates = np.flatnonzero(in_month)
    sample = np.sort(rng.choice(candidates, size=min(2_000_000, len(candidates)), replace=False))
    ic_rows = []
    for tau in TAUS:
        signals = {
            "S1_signed_sweep_bps": (s * sweep_bps, responses[tau] * s),       # post-event mid change (unsigned)
            "S2_burst_intensity": (burst, pre_responses[tau]),
            "S3_signed_hits": (s * hits, responses[tau] * s),
            "S1_signed_levels": (s * levels, responses[tau] * s),
        }
        for name, (x, y) in signals.items():
            xs, ys = x[sample], y[sample]
            ok = np.isfinite(xs) & np.isfinite(ys)
            ic_rows.append({"month": month, "tau": tau, "signal": name, "subset": "all",
                            "spearman_ic": spearman(xs[ok], ys[ok])})
            ok_fresh = ok & fresh[tau][sample]
            ic_rows.append({"month": month, "tau": tau, "signal": name, "subset": "fresh",
                            "spearman_ic": spearman(xs[ok_fresh], ys[ok_fresh])})
    rows.append(pl.DataFrame(ic_rows).with_columns(pl.lit("IC").alias("table")))
    return rows, cuts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-data-root", type=Path, default=REPO / "shared-data")
    parser.add_argument("--start-month", default="2022-01")
    parser.add_argument("--end-month", default="2022-12")
    parser.add_argument("--output-dir", type=Path, default=REPO / "alpha-search" / "reports" / "seconds_signals_e002")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cuts = None
    tables: dict[str, list] = {}
    for month in iter_months(args.start_month, args.end_month):
        rows, cuts = process_month(args.shared_data_root, month, cuts)
        for frame in rows:
            name = frame["table"][0]
            tables.setdefault(name, []).append(frame.drop("table"))
        print(f"{month} done", flush=True)
    for name, frames in tables.items():
        pl.concat(frames, how="diagonal").write_parquet(args.output_dir / f"{name}.parquet")
    (args.output_dir / "state_cuts.txt").write_text(
        "\n".join(f"{k}: {np.round(v, 4).tolist()}" for k, v in cuts.items()) + "\n"
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
