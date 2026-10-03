"""Hourly execution-signature features for E-005 / H-013.

Step 1 (per month): aggregate reconstructed events (timestamp_direction_v1) into
UTC-hour buckets, using the frozen H-014 to H-016 definitions and the frozen
E-002/E-003 cuts:

- notional = quantity × vwap; signed by the aggressor;
- multi-level (ML) events: price_level_count >= 2; sweep_bps = s × 1e4 × ln(last/first);
- level runs: consecutive single-level same-side events at an unchanged price;
  a run "qualifies" once it has >= 3 hits in the high absorption tercile, and is
  counted once, in the hour it first qualifies, by its aggressor side;
- burst decile of each event (signed 5 s intensity / 24 h rate);
- large events: notional >= $1m;
- hour close: last price of the last event in the hour.

Step 2 (per symbol): trailing-window features at each decision hour T over
[T - W, T) for W = 8 h and 24 h, each z-scored (catalogue z30) against its own
previous 720 hourly values (min 360).

Months run only to 2026-05: the locked 2026-06..08 months are never built.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np
import polars as pl


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "alpha-search" / "scripts"))
sys.path.insert(0, str(REPO / "shared-methodology" / "src"))

from crypto_market_data.months import iter_months  # noqa: E402
from explore_seconds_signals import DAY_MS, load_events, month_start_ms, trailing_count, trailing_mean_at  # noqa: E402


HOUR_MS = 3_600_000
LAST_MONTH = "2026-05"
FIRST_MONTH = {"BTCUSDT": "2019-10", "ETHUSDT": "2019-12", "SOLUSDT": "2020-10"}
LARGE_NOTIONAL = 1_000_000.0
MIN_HITS = 3
WINDOWS = {"8h": 8, "24h": 24}
Z_WINDOW, Z_MIN = 720, 360
FEATURES = ("ml_share", "ml_imbalance", "sweep_pressure", "absorption_imbalance", "burst_skew",
            "large_imbalance", "flow_imbalance")


def hourly_path(root: Path, symbol: str, month: str) -> Path:
    year, number = month.split("-")
    return (root / "features" / "alpha" / f"symbol={symbol}" / "signature_hourly_v2"
            / f"year={year}" / f"month={number}" / "part-000.parquet")


def panel_path(root: Path, symbol: str) -> Path:
    return root / "features" / "alpha" / f"symbol={symbol}" / "signature_panel_v2" / "part-000.parquet"


def aggregate_month(task: tuple) -> str:
    root, symbol, month, cuts = task
    target = hourly_path(root, symbol, month)
    if target.exists():
        return f"{symbol} {month}: exists"
    events = load_events(root, month, symbol)
    t = events["timestamp_ms"].to_numpy()
    s = events["aggressor_sign"].to_numpy().astype(np.int8)
    qty = events["quantity"].to_numpy()
    notional = qty * events["vwap"].to_numpy()
    first = events["first_price"].to_numpy()
    last = events["last_price"].to_numpy()
    levels = events["price_level_count"].to_numpy()
    del events
    sf = s.astype(np.float64)
    multi = levels >= 2
    sweep_bps = sf * 1e4 * np.log(last / first)

    # Level runs and absorption (frozen H-015 / E-003 definitions, as in build_quote_triggers.py).
    same = np.zeros(len(t), bool)
    same[1:] = (s[1:] == s[:-1]) & (levels[1:] == 1) & (levels[:-1] == 1) & (first[1:] == last[:-1])
    run_id = np.cumsum(~same)
    run_start = np.flatnonzero(~same)
    hits = np.arange(len(t)) - run_start[run_id - 1] + 1
    cumulative = np.cumsum(qty)
    run_qty = cumulative - np.concatenate([[0.0], cumulative])[run_start[run_id - 1]]
    depth = {}
    for side in (1, -1):
        chosen = multi & (s == side)
        depth[side] = np.exp(trailing_mean_at(t, t[chosen], np.log(qty[chosen] / levels[chosen]), 1_800_000))
    absorption = run_qty / np.where(s > 0, depth[1], depth[-1])
    high = (np.digitize(absorption, cuts["absorption"]) + 1 == 3) & np.isfinite(absorption)
    qualifies = (levels == 1) & (hits >= MIN_HITS) & high
    first_qualifying = np.zeros(len(t), bool)
    q_index = np.flatnonzero(qualifies)
    if len(q_index):
        _, first_pos = np.unique(run_id[q_index], return_index=True)
        first_qualifying[q_index[first_pos]] = True

    rate_24h = trailing_count(t, np.ones(len(t), bool), DAY_MS) / 86_400.0
    burst = (trailing_count(t, sf > 0, 5_000) - trailing_count(t, sf < 0, 5_000)) / np.maximum(rate_24h * 5.0, 1e-9)
    decile = np.digitize(burst, cuts["burst"]) + 1

    keep = t >= month_start_ms(month)
    hour = (t[keep] // HOUR_MS).astype(np.int64)
    frame = pl.DataFrame({
        "hour": hour,
        "notional": notional[keep],
        "signed_notional": (sf * notional)[keep],
        "ml_notional": np.where(multi, notional, 0.0)[keep],
        "ml_signed_notional": np.where(multi, sf * notional, 0.0)[keep],
        # sweep_bps is side-aligned (always >= 0); the registered feature is signed depth s × sweep_bps.
        "sweep_signed_bps": np.where(multi, sf * sweep_bps, 0.0)[keep],
        "large_signed_notional": np.where(notional >= LARGE_NOTIONAL, sf * notional, 0.0)[keep],
        "absorb_sell_runs": (first_qualifying & (s < 0))[keep].astype(np.int32),
        "absorb_buy_runs": (first_qualifying & (s > 0))[keep].astype(np.int32),
        "burst_top": (decile == 10)[keep].astype(np.int32),
        "burst_bottom": (decile == 1)[keep].astype(np.int32),
        "price": last[keep],
    })
    hourly = (frame.group_by("hour").agg(
        pl.len().alias("events"),
        *[pl.col(c).sum() for c in frame.columns if c not in ("hour", "price")],
        pl.col("price").last().alias("close"))
        .sort("hour"))
    target.parent.mkdir(parents=True, exist_ok=True)
    hourly.write_parquet(target, compression="zstd")
    return f"{symbol} {month}: {len(hour)} events, {hourly.height} hours"


def rolling_z(values: np.ndarray) -> np.ndarray:
    """Catalogue z30: standardise against the previous 720 hourly values (strictly earlier)."""

    series = pl.Series(values, nan_to_null=True).shift(1)
    mean = series.rolling_mean(Z_WINDOW, min_samples=Z_MIN).fill_null(np.nan).to_numpy()
    std = series.rolling_std(Z_WINDOW, min_samples=Z_MIN).fill_null(np.nan).to_numpy()
    return (values - mean) / np.where(std > 0, std, np.nan)


def build_panel(root: Path, symbol: str) -> pl.DataFrame:
    hourly = pl.concat([pl.read_parquet(hourly_path(root, symbol, m))
                        for m in iter_months(FIRST_MONTH[symbol], LAST_MONTH)]).sort("hour")
    first, last = int(hourly["hour"].min()), int(hourly["hour"].max())
    grid = pl.DataFrame({"hour": np.arange(first, last + 1, dtype=np.int64)})
    h = grid.join(hourly, on="hour", how="left").with_columns(
        [pl.col(c).fill_null(0) for c in hourly.columns if c not in ("hour", "close")])
    close = h["close"].fill_null(strategy="forward").to_numpy()
    out = {"decision_ms": (h["hour"].to_numpy() + 1) * HOUR_MS, "close": close, "events": h["events"].to_numpy()}

    def window_sum(column: str, hours: int) -> np.ndarray:
        return h[column].cast(pl.Float64).rolling_sum(hours, min_samples=hours).fill_null(np.nan).to_numpy()

    for label, hours in WINDOWS.items():
        notional = window_sum("notional", hours)
        ml = window_sum("ml_notional", hours)
        events = window_sum("events", hours)
        runs = window_sum("absorb_sell_runs", hours) + window_sum("absorb_buy_runs", hours)
        raw = {
            "ml_share": ml / notional,
            "ml_imbalance": window_sum("ml_signed_notional", hours) / ml,
            "sweep_pressure": window_sum("sweep_signed_bps", hours) / events,
            "absorption_imbalance": np.where(runs > 0, (window_sum("absorb_sell_runs", hours)
                                                        - window_sum("absorb_buy_runs", hours)) / np.maximum(runs, 1), 0.0),
            "burst_skew": (window_sum("burst_top", hours) - window_sum("burst_bottom", hours)) / events,
            "large_imbalance": window_sum("large_signed_notional", hours) / notional,
            "flow_imbalance": window_sum("signed_notional", hours) / notional,
        }
        lagged = np.concatenate([np.full(hours, np.nan), close[:-hours]])
        raw["own_return"] = np.log(close / lagged)
        for name, values in raw.items():
            values = np.where(np.isfinite(values), values, np.nan)
            out[f"{name}_{label}"] = values
            out[f"{name}_{label}_z"] = rolling_z(values)
    panel = pl.DataFrame(out).with_columns(pl.lit(symbol).alias("symbol"))
    path = panel_path(root, symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.write_parquet(path, compression="zstd")
    return panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-data-root", type=Path, default=REPO / "shared-data")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    cuts = json.loads((REPO / "alpha-search" / "configs" / "seconds_signals_test_v1.json").read_text())["frozen_cuts"]
    tasks = [(args.shared_data_root, s, m, cuts) for s in args.symbols for m in iter_months(FIRST_MONTH[s], LAST_MONTH)]
    with ProcessPoolExecutor(args.workers) as pool:
        for message in pool.map(aggregate_month, tasks):
            print(message, flush=True)
    for symbol in args.symbols:
        panel = build_panel(args.shared_data_root, symbol)
        print(f"{symbol} panel: {panel.height} hours", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
