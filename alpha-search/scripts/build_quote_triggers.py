"""Per-event trigger features for the H-017/H-018 simulator.

For each reconstructed event in the quote year this stores the quantities the
registered policies need, using the frozen H-014 to H-016 definitions:

- sweep_bps: side-aligned first-to-last execution-price move of the event
  (scale-free sweep depth, H-014);
- hits and absorption_tercile: the level-run length and its absorption tercile
  (frozen E-003 cuts) for single-level events (H-015);
- burst_decile: signed 5-second burst-intensity decile (frozen E-002 cuts, H-016).

Thresholds (D, k) are applied later by the simulator.
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

from crypto_market_data.months import iter_months  # noqa: E402
from explore_seconds_signals import DAY_MS, load_events, month_start_ms, trailing_count, trailing_mean_at  # noqa: E402


def output_path(root: Path, symbol: str, month: str) -> Path:
    year, number = month.split("-")
    return (root / "features" / "alpha" / f"symbol={symbol}" / "quote_triggers_v1"
            / f"year={year}" / f"month={number}" / "part-000.parquet")


def build_month(root: Path, symbol: str, month: str, cuts: dict) -> pl.DataFrame:
    events = load_events(root, month, symbol)
    t = events["timestamp_ms"].to_numpy()
    s = events["aggressor_sign"].to_numpy().astype(np.int8)
    qty = events["quantity"].to_numpy()
    first = events["first_price"].to_numpy()
    last = events["last_price"].to_numpy()
    levels = events["price_level_count"].to_numpy()
    del events
    sf = s.astype(np.float64)
    sweep_bps = sf * 1e4 * np.log(last / first)

    same = np.zeros(len(t), bool)
    same[1:] = (s[1:] == s[:-1]) & (levels[1:] == 1) & (levels[:-1] == 1) & (first[1:] == last[:-1])
    run_id = np.cumsum(~same)
    run_start = np.flatnonzero(~same)
    hits = np.arange(len(t)) - run_start[run_id - 1] + 1
    cumulative = np.cumsum(qty)
    run_qty = cumulative - np.concatenate([[0.0], cumulative])[run_start[run_id - 1]]
    depth = {}
    for side in (1, -1):
        chosen = (levels >= 2) & (s == side)
        depth[side] = np.exp(trailing_mean_at(t, t[chosen], np.log(qty[chosen] / levels[chosen]), 1_800_000))
    absorption = run_qty / np.where(s > 0, depth[1], depth[-1])
    absorption_tercile = np.digitize(absorption, cuts["absorption"]) + 1
    absorption_tercile[~np.isfinite(absorption)] = 0

    rate_24h = trailing_count(t, np.ones(len(t), bool), DAY_MS) / 86_400.0
    burst = (trailing_count(t, sf > 0, 5_000) - trailing_count(t, sf < 0, 5_000)) / np.maximum(rate_24h * 5.0, 1e-9)
    burst_decile = np.digitize(burst, cuts["burst"]) + 1

    keep = t >= month_start_ms(month)
    return pl.DataFrame({
        "time_ms": t[keep], "sign": s[keep], "price": last[keep],
        "sweep_bps": sweep_bps[keep].astype(np.float32),
        "levels": levels[keep].astype(np.uint16),
        "hits": hits[keep].astype(np.uint32),
        "absorption_tercile": absorption_tercile[keep].astype(np.int8),
        "burst_decile": burst_decile[keep].astype(np.int8),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-data-root", type=Path, default=REPO / "shared-data")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "SOLUSDT", "ETHUSDT"])
    parser.add_argument("--start-month", default="2023-05")
    parser.add_argument("--end-month", default="2024-03")
    args = parser.parse_args()
    config = json.loads((REPO / "alpha-search" / "configs" / "seconds_signals_test_v1.json").read_text())
    cuts = config["frozen_cuts"]
    for symbol in args.symbols:
        for month in iter_months(args.start_month, args.end_month):
            target = output_path(args.shared_data_root, symbol, month)
            if target.exists():
                continue
            frame = build_month(args.shared_data_root, symbol, month, cuts)
            target.parent.mkdir(parents=True, exist_ok=True)
            frame.write_parquet(target, compression="zstd")
            print(f"{symbol} {month}: {frame.height} events", flush=True)


if __name__ == "__main__":
    main()
