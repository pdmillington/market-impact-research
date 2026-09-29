"""Build a 1-second decision grid from reconstructed aggressor events.

Each row summarises every `timestamp_direction_v1` event whose timestamp falls in
that UTC second. Seconds without events are absent. The grid supports clock-time
stress detection without losing the event-level execution information:

- close/high/low use event last, maximum and minimum execution prices;
- buy/sell quantities and event counts are split by aggressor side;
- deep events have side-aligned endpoint sweep of at least the frozen decay-v2
  top-quartile cut (0.686 bps) or at least five price levels;
- the flip-gap spread proxy is the absolute price change from the last fill of
  one aggressor event to the first fill of the next event on the opposite side,
  counted only when the two events are at most 250 ms apart and capped at 100 bps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import polars as pl


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "shared-methodology" / "src"))

from crypto_market_data.layout import DataLayout  # noqa: E402
from crypto_market_data.months import iter_months  # noqa: E402


GRID_VERSION = "second_grid_v1"
DEEP_SWEEP_BPS = 0.686
DEEP_PRICE_LEVELS = 5
FLIP_MAX_GAP_MS = 250
FLIP_CAP_BPS = 100.0


def second_grid(events: pl.LazyFrame) -> pl.LazyFrame:
    side = pl.col("aggressor_sign")
    buy = side > 0
    sell = side < 0
    ordered = events.sort("timestamp_ms", "first_trade_id").with_columns(
        (10_000.0 * side * (pl.col("last_price") / pl.col("first_price")).log()).alias(
            "sweep_bps"
        ),
        (
            (side != side.shift(1))
            & side.shift(1).is_not_null()
            & (pl.col("timestamp_ms") - pl.col("timestamp_ms").shift(1) <= FLIP_MAX_GAP_MS)
        ).alias("flip"),
        (10_000.0 * (pl.col("first_price") / pl.col("last_price").shift(1)).log().abs())
        .clip(upper_bound=FLIP_CAP_BPS)
        .alias("flip_gap_bps"),
    )
    deep = (pl.col("sweep_bps") >= DEEP_SWEEP_BPS) | (
        pl.col("price_level_count") >= DEEP_PRICE_LEVELS
    )
    return (
        ordered.group_by((pl.col("timestamp_ms") // 1_000).alias("second"))
        .agg(
            pl.col("last_price").last().alias("close"),
            pl.col("max_price").max().alias("high"),
            pl.col("min_price").min().alias("low"),
            pl.col("quantity").filter(buy).sum().alias("buy_qty"),
            pl.col("quantity").filter(sell).sum().alias("sell_qty"),
            buy.sum().cast(pl.UInt32).alias("buy_events"),
            sell.sum().cast(pl.UInt32).alias("sell_events"),
            (deep & buy).sum().cast(pl.UInt32).alias("buy_deep"),
            (deep & sell).sum().cast(pl.UInt32).alias("sell_deep"),
            pl.col("quantity").filter(deep & buy).sum().alias("buy_deep_qty"),
            pl.col("quantity").filter(deep & sell).sum().alias("sell_deep_qty"),
            pl.col("sweep_bps").filter(buy).sum().alias("buy_sweep_bps"),
            pl.col("sweep_bps").filter(sell).sum().alias("sell_sweep_bps"),
            pl.col("flip").sum().cast(pl.UInt32).alias("flips"),
            pl.col("flip_gap_bps").filter(pl.col("flip")).sum().alias("flip_gap_sum_bps"),
        )
        .sort("second")
    )


def output_path(root: Path, symbol: str, month: str) -> Path:
    year, number = month.split("-")
    return (
        root
        / "features"
        / "alpha"
        / f"symbol={symbol}"
        / GRID_VERSION
        / f"year={year}"
        / f"month={number}"
        / "part-000.parquet"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-data-root", type=Path, default=REPO / "shared-data")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start-month", required=True)
    parser.add_argument("--end-month", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    layout = DataLayout(args.shared_data_root)
    for month in iter_months(args.start_month, args.end_month):
        target = output_path(args.shared_data_root, args.symbol, month)
        if target.exists() and not args.overwrite:
            print(f"{month}: exists", flush=True)
            continue
        source = layout.event_month(args.symbol, month)
        grid = second_grid(pl.scan_parquet(source)).collect()
        target.parent.mkdir(parents=True, exist_ok=True)
        grid.write_parquet(target, compression="zstd")
        print(f"{month}: {grid.height} seconds", flush=True)
    manifest = {
        "grid_version": GRID_VERSION,
        "event_rule": "timestamp_direction_v1",
        "deep_sweep_bps": DEEP_SWEEP_BPS,
        "deep_price_levels": DEEP_PRICE_LEVELS,
        "flip_max_gap_ms": FLIP_MAX_GAP_MS,
        "flip_cap_bps": FLIP_CAP_BPS,
    }
    manifest_path = output_path(args.shared_data_root, args.symbol, "2000-01").parents[2]
    (manifest_path / "grid_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
