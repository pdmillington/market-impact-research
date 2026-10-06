"""Audit of the trade-event construction, for the paper's data description.

Two parts, both written to reports/redo/event_construction/:

1. monthly.csv / summary.csv: statistics of the shared `timestamp_direction_v1`
   events (one event per contiguous same-sign run of fills at a millisecond
   timestamp; price is not a grouping key), BTCUSDT 2021-01..2026-08:
   - fills, distinct timestamps, events;
   - multi-fill and multi-price events (sweeps across price levels) and their volume share;
   - timestamps with both aggressor sides (mixed-direction) and the volume that a
     one-event-per-timestamp rule taking the first fill's sign would mis-sign;
   - how much of the absolute log-price path happens inside events
     (first -> last fill price) rather than between events.
   price_levels.csv: distribution of price levels per event.

2. old_inputs.csv: what the paper's original input files
   (crypto-impact-study/data/raw/BTCUSDT_monthly, 2025-01..2026-05) contain,
   checked month by month against the fills: rows vs distinct timestamps, and
   whether each row's sign, price and quantity equal the first fill's sign, the
   first fill's price and the gross quantity at that timestamp.

Months from 2026-09 onward are refused (alpha-search P-006).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import polars as pl


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "shared-methodology" / "src"))

from crypto_market_data.layout import DataLayout  # noqa: E402
from crypto_market_data.months import iter_months  # noqa: E402

OUT = REPO / "crypto-impact-study" / "reports" / "redo" / "event_construction"
OLD = REPO / "crypto-impact-study" / "data" / "raw" / "BTCUSDT_monthly"
LEVEL_BUCKETS = [1, 2, 3, 5, 10, 20, 50]


def month_stats(layout: DataLayout, month: str) -> tuple[dict, pd.DataFrame]:
    events = pl.scan_parquet(layout.event_month("BTCUSDT", month))
    ev = events.with_columns(
        (pl.col("last_price") / pl.col("first_price")).log().abs().alias("within"),
        (pl.col("first_price") / pl.col("last_price").shift(1)).log().abs().fill_null(0).alias("between"),
        (pl.col("aggressor_sign") != pl.col("aggressor_sign").first().over("timestamp_ms")).alias("off_first_sign"),
        (pl.len().over("timestamp_ms") > 1).alias("mixed_ts"),
    )
    s = ev.select(
        pl.len().alias("events"),
        pl.col("fill_count").sum().alias("fills"),
        pl.col("timestamp_ms").n_unique().alias("timestamps"),
        pl.col("quantity").sum().alias("volume"),
        (pl.col("fill_count") > 1).sum().alias("multi_fill_events"),
        (pl.col("price_level_count") > 1).sum().alias("multi_price_events"),
        pl.col("quantity").filter(pl.col("fill_count") > 1).sum().alias("multi_fill_volume"),
        pl.col("quantity").filter(pl.col("price_level_count") > 1).sum().alias("multi_price_volume"),
        pl.col("fill_count").filter(pl.col("price_level_count") > 1).sum().alias("multi_price_fills"),
        pl.col("timestamp_ms").filter(pl.col("mixed_ts")).n_unique().alias("mixed_direction_timestamps"),
        pl.col("mixed_ts").sum().alias("events_at_mixed_timestamps"),
        pl.col("quantity").filter(pl.col("off_first_sign")).sum().alias("volume_missigned_by_first_sign_rule"),
        pl.col("within").sum().alias("abs_logprice_within_events"),
        pl.col("between").sum().alias("abs_logprice_between_events"),
    ).collect().to_dicts()[0]
    s["month"] = month
    levels = (ev.select(pl.col("price_level_count").cut(
        [b + 0.5 for b in LEVEL_BUCKETS], labels=["1", "2", "3", "4-5", "6-10", "11-20", "21-50", ">50"]).alias("levels"),
        "quantity")
        .group_by("levels").agg(pl.len().alias("events"), pl.col("quantity").sum().alias("volume"))
        .collect().to_pandas())
    levels["month"] = month
    return s, levels


def old_input_check(layout: DataLayout, month: str) -> dict:
    old = pl.scan_parquet(OLD / f"BTCUSDT-trades-{month}.parquet").select(
        pl.col("time").alias("timestamp_ms"), "price", "qty", (pl.when(pl.col("is_buyer_maker")).then(-1).otherwise(1)).alias("sign"))
    fills = pl.scan_parquet(layout.canonical_month("BTCUSDT", month)).sort("trade_id")
    first = fills.group_by("timestamp_ms", maintain_order=True).agg(
        pl.col("price").first().alias("first_price"), pl.col("price").last().alias("last_price"),
        pl.col("aggressor_sign").first().alias("first_sign"), pl.col("quantity").sum().alias("gross"),
        pl.col("aggressor_sign").n_unique().alias("sides"))
    j = old.join(first, on="timestamp_ms", how="full", coalesce=True)
    r = j.select(
        pl.col("price").is_not_null().sum().alias("old_rows"),
        pl.col("first_price").is_not_null().sum().alias("fill_timestamps"),
        (pl.col("price").is_not_null() & pl.col("first_price").is_null()).sum().alias("old_rows_without_fills"),
        (pl.col("price").is_null() & pl.col("first_price").is_not_null()).sum().alias("fill_timestamps_missing_from_old"),
        (pl.col("price") == pl.col("first_price")).sum().alias("old_price_eq_first_fill"),
        ((pl.col("price") == pl.col("last_price")) & (pl.col("first_price") != pl.col("last_price"))).sum().alias("old_price_eq_last_fill_where_different"),
        (pl.col("first_price") != pl.col("last_price")).sum().alias("timestamps_first_ne_last_price"),
        (pl.col("sign") == pl.col("first_sign")).sum().alias("old_sign_eq_first_fill"),
        ((pl.col("qty") - pl.col("gross")).abs() < 1e-9).sum().alias("old_qty_eq_gross"),
        (pl.col("sides") > 1).sum().alias("mixed_direction_timestamps"),
    ).collect().to_dicts()[0]
    r["month"] = month
    return r


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2021-01")
    parser.add_argument("--end", default="2026-08")
    parser.add_argument("--skip-old", action="store_true")
    args = parser.parse_args()
    if args.end > "2026-08":
        raise SystemExit("Months after 2026-08 are reserved for the alpha-search P-006 test.")
    OUT.mkdir(parents=True, exist_ok=True)
    layout = DataLayout(REPO / "shared-data")

    if not args.skip_old:
        old = [old_input_check(layout, m) for m in iter_months("2025-01", "2026-05")]
        pd.DataFrame(old).to_csv(OUT / "old_inputs.csv", index=False)
        print(pd.DataFrame(old).set_index("month").sum().to_string(), flush=True)

    rows, levels = [], []
    for month in iter_months(args.start, args.end):
        s, lv = month_stats(layout, month)
        rows.append(s)
        levels.append(lv)
        print(f"{month}: {s['events']:,} events, multi-price {s['multi_price_events'] / s['events']:.1%}", flush=True)
    monthly = pd.DataFrame(rows).set_index("month")
    monthly.to_csv(OUT / "monthly.csv")
    levels = pd.concat(levels).groupby("levels", observed=True)[["events", "volume"]].sum()
    levels = levels.reindex(["1", "2", "3", "4-5", "6-10", "11-20", "21-50", ">50"])
    levels["event_share"] = levels["events"] / levels["events"].sum()
    levels["volume_share"] = levels["volume"] / levels["volume"].sum()
    levels.to_csv(OUT / "price_levels.csv")

    t = monthly.sum(numeric_only=True)
    summary = {
        "sample": f"{args.start}..{args.end}",
        "fills": t["fills"], "timestamps": t["timestamps"], "events": t["events"],
        "fills_per_event": t["fills"] / t["events"],
        "events_per_timestamp_minus_1": t["events"] / t["timestamps"] - 1,
        "multi_fill_event_share": t["multi_fill_events"] / t["events"],
        "multi_fill_volume_share": t["multi_fill_volume"] / t["volume"],
        "multi_price_event_share": t["multi_price_events"] / t["events"],
        "multi_price_volume_share": t["multi_price_volume"] / t["volume"],
        "multi_price_fill_share": t["multi_price_fills"] / t["fills"],
        "mixed_direction_timestamp_share": t["mixed_direction_timestamps"] / t["timestamps"],
        "events_at_mixed_timestamps_share": t["events_at_mixed_timestamps"] / t["events"],
        "volume_missigned_share_first_sign_rule": t["volume_missigned_by_first_sign_rule"] / t["volume"],
        "share_abs_logprice_path_within_events": t["abs_logprice_within_events"]
        / (t["abs_logprice_within_events"] + t["abs_logprice_between_events"]),
    }
    pd.Series(summary).to_frame("value").to_csv(OUT / "summary.csv")
    print(pd.Series(summary).to_string())
    print(levels.round(4).to_string())


if __name__ == "__main__":
    main()
