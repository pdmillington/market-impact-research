"""Trading gaps and long bars: does the BTCUSDT perpetual market ever stop?

Binance perpetuals trade 24/7 with no holiday closures, but the exchange has
maintenance windows and occasional outages. For every pair of consecutive trade
events (`timestamp_direction_v1`) this script measures the time gap and the jump
in exchange trade IDs across it:

- trade IDs continuous (jump 0) across a long gap: the exchange printed no
  trades, i.e. a genuine halt or an empty market;
- trade IDs jump: trades exist that are missing from our data.

Outputs (reports/redo/data_description/):
- trading_gaps.csv: every gap longer than GAP_SECONDS, with trade-ID jump;
- gap_distribution.csv: counts of gaps by length, by year;
- long_bars_4000.csv: 4,000-event bars (the notebooks' construction, 2021-01..2026-08)
  lasting longer than LONG_BAR_HOURS, with the largest internal gap;
- holiday_activity.csv: events and volume on 24-26 Dec and 31 Dec-1 Jan against
  the median of the surrounding 28 days.

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
sys.path[:0] = [str(REPO / "shared-methodology" / "src"), str(REPO / "crypto-impact-study"),
                str(REPO / "crypto-impact-study" / "src")]

from crypto_market_data.layout import DataLayout  # noqa: E402
from crypto_market_data.months import iter_months  # noqa: E402

OUT = REPO / "crypto-impact-study" / "reports" / "redo" / "data_description"
PAPER_EVENTS = REPO / "shared-data" / "features" / "academic" / "symbol=BTCUSDT" / "paper_events_v2"
GAP_SECONDS = 60
LONG_BAR_HOURS = 3
GAP_BUCKETS = [10.0, 30.0, 60.0, 300.0, 1800.0, 3600.0, 6 * 3600.0, np.inf]


def month_gaps(layout: DataLayout, month: str, previous: dict | None) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    ev = (pl.scan_parquet(layout.event_month("BTCUSDT", month))
          .select("timestamp_ms", "first_trade_id", "last_trade_id").collect())
    if previous is not None:
        ev = pl.concat([pl.DataFrame([previous], schema=ev.schema), ev])
    g = ev.with_columns(
        (pl.col("timestamp_ms") - pl.col("timestamp_ms").shift(1)).alias("gap_ms"),
        (pl.col("first_trade_id").cast(pl.Int64) - pl.col("last_trade_id").shift(1).cast(pl.Int64) - 1).alias("trade_id_jump"),
        pl.col("timestamp_ms").shift(1).alias("gap_start_ms"),
    ).drop_nulls("gap_ms")
    s = g["gap_ms"].to_numpy() / 1000
    counts = pl.DataFrame({"month": [month] * (len(GAP_BUCKETS) - 1),
                           "gap_from_s": GAP_BUCKETS[:-1], "gap_to_s": GAP_BUCKETS[1:],
                           "gaps": [int(((s >= a) & (s < b)).sum()) for a, b in zip(GAP_BUCKETS[:-1], GAP_BUCKETS[1:])]})
    long = g.filter(pl.col("gap_ms") >= GAP_SECONDS * 1000).select("gap_start_ms", "timestamp_ms", "gap_ms", "trade_id_jump")
    last = ev.row(-1, named=True)
    return long, counts, last


def long_bars(start: str, end: str, gaps: pd.DataFrame) -> pd.DataFrame:
    from src.crypto_impact.monthly_eda import build_fixed_event_bars
    files = [p for p in sorted(PAPER_EVENTS.glob("BTCUSDT-trades-*.parquet")) if start <= p.stem[-7:] <= end]
    bars = build_fixed_event_bars(files, events_per_bar=4000).reset_index()
    out = bars.loc[bars["duration_seconds"] >= LONG_BAR_HOURS * 3600,
                   ["start_time", "end_time", "duration_seconds", "gross_volume", "log_return_bps"]].copy()
    rows = []
    for row in out.itertuples():
        inside = gaps.loc[(gaps["gap_start"] >= row.start_time) & (gaps["gap_end"] <= row.end_time)]
        top = inside.sort_values("gap_seconds").iloc[-1] if len(inside) else None
        rows.append({"largest_internal_gap_seconds": top["gap_seconds"] if top is not None else np.nan,
                     "largest_gap_start": top["gap_start"] if top is not None else pd.NaT,
                     "largest_gap_trade_id_jump": top["trade_id_jump"] if top is not None else np.nan})
    out = pd.concat([out.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    out["duration_hours"] = out["duration_seconds"] / 3600
    out.attrs["bars"] = len(bars)
    out.attrs["duration_quantiles"] = bars["duration_seconds"].quantile([0.5, 0.99, 0.999, 0.9999]).to_dict()
    return out.sort_values("duration_seconds", ascending=False)


def holiday_activity() -> pd.DataFrame:
    daily = pd.read_csv(OUT / "daily.csv", parse_dates=["day"])
    daily = daily.set_index("day")
    rows = []
    for year in range(2019, 2027):
        for month, day, label in ((12, 24, "Christmas Eve"), (12, 25, "Christmas Day"), (12, 26, "Boxing Day"),
                                  (12, 31, "New Year's Eve"), (1, 1, "New Year's Day")):
            d = pd.Timestamp(year=year, month=month, day=day, tz=daily.index.tz)
            if d not in daily.index:
                continue
            around = daily.loc[(daily.index >= d - pd.Timedelta(days=14)) & (daily.index <= d + pd.Timedelta(days=14)) & (daily.index != d)]
            rows.append({"day": d.date(), "holiday": label, "events": daily.loc[d, "events"],
                         "events_vs_28d_median": daily.loc[d, "events"] / around["events"].median(),
                         "volume_btc": daily.loc[d, "volume_btc"],
                         "volume_vs_28d_median": daily.loc[d, "volume_btc"] / around["volume_btc"].median(),
                         "range_pct": daily.loc[d, "range_pct"]})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2019-10")
    parser.add_argument("--end", default="2026-08")
    args = parser.parse_args()
    if args.end > "2026-08":
        raise SystemExit("Months after 2026-08 are reserved for the alpha-search P-006 test.")
    OUT.mkdir(parents=True, exist_ok=True)
    layout = DataLayout(REPO / "shared-data")
    longs, counts, previous = [], [], None
    for month in iter_months(args.start, args.end):
        long, count, previous = month_gaps(layout, month, previous)
        longs.append(long); counts.append(count)
    gaps = pl.concat(longs).to_pandas()
    gaps["gap_start"] = pd.to_datetime(gaps["gap_start_ms"], unit="ms", utc=True)
    gaps["gap_end"] = pd.to_datetime(gaps["timestamp_ms"], unit="ms", utc=True)
    gaps["gap_seconds"] = gaps["gap_ms"] / 1000
    gaps["data_missing"] = gaps["trade_id_jump"] != 0
    gaps = gaps[["gap_start", "gap_end", "gap_seconds", "trade_id_jump", "data_missing"]].sort_values("gap_seconds", ascending=False)
    gaps.to_csv(OUT / "trading_gaps.csv", index=False)
    dist = pl.concat(counts).to_pandas()
    dist["year"] = dist["month"].str[:4]
    dist = dist.groupby(["year", "gap_from_s", "gap_to_s"])["gaps"].sum().unstack(["gap_from_s", "gap_to_s"])
    dist.to_csv(OUT / "gap_distribution.csv")

    bars = long_bars("2021-01", min(args.end, "2026-08"), gaps)
    bars.to_csv(OUT / "long_bars_4000.csv", index=False)
    holidays = holiday_activity()
    holidays.to_csv(OUT / "holiday_activity.csv", index=False)

    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
    print(f"gaps >= {GAP_SECONDS}s: {len(gaps)}; of which trade IDs jump (data missing): {int(gaps['data_missing'].sum())}")
    print(gaps.head(25).to_string(index=False))
    print(dist.to_string())
    print(f"\n4,000-event bars: {bars.attrs['bars']:,}; duration quantiles (s): {bars.attrs['duration_quantiles']}")
    print(f"bars >= {LONG_BAR_HOURS}h: {len(bars)}")
    print(bars.head(25).to_string(index=False))
    print(holidays.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
