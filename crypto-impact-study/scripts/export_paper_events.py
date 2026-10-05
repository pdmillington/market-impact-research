"""Export corrected monthly event files in the paper notebooks' input schema.

The market-impact notebooks read `BTCUSDT-trades-YYYY-MM.parquet` with columns
price, qty, time (ms), time_str (UTC datetime) and is_buyer_maker, one row per
aggressor event. The original files merged every timestamp into one event (the
first fill's sign, gross volume of all fills). These files are built from the
shared `timestamp_direction_v1` events instead: one row per contiguous same-sign
run at a timestamp, price = last fill price of the run, qty = run volume.

Output: shared-data/features/academic/symbol=BTCUSDT/paper_events_v2/BTCUSDT-trades-YYYY-MM.parquet
Months after 2026-08 are refused (reserved for the alpha-search P-006 test).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import polars as pl


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "shared-methodology" / "src"))

from crypto_market_data.layout import DataLayout  # noqa: E402
from crypto_market_data.months import iter_months  # noqa: E402


def output_dir(root: Path, symbol: str) -> Path:
    return root / "features" / "academic" / f"symbol={symbol}" / "paper_events_v2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start-month", default="2021-01")
    parser.add_argument("--end-month", default="2026-08")
    parser.add_argument("--shared-data-root", type=Path, default=REPO / "shared-data")
    args = parser.parse_args()
    if args.end_month > "2026-08":
        raise SystemExit("Months after 2026-08 are reserved for the alpha-search P-006 test.")
    layout = DataLayout(args.shared_data_root)
    out = output_dir(args.shared_data_root, args.symbol)
    out.mkdir(parents=True, exist_ok=True)
    for month in iter_months(args.start_month, args.end_month):
        target = out / f"{args.symbol}-trades-{month}.parquet"
        frame = (pl.scan_parquet(layout.event_month(args.symbol, month))
                 .sort("timestamp_ms", "first_trade_id")
                 .select(pl.col("last_price").alias("price"), pl.col("quantity").alias("qty"),
                         pl.col("timestamp_ms").alias("time"),
                         pl.from_epoch("timestamp_ms", time_unit="ms").alias("time_str"),
                         (pl.col("aggressor_sign") < 0).alias("is_buyer_maker"))
                 .collect())
        frame.write_parquet(target, compression="zstd")
        print(f"{month}: {frame.height:,} events", flush=True)


if __name__ == "__main__":
    main()
