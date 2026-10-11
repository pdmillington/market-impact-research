"""Write the P&B event blocks (pb_blocks_v1) used by the redo scripts.

Blocks are built by the shared builder (crypto_impact.bars) from the shared
`timestamp_direction_v1` event lake, so they follow the same construction rules as
every other bar in the study:

- one event per contiguous same-sign run of fills at a millisecond timestamp,
  priced at the run's last fill (the paper's original input files instead merged
  each timestamp under the first fill's price and sign);
- consecutive, non-overlapping blocks of N events, continuous across months;
- gross and signed volume rounded to the 0.001 BTC lot;
- blocks covering a known unrecoverable gap in the trade archives are kept with
  spans_data_gap = True (their observed volume enters trailing windows); analyses
  drop them.

Columns: n_events, start/end_time_ms, duration_seconds, start/end_price, gross_volume,
signed_volume, sign_sum, sign_bias, trade_imbalance, log_return = ln(end/start price),
price_change_fraction (share of adjacent events with a different price),
endpoint_price_changed, spans_data_gap.

Output: shared-data/features/academic/symbol=BTCUSDT/pb_blocks_v1/N=<n>/<YYYY-MM>.parquet,
holding the blocks completed while that month's events are read (a block that starts
in one month and ends in the next is in the later month's file).
Months from 2026-09 onward are refused (reserved for the alpha-search P-006 test).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import polars as pl


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "shared-methodology" / "src"))
sys.path.insert(0, str(REPO / "crypto-impact-study" / "src"))

from crypto_market_data.months import iter_months  # noqa: E402
from crypto_impact.bars import iter_event_bars, lake_event_sources  # noqa: E402


N_VALUES = (100, 250, 500, 1000, 2000, 4000, 8000, 16000)
LAST_ALLOWED_MONTH = "2026-08"


def output_path(root: Path, symbol: str, n: int, month: str) -> Path:
    return root / "features" / "academic" / f"symbol={symbol}" / "pb_blocks_v1" / f"N={n}" / f"{month}.parquet"


BLOCK_SCHEMA = {
    "n_events": pl.Int32, "start_time_ms": pl.Int64, "end_time_ms": pl.Int64, "duration_seconds": pl.Float64,
    "start_price": pl.Float64, "end_price": pl.Float64, "gross_volume": pl.Float64, "signed_volume": pl.Float64,
    "sign_sum": pl.Int32, "sign_bias": pl.Float64, "trade_imbalance": pl.Float64, "log_return": pl.Float64,
    "price_change_fraction": pl.Float64, "endpoint_price_changed": pl.Boolean, "spans_data_gap": pl.Boolean,
}


def block_frame(bars, n: int) -> pl.DataFrame:
    """Shared-builder bars in the pb_blocks_v1 schema."""

    if bars is None or bars.empty:
        return pl.DataFrame(schema=BLOCK_SCHEMA)
    start_ms = bars.index.as_unit("ms").asi8
    end_ms = bars["end_time"].dt.as_unit("ms").astype("int64").to_numpy()
    return pl.DataFrame({
        "n_events": np.full(len(bars), n, dtype=np.int32),
        "start_time_ms": start_ms,
        "end_time_ms": end_ms,
        "duration_seconds": (end_ms - start_ms) / 1_000.0,
        "start_price": bars["start_price"].to_numpy(),
        "end_price": bars["end_price"].to_numpy(),
        "gross_volume": bars["gross_volume"].to_numpy(),
        "signed_volume": bars["signed_imbalance"].to_numpy(),
        "sign_sum": bars["sign_sum"].to_numpy().astype(np.int32),
        "sign_bias": bars["sign_bias"].to_numpy(),
        "trade_imbalance": (bars["signed_imbalance"] / bars["gross_volume"]).to_numpy(),
        "log_return": np.log(bars["end_price"].to_numpy() / bars["start_price"].to_numpy()),
        "price_change_fraction": bars["price_change_fraction"].to_numpy(),
        "endpoint_price_changed": bars["endpoint_price_changed"].to_numpy(),
        "spans_data_gap": bars["spans_data_gap"].to_numpy(),
    }, schema=BLOCK_SCHEMA)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start-month", default="2019-09")
    parser.add_argument("--end-month", default="2026-08")
    parser.add_argument("--shared-data-root", type=Path, default=REPO / "shared-data")
    args = parser.parse_args()
    if args.end_month > LAST_ALLOWED_MONTH:
        raise SystemExit(f"Months after {LAST_ALLOWED_MONTH} are reserved for the alpha-search P-006 test.")
    root = args.shared_data_root
    months = list(iter_months(args.start_month, args.end_month))
    sources = lake_event_sources(root, args.symbol, months)
    for month, completed in iter_event_bars(sources, N_VALUES, drop_data_gaps=False):
        for n in N_VALUES:
            target = output_path(root, args.symbol, n, month)
            target.parent.mkdir(parents=True, exist_ok=True)
            block_frame(completed.get(n), n).write_parquet(target, compression="zstd")
        print(f"{month}: done", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
