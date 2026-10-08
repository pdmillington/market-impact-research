"""Build the paper's continuous P&B event blocks from the shared event lake.

Replicates `aggregate_complete_blocks` in notebooks/PatzeltBouchaudScaling.ipynb,
but reads the shared `timestamp_direction_v1` events instead of the paper's
pre-merged monthly files. Those files collapsed every timestamp into a single
event with the first fill's sign and the gross volume of all fills, which
mis-signs opposite-side volume at timestamps where both sides traded (about 1%
of timestamps). The shared events keep each contiguous same-sign run separate.

For each block of N consecutive events (continuous across months, no gaps reset):

- event price = the last fill price of the event (as in the paper);
- log_return = ln(last event price / first event price);
- gross and signed volume, sign sum, duration;
- price_change_fraction = share of adjacent events with a different price;
- endpoint_price_changed = last price differs from first;
- spans_data_gap = the block covers a known unrecoverable gap in the trade archives
  (config/btcusdt_data_gaps.csv); analyses drop these blocks.

Output: one parquet per N and month (month of the block's first event) under
shared-data/features/academic/symbol=BTCUSDT/pb_blocks_v1/N=<n>/<YYYY-MM>.parquet.
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

from crypto_market_data.layout import DataLayout  # noqa: E402
from crypto_market_data.months import iter_months  # noqa: E402
from crypto_impact.data_gaps import spans_data_gap  # noqa: E402


N_VALUES = (100, 250, 500, 1000, 2000, 4000, 8000, 16000)
LAST_ALLOWED_MONTH = "2026-08"
DAY_MS = 86_400_000


def output_path(root: Path, symbol: str, n: int, month: str) -> Path:
    return root / "features" / "academic" / f"symbol={symbol}" / "pb_blocks_v1" / f"N={n}" / f"{month}.parquet"


def complete_blocks(time_ms: np.ndarray, price: np.ndarray, quantity: np.ndarray, sign: np.ndarray, n: int) -> pl.DataFrame:
    rows = len(time_ms) // n
    time_m = time_ms[: rows * n].reshape(rows, n)
    price_m = price[: rows * n].reshape(rows, n)
    qty_m = quantity[: rows * n].reshape(rows, n)
    sign_m = sign[: rows * n].reshape(rows, n)
    gross = qty_m.sum(axis=1)
    signed = (qty_m * sign_m).sum(axis=1)
    sign_sum = sign_m.sum(axis=1)
    return pl.DataFrame({
        "n_events": np.full(rows, n, dtype=np.int32),
        "start_time_ms": time_m[:, 0],
        "end_time_ms": time_m[:, -1],
        "duration_seconds": (time_m[:, -1] - time_m[:, 0]) / 1_000.0,
        "start_price": price_m[:, 0],
        "end_price": price_m[:, -1],
        "gross_volume": gross,
        "signed_volume": signed,
        "sign_sum": sign_sum.astype(np.int32),
        "sign_bias": sign_sum / n,
        "trade_imbalance": signed / gross,
        "log_return": np.log(price_m[:, -1] / price_m[:, 0]),
        "price_change_fraction": (np.diff(price_m, axis=1) != 0).mean(axis=1) if n > 1 else np.full(rows, np.nan),
        "endpoint_price_changed": price_m[:, -1] != price_m[:, 0],
        "spans_data_gap": spans_data_gap(time_m[:, 0], time_m[:, -1]),
    })


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
    layout = DataLayout(root)
    carries = {n: None for n in N_VALUES}
    total_events = 0
    for month in iter_months(args.start_month, args.end_month):
        events = (pl.scan_parquet(layout.event_month(args.symbol, month))
                  .select("timestamp_ms", "first_trade_id", "last_price", "quantity", "aggressor_sign")
                  .sort("timestamp_ms", "first_trade_id").collect())
        t = events["timestamp_ms"].to_numpy()
        p = events["last_price"].to_numpy()
        q = events["quantity"].to_numpy()
        s = events["aggressor_sign"].to_numpy().astype(np.int8)
        total_events += len(t)
        for n in N_VALUES:
            if carries[n] is not None:
                ct, cp, cq, cs = carries[n]
                bt, bp, bq, bs = np.concatenate([ct, t]), np.concatenate([cp, p]), np.concatenate([cq, q]), np.concatenate([cs, s])
            else:
                bt, bp, bq, bs = t, p, q, s
            complete = (len(bt) // n) * n
            frame = complete_blocks(bt[:complete], bp[:complete], bq[:complete], bs[:complete], n)
            carries[n] = (bt[complete:], bp[complete:], bq[complete:], bs[complete:])
            target = output_path(root, args.symbol, n, month)
            target.parent.mkdir(parents=True, exist_ok=True)
            frame.write_parquet(target, compression="zstd")
        print(f"{month}: {len(t):,} events", flush=True)
    print(f"done: {total_events:,} events", flush=True)


if __name__ == "__main__":
    main()
