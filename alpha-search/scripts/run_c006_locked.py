"""C-006 locked-month test (2026-06 to 2026-08), single use, reject-only.

Rule fixed in HYPOTHESIS_LEDGER.md before any locked bar or return was built or
read. Reuses run_c006.py unchanged except for the dates and the cache root, and
computes only the frozen primary cell (return_pretrend, 1,000 events, top 1%,
5 minutes).

Reject if (a) the pooled mean gross edge (equal weight per instrument-day) is below
2 bps, or (b) fewer than 2 of the 3 instruments have a positive mean gross edge.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import polars as pl


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "alpha-search" / "scripts"))

import run_c006 as c  # noqa: E402


c.CONFIG = {**c.CONFIG, "first_feature_month": "2025-05", "first_test_month": "2026-06",
            "last_test_month": "2026-08", "data_end_exclusive": "2026-09-01"}
c.MODELS = {"return_pretrend": c.CONFIG["models"]["return_pretrend"]}
c.HORIZONS = (5,)
c.TAILS = (0.01,)
OUT = REPO / "alpha-search" / "reports" / "c006_locked"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    per_symbol, daily_rows, all_trades = {}, [], []
    for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        cache = REPO / "shared-data" / "features" / "alpha" / f"symbol={symbol}" / "impact_research_v2_locked"
        clock_time, clock_price = c.load_price_clock(cache, c.month_ms("2026-09"))
        data = c.features(cache, 1000, clock_time, clock_price)
        t = c.trades(data, 1000)
        t = t.filter(pl.col("month") >= "2026-06").with_columns(pl.lit(symbol).alias("symbol"))
        all_trades.append(t)
        s = c.score(t)
        per_symbol[symbol] = s
        daily = t.group_by("day").agg(pl.col("edge_simple_bps").mean().alias("mean_edge")).with_columns(pl.lit(symbol).alias("symbol"))
        daily_rows.append(daily)
        print(symbol, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in s.items() if k != "by_year"}, flush=True)
    days = pl.concat(daily_rows)
    v = days["mean_edge"].to_numpy()
    pooled_t = float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))))
    pooled = {"instrument_days": len(v), "mean_gross_bps": float(v.mean()), "net_bps": float(v.mean() - 4.0), "t": pooled_t}
    positive = sum(per_symbol[s]["gross_bps"] > 0 for s in per_symbol)
    reject_a = pooled["mean_gross_bps"] < 2.0
    reject_b = positive < 2
    result = {"rule": "reject if pooled mean gross < 2 bps, or fewer than 2 of 3 instruments positive",
              "per_symbol": per_symbol, "pooled": pooled, "instruments_positive": positive,
              "reject_a_pooled_below_threshold": reject_a, "reject_b_breadth": reject_b,
              "verdict": "REJECTED" if (reject_a or reject_b) else "NOT REJECTED"}
    (OUT / "results.json").write_text(json.dumps(result, indent=2, default=str))
    pl.concat(all_trades).write_parquet(OUT / "trades.parquet")
    days.write_parquet(OUT / "daily.parquet")
    print(json.dumps({k: v for k, v in result.items() if k != "per_symbol"}, indent=2, default=str))


if __name__ == "__main__":
    main()
