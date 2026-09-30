"""Fill-model validation on H-017/H-018 calibration days (2023-05-16 to 2023-07-31).

Runs the naive quoter (S0) and summarises fills, queue waits, spread capture and
markouts per day; measures how much of each day the registered pull triggers
would keep each side off the book for every (D, k, T) in the H-017 grid. No
test month is touched.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import polars as pl


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "alpha-search" / "src"))
sys.path.insert(0, str(REPO / "alpha-search" / "scripts"))

from alpha_search.simulator import DAY_MS, fill_markouts, load_day, pull_times, pulled_share, run_quoting  # noqa: E402
from build_quote_triggers import output_path as trigger_path  # noqa: E402


def utc_ms(day: str) -> int:
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1_000)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-data-root", type=Path, default=REPO / "shared-data")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "SOLUSDT"])
    parser.add_argument("--start", default="2023-05-16")
    parser.add_argument("--end", default="2023-07-31")
    parser.add_argument("--output-dir", type=Path, default=REPO / "alpha-search" / "reports" / "simulator_validation")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    root = args.shared_data_root
    days = range(utc_ms(args.start), utc_ms(args.end) + DAY_MS, DAY_MS)

    daily, coverage = [], []
    for symbol in args.symbols:
        month_cache: dict[str, pl.DataFrame] = {}
        for start in days:
            day = load_day(root, symbol, start)
            if day is None:
                continue
            result = run_quoting(day, unit_notional=1_000.0, inventory_limit=5, latency=100.0)
            fills = fill_markouts(day, result.fills)
            mid = float(np.nanmedian(0.5 * (day.bid + day.ask)))
            row = {"symbol": symbol, "day": datetime.fromtimestamp(start / 1_000, tz=timezone.utc).date().isoformat(),
                   "tick_bps": 1e4 * day.tick / mid, "episodes": result.episodes, "fills": fills.height,
                   "quoting_share_bid": result.time_quoting_share[1], "quoting_share_ask": result.time_quoting_share[-1]}
            if fills.height:
                row.update({
                    "through_share": float((fills["reason"] == "fill_through").mean()),
                    "median_wait_s": float(fills["wait_ms"].median() / 1_000),
                    "spread_capture_bps": float(fills["spread_capture_bps"].mean()),
                    "markout_1s_bps": float(fills["markout_1s_bps"].mean()),
                    "markout_5s_bps": float(fills["markout_5s_bps"].mean()),
                    "markout_60s_bps": float(fills["markout_60s_bps"].mean()),
                    "net_per_fill_60s_bps": float(fills["markout_60s_bps"].mean() - 2.0),
                    "queue_ahead_median_notional": float(fills["queue_ahead"].median() * mid),
                })
            daily.append(row)

            month = datetime.fromtimestamp(start / 1_000, tz=timezone.utc).strftime("%Y-%m")
            if month not in month_cache:
                month_cache[month] = pl.read_parquet(trigger_path(root, symbol, month))
            triggers = month_cache[month].filter((pl.col("time_ms") >= start) & (pl.col("time_ms") < start + DAY_MS))
            rules = {"sweep_only": dict(burst=False, min_hits=None), "level_only": dict(burst=False, sweep_bps=None),
                     "burst_only": dict(burst=True, sweep_bps=None, min_hits=None), "all": dict(burst=True)}
            for d in (2, 5, 10):
                for k in (3, 6):
                    for rule, kwargs in rules.items():
                        params = {"sweep_bps": d, "min_hits": k, **kwargs}
                        pulls = pull_times(triggers, sweep_bps=params.get("sweep_bps"),
                                           min_hits=params.get("min_hits"), burst=params["burst"])
                        for pause in (2, 5):
                            coverage.append({
                                "symbol": symbol, "day": row["day"], "D_bps": d, "k": k, "T_s": pause, "rule": rule,
                                "pulled_share_bid": pulled_share(pulls[1], pause * 1_000.0, start, start + DAY_MS),
                                "pulled_share_ask": pulled_share(pulls[-1], pause * 1_000.0, start, start + DAY_MS),
                            })
            print(f"{symbol} {row['day']} fills {row['fills']}", flush=True)
    pl.DataFrame(daily, infer_schema_length=None).write_csv(out / "s0_daily.csv")
    pl.DataFrame(coverage).write_csv(out / "trigger_coverage.csv")
    print(out)


if __name__ == "__main__":
    main()
