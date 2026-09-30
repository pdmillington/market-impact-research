"""H-017 signal-protected passive quoting, run as registered.

Phases:
- calibrate: S0 and the 12-point S1 grid (D bps x T s x k) on 2023-05-16..2023-07-31;
  S1 is chosen per symbol by mean calibration net P&L per day and frozen to
  `selection_<symbol>.json` before any test day is simulated.
- test: S0 and the frozen S1 on 2023-08-01..2024-03-31, plus the registered
  sensitivities (latency 250 ms, optimistic (pro-rata cancellation) queue, inventory +-2), for S0 and S1.

Daily rows are written per phase; `evaluate_quoting_h017.py` computes the
registered metrics and pass conditions.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import polars as pl


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "alpha-search" / "src"))
sys.path.insert(0, str(REPO / "alpha-search" / "scripts"))

from alpha_search.simulator import DAY_MS, daily_pnl, fill_markouts, load_day, pull_times, run_quoting  # noqa: E402
from build_quote_triggers import output_path as trigger_path  # noqa: E402


FROZEN = {
    "unit_notional_usd": 20.0,      # minimum-notional unit (Binance USD-M minimum order notional)
    "inventory_limit": 5,
    "latency_ms": 100.0,
    "maker_fee_bps": 2.0,
    "queue_model": "conservative",
    "calibration": ("2023-05-16", "2023-07-31"),
    "test": ("2023-08-01", "2024-03-31"),
    "grid": {"D_bps": [2, 5, 10], "T_s": [2, 5], "k": [3, 6]},
}
SENSITIVITIES = {
    "latency_250": {"latency_ms": 250.0},
    "optimistic_queue": {"queue_model": "proportional"},
    "inventory_2": {"inventory_limit": 2},
}


def utc_ms(day: str) -> int:
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1_000)


def iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1_000, tz=timezone.utc).date().isoformat()


def day_triggers(root: Path, symbol: str, start: int) -> pl.DataFrame:
    month = datetime.fromtimestamp(start / 1_000, tz=timezone.utc).strftime("%Y-%m")
    return (pl.scan_parquet(trigger_path(root, symbol, month))
            .filter((pl.col("time_ms") >= start - 10_000) & (pl.col("time_ms") < start + DAY_MS))
            .collect())


def funding_table(root: Path, symbol: str) -> tuple[np.ndarray, np.ndarray]:
    table = pl.read_parquet(root / "canonical" / "funding" / f"symbol={symbol}" / "all" / "part-000.parquet").sort("time_ms")
    return table["time_ms"].to_numpy().astype(np.float64), table["funding_rate"].to_numpy()


def simulate_day(task: tuple) -> list[dict]:
    root, symbol, start, runs = task
    day = load_day(root, symbol, start)
    if day is None:
        return []
    triggers = day_triggers(root, symbol, start)
    funding = funding_table(root, symbol)
    mid = float(np.nanmedian(0.5 * (day.bid + day.ask)))
    rows = []
    for run in runs:
        pulls = None
        if run["strategy"] == "S1":
            pulls = pull_times(triggers, sweep_bps=run["D_bps"], min_hits=run["k"], burst=True)
        result = run_quoting(day, unit_notional=run["unit_notional_usd"], inventory_limit=run["inventory_limit"],
                             latency=run["latency_ms"], pulls=pulls, pull_seconds=run.get("T_s", 0.0),
                             queue_model=run["queue_model"])
        pnl = daily_pnl(day, result.fills, maker_fee_bps=run["maker_fee_bps"], funding=funding,
                        unit_notional=run["unit_notional_usd"])
        row = {"symbol": symbol, "day": iso(start), "run": run["name"], "strategy": run["strategy"],
               "tick_bps": 1e4 * day.tick / mid, "mid": mid,
               "quoting_share_bid": result.time_quoting_share[1], "quoting_share_ask": result.time_quoting_share[-1],
               **pnl, "net_per_fill_bps": pnl["pnl_usd"] / max(pnl["fills"], 1) / run["unit_notional_usd"] * 1e4}
        fills = fill_markouts(day, result.fills)
        if fills.height:
            for column in ("spread_capture_bps", "markout_1s_bps", "markout_5s_bps", "markout_60s_bps"):
                row[column] = float(fills[column].mean())
            row["through_share"] = float((fills["reason"] == "fill_through").mean())
        rows.append(row)
    return rows


def run_specs(phase: str, selection: dict | None) -> list[dict]:
    base = {k: FROZEN[k] for k in ("unit_notional_usd", "inventory_limit", "latency_ms", "maker_fee_bps", "queue_model")}
    if phase == "calibrate":
        runs = [{**base, "name": "S0", "strategy": "S0"}]
        grid = FROZEN["grid"]
        for d, t, k in itertools.product(grid["D_bps"], grid["T_s"], grid["k"]):
            runs.append({**base, "name": f"S1_D{d}_T{t}_k{k}", "strategy": "S1", "D_bps": d, "T_s": t, "k": k})
        return runs
    s1 = {"strategy": "S1", "D_bps": selection["D_bps"], "T_s": selection["T_s"], "k": selection["k"]}
    runs = [{**base, "name": "S0", "strategy": "S0"}, {**base, **s1, "name": "S1"}]
    for label, override in SENSITIVITIES.items():
        runs.append({**base, **override, "name": f"S0_{label}", "strategy": "S0"})
        runs.append({**base, **s1, **override, "name": f"S1_{label}"})
    return runs


def select(calibration: pl.DataFrame, symbol: str) -> dict:
    table = (calibration.filter((pl.col("symbol") == symbol) & (pl.col("strategy") == "S1"))
             .filter(pl.col("pnl_bps_of_unit").is_finite())
             .group_by("run").agg(pl.col("pnl_bps_of_unit").mean().alias("mean_pnl_bps_per_day"), pl.len().alias("days"))
             .sort("mean_pnl_bps_per_day", descending=True))
    if table.height != 12 or table["days"].n_unique() != 1:
        raise SystemExit(f"incomplete calibration grid for {symbol}: {table}")
    best = table.row(0, named=True)
    _, d, t, k = best["run"].split("_")
    return {"symbol": symbol, "run": best["run"], "D_bps": int(d[1:]), "T_s": int(t[1:]), "k": int(k[1:]),
            "calibration_mean_pnl_bps_per_day": best["mean_pnl_bps_per_day"], "calibration_days": best["days"],
            "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "grid_ranking": table.to_dicts()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["calibrate", "test"])
    parser.add_argument("--symbols", nargs="+", default=["SOLUSDT", "ETHUSDT"])
    parser.add_argument("--shared-data-root", type=Path, default=REPO / "shared-data")
    parser.add_argument("--output-dir", type=Path, default=REPO / "alpha-search" / "reports" / "quoting_h017")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    first, last = FROZEN["calibration"] if args.phase == "calibrate" else FROZEN["test"]
    days = list(range(utc_ms(first), utc_ms(last) + DAY_MS, DAY_MS))

    for symbol in args.symbols:
        target = out / f"{args.phase}_daily_{symbol}.csv"
        selection = None
        if args.phase == "test":
            selection = json.loads((out / f"selection_{symbol}.json").read_text())
        runs = run_specs(args.phase, selection)
        rows = []
        tasks = [(args.shared_data_root, symbol, start, runs) for start in days]
        with ProcessPoolExecutor(args.workers) as pool:
            for result in pool.map(simulate_day, tasks):
                rows.extend(result)
                if result:
                    print(f"{symbol} {result[0]['day']} S0 {result[0]['pnl_bps_of_unit']:.1f} bps/unit", flush=True)
        if not rows:
            print(f"{symbol}: no simulated days (quotes missing); skipped", flush=True)
            continue
        frame = pl.DataFrame(rows, infer_schema_length=None)
        if frame["pnl_bps_of_unit"].is_nan().any():
            raise SystemExit(f"{symbol}: NaN daily P&L rows; fix before selecting")
        frame.write_csv(target)
        if args.phase == "calibrate":
            selection = select(frame, symbol)
            path = out / f"selection_{symbol}.json"
            if path.exists():
                raise SystemExit(f"{path} already frozen; refusing to overwrite")
            path.write_text(json.dumps(selection, indent=2))
            print(json.dumps({k: v for k, v in selection.items() if k != "grid_ranking"}), flush=True)
        (out / f"run_manifest_{args.phase}_{symbol}.json").write_text(json.dumps(
            {"frozen": FROZEN, "sensitivities": SENSITIVITIES, "runs": runs, "days_requested": len(days),
             "days_simulated": frame["day"].n_unique() if frame.height else 0}, indent=2, default=str))


if __name__ == "__main__":
    main()
