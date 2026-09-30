"""H-018 signal-timed execution, run as registered (with the pre-test amendment).

Phases:
- calibrate: E0, E1 and the 12-point E2 grid (m ticks x T s x k) on
  2023-05-16..2023-07-31; E2 is chosen per instrument by lowest mean IS and
  frozen to `selection_<symbol>.json` before any test day is simulated.
- test: E0, E1 and the frozen E2 on 2023-08-01..2024-03-31, plus the registered
  sensitivities (latency 250 ms, optimistic (pro-rata cancellation) queue, $50k, W = 60 min).

Parent orders: one per 2-hour block at a uniform random time, fair-coin side;
the random stream is keyed by (seed, day) so it is identical across
strategies, instruments and runs. E3 (secondary) is not built here.
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

from alpha_search.simulator import DAY_MS, execute_parent, execution_triggers, load_day, shortfall  # noqa: E402
from build_quote_triggers import output_path as trigger_path  # noqa: E402


SEED = 20260930
FROZEN = {
    "seed": SEED,
    "blocks_per_day": 12,
    "notional_usd": 10_000.0,
    "window_min": 30.0,
    "latency_ms": 100.0,
    "queue_model": "conservative",
    "maker_fee_bps": 2.0,
    "taker_fee_bps": 5.0,
    "calibration": ("2023-05-16", "2023-07-31"),
    "test": ("2023-08-01", "2024-03-31"),
    "grid": {"m_ticks": [2, 5, 10], "T_s": [2, 5], "k": [6, 12]},
}
SENSITIVITIES = {
    "latency_250": {"latency_ms": 250.0},
    "optimistic_queue": {"queue_model": "proportional"},
    "size_50k": {"notional_usd": 50_000.0},
    "window_60": {"window_min": 60.0},
}
MAX_WINDOW_MS = 60 * 60_000


def utc_ms(day: str) -> int:
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1_000)


def month_of(ms: float) -> str:
    return datetime.fromtimestamp(ms / 1_000, tz=timezone.utc).strftime("%Y-%m")


def parent_orders(day_start: int) -> list[tuple[float, int]]:
    rng = np.random.default_rng([SEED, day_start // DAY_MS])
    block = DAY_MS / FROZEN["blocks_per_day"]
    offsets = rng.uniform(0.0, block, FROZEN["blocks_per_day"])
    sides = np.where(rng.uniform(size=FROZEN["blocks_per_day"]) < 0.5, 1, -1)
    return [(float(day_start + b * block + offsets[b]), int(sides[b])) for b in range(FROZEN["blocks_per_day"])]


def load_triggers(root: Path, symbol: str, start: int, end: int) -> pl.DataFrame:
    frames = []
    for month in sorted({month_of(start), month_of(end - 1)}):
        path = trigger_path(root, symbol, month)
        if path.exists():
            frames.append(pl.scan_parquet(path).filter((pl.col("time_ms") >= start) & (pl.col("time_ms") < end)).collect())
    return pl.concat(frames) if frames else pl.DataFrame()


def simulate_day(task: tuple) -> list[dict]:
    root, symbol, start, runs = task
    day = load_day(root, symbol, start, tail_ms=MAX_WINDOW_MS + 300_000)
    if day is None:
        return []
    triggers = load_triggers(root, symbol, start - 10_000, start + DAY_MS + MAX_WINDOW_MS + 300_000)
    mid = float(np.nanmedian(0.5 * (day.bid + day.ask)))
    cache: dict = {}
    rows = []
    for order, (arrival, direction) in enumerate(parent_orders(start)):
        if day.quote_index(arrival) < 0:
            continue
        for run in runs:
            runaway = adverse = None
            if run["strategy"] == "E2":
                key = (direction, run["m_ticks"], run["k"])
                if key not in cache:
                    cache[key] = execution_triggers(triggers, direction, day.tick, sweep_ticks=run["m_ticks"], min_hits=run["k"])
                runaway, adverse = cache[key]
            size = run["notional_usd"] / mid
            execution = execute_parent(day, direction, arrival, size, run["window_min"] * 60_000, run["latency_ms"],
                                       strategy=run["strategy"], runaway=runaway, adverse=adverse,
                                       pull_seconds=run.get("T_s", 0.0), queue_model=run["queue_model"])
            result = shortfall(day, direction, arrival, execution, maker_fee_bps=run["maker_fee_bps"],
                               taker_fee_bps=run["taker_fee_bps"])
            rows.append({"symbol": symbol, "day": datetime.fromtimestamp(start / 1_000, tz=timezone.utc).date().isoformat(),
                         "order": order, "arrival_ms": arrival, "direction": direction, "run": run["name"],
                         "strategy": run["strategy"], "tick_bps": 1e4 * day.tick / mid, **result})
    return rows


def run_specs(phase: str, selection: dict | None) -> list[dict]:
    base = {k: FROZEN[k] for k in ("notional_usd", "window_min", "latency_ms", "queue_model", "maker_fee_bps", "taker_fee_bps")}
    runs = [{**base, "name": "E0", "strategy": "E0"}, {**base, "name": "E1", "strategy": "E1"}]
    if phase == "calibrate":
        grid = FROZEN["grid"]
        for m, t, k in itertools.product(grid["m_ticks"], grid["T_s"], grid["k"]):
            runs.append({**base, "name": f"E2_m{m}_T{t}_k{k}", "strategy": "E2", "m_ticks": m, "T_s": t, "k": k})
        return runs
    e2 = {"strategy": "E2", "m_ticks": selection["m_ticks"], "T_s": selection["T_s"], "k": selection["k"]}
    runs.append({**base, **e2, "name": "E2"})
    for label, override in SENSITIVITIES.items():
        runs.append({**base, **override, "name": f"E0_{label}", "strategy": "E0"})
        runs.append({**base, **override, "name": f"E1_{label}", "strategy": "E1"})
        runs.append({**base, **e2, **override, "name": f"E2_{label}"})
    return runs


def select(calibration: pl.DataFrame, symbol: str) -> dict:
    table = (calibration.filter(pl.col("strategy") == "E2").group_by("run")
             .agg(pl.col("is_bps").mean().alias("mean_is_bps"), pl.len().alias("orders"))
             .sort("mean_is_bps"))
    if table.height != 12 or table["orders"].n_unique() != 1 or table["mean_is_bps"].is_nan().any():
        raise SystemExit(f"incomplete or invalid calibration grid for {symbol}: {table}")
    best = table.row(0, named=True)
    _, m, t, k = best["run"].split("_")
    return {"symbol": symbol, "run": best["run"], "m_ticks": int(m[1:]), "T_s": int(t[1:]), "k": int(k[1:]),
            "calibration_mean_is_bps": best["mean_is_bps"], "calibration_orders": best["orders"],
            "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "grid_ranking": table.to_dicts()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["calibrate", "test"])
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "SOLUSDT", "ETHUSDT"])
    parser.add_argument("--shared-data-root", type=Path, default=REPO / "shared-data")
    parser.add_argument("--output-dir", type=Path, default=REPO / "alpha-search" / "reports" / "execution_h018")
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    first, last = FROZEN["calibration"] if args.phase == "calibrate" else FROZEN["test"]
    days = list(range(utc_ms(first), utc_ms(last) + DAY_MS, DAY_MS))
    for symbol in args.symbols:
        selection = json.loads((out / f"selection_{symbol}.json").read_text()) if args.phase == "test" else None
        runs = run_specs(args.phase, selection)
        rows = []
        with ProcessPoolExecutor(args.workers) as pool:
            for result in pool.map(simulate_day, [(args.shared_data_root, symbol, s, runs) for s in days]):
                rows.extend(result)
                if result:
                    e0 = np.mean([r["is_bps"] for r in result if r["run"] == "E0"])
                    print(f"{symbol} {result[0]['day']} E0 IS {e0:.2f} bps", flush=True)
        if not rows:
            print(f"{symbol}: no simulated days (quotes missing); skipped", flush=True)
            continue
        frame = pl.DataFrame(rows, infer_schema_length=None)
        if frame["is_bps"].is_nan().any():
            raise SystemExit(f"{symbol}: NaN shortfall rows")
        expected = len(days)
        simulated = frame["day"].n_unique()
        frame.write_parquet(out / f"{args.phase}_orders_{symbol}.parquet")
        if args.phase == "calibrate":
            path = out / f"selection_{symbol}.json"
            if path.exists():
                raise SystemExit(f"{path} already frozen; refusing to overwrite")
            selection = select(frame, symbol)
            path.write_text(json.dumps(selection, indent=2))
            print(json.dumps({k: v for k, v in selection.items() if k != "grid_ranking"}), flush=True)
        (out / f"run_manifest_{args.phase}_{symbol}.json").write_text(json.dumps(
            {"frozen": FROZEN, "sensitivities": SENSITIVITIES, "runs": runs, "days_requested": expected,
             "days_simulated": simulated}, indent=2, default=str))


if __name__ == "__main__":
    main()
