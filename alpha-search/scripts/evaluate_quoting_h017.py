"""Registered H-017 metrics and pass conditions from `test_daily_<symbol>.csv`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import polars.selectors  # noqa: F401


REPO = Path(__file__).resolve().parents[2]


def t_stat(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(np.mean(x) / (np.std(x, ddof=1) / np.sqrt(len(x)))) if len(x) > 2 and np.std(x) > 0 else float("nan")


def trimmed(x: np.ndarray, n: int = 10) -> float:
    order = np.sort(x)
    return float(order[n:-n].mean()) if len(order) > 2 * n else float("nan")


def max_drawdown(x: np.ndarray) -> float:
    path = np.cumsum(x)
    return float(np.max(np.maximum.accumulate(path) - path)) if len(path) else float("nan")


def summarise(daily: pl.DataFrame, run: str) -> dict:
    frame = daily.filter(pl.col("run") == run).sort("day")
    pnl = frame["pnl_bps_of_unit"].to_numpy()
    monthly = frame.group_by(pl.col("day").str.slice(0, 7).alias("month")).agg(pl.col("pnl_bps_of_unit").mean())
    return {
        "run": run, "days": len(pnl),
        "pnl_per_day_bps_of_unit": float(pnl.mean()), "t_daily": t_stat(pnl),
        "months_positive_share": float((monthly["pnl_bps_of_unit"] > 0).mean()),
        "pnl_ex_best_worst_10": trimmed(pnl),
        "sharpe_daily_ann": float(pnl.mean() / pnl.std(ddof=1) * np.sqrt(365)) if pnl.std() > 0 else float("nan"),
        "max_drawdown_bps_of_unit": max_drawdown(pnl),
        "fills_per_day": float(frame["fills"].mean()),
        "quoting_share": float(((frame["quoting_share_bid"] + frame["quoting_share_ask"]) / 2).mean()),
        "net_per_fill_bps": float(frame["pnl_usd"].sum() / frame["fills"].sum() / 20.0 * 1e4),
        "spread_per_day": float(frame["spread_usd"].mean() / 20.0 * 1e4),
        "mark_per_day": float(frame["mark_usd"].mean() / 20.0 * 1e4),
        "fees_per_day": float(frame["fees_usd"].mean() / 20.0 * 1e4),
        "funding_per_day": float(frame["funding_usd"].mean() / 20.0 * 1e4),
        "markout_1s_bps": float(frame["markout_1s_bps"].mean()),
        "markout_5s_bps": float(frame["markout_5s_bps"].mean()),
        "markout_60s_bps": float(frame["markout_60s_bps"].mean()),
        "spread_capture_bps": float(frame["spread_capture_bps"].mean()),
    }


def paired(daily: pl.DataFrame, a: str, b: str) -> dict:
    wide = (daily.filter(pl.col("run").is_in([a, b])).pivot(on="run", index="day", values="pnl_bps_of_unit")
            .drop_nulls().sort("day"))
    diff = (wide[a] - wide[b]).to_numpy()
    monthly = (wide.with_columns((pl.col(a) - pl.col(b)).alias("d"))
               .group_by(pl.col("day").str.slice(0, 7)).agg(pl.col("d").mean()))
    return {"diff_per_day": float(diff.mean()), "t_daily": t_stat(diff),
            "months_positive_share": float((monthly["d"] > 0).mean())}


def break_even_tick(daily: pl.DataFrame, run: str, fee: float = 2.0) -> float:
    """Tick (bps) at which a fill capturing half a tick breaks even, holding the
    measured adverse selection (60 s markout minus spread capture) fixed."""

    frame = daily.filter(pl.col("run") == run)
    adverse = float((frame["markout_60s_bps"] - frame["spread_capture_bps"]).mean())
    return 2.0 * (fee - adverse)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=["SOLUSDT", "ETHUSDT"])
    parser.add_argument("--phase", default="test")
    parser.add_argument("--dir", type=Path, default=REPO / "alpha-search" / "reports" / "quoting_h017")
    args = parser.parse_args()
    report = {}
    for symbol in args.symbols:
        path = args.dir / f"{args.phase}_daily_{symbol}.csv"
        if not path.exists():
            continue
        daily = pl.read_csv(path).with_columns(pl.selectors.float().fill_nan(None))
        runs = sorted(daily["run"].unique().to_list())
        summary = [summarise(daily, r) for r in runs]
        result = {"summary": summary, "break_even_tick_bps": {r: break_even_tick(daily, r) for r in runs}}
        if "S1" in runs:
            c1 = paired(daily, "S1", "S0")
            s1 = next(s for s in summary if s["run"] == "S1")
            result["condition_1"] = {**c1, "pass": c1["diff_per_day"] > 0 and c1["t_daily"] > 2 and c1["months_positive_share"] > 0.6}
            result["condition_2"] = {"pass": s1["pnl_per_day_bps_of_unit"] > 0 and s1["t_daily"] > 2
                                     and s1["months_positive_share"] > 0.6 and s1["pnl_ex_best_worst_10"] > 0}
            for label in ("latency_250", "optimistic_queue", "inventory_2"):
                if f"S1_{label}" in runs:
                    result[f"condition_1_{label}"] = paired(daily, f"S1_{label}", f"S0_{label}")
        result["monthly"] = (daily.group_by(pl.col("day").str.slice(0, 7).alias("month"), "run")
                             .agg(pl.col("pnl_bps_of_unit").mean(), pl.col("tick_bps").mean(), pl.col("fills").mean(),
                                  pl.col("net_per_fill_bps").mean())
                             .sort("month", "run").to_dicts())
        report[symbol] = result
        pl.DataFrame(summary).write_csv(args.dir / f"{args.phase}_summary_{symbol}.csv")
    (args.dir / f"{args.phase}_evaluation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({s: {k: v for k, v in r.items() if k != "monthly"} for s, r in report.items()}, indent=2))


if __name__ == "__main__":
    main()
