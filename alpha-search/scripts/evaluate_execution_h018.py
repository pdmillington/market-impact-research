"""Registered H-018 metrics and pass conditions from `test_orders_<symbol>.parquet`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl


REPO = Path(__file__).resolve().parents[2]


def day_t(frame: pl.DataFrame, column: str) -> tuple[float, float]:
    daily = frame.group_by("day").agg(pl.col(column).mean())[column].to_numpy()
    return float(daily.mean()), float(daily.mean() / (daily.std(ddof=1) / np.sqrt(len(daily))))


def difference(orders: pl.DataFrame, a: str, b: str) -> dict:
    """Mean of IS(a) - IS(b) per order, day-clustered t, and months positive."""

    wide = orders.filter(pl.col("run").is_in([a, b])).pivot(on="run", index=["day", "order"], values="is_bps")
    wide = wide.with_columns((pl.col(a) - pl.col(b)).alias("d"))
    mean, t = day_t(wide, "d")
    months = wide.group_by(pl.col("day").str.slice(0, 7)).agg(pl.col("d").mean())
    return {"diff_bps": mean, "t_day": t, "months_positive_share": float((months["d"] > 0).mean())}


def summary(orders: pl.DataFrame) -> list[dict]:
    return (orders.group_by("run").agg(
        pl.col("is_bps").mean().alias("is_bps"), pl.col("is_bps").std().alias("is_sd"),
        pl.col("is_bps").quantile(0.95).alias("is_p95"), pl.col("is_bps").quantile(0.99).alias("is_p99"),
        pl.col("spread_bps").mean(), pl.col("drift_bps").mean(), pl.col("fee_bps").mean(),
        pl.col("passive").mean().alias("passive_share"), pl.col("time_to_fill_s").median().alias("median_time_s"),
        (pl.col("reason") == "runaway").mean().alias("runaway_share"), pl.col("excess_share").mean(),
        pl.len().alias("orders")).sort("run").to_dicts())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "SOLUSDT", "ETHUSDT"])
    parser.add_argument("--phase", default="test")
    parser.add_argument("--dir", type=Path, default=REPO / "alpha-search" / "reports" / "execution_h018")
    args = parser.parse_args()
    report = {}
    for symbol in args.symbols:
        path = args.dir / f"{args.phase}_orders_{symbol}.parquet"
        if not path.exists():
            continue
        orders = pl.read_parquet(path)
        runs = set(orders["run"].unique().to_list())
        result = {"summary": summary(orders)}
        c1 = difference(orders, "E0", "E1")
        c1["pass"] = c1["diff_bps"] > 0 and c1["t_day"] > 2
        result["condition_1_E0_minus_E1"] = c1
        if "E2" in runs:
            c2 = difference(orders, "E1", "E2")
            c2["pass"] = c2["diff_bps"] > 0 and c2["t_day"] > 2 and c2["months_positive_share"] > 0.6
            c2["significantly_negative"] = c2["t_day"] < -2
            result["condition_2_E1_minus_E2"] = c2
            p95 = {r["run"]: r["is_p95"] for r in result["summary"]}
            result["condition_3_p95_E2_minus_E1"] = {"diff_bps": p95["E2"] - p95["E1"], "pass": p95["E2"] - p95["E1"] <= 1.0}
            for label in ("latency_250", "optimistic_queue", "size_50k", "window_60"):
                if f"E2_{label}" in runs:
                    result[f"sensitivity_{label}"] = {"E0_minus_E1": difference(orders, f"E0_{label}", f"E1_{label}"),
                                                      "E1_minus_E2": difference(orders, f"E1_{label}", f"E2_{label}")}
        result["monthly"] = (orders.filter(pl.col("run").is_in(["E0", "E1", "E2"]))
                             .group_by(pl.col("day").str.slice(0, 7).alias("month"), "run")
                             .agg(pl.col("is_bps").mean(), pl.col("passive").mean(), pl.col("tick_bps").mean())
                             .sort("month", "run").to_dicts())
        report[symbol] = result
        pl.DataFrame(result["summary"]).write_csv(args.dir / f"{args.phase}_summary_{symbol}.csv")
    (args.dir / f"{args.phase}_evaluation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({s: {k: v for k, v in r.items() if k not in ("monthly", "summary")} for s, r in report.items()}, indent=2))


if __name__ == "__main__":
    main()
