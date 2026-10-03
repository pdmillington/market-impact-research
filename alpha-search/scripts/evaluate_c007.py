"""C-007 evaluation: stress reversion (H-007 v1 frozen) on ETHUSDT and SOLUSDT.

Registered rules (HYPOTHESIS_LEDGER.md, C-006/C-007):
- co-primary rates 100 and 200 episodes/yr, trigger entry, 5-minute hold;
- condition 1: mean gross >= T(5 min) = 2 bps, net of 4 bps > 0, and Holm across
  the two rates on one-sided p from the day-block t (alpha 0.05);
- condition 2: ordinary-day gross (best 10 days removed) >= 2 bps;
- condition 3: positive mean in at least 3 of the 4 calendar years 2023-2026;
- a rate passes if 1-3 hold; an instrument passes if at least one rate passes;
  C-007 is confirmed only if both instruments pass.
Reversion is measured as a simple return, -d (P1/P0 - 1); the pipeline's log
version is reported alongside. BTC (H-007 v1 outputs) is evaluated the same
way for reference only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import norm


REPO = Path(__file__).resolve().parents[2]
REPORTS = REPO / "alpha-search" / "reports"
TEST_START = 1_672_531_200            # 2023-01-01 UTC, seconds
TEST_END = 1_780_272_000              # 2026-06-01 UTC
THRESHOLD_BPS, ROUND_TRIP_BPS = 2.0, 4.0
PRIMARY_RATES = (100, 200)


def load(directory: Path) -> pl.DataFrame:
    trades = pl.read_parquet(directory / "trades.parquet")
    episodes = pl.read_parquet(directory / "episodes.parquet").select("trigger_second", "target_per_year", "direction")
    frame = trades.join(episodes, on=["trigger_second", "target_per_year"], how="left")
    # gross_bps = -d * 1e4 * ln(P1/P0)  =>  ln(P1/P0) = -d * gross / 1e4  (d = +-1)
    log_move = -pl.col("direction") * pl.col("gross_bps") / 1e4
    return frame.with_columns(
        (-pl.col("direction") * (log_move.exp() - 1.0) * 1e4).alias("simple_bps"),
        pl.from_epoch("trigger_second").alias("time"),
    ).with_columns(pl.col("time").dt.date().alias("day"), pl.col("time").dt.year().alias("year"))


def cell(frame: pl.DataFrame, rate: int, rule: str, hold: int, column: str = "simple_bps") -> dict:
    g = frame.filter((pl.col("target_per_year") == rate) & (pl.col("rule") == rule) & (pl.col("holding_seconds") == hold)
                     & (pl.col("trigger_second") >= TEST_START) & (pl.col("trigger_second") < TEST_END)
                     & pl.col(column).is_finite())
    if g.height < 3:
        return {"trades": g.height}
    daily = g.group_by("day").agg(pl.col(column).sum().alias("s"), pl.len().alias("n")).sort("s", descending=True)
    per_day = (daily["s"] / daily["n"]).to_numpy()
    t = float(per_day.mean() / (per_day.std(ddof=1) / np.sqrt(len(per_day)))) if len(per_day) > 2 else float("nan")
    ordinary = daily.tail(max(daily.height - 10, 0))
    years = {int(y): float(v) for y, v in g.group_by("year").agg(pl.col(column).mean()).sort("year").iter_rows()}
    mean = float(g[column].mean())
    return {"trades": g.height, "days": daily.height, "gross_bps": mean, "net_bps": mean - ROUND_TRIP_BPS,
            "t": t, "p_one_sided": float(1 - norm.cdf(t)),
            "ordinary_gross_bps": float(ordinary["s"].sum() / ordinary["n"].sum()) if ordinary.height else float("nan"),
            "best10_share": float(daily.head(10)["s"].sum() / daily["s"].sum()) if daily["s"].sum() else float("nan"),
            "by_year": years, "years_positive": int(sum(v > 0 for y, v in years.items() if 2023 <= y <= 2026))}


def evaluate(frame: pl.DataFrame) -> dict:
    primaries = {rate: cell(frame, rate, "trigger", 300) for rate in PRIMARY_RATES}
    order = sorted(PRIMARY_RATES, key=lambda r: primaries[r]["p_one_sided"])
    holm = {order[0]: primaries[order[0]]["p_one_sided"] <= 0.05 / 2,
            order[1]: primaries[order[0]]["p_one_sided"] <= 0.05 / 2 and primaries[order[1]]["p_one_sided"] <= 0.05}
    rates = {}
    for rate, c in primaries.items():
        conditions = {
            "1_gross_net_holm": c["gross_bps"] >= THRESHOLD_BPS and c["net_bps"] > 0 and holm[rate],
            "2_ordinary": c["ordinary_gross_bps"] >= THRESHOLD_BPS,
            "3_years": c["years_positive"] >= 3,
        }
        rates[rate] = {**c, "conditions": conditions, "pass": all(conditions.values()),
                       "log_gross_bps": cell(frame, rate, "trigger", 300, "gross_bps").get("gross_bps")}
    reported = {f"{rule}_{rate}_hold{hold // 60}m": cell(frame, rate, rule, hold)
                for rate in (50, 100, 200) for rule, hold in (("trigger", 900), ("delay_10s", 300), ("trigger", 300))
                if not (rule == "trigger" and hold == 300 and rate in PRIMARY_RATES)}
    return {"primaries": rates, "instrument_pass": any(r["pass"] for r in rates.values()), "reported": reported}


def main() -> None:
    out = REPORTS / "c007_evaluation"
    out.mkdir(parents=True, exist_ok=True)
    results = {s: evaluate(load(REPORTS / f"stress_c007_{s}")) for s in ("ETHUSDT", "SOLUSDT")}
    results["BTCUSDT_reference"] = evaluate(load(REPORTS / "stress_episodes_v1"))
    results["confirmed"] = results["ETHUSDT"]["instrument_pass"] and results["SOLUSDT"]["instrument_pass"]
    (out / "results.json").write_text(json.dumps(results, indent=2, default=str))
    for s in ("ETHUSDT", "SOLUSDT", "BTCUSDT_reference"):
        for rate, r in results[s]["primaries"].items():
            print(f"{s:18s} rate {rate}: trades {r['trades']:4d} gross {r['gross_bps']:6.2f} (log {r['log_gross_bps']:6.2f}) "
                  f"ordinary {r['ordinary_gross_bps']:6.2f} t {r['t']:5.2f} p {r['p_one_sided']:.4f} years+ {r['years_positive']} "
                  f"{ {y: round(v, 1) for y, v in r['by_year'].items()} } -> {r['conditions']} pass={r['pass']}")
        print(f"{s} instrument pass: {results[s]['instrument_pass']}")
    print("C-007 confirmed:", results["confirmed"])


if __name__ == "__main__":
    main()
