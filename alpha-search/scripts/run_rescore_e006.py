"""E-006: re-score saved results under the revised objective (researcher, 2026-10-01).

Uses saved outputs only: no new data is read and no new return is examined.

Per-trade results are put on one footing:

- tau: holding horizon;
- gross mean return per trade (bps, mid to mid as saved by each experiment);
- gross on ordinary days (excluding the best 10 days where available);
- day-block t and sample period;
- threshold T(tau) = max(1.5, min(2 * sqrt(tau / 5 min), 10)) bps;
- net at the researcher's target fees: a 4 bp round trip (taker 2 + 2; or maker
  1 + 1 plus about 2 bp passive drift from H-018).

Book strategies (H-010, H-019, H-020) are summarised separately: annual return
and Sharpe gross and at 1 bp / 2 bp per side.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import polars as pl


REPO = Path(__file__).resolve().parents[2]
REPORTS = REPO / "alpha-search" / "reports"
OUT = REPORTS / "e006_rescore"
ROUND_TRIP_BPS = 4.0


def threshold(tau_minutes: float) -> float:
    return max(1.5, min(2.0 * np.sqrt(tau_minutes / 5.0), 10.0))


def h006() -> pl.DataFrame:
    d = pl.read_csv(REPORTS / "residual_decomposition_v1" / "tail_breakeven.csv")
    return d.select(
        pl.lit("H-006").alias("source"),
        pl.format("{} ev, {} top {}", "n_events", "model", "tail_fraction").alias("cell"),
        pl.lit("BTCUSDT").alias("instrument"), pl.lit("2023-01..2026-05").alias("period"),
        pl.col("horizon_minutes").cast(pl.Float64).alias("tau_minutes"),
        pl.col("mean_edge_immediate_bps").alias("gross_bps"),
        pl.col("edge_excluding_best_10_days_bps").alias("gross_ordinary_bps"),
        pl.col("daily_block_t").alias("t"), pl.col("trades_per_day"),
        pl.col("best_10_day_edge_share").alias("best10_share"),
        pl.concat_str([pl.lit("2023 "), pl.col("edge_2023_bps").round(1), pl.lit(" | 2024 "), pl.col("edge_2024_bps").round(1),
                       pl.lit(" | 2025 "), pl.col("edge_2025_bps").round(1), pl.lit(" | 2026 "), pl.col("edge_2026_bps").round(1)]).alias("by_year"))


def h012() -> pl.DataFrame:
    d = pl.read_csv(REPORTS / "intraday_flow_v1" / "economics.csv").filter(pl.col("measure") == "unconditional_gross")
    return d.select(
        pl.lit("H-012").alias("source"), pl.format("{} top {}", "cell", "tail_fraction").alias("cell"),
        pl.lit("BTCUSDT").alias("instrument"), pl.lit("2023-01..2026-05").alias("period"),
        pl.col("horizon_minutes").cast(pl.Float64).alias("tau_minutes"),
        pl.col("mean_bps").alias("gross_bps"), pl.col("ordinary_day_mean_bps").alias("gross_ordinary_bps"),
        pl.col("daily_block_t").alias("t"), pl.col("trades_per_day"), pl.col("best_10_day_share").alias("best10_share"),
        pl.concat_str([pl.lit("2023 "), pl.col("mean_2023_bps").round(1), pl.lit(" | 2024 "), pl.col("mean_2024_bps").round(1),
                       pl.lit(" | 2025 "), pl.col("mean_2025_bps").round(1), pl.lit(" | 2026 "), pl.col("mean_2026_bps").round(1)]).alias("by_year"))


def stress() -> pl.DataFrame:
    rows = []
    for version, source in (("v1", "H-007"), ("v2", "H-008")):
        trades = pl.read_parquet(REPORTS / f"stress_episodes_{version}" / "trades.parquet").with_columns(
            pl.from_epoch("trigger_second").alias("time"))
        trades = trades.with_columns(pl.col("time").dt.date().alias("day"), pl.col("time").dt.year().alias("year"))
        test = trades.filter(pl.col("year") >= 2023)
        for (rule, hold), g in test.group_by("rule", "holding_seconds"):
            daily = g.group_by("day").agg(pl.col("gross_bps").sum().alias("s"), pl.len().alias("n")).sort("s", descending=True)
            mean = g["gross_bps"].mean()
            per_day = (daily["s"] / daily["n"]).to_numpy()
            t = per_day.mean() / (per_day.std(ddof=1) / np.sqrt(len(per_day))) if len(per_day) > 2 else np.nan
            ordinary = daily.tail(max(daily.height - 10, 0))
            ordinary_mean = ordinary["s"].sum() / ordinary["n"].sum() if ordinary.height else np.nan
            years = g.group_by("year").agg(pl.col("gross_bps").mean()).sort("year")
            rows.append({"source": source, "cell": f"{rule}, hold {hold // 60} min", "instrument": "BTCUSDT",
                         "period": "2023..2026-05", "tau_minutes": hold / 60, "gross_bps": mean,
                         "gross_ordinary_bps": ordinary_mean, "t": t, "trades_per_day": g.height / 1_247,
                         "best10_share": float(daily.head(10)["s"].sum() / daily["s"].sum()) if daily["s"].sum() else np.nan,
                         "by_year": " | ".join(f"{y} {v:.1f}" for y, v in years.iter_rows())})
    return pl.DataFrame(rows)


def h009() -> pl.DataFrame:
    d = pl.read_csv(REPORTS / "funding_positioning_v1" / "tail_breakeven.csv")
    return d.select(
        pl.lit("H-009").alias("source"), pl.format("{} top {}", "model", "tail_fraction").alias("cell"),
        pl.lit("BTCUSDT").alias("instrument"), pl.lit("2023-01..2026-05").alias("period"),
        (pl.col("horizon_hours") * 60.0).alias("tau_minutes"),
        (pl.col("mean_price_edge_bps") + pl.col("mean_carry_edge_bps")).alias("gross_bps"),
        pl.col("edge_excluding_best_10_days_bps").alias("gross_ordinary_bps"),
        pl.col("daily_block_t").alias("t"), pl.col("trades_per_day"), pl.col("best_10_day_share").alias("best10_share"),
        pl.concat_str([pl.lit("2023 "), pl.col("edge_2023_bps").round(1), pl.lit(" | 2024 "), pl.col("edge_2024_bps").round(1),
                       pl.lit(" | 2025 "), pl.col("edge_2025_bps").round(1), pl.lit(" | 2026 "), pl.col("edge_2026_bps").round(1)]).alias("by_year"))


def h003() -> pl.DataFrame:
    d = pl.read_csv(REPORTS / "excess_impact_v1" / "signal_summary.csv")
    return d.select(
        pl.lit("H-003").alias("source"), pl.format("{} {} {}", "construction", "side", "strategy").alias("cell"),
        pl.lit("BTCUSDT").alias("instrument"),
        pl.when(pl.col("period") == "validation").then(pl.lit("2020-09..2021-06")).otherwise(pl.lit("2021-07..2021-12")).alias("period"),
        pl.col("horizon_minutes").cast(pl.Float64).alias("tau_minutes"),
        pl.col("mean_return_bps").alias("gross_bps"), pl.lit(None, dtype=pl.Float64).alias("gross_ordinary_bps"),
        pl.col("day_t_stat").alias("t"), pl.lit(None, dtype=pl.Float64).alias("trades_per_day"),
        pl.lit(None, dtype=pl.Float64).alias("best10_share"), pl.lit("overlapping signals; no costs").alias("by_year"))


def h004() -> pl.DataFrame:
    d = (pl.read_csv(REPORTS / "impact_surface_residual_alpha_v1" / "residual_strategies.csv")
         .filter(pl.col("strategy") != "aligned_q5_minus_q1")
         .group_by("construction", "side", "model", "horizon_minutes", "strategy")
         .agg(pl.col("mean_return_bps").mean(), pl.col("day_t_stat").mean()))
    return d.select(
        pl.lit("H-004").alias("source"), pl.format("{} {} {} {}", "construction", "side", "model", "strategy").alias("cell"),
        pl.lit("BTCUSDT").alias("instrument"), pl.lit("walk-forward folds (2020-21)").alias("period"),
        pl.col("horizon_minutes").cast(pl.Float64).alias("tau_minutes"), pl.col("mean_return_bps").alias("gross_bps"),
        pl.lit(None, dtype=pl.Float64).alias("gross_ordinary_bps"), pl.col("day_t_stat").alias("t"),
        pl.lit(None, dtype=pl.Float64).alias("trades_per_day"), pl.lit(None, dtype=pl.Float64).alias("best10_share"),
        pl.lit("mean over folds").alias("by_year"))


def seconds_scale() -> pl.DataFrame:
    """E-002 (BTC 2022 exploration): continuation of the aligned mid proxy by bucket."""

    rows = []
    specs = [("S1_levels_notional.parquet", ["level_class", "notional_class"], "sum_response", "sweep levels x notional"),
             ("S2_burst.parquet", ["burst_decile"], "sum_burst_response", "burst decile"),
             ("S3_level_hits.parquet", ["hit_class", "state"], "sum_response", "level hits x state")]
    for name, keys, column, label in specs:
        d = pl.read_parquet(REPORTS / "seconds_signals_e002" / name)
        for key, g in d.group_by(["tau"] + keys):
            daily = g.group_by("day").agg(pl.col(column).sum().alias("s"), pl.col("n").sum().alias("n"))
            per_day = (daily["s"] / daily["n"]).to_numpy()
            mean = float(daily["s"].sum() / daily["n"].sum())
            t = per_day.mean() / (per_day.std(ddof=1) / np.sqrt(len(per_day))) if len(per_day) > 2 else np.nan
            ordered = daily.sort("s", descending=True)
            rows.append({"source": "E-002", "cell": f"{label} {dict(zip(keys, key[1:]))}", "instrument": "BTCUSDT",
                         "period": "2022 (exploration)", "tau_minutes": key[0] / 60, "gross_bps": mean,
                         "gross_ordinary_bps": float(ordered.tail(max(ordered.height - 10, 1))["s"].sum() / ordered.tail(max(ordered.height - 10, 1))["n"].sum()),
                         "t": t, "trades_per_day": float(g["n"].sum() / daily.height), "best10_share": None,
                         "by_year": "event-level continuation; 1 event = 1 trade"})
    return pl.DataFrame(rows)


def books() -> pl.DataFrame:
    rows = []
    for name, path in (("H-010 BTC (primary, log-return P&L)", REPORTS / "crowding_backtest_v1" / "backtest_summary.csv"),):
        d = pl.read_csv(path).filter(pl.col("primary"))
        r = d.row(0, named=True)
        cost_per_bp = r["turnover_per_day"] * 365 / 100      # %/yr per bp per side
        sigma = r["gross_return_pct_per_year"] / r["gross_sharpe"]
        for label, bp in (("gross", 0.0), ("2 bp", 2.0), ("1 bp", 1.0)):
            ret = r["gross_return_pct_per_year"] - bp * cost_per_bp
            rows.append({"strategy": name, "fees": label, "return_pct_per_year": ret, "sharpe": ret / sigma})
    for venue, signal, window in (("bybit", "C", ("2022-06-01", "2026-06-01")), ("bybit", "funding_level", ("2022-06-01", "2026-06-01")),
                                  ("binance", "funding_level", ("2022-01-01", "2026-06-01"))):
        d = pl.read_parquet(REPORTS / "carry_h020" / f"daily_{venue}_{signal}.parquet").with_columns(pl.col("date").cast(pl.Date))
        d = d.filter((pl.col("date") >= pl.lit(window[0]).str.to_date()) & (pl.col("date") < pl.lit(window[1]).str.to_date()))
        gate = 6.5 if venue == "bybit" else 6.0
        per_bp = ((d["net_gross"] - d["net_gate"]) / gate).to_numpy()
        for label, bp in (("gross", 0.0), ("2 bp", 2.0), ("1 bp", 1.0)):
            v = d["net_gross"].to_numpy() - bp * per_bp
            rows.append({"strategy": f"H-020 {venue} {signal}", "fees": label, "return_pct_per_year": 365 * 100 * v.mean(),
                         "sharpe": v.mean() / v.std(ddof=1) * np.sqrt(365)})
    return pl.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = [h006(), h012(), stress(), h009(), h003(), h004(), seconds_scale()]
    cells = pl.concat([f.with_columns(pl.col("trades_per_day").cast(pl.Float64), pl.col("best10_share").cast(pl.Float64),
                                      pl.col("gross_ordinary_bps").cast(pl.Float64), pl.col("t").cast(pl.Float64))
                       for f in frames], how="diagonal")
    cells = cells.with_columns(
        pl.col("tau_minutes").map_elements(threshold, return_dtype=pl.Float64).alias("threshold_bps"),
        (pl.col("gross_bps") - ROUND_TRIP_BPS).alias("net_target_bps"),
        (pl.col("gross_ordinary_bps") - ROUND_TRIP_BPS).alias("net_target_ordinary_bps"))
    cells = cells.with_columns(
        (pl.col("gross_bps") >= pl.col("threshold_bps")).alias("pass_gross"),
        ((pl.col("gross_bps") >= pl.col("threshold_bps")) & (pl.col("net_target_bps") > 0)).alias("pass_gross_and_net"),
        ((pl.col("gross_ordinary_bps") >= pl.col("threshold_bps")) & (pl.col("net_target_ordinary_bps") > 0)
         & (pl.col("t") > 2)).alias("robust_candidate"))
    cells.write_parquet(OUT / "cells.parquet")
    cells.write_csv(OUT / "cells.csv")
    book = books()
    book.write_csv(OUT / "books.csv")
    summary = (cells.group_by("source").agg(pl.len().alias("cells"), pl.col("pass_gross").sum(), pl.col("pass_gross_and_net").sum(),
                                           pl.col("robust_candidate").sum()).sort("source"))
    summary.write_csv(OUT / "summary.csv")
    (OUT / "manifest.json").write_text(json.dumps({"built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                                   "round_trip_bps": ROUND_TRIP_BPS,
                                                   "threshold": "max(1.5, min(2*sqrt(tau/5min), 10)) bps"}, indent=2))
    print(summary)


if __name__ == "__main__":
    main()
