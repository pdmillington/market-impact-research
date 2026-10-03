"""C-006: 5-minute pretrend reversal (H-006 frozen) on ETHUSDT and SOLUSDT, with BTC as reference.

Reuses H-006 (run_residual_decomposition.py) definitions unchanged:
- event bars (impact_research_v2), R = bar return / trailing 24 h volatility,
  pre15/pre60 = clock returns ending at the bar start / trailing volatility,
  prices from the event_200 clock;
- the same row usability filter (relative imbalance > 0, activity > 0, side != 0);
- monthly OLS on the preceding 12 months with the horizon embargo; training-only
  winsorisation; tail cut = training quantile of |fitted|; trade in the forecast
  direction; delayed entry at the first event_200 endpoint >= 1 s after the decision.

The impact-model component P is not built (the frozen primary `return_pretrend`
does not use it). Edges are measured as simple returns (linear perp); log edges
are reported alongside. Model fitting uses log targets exactly as H-006.

Registered primary: return_pretrend, 1,000-event bars, top 1%, 5 minutes.
Every input stops before 2026-06-01 (locked months excluded).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import polars as pl


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "alpha-search" / "src"))
sys.path.insert(0, str(REPO / "alpha-search" / "scripts"))

from alpha_search.perp_pnl import book_pnl  # noqa: E402
from alpha_search.research import add_months  # noqa: E402
from run_residual_decomposition import (  # noqa: E402
    DAY_MS, load_price_clock, month_ms, month_range, ols, price_at_or_after, price_at_or_before,
)


CONFIG = json.loads((REPO / "alpha-search" / "configs" / "residual_decomposition_v1.json").read_text())
MODELS = {k: CONFIG["models"][k] for k in ("return_pretrend", "pretrend_only", "bar_return", "return_pretrend_flow")}
FEATURES = ["R", "q", "pre15", "pre60"]
SIZES = (200, 500, 1000, 2000, 4000, 8000)
HORIZONS = (5, 10)
TAILS = (0.01, 0.05)
PRIMARY = {"n_events": 1000, "model": "return_pretrend", "tail": 0.01, "horizon": 5}
THRESHOLD_BPS, ROUND_TRIP_BPS = 2.0, 4.0
OUT = REPO / "alpha-search" / "reports" / "c006"


def features(cache_root: Path, n_events: int, clock_time, clock_price) -> pl.DataFrame:
    start_ms = month_ms(CONFIG["first_feature_month"])
    end_ms = month_ms(CONFIG["data_end_exclusive"][:7])
    frame = (pl.scan_parquet(cache_root / f"event_{n_events}" / "part-000.parquet")
             .select("start_time_ms", "end_time_ms", "start_price", "end_price", "signed_imbalance",
                     "current_return_bps", "trailing_realised_volatility_bps", "normalised_imbalance",
                     "relative_imbalance", "relative_activity")
             .filter((pl.col("end_time_ms") >= start_ms) & (pl.col("end_time_ms") < end_ms))
             .sort("end_time_ms").collect())
    time = frame["end_time_ms"].to_numpy()
    side = np.sign(frame["signed_imbalance"].to_numpy()).astype(np.int8)
    sigma = frame["trailing_realised_volatility_bps"].to_numpy()
    z, activity = frame["relative_imbalance"].to_numpy(), frame["relative_activity"].to_numpy()
    usable = np.isfinite(z) & np.isfinite(activity) & (z > 0) & (activity > 0) & (side != 0)
    month = np.full(len(time), "", dtype=object)
    for m in month_range(CONFIG["first_feature_month"], CONFIG["last_test_month"]):
        month[(time >= month_ms(m)) & (time < month_ms(add_months(m, 1)))] = m
    start_price, end_price = frame["start_price"].to_numpy(), frame["end_price"].to_numpy()
    columns = {"month": month.astype(str), "day": time // DAY_MS, "time_ms": time,
               "R": frame["current_return_bps"].to_numpy() / sigma, "q": side * frame["normalised_imbalance"].to_numpy()}
    for minutes in CONFIG["pretrend_lookbacks_minutes"]:
        past = price_at_or_before(clock_time, clock_price, frame["start_time_ms"].to_numpy() - int(minutes) * 60_000)
        columns[f"pre{minutes}"] = 1e4 * np.log(start_price / past) / sigma
    entry_price, entry_time = price_at_or_after(clock_time, clock_price, time + int(CONFIG["entry_delay_ms"]))
    columns["entry_time_ms"] = entry_time
    for minutes in HORIZONS:
        horizon = minutes * 60_000
        exit_now, _ = price_at_or_after(clock_time, clock_price, time + horizon)
        columns[f"F{minutes}"] = 1e4 * np.log(exit_now / end_price)
        exit_late, _ = price_at_or_after(clock_time, clock_price, np.where(entry_time >= 0, entry_time, 0) + horizon)
        ratio = np.where(entry_time >= 0, exit_late / entry_price, np.nan)
        columns[f"D{minutes}"] = 1e4 * np.log(ratio)
        columns[f"S{minutes}"] = 1e4 * (ratio - 1.0)          # simple (linear perp) delayed return
    out = pl.DataFrame(columns).filter(pl.Series(usable))
    required = FEATURES + [f"F{m}" for m in HORIZONS]
    return out.filter(pl.all_horizontal([pl.col(c).is_finite() for c in required]) & (pl.col("month") != ""))


def trades(data: pl.DataFrame, n_events: int) -> pl.DataFrame:
    """Walk-forward tail trades for every model x horizon x tail (H-006 procedure)."""

    low, high = CONFIG["winsor_quantiles"]
    window = int(CONFIG["regression_training_months"])
    embargo_ms = (max(CONFIG["future_horizons_minutes"]) + 5) * 60_000
    rows = []
    for month in month_range(CONFIG["first_test_month"], CONFIG["last_test_month"]):
        train = data.filter(pl.col("month").is_in(month_range(add_months(month, -window), add_months(month, -1)))
                            & (pl.col("time_ms") < month_ms(month) - embargo_ms))
        test = data.filter(pl.col("month") == month)
        if train.height < 1_000 or test.height < 20:
            continue
        ftrain = train.select(FEATURES).to_numpy()
        bounds = np.quantile(ftrain, [low, high], axis=0)
        ctrain = dict(zip(FEATURES, np.clip(ftrain, bounds[0], bounds[1]).T))
        ctest = dict(zip(FEATURES, np.clip(test.select(FEATURES).to_numpy(), bounds[0], bounds[1]).T))
        for minutes in HORIZONS:
            y = train[f"F{minutes}"].to_numpy()
            y = np.clip(y, *np.quantile(y, [low, high]))
            for model, names in MODELS.items():
                xtr = np.column_stack([ctrain[c] for c in names])
                xte = np.column_stack([ctest[c] for c in names])
                centre = xtr.mean(axis=0)
                beta = ols(xtr - centre, y - y.mean())
                fitted, forecast = (xtr - centre) @ beta, (xte - centre) @ beta
                for tail in TAILS:
                    cut = float(np.quantile(np.abs(fitted), 1.0 - tail))
                    chosen = (np.abs(forecast) >= cut) & np.isfinite(test[f"S{minutes}"].to_numpy())
                    if not chosen.any():
                        continue
                    d = np.sign(forecast[chosen])
                    rows.append(pl.DataFrame({
                        "n_events": n_events, "model": model, "horizon": minutes, "tail": tail, "month": month,
                        "day": test["day"].to_numpy()[chosen], "time_ms": test["time_ms"].to_numpy()[chosen],
                        "entry_time_ms": test["entry_time_ms"].to_numpy()[chosen], "direction": d.astype(np.int8),
                        "edge_simple_bps": d * test[f"S{minutes}"].to_numpy()[chosen],
                        "edge_log_bps": d * test[f"D{minutes}"].to_numpy()[chosen]}))
    return pl.concat(rows) if rows else pl.DataFrame()


def score(t: pl.DataFrame) -> dict:
    daily = t.group_by("day").agg(pl.col("edge_simple_bps").sum().alias("s"), pl.len().alias("n")).sort("s", descending=True)
    per_day = (daily["s"] / daily["n"]).to_numpy()
    tstat = float(per_day.mean() / (per_day.std(ddof=1) / np.sqrt(len(per_day)))) if len(per_day) > 2 else float("nan")
    ordinary = daily.tail(max(daily.height - 10, 0))
    monthly = t.group_by("month").agg(pl.col("edge_simple_bps").mean())
    years = {int(y): float(v) for y, v in t.with_columns(pl.col("month").str.slice(0, 4).cast(pl.Int32).alias("y"))
             .group_by("y").agg(pl.col("edge_simple_bps").mean()).sort("y").iter_rows()}
    gross = float(t["edge_simple_bps"].mean())
    early = t.filter(pl.col("month") < "2025-01")["edge_simple_bps"].mean()
    late = t.filter(pl.col("month") >= "2025-01")["edge_simple_bps"].mean()
    return {"trades": t.height, "days": daily.height, "trades_per_day": t.height / max(daily.height, 1),
            "gross_bps": gross, "gross_log_bps": float(t["edge_log_bps"].mean()), "net_bps": gross - ROUND_TRIP_BPS,
            "t": tstat, "ordinary_gross_bps": float(ordinary["s"].sum() / ordinary["n"].sum()) if ordinary.height else float("nan"),
            "best10_share": float(daily.head(10)["s"].sum() / daily["s"].sum()) if daily["s"].sum() else float("nan"),
            "months_positive_share": float((monthly["edge_simple_bps"] > 0).mean()),
            "by_year": years, "years_positive": int(sum(v > 0 for y, v in years.items() if 2023 <= y <= 2026)),
            "mean_2023_24_bps": float(early) if early is not None else None, "mean_2025_26_bps": float(late) if late is not None else None}


def conditions(s: dict) -> dict:
    return {"1_gross_net_t": s["gross_bps"] >= THRESHOLD_BPS and s["net_bps"] > 0 and s["t"] > 2,
            "2_ordinary_months": s["ordinary_gross_bps"] >= THRESHOLD_BPS and s["months_positive_share"] > 0.60,
            "3_years": s["years_positive"] >= 3}


def book(t: pl.DataFrame, clock_time, clock_price) -> dict:
    """Netted overlapping positions on a 1-minute grid: each trade 1/24 unit, held 5 min from its entry minute."""

    first, last = month_ms(CONFIG["first_test_month"]), month_ms(CONFIG["data_end_exclusive"][:7])
    minutes = np.arange(first, last, 60_000, dtype=np.int64)
    price = price_at_or_before(clock_time, clock_price, minutes)
    target = np.zeros(len(minutes))
    start = np.searchsorted(minutes, t["entry_time_ms"].to_numpy(), side="left")
    for s, d in zip(start, t["direction"].to_numpy()):
        target[s: s + 5] += d / 24.0
    result = book_pnl(target[None, :], price[None, :], np.zeros((1, len(minutes))), np.ones(len(minutes), bool))
    gross = (result.price + result.funding).sum(axis=0)
    traded = result.traded.sum(axis=0)
    day = minutes // DAY_MS
    out = {"mean_abs_position": float(np.abs(target).mean()), "max_abs_position": float(np.abs(target).max()),
           "turnover_per_day": float(traded.sum() / (len(minutes) / 1440))}
    for label, bps in (("gross", 0.0), ("2bp_per_side", 2.0), ("1bp_per_side", 1.0)):
        pnl = gross - bps * 1e-4 * traded
        daily = pl.DataFrame({"d": day, "v": pnl}).group_by("d").agg(pl.col("v").sum())["v"].to_numpy()
        out[label] = {"ann_return_per_unit_max_pct": float(daily.mean() * 365 * 100 / max(out["max_abs_position"], 1e-9)),
                      "sharpe": float(daily.mean() / daily.std(ddof=1) * np.sqrt(365))}
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    all_trades = []
    for symbol in ("ETHUSDT", "SOLUSDT", "BTCUSDT"):
        cache = REPO / "shared-data" / "features" / "alpha" / f"symbol={symbol}" / "impact_research_v2"
        clock_time, clock_price = load_price_clock(cache, month_ms(CONFIG["data_end_exclusive"][:7]))
        durations = {}
        sym_trades = []
        for n in SIZES:
            data = features(cache, n, clock_time, clock_price)
            y2022 = pl.scan_parquet(cache / f"event_{n}" / "part-000.parquet").filter(
                (pl.col("end_time_ms") >= month_ms("2022-01")) & (pl.col("end_time_ms") < month_ms("2023-01"))
            ).select((pl.col("end_time_ms") - pl.col("start_time_ms")).median()).collect().item()
            durations[n] = float(y2022) / 1_000.0
            t = trades(data, n).with_columns(pl.lit(symbol).alias("symbol"))
            sym_trades.append(t)
            print(symbol, n, "bars", data.height, "trades", t.height, flush=True)
        st = pl.concat(sym_trades)
        all_trades.append(st)
        grid = {}
        for (n, model, horizon, tail), g in st.group_by("n_events", "model", "horizon", "tail"):
            grid[f"{n}_{model}_{horizon}m_top{tail}"] = score(g)
        p = st.filter((pl.col("n_events") == PRIMARY["n_events"]) & (pl.col("model") == PRIMARY["model"])
                      & (pl.col("horizon") == PRIMARY["horizon"]) & (pl.col("tail") == PRIMARY["tail"]))
        primary = score(p)
        results[symbol] = {"primary": primary, "conditions": conditions(primary), "pass": all(conditions(primary).values()),
                           "median_bar_seconds_2022": durations, "book": book(p, clock_time, clock_price), "grid": grid}
        print(symbol, json.dumps({k: v for k, v in results[symbol].items() if k != "grid"}, default=str)[:900], flush=True)
    btc_1000_seconds = results["BTCUSDT"]["median_bar_seconds_2022"][1000]
    for symbol in ("ETHUSDT", "SOLUSDT"):
        d = results[symbol]["median_bar_seconds_2022"]
        matched = min(d, key=lambda n: abs(np.log(d[n] / btc_1000_seconds)))
        results[symbol]["duration_matched_size"] = matched
        results[symbol]["duration_matched"] = results[symbol]["grid"].get(f"{matched}_return_pretrend_5m_top0.01")
    results["confirmed"] = results["ETHUSDT"]["pass"] and results["SOLUSDT"]["pass"]
    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=str))
    pl.concat(all_trades).write_parquet(OUT / "trades.parquet")
    print("C-006 confirmed:", results["confirmed"])


if __name__ == "__main__":
    main()
