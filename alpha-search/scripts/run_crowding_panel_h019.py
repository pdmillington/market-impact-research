"""H-019 crowding contrarian on the pooled alt-perp panel, as registered.

Every input is truncated before 2026-06-01 (the locked confirmation months are
never read). Warm-up 2021; test 2022-01-01 to 2026-05-31.

Implementation choices (recorded in the ledger run notes):
- decision hours are kline close times; hour h's P&L is realised over (T_h, T_h+1h];
- minimum samples: rolling z-scores need 360 of 720 hourly values; the funding z
  needs 20 settlements in its 30-day window; the |C| threshold uses the
  trailing 365 days *excluding* the current hour and needs 180 days;
- an hour with no kline close has zero target position;
- the funding settlement at exactly 2026-06-01 00:00 is outside the data cut.

P&L (v2, 2026-09-30): linear-perp accounting via `alpha_search.perp_pnl`
(quantity × price change, funding on notional at settlement, costs on the trade
from drifted notional). v1 used log returns; its outputs are kept as
`v1_log_returns_*`.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import polars as pl


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "shared-methodology" / "src"))
sys.path.insert(0, str(REPO / "alpha-search" / "src"))

from alpha_search.perp_pnl import book_pnl  # noqa: E402
from crypto_market_data.series import DATASETS  # noqa: E402


HOUR_MS = 3_600_000
DAY_MS = 86_400_000
GRID_START = "2020-01-01"
CUTOFF = "2026-06-01"                      # locked months start here; nothing at or after is read
TEST = ("2022-01-01", "2026-06-01")
CURRENT = ("2024-06-01", "2026-06-01")
Z_WINDOW, Z_MIN = 720, 360
FUNDING_WINDOW_MS, FUNDING_MIN = 30 * DAY_MS, 20
TAIL_WINDOW, TAIL_MIN, TAIL_Q = 8_760, 4_320, 0.80
VOL_TARGET, SCALE_CAP = 0.20, 3.0
HOLD_HOURS = 24
WINSOR = 4.0
COSTS_BPS = {"gate_6": 6.0, "maker_3": 3.0, "gross": 0.0}
VARIANTS = {"full": ("F", "P", "R"), "drop_F": ("P", "R"), "drop_P": ("F", "R"), "drop_R": ("F", "P")}


def utc_ms(day: str) -> int:
    return int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp() * 1_000)


GRID0, CUT = utc_ms(GRID_START), utc_ms(CUTOFF)
N_HOURS = (CUT - GRID0) // HOUR_MS
T = GRID0 + HOUR_MS * np.arange(N_HOURS, dtype=np.int64)           # decision times


def on_grid(root: Path, dataset: str, symbol: str) -> np.ndarray:
    """Hourly close aligned to decision times (kline close = open + 1h)."""

    table = pl.read_parquet(DATASETS[dataset].table(root, symbol), columns=["open_time_ms", "close"])
    table = table.filter(pl.col("open_time_ms") < CUT - HOUR_MS + 1)
    index = (table["open_time_ms"].to_numpy() + HOUR_MS - GRID0) // HOUR_MS
    out = np.full(N_HOURS, np.nan)
    ok = (index >= 0) & (index < N_HOURS)
    out[index[ok]] = table["close"].to_numpy()[ok]
    return out


def rolling(values: np.ndarray, window: int, minimum: int, how: str) -> np.ndarray:
    series = pl.Series(values, nan_to_null=True)
    if how == "mean":
        result = series.rolling_mean(window, min_samples=minimum)
    elif how == "std":
        result = series.rolling_std(window, min_samples=minimum)
    else:
        result = series.rolling_quantile(TAIL_Q, window_size=window, min_samples=minimum, interpolation="linear")
    return result.fill_null(np.nan).to_numpy()


def zscore(values: np.ndarray) -> np.ndarray:
    std = rolling(values, Z_WINDOW, Z_MIN, "std")
    return (values - rolling(values, Z_WINDOW, Z_MIN, "mean")) / np.where(std > 0, std, np.nan)


def funding_features(root: Path, symbol: str) -> tuple[np.ndarray, np.ndarray]:
    """(z-score of the last settled rate at each decision, funding paid over the next hour)."""

    table = (pl.read_parquet(DATASETS["funding"].table(root, symbol)).filter(pl.col("time_ms") < CUT)
             .sort("time_ms"))
    times = table["time_ms"].to_numpy()
    rates = table["funding_rate"].to_numpy()
    lo = np.searchsorted(times, times - FUNDING_WINDOW_MS, side="right")
    cum = np.concatenate([[0.0], np.cumsum(rates)])
    cum2 = np.concatenate([[0.0], np.cumsum(rates**2)])
    count = np.arange(1, len(times) + 1) - lo
    mean = (cum[1:] - cum[lo]) / np.maximum(count, 1)
    var = (cum2[1:] - cum2[lo]) / np.maximum(count, 1) - mean**2
    std = np.sqrt(np.maximum(var * count / np.maximum(count - 1, 1), 0.0))
    z = np.where((count >= FUNDING_MIN) & (std > 0), (rates - mean) / np.where(std > 0, std, 1.0), np.nan)
    # Settlements are stamped a few ms after the hour; 1 s tolerance assigns them to that hour.
    last = np.searchsorted(times, T + 1_000, side="right") - 1
    z_at = np.where(last >= 0, z[np.clip(last, 0, None)], np.nan)
    upper = np.searchsorted(times, T + HOUR_MS + 1_000, side="right")
    lower = np.searchsorted(times, T + 1_000, side="right")
    paid = cum[upper] - cum[lower]
    return z_at, paid


def symbol_panel(root: Path, symbol: str, member: np.ndarray) -> dict:
    close = on_grid(root, "perp_klines_1h", symbol)
    premium = on_grid(root, "premium_index_1h", symbol)
    log_close = np.log(close)
    ret_prev = np.full(N_HOURS, np.nan)
    ret_prev[1:] = log_close[1:] - log_close[:-1]
    r24 = np.full(N_HOURS, np.nan)
    r24[24:] = log_close[24:] - log_close[:-24]
    p8 = rolling(premium, 8, 8, "mean")
    z_f, paid = funding_features(root, symbol)
    z = {"F": np.clip(z_f, -WINSOR, WINSOR), "P": np.clip(zscore(p8), -WINSOR, WINSOR),
         "R": np.clip(zscore(r24), -WINSOR, WINSOR)}
    vol = rolling(ret_prev, Z_WINDOW, Z_MIN, "std") * np.sqrt(8_760)
    scale = np.where(vol > 0, np.minimum(VOL_TARGET / np.where(vol > 0, vol, np.nan), SCALE_CAP), np.nan)
    tradable = member & np.isfinite(close) & np.isfinite(scale)
    simple_next = np.full(N_HOURS, np.nan)
    simple_next[:-1] = close[1:] / close[:-1] - 1.0
    out = {"close": close, "simple_next": np.nan_to_num(simple_next), "funding_next": np.nan_to_num(paid), "tradable": tradable,
           "scale": np.nan_to_num(scale), "C": {}, "pos": {}}
    for name, parts in VARIANTS.items():
        stack = np.vstack([z[p] for p in parts])
        valid = np.isfinite(stack).sum(axis=0)
        c = np.where(valid > 0, -np.nansum(stack, axis=0) / np.maximum(valid, 1), np.nan)
        magnitude = np.abs(c)
        shifted = np.concatenate([[np.nan], magnitude[:-1]])
        threshold = rolling(shifted, TAIL_WINDOW, TAIL_MIN, "quantile")
        signal = np.where(tradable & np.isfinite(c) & np.isfinite(threshold) & (magnitude >= threshold),
                          np.sign(c), 0.0)
        tranche = pl.Series(signal).rolling_mean(HOLD_HOURS, min_samples=1).to_numpy()
        out["C"][name] = c
        out["pos"][name] = np.where(tradable, tranche * out["scale"], 0.0)
    return out


def daily(values: np.ndarray) -> pl.DataFrame:
    day = (T // DAY_MS).astype(np.int64)
    return (pl.DataFrame({"day": day, "v": values}).group_by("day").agg(pl.col("v").sum()).sort("day")
            .with_columns(pl.from_epoch(pl.col("day") * 86_400, time_unit="s").dt.date().alias("date")))


def newey_west_t(y: np.ndarray, x: np.ndarray | None = None, lag: int = 1) -> tuple[float, float]:
    """Coefficient and NW t of the intercept (mean, or alpha when x is given)."""

    X = np.ones((len(y), 1)) if x is None else np.column_stack([np.ones(len(y)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    xe = X * e[:, None]
    S = xe.T @ xe
    for l in range(1, lag + 1):
        w = 1 - l / (lag + 1)
        G = xe[l:].T @ xe[:-l]
        S += w * (G + G.T)
    inv = np.linalg.inv(X.T @ X)
    cov = inv @ S @ inv
    return float(beta[0]), float(beta[0] / np.sqrt(cov[0, 0]))


def window(frame: pl.DataFrame, bounds: tuple[str, str]) -> pl.DataFrame:
    lo, hi = (utc_ms(b) // DAY_MS for b in bounds)
    return frame.filter((pl.col("day") >= lo) & (pl.col("day") < hi))


def stats(frame: pl.DataFrame) -> dict:
    v = frame["v"].to_numpy()
    mean, t = newey_west_t(v)
    return {"days": len(v), "ann_return_pct": 365 * 100 * v.mean(), "sharpe": float(v.mean() / v.std(ddof=1) * np.sqrt(365)),
            "nw_t": t}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-data-root", type=Path, default=REPO / "shared-data")
    parser.add_argument("--output-dir", type=Path, default=REPO / "alpha-search" / "reports" / "crowding_panel_h019")
    args = parser.parse_args()
    root, out = args.shared_data_root, args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    members = pl.read_parquet(root / "features" / "alpha" / "alt_panel_v1" / "membership.parquet")
    members = members.filter(pl.col("month") < CUTOFF[:7])
    month_of_hour = np.array([datetime.fromtimestamp(t / 1_000, tz=timezone.utc).strftime("%Y-%m") for t in T[::24]])
    month_of_hour = np.repeat(month_of_hour, 24)[:N_HOURS]
    symbols = sorted(members["symbol"].unique().to_list())
    rank_half = {}
    panels = {}
    for symbol in symbols:
        rows = members.filter(pl.col("symbol") == symbol)
        member = np.isin(month_of_hour, rows["month"].to_list())
        low = np.isin(month_of_hour, rows.filter(pl.col("rank") > 15)["month"].to_list())
        panels[symbol] = symbol_panel(root, symbol, member)
        rank_half[symbol] = low
        print(symbol, flush=True)

    syms = list(panels)
    close = np.vstack([panels[s]["close"] for s in syms])
    fund = np.vstack([panels[s]["funding_next"] for s in syms])
    tradable = np.vstack([panels[s]["tradable"] for s in syms])
    every_hour = np.ones(N_HOURS, bool)
    n_active = tradable.sum(axis=0)
    denom = np.where(n_active > 0, n_active, np.nan)
    market = np.where(tradable, np.vstack([panels[s]["simple_next"] for s in syms]), 0.0).sum(axis=0) / denom
    market_daily = daily(np.nan_to_num(market))
    test_mask = T >= utc_ms(TEST[0])

    results: dict = {"pnl_model": "linear perp v2 (alpha_search.perp_pnl)", "n_symbols": len(symbols),
                     "mean_active_members_test": float(np.nanmean(np.where(test_mask, n_active, np.nan)))}
    daily_frames = {}
    for variant in VARIANTS:
        position = np.vstack([panels[s]["pos"][variant] for s in syms])
        # Portfolio: equal weight across tradable members, re-targeted every hour.
        book = book_pnl(np.nan_to_num(position / denom), close, fund, every_hour)
        for cost_name, cost in COSTS_BPS.items():
            if variant != "full" and cost_name != "gate_6":
                continue
            d = daily(book.net(cost))
            key = f"{variant}_{cost_name}"
            daily_frames[key] = d
            test, current = window(d, TEST), window(d, CURRENT)
            entry = {"test": stats(test), "current": stats(current)}
            entry["by_year"] = {str(y): stats(g) for (y,), g in test.group_by(pl.col("date").dt.year()) if g.height > 30}
            if key == "full_gate_6":
                x = window(market_daily, TEST)["v"].to_numpy()
                alpha, alpha_t = newey_west_t(test["v"].to_numpy(), x)
                entry["alpha_daily"] = alpha
                entry["alpha_ann_pct"] = alpha * 365 * 100
                entry["alpha_nw_t"] = alpha_t
                # Breadth: each instrument's own unscaled book, net of costs.
                own = book_pnl(position, close, fund, every_hour)
                own_net = own.price + own.funding - cost * 1e-4 * own.traded
                breadth = []
                for i, s in enumerate(syms):
                    months = tradable[i, test_mask].sum() / (24 * 30.4)
                    if months >= 12:
                        breadth.append({"symbol": s, "member_months": months, "net": float(own_net[i, test_mask].sum())})
                entry["breadth"] = {"instruments": len(breadth),
                                    "positive_share": float(np.mean([b["net"] > 0 for b in breadth]))}
                pl.DataFrame(breadth).write_csv(out / "breadth_full_gate_6.csv")
            if key == "full_gross":
                halves = {}
                for label, pick in (("low_volume", True), ("high_volume", False)):
                    in_half = np.vstack([rank_half[s] == pick for s in syms]) & tradable
                    count = in_half.sum(axis=0)
                    target = np.where(in_half, position, 0.0) / np.where(count > 0, count, np.nan)
                    halves[label] = book_pnl(np.nan_to_num(target), close, fund, every_hour).net(0.0)
                diff = daily(halves["low_volume"] - halves["high_volume"])
                results["S1_low_minus_high_volume_gross"] = {"test": stats(window(diff, TEST)),
                                                             "current": stats(window(diff, CURRENT))}
            results[key] = entry
            print(key, json.dumps(entry["test"]), flush=True)

    # S2: cross-sectional, dollar-neutral quintiles, with a saved decomposition.
    C = np.vstack([np.where(panels[s]["tradable"], panels[s]["C"]["full"], np.nan) for s in syms])
    scale = np.vstack([panels[s]["scale"] for s in syms])
    weights = np.zeros_like(C)
    valid = np.isfinite(C)
    for h in np.flatnonzero(valid.sum(axis=0) >= 10):
        idx = np.flatnonzero(valid[:, h])
        k = max(int(round(len(idx) / 5)), 1)
        order = idx[np.argsort(C[idx, h])]
        short, long = order[:k], order[-k:]
        weights[long, h] = scale[long, h] / scale[long, h].sum()
        weights[short, h] = -scale[short, h] / scale[short, h].sum()
    held = np.vstack([pl.Series(w).rolling_mean(HOLD_HOURS, min_samples=1).to_numpy() for w in weights])
    held = np.where(tradable, held, 0.0)
    s2 = book_pnl(held, close, fund, every_hour)
    s2_legs = {"price_long": s2.leg("price", 1), "price_short": s2.leg("price", -1),
               "funding_long": s2.leg("funding", 1), "funding_short": s2.leg("funding", -1),
               "cost_gate_6": -COSTS_BPS["gate_6"] * 1e-4 * s2.traded.sum(axis=0)}
    s2_legs["gross"] = s2_legs["price_long"] + s2_legs["price_short"] + s2_legs["funding_long"] + s2_legs["funding_short"]
    s2_legs["net_gate_6"] = s2_legs["gross"] + s2_legs["cost_gate_6"]
    s2_daily = {k: daily(v) for k, v in s2_legs.items()}
    results["S2_cross_sectional_gate_6"] = {"test": stats(window(s2_daily["net_gate_6"], TEST)),
                                            "current": stats(window(s2_daily["net_gate_6"], CURRENT))}
    results["S2_decomposition"] = {
        "note": "Reported after the H-019 verdict (the post-hoc diagnostic behind lead L-H019a), saved reproducibly from v2.",
        "legs_test": {k: stats(window(v, TEST)) for k, v in s2_daily.items()},
        "turnover_per_day_test": float(s2.traded.sum(axis=0)[test_mask].mean() * 24),
        "gross_exposure_test": float(np.abs(s2.notional).sum(axis=0)[test_mask].mean()),
    }
    pl.DataFrame({"date": s2_daily["gross"]["date"]} | {k: v["v"] for k, v in s2_daily.items()}).write_parquet(out / "s2_decomposition_daily.parquet")
    print("S2", json.dumps(results["S2_decomposition"]["legs_test"]["gross"]), flush=True)

    g = results["full_gate_6"]
    results["conditions"] = {
        "1_existence": g["test"]["sharpe"] > 0 and g["test"]["nw_t"] > 2,
        "2_not_beta": g["alpha_daily"] > 0 and g["alpha_nw_t"] > 2,
        "3_breadth": g["breadth"]["positive_share"] > 0.55,
        "4_current": g["current"]["sharpe"] > 0,
    }
    results["pass"] = all(results["conditions"].values())
    (out / "results.json").write_text(json.dumps(results, indent=2))
    pl.DataFrame({k: v["v"] for k, v in daily_frames.items()} | {"date": daily_frames["full_gate_6"]["date"]}).write_parquet(out / "daily_pnl.parquet")
    print(json.dumps({"conditions": results["conditions"], "pass": results["pass"]}, indent=2))


if __name__ == "__main__":
    main()
