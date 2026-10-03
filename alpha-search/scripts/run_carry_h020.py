"""H-020 Stage 1: low-turnover cross-sectional crowding carry, as registered.

Primary signal: the H-019 composite C (long high C = uncrowded, short low C).
Secondary S1: funding level (long low trailing-72h funding, short high).
Book: daily rebalance at 00:00 UTC, enter top/bottom quintile, hold while in the
top/bottom 40%, risk-scaled weights normalised to one unit per side.

Venues: Bybit (the Stage 1 test) and Binance (consistency, condition 4).
Every input is truncated before 2026-06-01 through the shared H-019 grid.

Implementation choices (recorded in the ledger run notes):
- the book is flat on any day with fewer than 10 members with a defined score
  (the H-019 S2 precedent);
- quintile and 40% band sizes are round(n/5) and round(0.4 n), at least 1;
- a held name with no defined score or no close at the rebalance is exited;
- quantities are fixed between daily rebalances (notional drifts with price);
- S1 "funding level" is the funding accrued over the trailing 72 hours (sum of
  settled rates), which is invariant to the settlement interval.

P&L uses linear-perp accounting (`alpha_search.perp_pnl`): quantity × price
change, funding on notional at settlement, and costs on the notional traded from
the drifted position at each rebalance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import polars as pl


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "alpha-search" / "scripts"))
sys.path.insert(0, str(REPO / "shared-methodology" / "src"))
sys.path.insert(0, str(REPO / "alpha-search" / "src"))

import run_crowding_panel_h019 as base  # noqa: E402
from alpha_search.perp_pnl import book_pnl  # noqa: E402
from crypto_market_data import bybit  # noqa: E402
from crypto_market_data.series import DATASETS  # noqa: E402


VENUES = {
    "bybit": {"gate_bps": 6.5, "test": ("2022-06-01", "2026-06-01"),
              "membership": "features/alpha/bybit_panel_v1/membership.parquet"},
    "binance": {"gate_bps": 6.0, "test": ("2022-01-01", "2026-06-01"),
                "membership": "features/alpha/alt_panel_v1/membership_h020.parquet"},
}
CURRENT = ("2024-06-01", "2026-06-01")
MIN_NAMES, ENTER, HOLD = 10, 0.20, 0.40
FUNDING_LEVEL_HOURS = 72


def table(root: Path, venue: str, kind: str, symbol: str) -> Path:
    if venue == "bybit":
        return bybit.table_path(root, {"close": "kline_1h", "premium": "premium_1h", "funding": "funding"}[kind], symbol)
    return DATASETS[{"close": "perp_klines_1h", "premium": "premium_index_1h", "funding": "funding"}[kind]].table(root, symbol)


def grid(path: Path, column: str) -> np.ndarray:
    frame = pl.read_parquet(path, columns=["open_time_ms", column]).filter(pl.col("open_time_ms") < base.CUT - base.HOUR_MS + 1)
    index = (frame["open_time_ms"].to_numpy() + base.HOUR_MS - base.GRID0) // base.HOUR_MS
    out = np.full(base.N_HOURS, np.nan)
    ok = (index >= 0) & (index < base.N_HOURS)
    out[index[ok]] = frame[column].to_numpy()[ok]
    return out


def funding_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(z of last settled rate, funding paid over the next hour, funding accrued over the last 72 h)."""

    frame = pl.read_parquet(path).filter(pl.col("time_ms") < base.CUT).sort("time_ms")
    times, rates = frame["time_ms"].to_numpy(), frame["funding_rate"].to_numpy()
    lo = np.searchsorted(times, times - base.FUNDING_WINDOW_MS, side="right")
    cum = np.concatenate([[0.0], np.cumsum(rates)])
    cum2 = np.concatenate([[0.0], np.cumsum(rates**2)])
    count = np.arange(1, len(times) + 1) - lo
    mean = (cum[1:] - cum[lo]) / np.maximum(count, 1)
    var = (cum2[1:] - cum2[lo]) / np.maximum(count, 1) - mean**2
    std = np.sqrt(np.maximum(var * count / np.maximum(count - 1, 1), 0.0))
    z = np.where((count >= base.FUNDING_MIN) & (std > 0), (rates - mean) / np.where(std > 0, std, 1.0), np.nan)
    last = np.searchsorted(times, base.T + 1_000, side="right") - 1
    z_at = np.where(last >= 0, z[np.clip(last, 0, None)], np.nan)
    paid = cum[np.searchsorted(times, base.T + base.HOUR_MS + 1_000, side="right")] - cum[np.searchsorted(times, base.T + 1_000, side="right")]
    upper = np.searchsorted(times, base.T + 1_000, side="right")
    lower = np.searchsorted(times, base.T + 1_000 - FUNDING_LEVEL_HOURS * base.HOUR_MS, side="right")
    level = np.where(upper > 0, cum[upper] - cum[lower], np.nan)
    return z_at, paid, level


def symbol_arrays(root: Path, venue: str, symbol: str, member: np.ndarray) -> dict:
    close = grid(table(root, venue, "close", symbol), "close")
    volume_column = "turnover" if venue == "bybit" else "quote_volume"
    value = grid(table(root, venue, "close", symbol), volume_column)
    premium = grid(table(root, venue, "premium", symbol), "close")
    log_close = np.log(close)
    ret_prev = np.full(base.N_HOURS, np.nan)
    ret_prev[1:] = log_close[1:] - log_close[:-1]
    r24 = np.full(base.N_HOURS, np.nan)
    r24[24:] = log_close[24:] - log_close[:-24]
    z_f, paid, level = funding_arrays(table(root, venue, "funding", symbol))
    z = np.vstack([np.clip(z_f, -base.WINSOR, base.WINSOR),
                   np.clip(base.zscore(base.rolling(premium, 8, 8, "mean")), -base.WINSOR, base.WINSOR),
                   np.clip(base.zscore(r24), -base.WINSOR, base.WINSOR)])
    valid = np.isfinite(z).sum(axis=0)
    composite = np.where(valid > 0, -np.nansum(z, axis=0) / np.maximum(valid, 1), np.nan)
    vol = base.rolling(ret_prev, base.Z_WINDOW, base.Z_MIN, "std") * np.sqrt(8_760)
    scale = np.where(vol > 0, np.minimum(base.VOL_TARGET / np.where(vol > 0, vol, np.nan), base.SCALE_CAP), np.nan)
    tradable = member & np.isfinite(close) & np.isfinite(scale)
    return {"close": close, "funding_next": np.nan_to_num(paid), "tradable": tradable,
            "scale": scale, "scores": {"C": composite, "funding_level": -level}, "value": np.nan_to_num(value)}


def book(scores: np.ndarray, tradable: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Daily hysteresis book. scores/tradable/scale are [symbols x hours]; returns weights [symbols x hours]."""

    n_sym, n_hours = scores.shape
    weights = np.zeros((n_sym, n_hours))
    long, short = set(), set()
    holding_days: dict[int, int] = {}
    finished: list[int] = []
    for h in range(0, n_hours, 24):
        ok = tradable[:, h] & np.isfinite(scores[:, h]) & np.isfinite(scale[:, h])
        idx = np.flatnonzero(ok)
        if len(idx) < MIN_NAMES:
            new_long, new_short = set(), set()
        else:
            order = idx[np.argsort(-scores[idx, h], kind="stable")]       # best (long) first
            rank = {int(i): r for r, i in enumerate(order)}
            k_in = max(int(round(ENTER * len(idx))), 1)
            k_hold = max(int(round(HOLD * len(idx))), 1)
            n = len(idx)
            new_long = {i for i in long if i in rank and rank[i] < k_hold} | {int(i) for i in order[:k_in]}
            new_short = {i for i in short if i in rank and rank[i] >= n - k_hold} | {int(i) for i in order[-k_in:]}
        for i in (long | short) - (new_long | new_short):
            finished.append(holding_days.pop(i, 0))
        for i in new_long | new_short:
            holding_days[i] = holding_days.get(i, 0) + 1
        long, short = new_long, new_short
        w = np.zeros(n_sym)
        if long:
            lw = np.array(sorted(long))
            w[lw] = scale[lw, h] / scale[lw, h].sum()
        if short:
            sw = np.array(sorted(short))
            w[sw] = -scale[sw, h] / scale[sw, h].sum()
        weights[:, h] = w                       # read by the P&L only at rebalance hours
    return weights, finished + list(holding_days.values())


def run_venue(root: Path, venue: str) -> dict:
    spec = VENUES[venue]
    members = pl.read_parquet(root / spec["membership"]).filter(pl.col("month") < base.CUTOFF[:7])
    month = np.repeat(np.array([np.datetime64(int(t), "ms").astype("datetime64[M]").astype(str) for t in base.T[::24]]), 24)[: base.N_HOURS]
    symbols = sorted(members["symbol"].unique().to_list())
    arrays = [symbol_arrays(root, venue, s, np.isin(month, members.filter(pl.col("symbol") == s)["month"].to_list()))
              for s in symbols]
    stack = lambda key: np.vstack([a[key] for a in arrays])  # noqa: E731
    close, fund, tradable, scale, value = (stack(k) for k in ("close", "funding_next", "tradable", "scale", "value"))
    rebalance = (np.arange(base.N_HOURS) % 24) == 0          # grid starts at 00:00 UTC
    test_hours = base.T >= base.utc_ms(spec["test"][0])
    out = {"symbols": len(symbols)}
    for signal in ("C", "funding_level"):
        scores = np.vstack([a["scores"][signal] for a in arrays])
        w, holdings = book(scores, tradable, scale)
        pnl = book_pnl(w, close, fund, rebalance)
        legs = {"price_long": pnl.leg("price", 1), "price_short": pnl.leg("price", -1),
                "funding_long": pnl.leg("funding", 1), "funding_short": pnl.leg("funding", -1)}
        gross = sum(legs.values())
        traded = pnl.traded.sum(axis=0)
        legs_daily = {k: base.window(base.daily(v), spec["test"]) for k, v in legs.items()}
        result = {"turnover_per_day": float(traded[test_hours].sum() / (test_hours.sum() / 24)),
                  "gross_exposure": float(np.abs(pnl.notional).sum(axis=0)[test_hours].mean()),
                  "mean_holding_days": float(np.mean(holdings)) if holdings else None,
                  "median_holding_days": float(np.median(holdings)) if holdings else None,
                  "legs_test": {k: base.stats(v) for k, v in legs_daily.items()}}
        saved = {"date": base.daily(gross)["date"]} | {k: base.daily(v)["v"] for k, v in legs.items()}
        for label, cost in (("gate", spec["gate_bps"]), ("maker_3", 3.0), ("gross", 0.0)):
            d = base.daily(gross - cost * 1e-4 * traded)
            saved[f"net_{label}"] = d["v"]
            test = base.window(d, spec["test"])
            v = test["v"].to_numpy()
            months = test.group_by(pl.col("date").dt.strftime("%Y-%m")).agg(pl.col("v").sum())
            result[label] = {
                "test": base.stats(test), "current": base.stats(base.window(d, CURRENT)),
                "ex_best_worst_10_sum_pct": float(np.sort(v)[10:-10].sum() * 100),
                "months_positive_share": float((months["v"] > 0).mean()),
                "by_year": {str(y): base.stats(g) for (y,), g in test.group_by(pl.col("date").dt.year()) if g.height > 30},
            }
        pl.DataFrame(saved).write_parquet(OUT / f"daily_{venue}_{signal}.parquet")
        change = pnl.traded
        # Capacity: participation = traded notional per day / member daily traded value.
        test_hours = base.T >= base.utc_ms(spec["test"][0])
        day_value = value[:, test_hours].reshape(len(symbols), -1, 24).sum(axis=2)
        day_trade = change[:, test_hours].reshape(len(symbols), -1, 24).sum(axis=2)
        traded = day_trade > 0
        result["capacity"] = {f"book_{b}": {"median_participation": float(np.median((day_trade[traded] * b) / np.maximum(day_value[traded], 1))),
                                            "p95_participation": float(np.quantile((day_trade[traded] * b) / np.maximum(day_value[traded], 1), 0.95))}
                              for b in (100_000, 1_000_000)}
        out[signal] = result
        print(venue, signal, json.dumps(result["gate"]["test"]), flush=True)
    return out


def conditions(bybit_c: dict, binance_c: dict) -> dict:
    gate = bybit_c["gate"]
    return {
        "1_net_sharpe_t": gate["test"]["sharpe"] > 0 and gate["test"]["nw_t"] > 2,
        "2_not_concentrated": gate["ex_best_worst_10_sum_pct"] > 0 and gate["months_positive_share"] > 0.55,
        "3_current": gate["current"]["sharpe"] > 0,
        "4_binance_consistency": binance_c["gate"]["test"]["sharpe"] > 0,
    }


OUT = REPO / "alpha-search" / "reports" / "carry_h020"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-data-root", type=Path, default=REPO / "shared-data")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    results = {venue: run_venue(args.shared_data_root, venue) for venue in ("bybit", "binance")}
    for signal in ("C", "funding_level"):
        c = conditions(results["bybit"][signal], results["binance"][signal])
        results[f"conditions_{signal}"] = c
        results[f"pass_{signal}"] = all(c.values())
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in results.items() if k.startswith(("conditions", "pass"))}, indent=2))


if __name__ == "__main__":
    main()
