"""C-006 slippage check (indicative only; researcher caveat 2026-10-01).

Level-1 quotes only (bookTicker 1-second grid, 2023-05 to 2024-03, with known
archive gaps), so depth beyond the touch is unknown. Costs are bracketed:

- touch coverage: share of trades whose size fits within the best-level quantity
  on the side we take, at entry (and at the 5-minute exit);
- optimistic cost: quantity beyond the touch fills one tick worse (H-018 taker rule);
- conservative cost: square-root impact, Y * sigma_24h * sqrt(Q / V_24h) per side,
  Y = 1, with trailing 24 h realised volatility (hourly returns) and trailing 24 h
  traded notional, both from the signature hourly tables (known at entry).

Net edge per trade = executable taker edge (spread paid on both legs) - 2 x 2 bp
fees - entry and exit slippage. Day-block t as elsewhere.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import polars as pl


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "shared-methodology" / "src"))
sys.path.insert(0, str(REPO / "alpha-search" / "scripts"))

from crypto_market_data.book_ticker import quote_grid_month  # noqa: E402
import build_signature_features as sig  # noqa: E402


ROOT = REPO / "shared-data"
OUT = REPO / "alpha-search" / "reports" / "c006"
MONTHS = ["2023-05", "2023-06", "2023-07", "2023-08", "2023-09", "2023-10", "2023-11", "2023-12", "2024-01", "2024-02", "2024-03"]
LO, HI = 1_682_899_200_000, 1_711_929_600_000
TICK = {"BTCUSDT": 0.1, "ETHUSDT": 0.01, "SOLUSDT": 0.001}
SIZES = (10_000, 50_000, 100_000, 250_000, 1_000_000)
FEE_ROUND_TRIP_BPS = 4.0
Y = 1.0


def quotes(symbol: str) -> dict:
    g = pl.concat([pl.read_parquet(quote_grid_month(ROOT, symbol, m)) for m in MONTHS]).sort("second")
    return {c: g[c].to_numpy() for c in ("second", "best_bid_price", "best_bid_qty", "best_ask_price", "best_ask_qty")}


def at(q: dict, ms: np.ndarray) -> dict:
    i = np.searchsorted(q["second"], ms // 1000, side="left") - 1
    ok = (i >= 0) & ((ms // 1000 - q["second"][np.clip(i, 0, None)]) <= 5)
    j = np.clip(i, 0, None)
    return {k: np.where(ok, v[j], np.nan) for k, v in q.items() if k != "second"}


def trailing_state(symbol: str, ms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trailing 24 h realised volatility (bps) and traded notional, from completed hours before entry."""

    hourly = pl.concat([pl.read_parquet(sig.hourly_path(ROOT, symbol, m), columns=["hour", "notional", "close"])
                        for m in ["2023-04"] + MONTHS]).sort("hour")
    hour, notional, close = hourly["hour"].to_numpy(), hourly["notional"].to_numpy(), hourly["close"].to_numpy()
    r = np.concatenate([[np.nan], np.diff(np.log(close))])
    cum_n = np.concatenate([[0.0], np.cumsum(notional)])
    cum_r2 = np.concatenate([[0.0], np.cumsum(np.nan_to_num(r**2))])
    k = np.searchsorted(hour, ms // 3_600_000, side="left")          # hours strictly before the entry hour
    lo = np.searchsorted(hour, ms // 3_600_000 - 24, side="left")
    volume = cum_n[k] - cum_n[lo]
    sigma_bps = 1e4 * np.sqrt(cum_r2[k] - cum_r2[lo])
    return sigma_bps, volume


def day_t(days: np.ndarray, values: np.ndarray) -> float:
    d = pl.DataFrame({"d": days, "v": values}).group_by("d").agg(pl.col("v").mean())["v"].to_numpy()
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))


def main() -> None:
    trades = pl.read_parquet(OUT / "trades.parquet").filter(
        (pl.col("n_events") == 1000) & (pl.col("model") == "return_pretrend") & (pl.col("horizon") == 5) & (pl.col("tail") == 0.01))
    results, rows = {}, []
    for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        q = quotes(symbol)
        t = trades.filter((pl.col("symbol") == symbol) & (pl.col("entry_time_ms") >= LO) & (pl.col("entry_time_ms") + 300_000 < HI))
        entry_ms = t["entry_time_ms"].to_numpy()
        d = t["direction"].to_numpy().astype(float)
        e, x = at(q, entry_ms), at(q, entry_ms + 300_000)
        ok = np.isfinite(e["best_ask_price"]) & np.isfinite(x["best_bid_price"])
        mid_e = (e["best_bid_price"] + e["best_ask_price"]) / 2
        taker_edge = 1e4 * np.where(d > 0, x["best_bid_price"] / e["best_ask_price"] - 1, 1 - x["best_ask_price"] / e["best_bid_price"])
        # Touch notional on the side we take: entry buy lifts the ask; exit sell hits the bid (and vice versa).
        touch_entry = np.where(d > 0, e["best_ask_qty"] * e["best_ask_price"], e["best_bid_qty"] * e["best_bid_price"])
        touch_exit = np.where(d > 0, x["best_bid_qty"] * x["best_bid_price"], x["best_ask_qty"] * x["best_ask_price"])
        tick_bps = 1e4 * TICK[symbol] / mid_e
        sigma_bps, volume = trailing_state(symbol, entry_ms)
        ok &= np.isfinite(sigma_bps) & (volume > 0)
        days = t["day"].to_numpy()[ok]
        per_symbol = {"trades": int(ok.sum()),
                      "touch_notional_entry_usd": {"p10": float(np.nanquantile(touch_entry[ok], 0.1)), "median": float(np.nanmedian(touch_entry[ok]))},
                      "touch_notional_exit_usd": {"p10": float(np.nanquantile(touch_exit[ok], 0.1)), "median": float(np.nanmedian(touch_exit[ok]))},
                      "median_sigma_24h_bps": float(np.median(sigma_bps[ok])), "median_volume_24h_usd": float(np.median(volume[ok])),
                      "taker_edge_bps": float(taker_edge[ok].mean()), "by_size": {}}
        for size in SIZES:
            excess_e = np.clip(1 - touch_entry / size, 0, 1)
            excess_x = np.clip(1 - touch_exit / size, 0, 1)
            optimistic = (excess_e + excess_x) * tick_bps
            sqrt_law = 2 * Y * sigma_bps * np.sqrt(size / volume)
            net_opt = taker_edge - FEE_ROUND_TRIP_BPS - optimistic
            net_sqrt = taker_edge - FEE_ROUND_TRIP_BPS - sqrt_law
            entry = {"fits_at_touch_entry": float((touch_entry[ok] >= size).mean()),
                     "fits_at_touch_both": float(((touch_entry >= size) & (touch_exit >= size))[ok].mean()),
                     "optimistic_slippage_bps": float(optimistic[ok].mean()),
                     "sqrt_law_slippage_bps": float(sqrt_law[ok].mean()),
                     "net_edge_optimistic_bps": float(net_opt[ok].mean()), "t_optimistic": day_t(days, net_opt[ok]),
                     "net_edge_sqrt_law_bps": float(net_sqrt[ok].mean()), "t_sqrt_law": day_t(days, net_sqrt[ok])}
            per_symbol["by_size"][size] = entry
            rows.append({"symbol": symbol, "size_usd": size, **entry})
        results[symbol] = per_symbol
    (OUT / "slippage_check.json").write_text(json.dumps(results, indent=2))
    table = pl.DataFrame(rows)
    table.write_csv(OUT / "slippage_check.csv")
    pl.Config.set_tbl_width_chars(220); pl.Config.set_tbl_cols(12); pl.Config.set_tbl_rows(20)
    for s, r in results.items():
        print(s, {k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items() if k != "by_size"})
    print(table.with_columns(pl.selectors.float().round(2)))


if __name__ == "__main__":
    main()
