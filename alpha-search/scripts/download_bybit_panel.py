"""Download and build the H-020 Bybit panel (Stage 1 test venue).

1. Instrument metadata for every USDT symbol in Bybit's public trading archive
   (delisted included); keep linear perpetuals and apply the H-020 exclusions.
2. Daily klines for all of them, used only for liquidity ranking.
3. Monthly membership: top 30 by trailing 30-day turnover, at least 365 days of
   history, trading in the last 2 days (the H-019/H-020 rule).
4. Hourly klines, hourly premium-index klines and funding for every member.

Data run to 2026-08-31. The locked months (2026-06..08) are downloaded but only
H-020 Stage 2 may read them.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import polars as pl


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "shared-methodology" / "src"))

from crypto_market_data import bybit  # noqa: E402
from crypto_market_data.months import iter_months  # noqa: E402


END_MS = int(datetime(2026, 9, 1, tzinfo=timezone.utc).timestamp() * 1_000)
EXCLUDED_SYMBOLS = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
STABLE_AND_INDEX_BASES = {"USDC", "BUSD", "TUSD", "FDUSD", "USDP", "DAI", "USDE", "PYUSD", "EUR", "DEFI", "BTCDOM",
                          "FOOTBALL", "BLUEBIRD"}
NON_CRYPTO_BASES = {"PAXG", "XAUT", "XAU", "XAG"}          # tokenised commodities typed as crypto by Bybit
NON_CRYPTO_TYPES = {"stock", "ETF", "commodity", "forex"}
TOP_N, ELIGIBLE_HISTORY_DAYS, RANK_WINDOW_DAYS = 30, 365, 30
FIRST_MEMBER_MONTH, LAST_MEMBER_MONTH = "2021-05", "2026-08"
DAY_MS = 86_400_000


def exclusion_reason(info: dict) -> str | None:
    if info.get("contractType") != "LinearPerpetual" or info.get("quoteCoin") != "USDT":
        return "not a USDT linear perpetual"
    if info["symbol"] in EXCLUDED_SYMBOLS:
        return "BTC/ETH/SOL"
    if info.get("baseCoin") in STABLE_AND_INDEX_BASES:
        return "stablecoin or index"
    if info.get("symbolType") in NON_CRYPTO_TYPES or info.get("baseCoin") in NON_CRYPTO_BASES:
        return "non-crypto underlying"
    return None


def month_start_ms(month: str) -> int:
    return int(datetime.strptime(month, "%Y-%m").replace(tzinfo=timezone.utc).timestamp() * 1_000)


def membership(root: Path, symbols: list[str]) -> pl.DataFrame:
    frames = [pl.read_parquet(bybit.table_path(root, "kline_1d", s), columns=["open_time_ms", "turnover"])
              .with_columns(pl.lit(s).alias("symbol")) for s in symbols
              if bybit.table_path(root, "kline_1d", s).exists()]
    daily = pl.concat([f for f in frames if f.height])
    first = daily.group_by("symbol").agg(pl.col("open_time_ms").min().alias("first_ms"))
    rows = []
    for month in iter_months(FIRST_MEMBER_MONTH, LAST_MEMBER_MONTH):
        start = month_start_ms(month)
        ranked = (daily.filter((pl.col("open_time_ms") >= start - RANK_WINDOW_DAYS * DAY_MS) & (pl.col("open_time_ms") < start))
                  .group_by("symbol").agg(pl.col("turnover").sum().alias("turnover_30d"),
                                          pl.col("open_time_ms").max().alias("last_ms"))
                  .join(first, on="symbol")
                  .filter((pl.col("first_ms") <= start - ELIGIBLE_HISTORY_DAYS * DAY_MS)
                          & (pl.col("last_ms") >= start - 2 * DAY_MS))
                  .sort("turnover_30d", descending=True).head(TOP_N)
                  .with_columns(pl.lit(month).alias("month"), pl.int_range(1, pl.len() + 1).alias("rank")))
        rows.append(ranked.select("month", "rank", "symbol", "turnover_30d"))
    return pl.concat(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-data-root", type=Path, default=REPO / "shared-data")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    root = args.shared_data_root
    out = root / "features" / "alpha" / "bybit_panel_v1"
    out.mkdir(parents=True, exist_ok=True)

    names = bybit.directory_symbols()
    with ThreadPoolExecutor(args.workers) as pool:
        infos = [i for i in pool.map(bybit.instrument, names) if i]
    reasons = {i["symbol"]: exclusion_reason(i) for i in infos}
    kept = sorted(s for s, r in reasons.items() if r is None)
    (out / "instruments.json").write_text(json.dumps({"directory_symbols": len(names), "instruments": infos,
                                                      "exclusions": reasons}, indent=1))
    launch = {i["symbol"]: int(i["launchTime"]) for i in infos}
    print(f"{len(names)} directory symbols, {len(infos)} with metadata, {len(kept)} kept", flush=True)

    def daily(symbol: str) -> int:
        path = bybit.table_path(root, "kline_1d", symbol)
        if path.exists():
            return pl.read_parquet(path).height
        return bybit.fetch_klines(root, "kline_1d", symbol, launch[symbol], END_MS).height

    with ThreadPoolExecutor(args.workers) as pool:
        counts = dict(zip(kept, pool.map(daily, kept)))
    print(f"daily klines: {sum(c > 0 for c in counts.values())} symbols with data", flush=True)

    members = membership(root, kept)
    members.write_parquet(out / "membership.parquet")
    ever = sorted(members["symbol"].unique().to_list())
    print(f"{len(ever)} symbols are ever members", flush=True)

    def series(task: tuple[str, str]) -> dict:
        dataset, symbol = task
        path = bybit.table_path(root, dataset, symbol)
        if path.exists():
            frame = pl.read_parquet(path)
        elif dataset == "funding":
            frame = bybit.fetch_funding(root, symbol, launch[symbol], END_MS)
        else:
            frame = bybit.fetch_klines(root, dataset, symbol, launch[symbol], END_MS)
        key = "time_ms" if dataset == "funding" else "open_time_ms"
        report = {"symbol": symbol, "dataset": dataset, "rows": frame.height,
                  "first_ms": int(frame[key].min()) if frame.height else None,
                  "last_ms": int(frame[key].max()) if frame.height else None}
        print(json.dumps(report), flush=True)
        return report

    with ThreadPoolExecutor(args.workers) as pool:
        reports = list(pool.map(series, [(d, s) for s in ever for d in ("kline_1h", "premium_1h", "funding")]))
    (out / "manifest.json").write_text(json.dumps({
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "end_ms": END_MS,
        "top_n": TOP_N, "eligible_history_days": ELIGIBLE_HISTORY_DAYS, "rank_window_days": RANK_WINDOW_DAYS,
        "excluded_symbols": sorted(EXCLUDED_SYMBOLS), "stable_and_index_bases": sorted(STABLE_AND_INDEX_BASES),
        "non_crypto_bases": sorted(NON_CRYPTO_BASES), "non_crypto_types": sorted(NON_CRYPTO_TYPES),
        "kept": len(kept), "ever_members": ever, "series": reports}, indent=1))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
