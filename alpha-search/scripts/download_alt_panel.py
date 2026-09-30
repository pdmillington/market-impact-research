"""Download and build the H-019 alt-perp panel.

1. Daily perp klines for every USD-M USDT perpetual (including delisted ones),
   used only for point-in-time liquidity ranking.
2. Monthly membership: the top 30 eligible perps by trailing 30-day quote
   volume, as registered in H-019. Only volumes are read here, never returns.
3. Hourly perp klines, hourly premium-index klines and funding for every perp
   that is ever a member (all available months, including the locked
   2026-06..08 confirmation months, which no analysis reads before the H-019
   test verdict).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import urllib.request

import polars as pl


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "shared-methodology" / "src"))

from crypto_market_data.download import download_verified  # noqa: E402
from crypto_market_data.months import iter_months  # noqa: E402
from crypto_market_data.series import DATASETS, parse_series  # noqa: E402


S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?delimiter=/&prefix="
EXCLUDED_SYMBOLS = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
EXCLUDED_BASES = {"USDC", "BUSD", "TUSD", "FDUSD", "USDP", "DAI", "USDE", "PYUSD", "EUR", "DEFI", "BTCDOM",
                  "FOOTBALL", "BLUEBIRD"}
TOP_N = 30
ELIGIBLE_HISTORY_DAYS = 365
RANK_WINDOW_DAYS = 30
FIRST_MEMBER_MONTH, LAST_MEMBER_MONTH = "2021-01", "2026-08"
DAY_MS = 86_400_000


def listing(prefix: str, tag: str) -> list[str]:
    out, marker = [], ""
    while True:
        url = S3 + prefix + (f"&marker={marker}" if marker else "")
        body = urllib.request.urlopen(url, timeout=120).read().decode()
        items = re.findall(rf"<{tag}>([^<]+)</{tag}>", body)
        out += items
        if "<IsTruncated>true</IsTruncated>" not in body:
            return out
        next_marker = re.findall(r"<NextMarker>([^<]+)</NextMarker>", body)
        marker = next_marker[0] if next_marker else items[-1]


def available_months(dataset: str, symbol: str) -> list[str]:
    spec = DATASETS[dataset]
    prefix = "data/" + spec.url_path.format(symbol=symbol, interval=spec.interval) + "/"
    return sorted({m for key in listing(prefix, "Key") for m in re.findall(r"-(\d{4}-\d{2})\.zip$", key)})


def perp_symbols() -> list[str]:
    prefixes = listing("data/futures/um/monthly/klines/", "Prefix")
    symbols = [p.rstrip("/").split("/")[-1] for p in prefixes]
    return sorted(s for s in symbols if s.endswith("USDT") and s.isascii() and s.isalnum())


def excluded(symbol: str) -> bool:
    base = symbol[: -len("USDT")]
    return symbol in EXCLUDED_SYMBOLS or base in EXCLUDED_BASES


def fetch_dataset(root: Path, dataset: str, symbol: str, last_month: str) -> dict:
    months = [m for m in available_months(dataset, symbol) if m <= last_month]
    spec = DATASETS[dataset]
    for month in months:
        download_verified(url=spec.url(symbol, month), destination=spec.source_archive(root, symbol, month))
    if not months:
        return {"symbol": symbol, "dataset": dataset, "months": 0}
    report = parse_series(root=root, dataset=dataset, symbol=symbol)
    return {"symbol": symbol, "dataset": dataset, "months": len(months), "rows": report.rows,
            "first": months[0], "last": months[-1], "conflicting_duplicate_times": report.conflicting_duplicate_times}


def month_start_ms(month: str) -> int:
    return int(datetime.strptime(month, "%Y-%m").replace(tzinfo=timezone.utc).timestamp() * 1_000)


def membership(root: Path, symbols: list[str]) -> pl.DataFrame:
    frames = []
    for symbol in symbols:
        path = DATASETS["perp_klines_1d"].table(root, symbol)
        if path.exists():
            frames.append(pl.read_parquet(path, columns=["open_time_ms", "quote_volume"])
                          .with_columns(pl.lit(symbol).alias("symbol")))
    daily = pl.concat(frames)
    first = daily.group_by("symbol").agg(pl.col("open_time_ms").min().alias("first_ms"))
    rows = []
    for month in iter_months(FIRST_MEMBER_MONTH, LAST_MEMBER_MONTH):
        start = month_start_ms(month)
        window = (daily.filter((pl.col("open_time_ms") >= start - RANK_WINDOW_DAYS * DAY_MS)
                               & (pl.col("open_time_ms") < start))
                  .group_by("symbol").agg(pl.col("quote_volume").sum().alias("volume_30d"),
                                          pl.col("open_time_ms").max().alias("last_ms")))
        ranked = (window.join(first, on="symbol")
                  .filter((pl.col("first_ms") <= start - ELIGIBLE_HISTORY_DAYS * DAY_MS)
                          & (pl.col("last_ms") >= start - 2 * DAY_MS))
                  .sort("volume_30d", descending=True).head(TOP_N)
                  .with_columns(pl.lit(month).alias("month"), pl.int_range(1, pl.len() + 1).alias("rank")))
        rows.append(ranked.select("month", "rank", "symbol", "volume_30d"))
    return pl.concat(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-data-root", type=Path, default=REPO / "shared-data")
    parser.add_argument("--last-month", default="2026-08")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    root = args.shared_data_root
    out = root / "features" / "alpha" / "alt_panel_v1"
    out.mkdir(parents=True, exist_ok=True)

    symbols = [s for s in perp_symbols() if not excluded(s)]
    print(f"{len(symbols)} eligible USDT perps", flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        daily = list(pool.map(lambda s: fetch_dataset(root, "perp_klines_1d", s, args.last_month), symbols))
    print(f"daily klines: {sum(r['months'] > 0 for r in daily)} symbols", flush=True)

    members = membership(root, symbols)
    members.write_parquet(out / "membership.parquet")
    ever = sorted(members["symbol"].unique().to_list())
    print(f"{len(ever)} symbols are ever members", flush=True)

    reports = []
    with ThreadPoolExecutor(args.workers) as pool:
        tasks = [(d, s) for s in ever for d in ("perp_klines_1h", "premium_index_1h", "funding")]
        for report in pool.map(lambda t: fetch_dataset(root, t[0], t[1], args.last_month), tasks):
            reports.append(report)
            print(json.dumps(report), flush=True)
    (out / "manifest.json").write_text(json.dumps({
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "top_n": TOP_N, "eligible_history_days": ELIGIBLE_HISTORY_DAYS, "rank_window_days": RANK_WINDOW_DAYS,
        "excluded_symbols": sorted(EXCLUDED_SYMBOLS), "excluded_bases": sorted(EXCLUDED_BASES),
        "candidates": len(symbols), "ever_members": ever, "daily": daily, "series": reports}, indent=1))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
