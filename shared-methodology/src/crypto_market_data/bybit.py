"""Fetch Bybit USDT linear-perpetual history through the public v5 API.

Datasets (canonical parquet under ``canonical/bybit/<dataset>/symbol=<S>/``):

- ``kline_1d`` and ``kline_1h``: traded-price klines with volume and turnover;
- ``premium_1h``: premium-index klines;
- ``funding``: settled funding rates.

Every raw API page is kept as gzipped JSON lines under ``source/bybit/`` as
source evidence. Delisted contracts are still served when queried by symbol, so
panels built from them are survivorship-free; the symbol list comes from the
``public.bybit.com/trading/`` directory, which keeps delisted contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import json
from pathlib import Path
import re
import time
from typing import Callable

import polars as pl
import requests


API = "https://api.bybit.com"
TRADING_DIRECTORY = "https://public.bybit.com/trading/"
KLINE_LIMIT = 1_000
FUNDING_LIMIT = 200
RATE_LIMIT_CODES = {10006, 10018}
MAX_ATTEMPTS = 8
TRANSIENT_ATTEMPTS = 4

Fetcher = Callable[[str, dict], dict]


@dataclass(frozen=True)
class KlineDataset:
    name: str
    endpoint: str
    interval: str
    step_ms: int
    columns: tuple[str, ...]


KLINES = {
    "kline_1d": KlineDataset("kline_1d", "/v5/market/kline", "D", 86_400_000,
                             ("open_time_ms", "open", "high", "low", "close", "volume", "turnover")),
    "kline_1h": KlineDataset("kline_1h", "/v5/market/kline", "60", 3_600_000,
                             ("open_time_ms", "open", "high", "low", "close", "volume", "turnover")),
    "premium_1h": KlineDataset("premium_1h", "/v5/market/premium-index-price-kline", "60", 3_600_000,
                               ("open_time_ms", "open", "high", "low", "close")),
}


def http_fetch(path: str, params: dict) -> dict:
    """GET one API page, retrying on rate limits and transient errors."""

    delay = 1.0
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.get(API + path, params=params, timeout=60)
            if response.status_code in (403, 429) or response.status_code >= 500:
                raise requests.exceptions.RequestException(f"HTTP {response.status_code}")
            payload = response.json()
            if payload.get("retCode") in RATE_LIMIT_CODES:
                raise requests.exceptions.RequestException(f"rate limited: {payload.get('retMsg')}")
            if payload.get("retCode") != 0 and attempt < TRANSIENT_ATTEMPTS:
                # Bybit returns server-side failures (e.g. "svc error") as non-zero codes;
                # retry a few times before handing the error to the caller.
                raise requests.exceptions.RequestException(f"API error: {payload.get('retMsg')}")
            return payload
        except (requests.exceptions.RequestException, ValueError):
            if attempt == MAX_ATTEMPTS:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
    raise RuntimeError("unreachable")


def directory_symbols() -> list[str]:
    """USDT symbols in Bybit's public trading archive (delisted included)."""

    body = requests.get(TRADING_DIRECTORY, timeout=120).text
    names = re.findall(r'href="([A-Za-z0-9]+)/"', body)
    return sorted(n for n in set(names) if n.endswith("USDT") and n.isascii() and n.isalnum())


def instrument(symbol: str, fetch: Fetcher = http_fetch) -> dict | None:
    """Instrument metadata (also served for closed contracts), or None."""

    payload = fetch("/v5/market/instruments-info", {"category": "linear", "symbol": symbol})
    rows = payload.get("result", {}).get("list", [])
    return rows[0] if rows else None


def _source(root: Path, dataset: str, symbol: str) -> Path:
    return Path(root) / "source" / "bybit" / dataset / f"symbol={symbol}" / "pages.jsonl.gz"


def table_path(root: Path, dataset: str, symbol: str) -> Path:
    return Path(root) / "canonical" / "bybit" / dataset / f"symbol={symbol}" / "part-000.parquet"


def _write(root: Path, dataset: str, symbol: str, pages: list[dict], frame: pl.DataFrame) -> Path:
    source = _source(root, dataset, symbol)
    source.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(source, "wt") as stream:
        for page in pages:
            stream.write(json.dumps(page) + "\n")
    output = table_path(root, dataset, symbol)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output, compression="zstd")
    return output


def fetch_klines(root: Path, dataset: str, symbol: str, start_ms: int, end_ms: int,
                 fetch: Fetcher = http_fetch) -> pl.DataFrame:
    """Page forward through [start_ms, end_ms) in windows of KLINE_LIMIT bars."""

    spec = KLINES[dataset]
    pages, rows = [], []
    cursor = start_ms - start_ms % spec.step_ms
    while cursor < end_ms:
        window_end = min(cursor + KLINE_LIMIT * spec.step_ms, end_ms) - 1
        params = {"category": "linear", "symbol": symbol, "interval": spec.interval,
                  "start": cursor, "end": window_end, "limit": KLINE_LIMIT}
        payload = fetch(spec.endpoint, params)
        if payload.get("retCode") != 0:
            raise RuntimeError(f"{symbol} {dataset}: {payload.get('retMsg')}")
        pages.append({"params": params, "payload": payload})
        rows.extend(payload["result"]["list"])
        cursor = window_end + 1
    frame = pl.DataFrame([r[: len(spec.columns)] for r in rows], schema=list(spec.columns), orient="row") if rows \
        else pl.DataFrame(schema=list(spec.columns))
    frame = (frame.with_columns(pl.col("open_time_ms").cast(pl.Int64),
                                *[pl.col(c).cast(pl.Float64) for c in spec.columns[1:]])
             .filter((pl.col("open_time_ms") >= start_ms) & (pl.col("open_time_ms") < end_ms))
             .unique("open_time_ms", keep="first").sort("open_time_ms"))
    _write(root, dataset, symbol, pages, frame)
    return frame


def fetch_funding(root: Path, symbol: str, start_ms: int, end_ms: int, fetch: Fetcher = http_fetch) -> pl.DataFrame:
    """Page backward from end_ms until start_ms or the first settlement."""

    pages, rows = [], []
    cursor = end_ms - 1
    while cursor >= start_ms:
        params = {"category": "linear", "symbol": symbol, "endTime": cursor, "limit": FUNDING_LIMIT}
        payload = fetch("/v5/market/funding/history", params)
        if payload.get("retCode") != 0:
            raise RuntimeError(f"{symbol} funding: {payload.get('retMsg')}")
        page = payload["result"]["list"]
        pages.append({"params": params, "payload": payload})
        if not page:
            break
        rows.extend(page)
        oldest = min(int(r["fundingRateTimestamp"]) for r in page)
        if oldest >= cursor:
            break
        cursor = oldest - 1
    frame = (pl.DataFrame({"time_ms": [int(r["fundingRateTimestamp"]) for r in rows],
                           "funding_rate": [float(r["fundingRate"]) for r in rows]},
                          schema={"time_ms": pl.Int64, "funding_rate": pl.Float64})
             .filter((pl.col("time_ms") >= start_ms) & (pl.col("time_ms") < end_ms))
             .unique("time_ms", keep="first").sort("time_ms"))
    _write(root, "funding", symbol, pages, frame)
    return frame
