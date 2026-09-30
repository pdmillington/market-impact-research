"""Download and consolidate Binance monthly price-series and funding archives.

Supported datasets:

- ``funding``: USD-M funding settlements (time, interval hours, rate);
- ``premium_index``, ``mark_price``, ``index_price``: USD-M 1-minute klines;
- ``perp_klines``: USD-M traded-price 1-minute klines (with taker-buy volume);
- ``spot_klines``: spot 1-minute klines;
- ``perp_klines_1h``, ``perp_klines_1d``, ``premium_index_1h``: coarser USD-M
  klines for multi-instrument panels (stored under ``interval=1h``/``1d``).

Monthly ZIPs are kept as immutable source evidence. Consolidation normalises
two archive quirks: older CSVs have no header row, and Binance spot archives
switched from millisecond to microsecond timestamps in 2025.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
import io
from pathlib import Path
import zipfile

import polars as pl
import requests

from .download import download_verified
from .months import iter_months


BASE_URL = "https://data.binance.vision/data"
KLINE_COLUMNS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
)
FUNDING_COLUMNS = ("calc_time", "funding_interval_hours", "last_funding_rate")
# Any epoch above this is in microseconds rather than milliseconds (year ~5138 in ms).
MICROSECOND_THRESHOLD = 100_000_000_000_000


@dataclass(frozen=True)
class Dataset:
    name: str
    url_path: str
    source_path: str
    columns: tuple[str, ...]
    interval: str | None = None

    def archive_name(self, symbol: str, month: str) -> str:
        if self.interval is None:
            return f"{symbol}-fundingRate-{month}.zip"
        return f"{symbol}-{self.interval}-{month}.zip"

    def url(self, symbol: str, month: str) -> str:
        path = self.url_path.format(symbol=symbol, interval=self.interval)
        return f"{BASE_URL}/{path}/{self.archive_name(symbol, month)}"

    def source_archive(self, root: Path, symbol: str, month: str) -> Path:
        path = self.source_path.format(symbol=symbol, interval=self.interval)
        return Path(root) / "source" / "binance" / path / self.archive_name(symbol, month)

    def daily_url(self, symbol: str, day: str) -> str:
        path = self.url_path.format(symbol=symbol, interval=self.interval)
        return f"{BASE_URL}/{path.replace('/monthly/', '/daily/', 1)}/{symbol}-{self.interval}-{day}.zip"

    def daily_archive(self, root: Path, symbol: str, day: str) -> Path:
        monthly = self.source_archive(root, symbol, "0000-00").parent
        return monthly / "daily" / f"{symbol}-{self.interval}-{day}.zip"

    def table(self, root: Path, symbol: str) -> Path:
        suffix = f"interval={self.interval}" if self.interval else "all"
        return (
            Path(root)
            / "canonical"
            / self.name
            / f"symbol={symbol}"
            / suffix
            / "part-000.parquet"
        )


DATASETS = {
    "funding": Dataset(
        "funding",
        "futures/um/monthly/fundingRate/{symbol}",
        "futures/um/fundingRate/{symbol}",
        FUNDING_COLUMNS,
    ),
    "premium_index": Dataset(
        "premium_index",
        "futures/um/monthly/premiumIndexKlines/{symbol}/{interval}",
        "futures/um/premiumIndexKlines/{symbol}/{interval}",
        KLINE_COLUMNS,
        "1m",
    ),
    "mark_price": Dataset(
        "mark_price",
        "futures/um/monthly/markPriceKlines/{symbol}/{interval}",
        "futures/um/markPriceKlines/{symbol}/{interval}",
        KLINE_COLUMNS,
        "1m",
    ),
    "index_price": Dataset(
        "index_price",
        "futures/um/monthly/indexPriceKlines/{symbol}/{interval}",
        "futures/um/indexPriceKlines/{symbol}/{interval}",
        KLINE_COLUMNS,
        "1m",
    ),
    "perp_klines": Dataset(
        "perp_klines",
        "futures/um/monthly/klines/{symbol}/{interval}",
        "futures/um/klines/{symbol}/{interval}",
        KLINE_COLUMNS,
        "1m",
    ),
    "perp_klines_1h": Dataset(
        "perp_klines",
        "futures/um/monthly/klines/{symbol}/{interval}",
        "futures/um/klines/{symbol}/{interval}",
        KLINE_COLUMNS,
        "1h",
    ),
    "perp_klines_1d": Dataset(
        "perp_klines",
        "futures/um/monthly/klines/{symbol}/{interval}",
        "futures/um/klines/{symbol}/{interval}",
        KLINE_COLUMNS,
        "1d",
    ),
    "premium_index_1h": Dataset(
        "premium_index",
        "futures/um/monthly/premiumIndexKlines/{symbol}/{interval}",
        "futures/um/premiumIndexKlines/{symbol}/{interval}",
        KLINE_COLUMNS,
        "1h",
    ),
    "spot_klines": Dataset(
        "spot_klines",
        "spot/monthly/klines/{symbol}/{interval}",
        "spot/klines/{symbol}/{interval}",
        KLINE_COLUMNS,
        "1m",
    ),
}


def download_series_range(
    *,
    root: Path,
    dataset: str,
    symbol: str,
    start_month: str,
    end_month: str,
    overwrite: bool = False,
    workers: int = 6,
) -> list[Path]:
    spec = DATASETS[dataset]

    def fetch(month: str) -> Path:
        return download_verified(
            url=spec.url(symbol, month),
            destination=spec.source_archive(root, symbol, month),
            overwrite=overwrite,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fetch, iter_months(start_month, end_month)))


def to_milliseconds(column: str) -> pl.Expr:
    value = pl.col(column).cast(pl.Int64)
    return (
        pl.when(value >= MICROSECOND_THRESHOLD).then(value // 1_000).otherwise(value)
    ).alias(f"{column}_ms")


def read_series_archive(path: Path, spec: Dataset) -> pl.DataFrame:
    """Read one monthly archive with or without a header row."""

    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"Expected one CSV in {path.name}, found {names}.")
        payload = archive.read(names[0])
    first_line = payload.split(b"\n", 1)[0].decode().strip()
    has_header = first_line.split(",")[0] == spec.columns[0]
    frame = pl.read_csv(
        io.BytesIO(payload),
        has_header=has_header,
        new_columns=None if has_header else list(spec.columns),
        infer_schema_length=0,
    )
    if tuple(frame.columns) != spec.columns:
        raise KeyError(f"{path.name} has columns {frame.columns}.")
    if spec.interval is None:
        return frame.select(
            to_milliseconds("calc_time").alias("time_ms"),
            pl.col("funding_interval_hours").cast(pl.Int32),
            pl.col("last_funding_rate").cast(pl.Float64).alias("funding_rate"),
        )
    return frame.select(
        to_milliseconds("open_time").alias("open_time_ms"),
        *[
            pl.col(name).cast(pl.Float64)
            for name in (
                "open",
                "high",
                "low",
                "close",
                "volume",
                "quote_volume",
                "taker_buy_volume",
                "taker_buy_quote_volume",
            )
        ],
        pl.col("count").cast(pl.Int64),
    )


def missing_days(table: pl.DataFrame, *, minimum_missing_minutes: int = 60) -> list[str]:
    """UTC days inside the table's span with at least this many absent minutes."""

    times = table["open_time_ms"]
    first_day = times.min() // 86_400_000
    last_day = times.max() // 86_400_000
    counts = (
        table.group_by((pl.col("open_time_ms") // 86_400_000).alias("day"))
        .len()
        .to_dict(as_series=False)
    )
    present = dict(zip(counts["day"], counts["len"]))
    output = []
    for day in range(first_day, last_day + 1):
        if 1_440 - present.get(day, 0) >= minimum_missing_minutes:
            output.append(str(date(1970, 1, 1) + timedelta(days=day)))
    return output


def backfill_series(*, root: Path, dataset: str, symbol: str) -> dict[str, list[str]]:
    """Download daily archives for days missing from a consolidated kline table."""

    spec = DATASETS[dataset]
    if spec.interval is None:
        raise ValueError("Backfill applies only to kline datasets.")
    table = pl.read_parquet(spec.table(root, symbol), columns=["open_time_ms"])
    fetched, unavailable = [], []
    for day in missing_days(table):
        url = spec.daily_url(symbol, day)
        if requests.head(url, timeout=60).status_code != 200:
            unavailable.append(day)
            continue
        download_verified(url=url, destination=spec.daily_archive(root, symbol, day))
        fetched.append(day)
    return {"fetched": fetched, "unavailable": unavailable}


@dataclass(frozen=True)
class SeriesReport:
    dataset: str
    symbol: str
    archives: int
    raw_rows: int
    rows: int
    duplicate_rows_removed: int
    conflicting_duplicate_times: int
    first_time_ms: int
    last_time_ms: int
    missing_minutes: int | None
    output: str


def parse_series(*, root: Path, dataset: str, symbol: str) -> SeriesReport:
    """Consolidate every downloaded archive of one dataset into one parquet table."""

    spec = DATASETS[dataset]
    source = spec.source_archive(root, symbol, "0000-00").parent
    # Monthly archives come first so they take precedence over daily backfills.
    paths = sorted(source.glob("*.zip")) + sorted((source / "daily").glob("*.zip"))
    if not paths:
        raise FileNotFoundError(f"No {dataset} archives have been downloaded.")
    raw = pl.concat([read_series_archive(path, spec) for path in paths])
    key = "time_ms" if spec.interval is None else "open_time_ms"
    exact = raw.unique(maintain_order=True)
    conflicting = exact.height - exact.select(key).n_unique()
    table = exact.unique(subset=key, keep="first").sort(key)
    times = table[key]
    missing = None
    if spec.interval == "1m":
        missing = int((times.max() - times.min()) // 60_000 + 1 - table.height)
    output = spec.table(root, symbol)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(output, compression="zstd")
    return SeriesReport(
        dataset=dataset,
        symbol=symbol,
        archives=len(paths),
        raw_rows=raw.height,
        rows=table.height,
        duplicate_rows_removed=raw.height - exact.height,
        conflicting_duplicate_times=int(conflicting),
        first_time_ms=int(times.min()),
        last_time_ms=int(times.max()),
        missing_minutes=missing,
        output=str(output),
    )
