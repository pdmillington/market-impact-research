"""Binance USD-M `bookTicker` (best bid/ask) archives.

Binance publishes tick-level best-bid/ask updates for USD-M perpetuals for
2023-05 to 2024-04 only. Monthly ZIPs are kept as source evidence. Each month is
parsed into a canonical parquet table and a derived 1-second quote grid holding
the last quote in each second (the quote in force at the end of that second).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

import polars as pl

from .download import download_verified
from .layout import DataLayout


BASE_URL = "https://data.binance.vision/data/futures/um/monthly/bookTicker"
COLUMNS = (
    "update_id",
    "best_bid_price",
    "best_bid_qty",
    "best_ask_price",
    "best_ask_qty",
    "transaction_time",
    "event_time",
)
SCHEMA = {
    "update_id": pl.UInt64,
    "best_bid_price": pl.Float64,
    "best_bid_qty": pl.Float64,
    "best_ask_price": pl.Float64,
    "best_ask_qty": pl.Float64,
    "transaction_time": pl.Int64,
    "event_time": pl.Int64,
}


def archive_url(symbol: str, month: str) -> str:
    return f"{BASE_URL}/{symbol}/{symbol}-bookTicker-{month}.zip"


def source_archive(root: Path, symbol: str, month: str) -> Path:
    return (
        Path(root) / "source" / "binance" / "futures" / "um" / "bookTicker" / symbol
        / f"{symbol}-bookTicker-{month}.zip"
    )


def canonical_month(root: Path, symbol: str, month: str) -> Path:
    year, number = month.split("-")
    return (
        Path(root) / "canonical" / "book_ticker" / f"symbol={symbol}"
        / f"year={year}" / f"month={number}" / "part-000.parquet"
    )


def quote_grid_month(root: Path, symbol: str, month: str) -> Path:
    year, number = month.split("-")
    return (
        Path(root) / "features" / "quotes" / f"symbol={symbol}" / "quote_grid_1s_v1"
        / f"year={year}" / f"month={number}" / "part-000.parquet"
    )


def report_month(root: Path, symbol: str, month: str) -> Path:
    return Path(root) / "metadata" / symbol / "book_ticker" / f"{month}.json"


@dataclass(frozen=True)
class BookTickerReport:
    symbol: str
    month: str
    rows: int
    grid_seconds: int
    crossed_or_locked_rows: int
    nonpositive_rows: int
    first_time_ms: int
    last_time_ms: int
    canonical_bytes: int
    grid_bytes: int


def _has_header(csv_path: Path) -> bool:
    with csv_path.open("rb") as stream:
        return stream.readline().split(b",", 1)[0].strip() == b"update_id"


def scan_book_ticker_csv(csv_path: Path) -> pl.LazyFrame:
    has_header = _has_header(csv_path)
    return pl.scan_csv(
        csv_path,
        has_header=has_header,
        new_columns=None if has_header else list(COLUMNS),
        schema_overrides=SCHEMA,
    ).select(list(COLUMNS))


GRID_CHUNK_ROWS = 50_000_000


def _chunk_grid(chunk: pl.DataFrame) -> pl.DataFrame:
    """Per-second last quote within one chunk, keeping its ordering keys."""

    return (
        chunk.sort("transaction_time", "update_id")
        .group_by((pl.col("transaction_time") // 1_000).alias("second"), maintain_order=True)
        .agg(
            pl.col("transaction_time").last(),
            pl.col("update_id").last(),
            pl.col("best_bid_price").last(),
            pl.col("best_bid_qty").last(),
            pl.col("best_ask_price").last(),
            pl.col("best_ask_qty").last(),
            pl.len().cast(pl.UInt32).alias("updates"),
        )
    )


def quote_grid(quotes: pl.LazyFrame, chunk_rows: int = GRID_CHUNK_ROWS) -> pl.DataFrame:
    """Last quote in each UTC second, by transaction time then update id.

    The month is processed in chunks so memory stays bounded; a second that
    spans two chunks is resolved by keeping the latest (time, id) and summing
    update counts.
    """

    total = quotes.select(pl.len()).collect().item()
    parts = [
        _chunk_grid(quotes.slice(offset, chunk_rows).collect())
        for offset in range(0, total, chunk_rows)
    ]
    combined = pl.concat(parts).sort("second", "transaction_time", "update_id")
    return (
        combined.group_by("second", maintain_order=True)
        .agg(
            pl.col("best_bid_price").last(),
            pl.col("best_bid_qty").last(),
            pl.col("best_ask_price").last(),
            pl.col("best_ask_qty").last(),
            pl.col("updates").sum(),
        )
        .sort("second")
    )


def parse_month(*, root: Path, symbol: str, month: str, overwrite: bool = False) -> BookTickerReport:
    layout = DataLayout(Path(root))
    report_path = report_month(root, symbol, month)
    if report_path.exists() and not overwrite:
        return BookTickerReport(**json.loads(report_path.read_text()))
    source = source_archive(root, symbol, month)
    canonical = canonical_month(root, symbol, month)
    grid = quote_grid_month(root, symbol, month)
    for path in (canonical, grid):
        path.parent.mkdir(parents=True, exist_ok=True)
    layout.temporary_root().mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="book-ticker-", dir=layout.temporary_root()) as temp:
        with zipfile.ZipFile(source) as archive:
            members = [m for m in archive.namelist() if m.endswith(".csv")]
            if len(members) != 1:
                raise ValueError(f"Expected one CSV in {source.name}, found {members}.")
            csv_path = Path(temp) / Path(members[0]).name
            with archive.open(members[0]) as src, csv_path.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        partial = canonical.with_suffix(".parquet.part")
        scan_book_ticker_csv(csv_path).sink_parquet(
            partial, compression="zstd", compression_level=6, row_group_size=500_000
        )
        partial.replace(canonical)
    quotes = pl.scan_parquet(canonical)
    summary = quotes.select(
        pl.len().alias("rows"),
        (pl.col("best_bid_price") >= pl.col("best_ask_price")).sum().alias("crossed"),
        (
            (pl.col("best_bid_price") <= 0) | (pl.col("best_ask_price") <= 0)
            | (pl.col("best_bid_qty") < 0) | (pl.col("best_ask_qty") < 0)
        ).sum().alias("nonpositive"),
        pl.col("transaction_time").min().alias("first"),
        pl.col("transaction_time").max().alias("last"),
    ).collect()
    grid_partial = grid.with_suffix(".parquet.part")
    quote_grid(quotes).write_parquet(grid_partial, compression="zstd")
    grid_partial.replace(grid)
    report = BookTickerReport(
        symbol=symbol,
        month=month,
        rows=int(summary["rows"][0]),
        grid_seconds=int(pl.scan_parquet(grid).select(pl.len()).collect().item()),
        crossed_or_locked_rows=int(summary["crossed"][0]),
        nonpositive_rows=int(summary["nonpositive"][0]),
        first_time_ms=int(summary["first"][0]),
        last_time_ms=int(summary["last"][0]),
        canonical_bytes=canonical.stat().st_size,
        grid_bytes=grid.stat().st_size,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(asdict(report), indent=2) + "\n")
    return report


def download_and_parse_month(*, root: Path, symbol: str, month: str, overwrite: bool = False) -> BookTickerReport:
    download_verified(
        url=archive_url(symbol, month),
        destination=source_archive(root, symbol, month),
        overwrite=overwrite,
    )
    return parse_month(root=root, symbol=symbol, month=month, overwrite=overwrite)
