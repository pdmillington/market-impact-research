"""Download and consolidate Binance USD-M daily futures metrics.

Binance publishes one daily archive of 5-minute snapshots containing open
interest and long/short ratios. The daily ZIPs are kept as immutable source
evidence; the consolidated table is de-duplicated (early archives repeat rows)
and keyed by UTC snapshot time in milliseconds.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
import io
from pathlib import Path
import zipfile

import polars as pl

from .download import download_verified
from .layout import DataLayout


BASE_URL = "https://data.binance.vision/data/futures/um/daily/metrics"
SNAPSHOT_MS = 5 * 60 * 1_000
NUMERIC_COLUMNS = (
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
)


def metrics_url(symbol: str, day: str) -> str:
    return f"{BASE_URL}/{symbol}/{symbol}-metrics-{day}.zip"


def iter_days(start_day: str, end_day: str) -> list[str]:
    start = date.fromisoformat(start_day)
    end = date.fromisoformat(end_day)
    if end < start:
        raise ValueError("end_day must not precede start_day.")
    return [
        (start + timedelta(days=offset)).isoformat()
        for offset in range((end - start).days + 1)
    ]


def download_metrics_range(
    *,
    root: Path,
    symbol: str,
    start_day: str,
    end_day: str,
    overwrite: bool = False,
    workers: int = 8,
) -> list[Path]:
    """Download an inclusive day range; existing verified archives are kept."""

    layout = DataLayout(Path(root))

    def fetch(day: str) -> Path:
        return download_verified(
            url=metrics_url(symbol, day),
            destination=layout.source_metrics_day(symbol, day),
            overwrite=overwrite,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fetch, iter_days(start_day, end_day)))


def read_metrics_archive(path: Path) -> pl.DataFrame:
    """Read one daily archive into typed columns."""

    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"Expected one CSV in {path.name}, found {names}.")
        payload = archive.read(names[0])
    frame = pl.read_csv(io.BytesIO(payload), infer_schema_length=0)
    missing = {"create_time", *NUMERIC_COLUMNS}.difference(frame.columns)
    if missing:
        raise KeyError(f"{path.name} lacks columns {sorted(missing)}.")
    return frame.select(
        pl.col("create_time")
        .str.to_datetime("%Y-%m-%d %H:%M:%S", time_zone="UTC")
        .dt.epoch("ms")
        .alias("time_ms"),
        # Some archives leave ratio fields blank; those become nulls, while any
        # other non-numeric text still fails loudly.
        *[
            pl.when(pl.col(name).str.strip_chars() == "")
            .then(None)
            .otherwise(pl.col(name))
            .cast(pl.Float64)
            .alias(name)
            for name in NUMERIC_COLUMNS
        ],
    )


@dataclass(frozen=True)
class MetricsReport:
    symbol: str
    archives: int
    raw_rows: int
    rows: int
    duplicate_rows_removed: int
    conflicting_duplicate_times: int
    first_time_ms: int
    last_time_ms: int
    missing_snapshots: int
    off_grid_rows: int
    output: str


def parse_metrics(*, root: Path, symbol: str) -> MetricsReport:
    """Consolidate every downloaded daily archive into one parquet table."""

    layout = DataLayout(Path(root))
    paths = sorted(layout.source_metrics_day(symbol, "x").parent.glob("*.zip"))
    if not paths:
        raise FileNotFoundError("No metrics archives have been downloaded.")
    raw = pl.concat([read_metrics_archive(path) for path in paths])
    exact = raw.unique(maintain_order=True)
    conflicting = exact.height - exact.select("time_ms").n_unique()
    # Where a timestamp appears with different values, keep the first archive's row.
    table = exact.unique(subset="time_ms", keep="first").sort("time_ms")
    times = table["time_ms"]
    on_grid = (times % SNAPSHOT_MS) == 0
    expected = (times.max() - times.min()) // SNAPSHOT_MS + 1
    output = layout.metrics_table(symbol)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(output, compression="zstd")
    return MetricsReport(
        symbol=symbol,
        archives=len(paths),
        raw_rows=raw.height,
        rows=table.height,
        duplicate_rows_removed=raw.height - exact.height,
        conflicting_duplicate_times=int(conflicting),
        first_time_ms=int(times.min()),
        last_time_ms=int(times.max()),
        missing_snapshots=int(expected - on_grid.sum()),
        off_grid_rows=int((~on_grid).sum()),
        output=str(output),
    )
