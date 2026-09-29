from __future__ import annotations

from pathlib import Path
import zipfile

import polars as pl
import pytest

from crypto_market_data import metrics
from crypto_market_data.layout import DataLayout


HEADER = (
    "create_time,symbol,sum_open_interest,sum_open_interest_value,"
    "count_toptrader_long_short_ratio,sum_toptrader_long_short_ratio,"
    "count_long_short_ratio,sum_taker_long_short_vol_ratio"
)


def write_archive(path: Path, day: str, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"BTCUSDT-metrics-{day}.csv", "\n".join([HEADER, *rows]))


def row(time: str, open_interest: float) -> str:
    return f"{time},BTCUSDT,{open_interest},1.0,1.0,1.0,1.0,1.0"


def test_metrics_url_and_days():
    assert metrics.metrics_url("BTCUSDT", "2020-09-01").endswith(
        "/daily/metrics/BTCUSDT/BTCUSDT-metrics-2020-09-01.zip"
    )
    assert metrics.iter_days("2024-02-28", "2024-03-01") == [
        "2024-02-28",
        "2024-02-29",
        "2024-03-01",
    ]
    with pytest.raises(ValueError):
        metrics.iter_days("2024-03-02", "2024-03-01")


def test_blank_ratio_fields_become_null(tmp_path):
    path = tmp_path / "BTCUSDT-metrics-2021-01-01.zip"
    write_archive(
        path,
        "2021-01-01",
        ["2021-01-01 00:00:00,BTCUSDT,10.0,1.0,,,1.0,1.0"],
    )
    frame = metrics.read_metrics_archive(path)
    assert frame["sum_open_interest"].to_list() == [10.0]
    assert frame["count_toptrader_long_short_ratio"].to_list() == [None]


def test_parse_removes_duplicates_and_reports_gaps(tmp_path):
    layout = DataLayout(tmp_path)
    write_archive(
        layout.source_metrics_day("BTCUSDT", "2020-09-01"),
        "2020-09-01",
        [
            row("2020-09-01 00:00:00", 10.0),
            row("2020-09-01 00:00:00", 10.0),
            row("2020-09-01 00:05:00", 11.0),
            row("2020-09-01 00:15:00", 12.0),
        ],
    )
    write_archive(
        layout.source_metrics_day("BTCUSDT", "2020-09-02"),
        "2020-09-02",
        [row("2020-09-01 00:15:00", 99.0), row("2020-09-01 00:20:00", 13.0)],
    )

    report = metrics.parse_metrics(root=tmp_path, symbol="BTCUSDT")
    table = pl.read_parquet(report.output)

    assert report.archives == 2
    assert report.raw_rows == 6
    assert report.duplicate_rows_removed == 1
    assert report.conflicting_duplicate_times == 1
    assert report.rows == 4
    assert report.missing_snapshots == 1
    assert report.off_grid_rows == 0
    assert table["time_ms"].is_sorted()
    assert table.filter(pl.col("time_ms") == 1_598_919_300_000)[
        "sum_open_interest"
    ].to_list() == [12.0]
