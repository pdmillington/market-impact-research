from __future__ import annotations

from pathlib import Path
import zipfile

import polars as pl

from crypto_market_data import series


def write_archive(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(path.with_suffix(".csv").name, "\n".join(lines))


def kline(open_time: int, close: float) -> str:
    return f"{open_time},{close},{close},{close},{close},1,{open_time + 59_999},1,3,0,0,0"


def test_urls_and_source_paths(tmp_path):
    funding = series.DATASETS["funding"]
    assert funding.url("BTCUSDT", "2020-01").endswith(
        "/futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-2020-01.zip"
    )
    perp = series.DATASETS["perp_klines"]
    assert perp.url("ETHUSDT", "2024-01").endswith(
        "/futures/um/monthly/klines/ETHUSDT/1m/ETHUSDT-1m-2024-01.zip"
    )
    assert perp.daily_url("ETHUSDT", "2024-01-02").endswith(
        "/futures/um/daily/klines/ETHUSDT/1m/ETHUSDT-1m-2024-01-02.zip"
    )
    spot = series.DATASETS["spot_klines"]
    assert spot.url("BTCUSDT", "2025-03").endswith(
        "/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2025-03.zip"
    )
    assert spot.source_archive(tmp_path, "BTCUSDT", "2025-03") == (
        tmp_path / "source" / "binance" / "spot" / "klines" / "BTCUSDT" / "1m"
        / "BTCUSDT-1m-2025-03.zip"
    )


def test_klines_with_and_without_header_and_microseconds(tmp_path):
    spec = series.DATASETS["spot_klines"]
    header = ",".join(series.KLINE_COLUMNS)
    write_archive(
        spec.source_archive(tmp_path, "BTCUSDT", "2024-12"),
        [kline(1_735_689_480_000, 1.0), kline(1_735_689_540_000, 2.0)],
    )
    write_archive(
        spec.source_archive(tmp_path, "BTCUSDT", "2025-01"),
        [header, kline(1_735_689_720_000_000, 4.0), kline(1_735_689_540_000, 2.0)],
    )

    report = series.parse_series(root=tmp_path, dataset="spot_klines", symbol="BTCUSDT")
    table = pl.read_parquet(report.output)

    assert table["open_time_ms"].to_list() == [
        1_735_689_480_000,
        1_735_689_540_000,
        1_735_689_720_000,
    ]
    assert table["close"].to_list() == [1.0, 2.0, 4.0]
    assert report.duplicate_rows_removed == 1
    assert report.missing_minutes == 2


def test_daily_backfill_paths_and_missing_days(tmp_path):
    spec = series.DATASETS["premium_index"]
    assert spec.daily_url("BTCUSDT", "2021-07-28").endswith(
        "/futures/um/daily/premiumIndexKlines/BTCUSDT/1m/BTCUSDT-1m-2021-07-28.zip"
    )
    assert spec.daily_archive(tmp_path, "BTCUSDT", "2021-07-28").parent.name == "daily"

    day = 86_400_000
    full_day = [i * 60_000 for i in range(1_440)]
    partial = [2 * day + i * 60_000 for i in range(1_000)]
    table = pl.DataFrame({"open_time_ms": full_day + partial + [3 * day]})
    # Day 1 is absent, day 2 is 440 minutes short and day 3 holds one minute.
    assert series.missing_days(table) == ["1970-01-02", "1970-01-03", "1970-01-04"]


def test_daily_backfill_is_parsed_after_monthly(tmp_path):
    spec = series.DATASETS["mark_price"]
    write_archive(spec.source_archive(tmp_path, "BTCUSDT", "2021-07"), [kline(0, 1.0)])
    write_archive(
        spec.daily_archive(tmp_path, "BTCUSDT", "1970-01-01"),
        [kline(0, 9.0), kline(60_000, 2.0)],
    )
    report = series.parse_series(root=tmp_path, dataset="mark_price", symbol="BTCUSDT")
    table = pl.read_parquet(report.output)
    assert table["close"].to_list() == [1.0, 2.0]
    assert report.conflicting_duplicate_times == 1


def test_funding_parse(tmp_path):
    spec = series.DATASETS["funding"]
    write_archive(
        spec.source_archive(tmp_path, "BTCUSDT", "2020-01"),
        [
            "calc_time,funding_interval_hours,last_funding_rate",
            "1577836800000,8,-0.00012359",
            "1577865600000,8,0.0001",
        ],
    )
    report = series.parse_series(root=tmp_path, dataset="funding", symbol="BTCUSDT")
    table = pl.read_parquet(report.output)
    assert table.columns == ["time_ms", "funding_interval_hours", "funding_rate"]
    assert table["funding_rate"].to_list() == [-0.00012359, 0.0001]
    assert report.missing_minutes is None
