from pathlib import Path
import zipfile

import polars as pl
import pytest

from crypto_market_data.layout import DataLayout
from crypto_market_data.parsing import canonical_fills_from_csv, parse_month_archive


def test_canonical_parser_handles_headerless_binance_file(tmp_path: Path) -> None:
    source = tmp_path / "trades.csv"
    source.write_text(
        "1,100.0,2.5,250.0,1000,false\n"
        "2,99.0,1.0,99.0,1001,true\n"
    )

    fills = canonical_fills_from_csv(source).collect()

    assert fills.columns == [
        "trade_id",
        "timestamp_ms",
        "price",
        "quantity",
        "aggressor_sign",
    ]
    assert fills["aggressor_sign"].to_list() == [1, -1]
    assert fills["trade_id"].dtype == pl.UInt64


def test_canonical_parser_handles_header(tmp_path: Path) -> None:
    source = tmp_path / "trades.csv"
    source.write_text(
        "id,price,qty,quote_qty,time,is_buyer_maker\n"
        "1,100.0,2.5,250.0,1000,false\n"
    )
    fills = canonical_fills_from_csv(source).collect()
    assert fills.height == 1
    assert fills["aggressor_sign"].item() == 1


def test_parse_month_archive_is_validated_and_resumable(tmp_path: Path) -> None:
    layout = DataLayout(tmp_path)
    source = layout.source_archive("BTCUSDT", "2020-01")
    source.parent.mkdir(parents=True)
    csv_text = (
        "1,100.0,1.0,100.0,1000,false\n"
        "2,101.0,2.0,202.0,1000,false\n"
        "2,101.0,2.0,202.0,1000,false\n"
        "3,99.0,4.0,396.0,1000,true\n"
    )
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("BTCUSDT-trades-2020-01.csv", csv_text)

    first = parse_month_archive(
        root=tmp_path, symbol="BTCUSDT", month="2020-01"
    )
    resumed = parse_month_archive(
        root=tmp_path, symbol="BTCUSDT", month="2020-01"
    )

    assert first == resumed
    assert first.fill_rows == 3
    assert first.event_rows == 2
    assert first.fill_count_difference == 0
    assert first.quantity_difference == 0
    assert first.signed_quantity_difference == 0
    assert first.exact_duplicate_rows_removed == 1
    assert first.reused_trade_ids == 0
    assert list(layout.temporary_root().iterdir()) == []


def write_archive(layout: DataLayout, csv_text: str) -> None:
    source = layout.source_archive("ETHUSDT", "2025-08")
    source.parent.mkdir(parents=True)
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("ETHUSDT-trades-2025-08.csv", csv_text)


def test_ids_reused_after_trading_gap_are_kept_in_time_order(tmp_path: Path) -> None:
    # Mirrors ETHUSDT 2025-08-29: the counter restarted two IDs early after a pause,
    # so the archive (sorted by ID) interleaves fills from both sides of the gap.
    layout = DataLayout(tmp_path)
    write_archive(
        layout,
        "10,100.0,1.0,100.0,1000,false\n"
        "11,100.0,2.0,200.0,2000,false\n"
        "11,101.0,3.0,303.0,900000,true\n"
        "12,100.5,4.0,402.0,2000,false\n"
        "12,101.0,5.0,505.0,900000,true\n"
        "13,101.0,6.0,606.0,900001,true\n",
    )

    report = parse_month_archive(root=tmp_path, symbol="ETHUSDT", month="2025-08")
    fills = pl.read_parquet(report.canonical_path)
    events = pl.read_parquet(report.event_path)

    assert report.reused_trade_ids == 2
    assert report.duplicate_trade_ids == 2
    assert fills["timestamp_ms"].is_sorted()
    assert events["timestamp_ms"].to_list() == [1000, 2000, 900000, 900001]
    # Before the gap: one buy event at 2000 ms containing IDs 11 and 12.
    assert events.filter(pl.col("timestamp_ms") == 2000)["fill_count"].item() == 2
    assert report.quantity_difference == 0


def test_duplicate_ids_without_a_gap_still_fail(tmp_path: Path) -> None:
    layout = DataLayout(tmp_path)
    write_archive(
        layout,
        "10,100.0,1.0,100.0,1000,false\n"
        "11,100.0,2.0,200.0,2000,false\n"
        "11,101.0,3.0,303.0,2500,true\n",
    )
    with pytest.raises(ValueError, match="identity"):
        parse_month_archive(root=tmp_path, symbol="ETHUSDT", month="2025-08")


def _write_month_and_daily(layout: DataLayout, monthly_csv: str, daily_csv: str) -> None:
    source = layout.source_archive("BTCUSDT", "2022-02")
    source.parent.mkdir(parents=True)
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("BTCUSDT-trades-2022-02.csv", monthly_csv)
    daily = layout.source_daily_archive("BTCUSDT", "2022-02-14")
    daily.parent.mkdir(parents=True)
    with zipfile.ZipFile(daily, "w") as archive:
        archive.writestr("BTCUSDT-trades-2022-02-14.csv", daily_csv)


def test_daily_archive_fills_gap_in_monthly_archive(tmp_path: Path) -> None:
    layout = DataLayout(tmp_path)
    monthly = (
        "1,100.0,1.0,100.0,1000,false\n"
        "2,101.0,2.0,202.0,1000,false\n"
        "5,99.0,4.0,396.0,9000,true\n"
    )
    daily = (  # the day is complete: IDs 3 and 4 are missing from the monthly file
        "1,100.0,1.0,100.0,1000,false\n"
        "2,101.0,2.0,202.0,1000,false\n"
        "3,102.0,1.0,102.0,2000,false\n"
        "4,98.0,3.0,294.0,3000,true\n"
    )
    _write_month_and_daily(layout, monthly, daily)
    report = parse_month_archive(root=tmp_path, symbol="BTCUSDT", month="2022-02")

    assert report.supplement_rows_added == 2
    assert report.supplement_archives == ["BTCUSDT-trades-2022-02-14.zip"]
    assert report.fill_rows == 5
    fills = pl.read_parquet(layout.canonical_month("BTCUSDT", "2022-02"))
    assert fills["trade_id"].to_list() == [1, 2, 3, 4, 5]
    assert report.event_rows == 4  # (1-2 buy @1000), (3 buy @2000), (4 sell @3000), (5 sell @9000)
    assert list(layout.temporary_root().iterdir()) == []


def test_daily_archive_that_disagrees_with_monthly_fails(tmp_path: Path) -> None:
    layout = DataLayout(tmp_path)
    monthly = "1,100.0,1.0,100.0,1000,false\n"
    daily = "1,100.0,9.0,900.0,1000,false\n2,101.0,1.0,101.0,2000,false\n"
    _write_month_and_daily(layout, monthly, daily)
    with pytest.raises(ValueError, match="disagree"):
        parse_month_archive(root=tmp_path, symbol="BTCUSDT", month="2022-02")
