from pathlib import Path
import zipfile

import polars as pl

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
    assert list(layout.temporary_root().iterdir()) == []
