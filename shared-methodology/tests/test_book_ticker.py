from __future__ import annotations

from pathlib import Path
import zipfile

import polars as pl

from crypto_market_data import book_ticker


def write_month(root: Path, lines: list[str]) -> None:
    source = book_ticker.source_archive(root, "BTCUSDT", "2023-05")
    source.parent.mkdir(parents=True)
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("BTCUSDT-bookTicker-2023-05.csv", "\n".join(lines))


def test_url_and_paths(tmp_path):
    assert book_ticker.archive_url("BTCUSDT", "2023-05").endswith(
        "/futures/um/monthly/bookTicker/BTCUSDT/BTCUSDT-bookTicker-2023-05.zip"
    )
    assert "quote_grid_1s_v1" in str(book_ticker.quote_grid_month(tmp_path, "BTCUSDT", "2023-05"))


def test_parse_builds_canonical_and_last_quote_per_second(tmp_path):
    header = ",".join(book_ticker.COLUMNS)
    write_month(
        tmp_path,
        [
            header,
            "3,100.0,1.0,100.1,2.0,1000500,1000501",
            "1,99.9,1.0,100.0,2.0,1000100,1000101",
            "4,100.1,5.0,100.2,1.0,1001200,1001201",
            "5,100.3,1.0,100.3,1.0,1001900,1001901",
        ],
    )
    report = book_ticker.parse_month(root=tmp_path, symbol="BTCUSDT", month="2023-05")
    grid = pl.read_parquet(book_ticker.quote_grid_month(tmp_path, "BTCUSDT", "2023-05"))

    assert report.rows == 4
    assert report.crossed_or_locked_rows == 1
    assert report.grid_seconds == 2
    assert grid["second"].to_list() == [1000, 1001]
    # Within second 1000 the later update (id 3, t=1000500) is the quote in force.
    assert grid["best_bid_price"].to_list() == [100.0, 100.3]
    assert grid["updates"].to_list() == [2, 2]


def test_chunked_grid_resolves_seconds_split_across_chunks():
    quotes = pl.DataFrame(
        {
            "update_id": [1, 2, 3, 4],
            "best_bid_price": [1.0, 2.0, 3.0, 4.0],
            "best_bid_qty": [1.0] * 4,
            "best_ask_price": [1.5, 2.5, 3.5, 4.5],
            "best_ask_qty": [1.0] * 4,
            "transaction_time": [1_000, 1_400, 1_800, 2_100],
            "event_time": [1_000, 1_400, 1_800, 2_100],
        }
    ).lazy()
    grid = book_ticker.quote_grid(quotes, chunk_rows=2)
    assert grid["second"].to_list() == [1, 2]
    assert grid["best_bid_price"].to_list() == [3.0, 4.0]
    assert grid["updates"].to_list() == [3, 1]


def test_headerless_file(tmp_path):
    write_month(tmp_path, ["1,99.9,1.0,100.0,2.0,1000100,1000101"])
    report = book_ticker.parse_month(root=tmp_path, symbol="BTCUSDT", month="2023-05")
    assert report.rows == 1
