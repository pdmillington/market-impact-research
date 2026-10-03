from __future__ import annotations

import polars as pl

from crypto_market_data import bybit


HOUR = 3_600_000


def fake_kline_api(first_ms: int, last_ms: int):
    """Serves hourly bars in [first_ms, last_ms], newest first, like Bybit."""

    calls = []

    def fetch(path, params):
        calls.append(params)
        start, end = params["start"], params["end"]
        times = [t for t in range(first_ms, last_ms + 1, HOUR) if start <= t <= end]
        times = sorted(times, reverse=True)[: params["limit"]]
        return {"retCode": 0, "result": {"list": [[str(t), "1", "2", "0.5", str(t / HOUR), "10", "100"] for t in times]}}

    return fetch, calls


def test_kline_pagination_covers_range_without_gaps_or_duplicates(tmp_path):
    first, last = 5 * HOUR, 2_600 * HOUR
    fetch, calls = fake_kline_api(first, last)
    frame = bybit.fetch_klines(tmp_path, "kline_1h", "TESTUSDT", 0, 3_000 * HOUR, fetch=fetch)
    assert frame.height == (last - first) // HOUR + 1
    assert frame["open_time_ms"].is_sorted() and frame["open_time_ms"].n_unique() == frame.height
    assert len(calls) == 3                      # 3,000 hours in windows of 1,000
    assert frame["close"][0] == 5.0
    assert bybit.table_path(tmp_path, "kline_1h", "TESTUSDT").exists()
    assert (tmp_path / "source" / "bybit" / "kline_1h" / "symbol=TESTUSDT" / "pages.jsonl.gz").exists()


def test_funding_pages_backward_to_first_settlement(tmp_path):
    settlements = list(range(8 * HOUR, 8 * HOUR * 500, 8 * HOUR))

    def fetch(path, params):
        rows = sorted([t for t in settlements if t <= params["endTime"]], reverse=True)[: params["limit"]]
        return {"retCode": 0, "result": {"list": [{"fundingRateTimestamp": str(t), "fundingRate": "0.0001"} for t in rows]}}

    frame = bybit.fetch_funding(tmp_path, "TESTUSDT", 0, 10**15, fetch=fetch)
    assert frame.height == len(settlements)
    assert frame["time_ms"].to_list() == settlements
    assert pl.read_parquet(bybit.table_path(tmp_path, "funding", "TESTUSDT")).height == len(settlements)


def test_kline_error_is_raised(tmp_path):
    def fetch(path, params):
        return {"retCode": 10001, "retMsg": "params error"}

    try:
        bybit.fetch_klines(tmp_path, "kline_1d", "TESTUSDT", 0, 10 * 86_400_000, fetch=fetch)
    except RuntimeError as error:
        assert "params error" in str(error)
    else:
        raise AssertionError("expected RuntimeError")
