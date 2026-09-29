import polars as pl
import pytest

from crypto_market_data.events import timestamp_direction_events


def test_timestamp_direction_events_preserve_sweep_path_and_direction_changes() -> None:
    fills = pl.DataFrame(
        {
            "trade_id": [1, 2, 3, 4, 5],
            "timestamp_ms": [1000, 1000, 1000, 1000, 1001],
            "price": [100.0, 101.0, 99.0, 102.0, 103.0],
            "quantity": [1.0, 2.0, 4.0, 1.0, 3.0],
            "aggressor_sign": [1, 1, -1, 1, 1],
        }
    )

    events = timestamp_direction_events(fills.lazy()).collect()

    assert events.height == 4
    first = events.row(0, named=True)
    assert first["quantity"] == 3.0
    assert first["vwap"] == pytest.approx((100.0 + 2 * 101.0) / 3)
    assert first["first_price"] == 100.0
    assert first["last_price"] == 101.0
    assert first["min_price"] == 100.0
    assert first["max_price"] == 101.0
    assert first["fill_count"] == 2
    assert first["price_level_count"] == 2

    # The buy after the intervening sell is a new event even at the same timestamp.
    assert events["aggressor_sign"].to_list() == [1, -1, 1, 1]
    assert events["first_trade_id"].to_list() == [1, 3, 4, 5]
