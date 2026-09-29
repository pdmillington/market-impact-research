from __future__ import annotations

import math
import sys
from pathlib import Path

import polars as pl


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from alpha_search.bars import (  # noqa: E402
    add_bar_horizon_targets,
    add_trailing_impact_features,
    build_fixed_event_bars,
    build_time_bars,
    coarsen_fixed_event_bars,
    marginal_impact_proxy,
)


def _events() -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "timestamp_ms": [0, 60_000, 300_000, 360_000, 600_000, 660_000],
            "aggressor_sign": [1, -1, 1, 1, -1, -1],
            "quantity": [1.0, 2.0, 1.0, 3.0, 2.0, 1.0],
            "first_price": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "last_price": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "min_price": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "max_price": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "fill_count": [1, 1, 1, 1, 1, 1],
            "price_level_count": [1, 1, 1, 1, 1, 1],
            "first_trade_id": [1, 2, 3, 4, 5, 6],
            "last_trade_id": [1, 2, 3, 4, 5, 6],
        },
        schema_overrides={
            "aggressor_sign": pl.Int8,
            "fill_count": pl.UInt32,
            "price_level_count": pl.UInt32,
            "first_trade_id": pl.UInt64,
            "last_trade_id": pl.UInt64,
        },
    ).lazy()


def test_fixed_event_bars_continue_in_event_order() -> None:
    result = build_fixed_event_bars(
        _events(), events_per_bar=2
    ).collect()

    assert result["event_count"].to_list() == [2, 2, 2]
    assert result["first_trade_id"].to_list() == [1, 3, 5]
    assert result["last_trade_id"].to_list() == [2, 4, 6]
    assert result["signed_imbalance"].to_list() == [-1.0, 4.0, -3.0]


def test_coarsened_event_bars_preserve_endpoints_and_totals() -> None:
    source = build_fixed_event_bars(_events(), events_per_bar=2)
    result = coarsen_fixed_event_bars(
        source,
        source_events_per_bar=2,
        target_events_per_bar=4,
    ).collect()

    assert result.height == 1
    assert result["event_count"][0] == 4
    assert result["first_trade_id"][0] == 1
    assert result["last_trade_id"][0] == 4
    assert result["gross_volume"][0] == 7.0
    assert result["signed_imbalance"][0] == 3.0


def test_event_horizon_keeps_realized_clock_duration() -> None:
    bars = build_fixed_event_bars(_events(), events_per_bar=2)
    result = add_bar_horizon_targets(bars, horizons=(1, 2)).collect()

    assert result["future_elapsed_1_minutes"].to_list()[:2] == [5.0, 5.0]
    expected = 10_000.0 * math.log(103.0 / 101.0)
    assert math.isclose(result["future_return_1_bps"][0], expected)


def test_five_minute_bars_use_the_same_corrected_events() -> None:
    result = build_time_bars(_events(), every="5m").collect()

    assert result.height == 3
    assert result["event_count"].to_list() == [2, 2, 2]
    assert result["signed_imbalance"].to_list() == [-1.0, 4.0, -3.0]


def test_marginal_impact_proxy_declines_with_existing_imbalance() -> None:
    data = pl.DataFrame(
        {
            "x": [0.01, 0.10],
            "volume": [1_000.0, 1_000.0],
            "volatility": [100.0, 100.0],
        }
    )
    result = data.select(
        marginal_impact_proxy(
            normalised_imbalance=pl.col("x"),
            trailing_volume=pl.col("volume"),
            trailing_volatility_bps=pl.col("volatility"),
            log_slope=0.5,
            imbalance_scale=0.01,
        ).alias("marginal")
    )

    assert result["marginal"][0] > result["marginal"][1] > 0


def test_trailing_impact_features_exclude_the_current_bar() -> None:
    bars = pl.DataFrame(
        {
            "bar_number": [0, 1, 2],
            "end_time_ms": [0, 60_000, 120_000],
            "gross_volume": [10.0, 20.0, 30.0],
            "current_return_bps": [3.0, 4.0, 12.0],
            "signed_imbalance": [1.0, -2.0, 3.0],
            "signed_current_impact_bps": [3.0, -4.0, 12.0],
            "duration_seconds": [60.0, 60.0, 60.0],
        }
    ).lazy()

    result = add_trailing_impact_features(
        bars, window="1d", min_samples=1
    ).collect()

    assert result["trailing_market_volume"].to_list()[1:] == [10.0, 30.0]
    assert math.isclose(result["trailing_realised_volatility_bps"][2], 5.0)
    assert math.isclose(result["normalised_imbalance"][2], 0.1)
