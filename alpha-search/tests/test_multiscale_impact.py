from __future__ import annotations

import math
import sys
from pathlib import Path

import polars as pl
import pytest


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from alpha_search.multiscale_impact import (  # noqa: E402
    build_trailing_impact_observations,
    resolve_measurement_windows,
)


def test_resolve_measurement_windows_deduplicates_same_scale() -> None:
    assert resolve_measurement_windows(1_000, ("same_bar", 1_000, 4_000)) == (
        1_000,
        4_000,
    )
    with pytest.raises(ValueError, match="exact multiple"):
        resolve_measurement_windows(500, (1_200,))


def test_trailing_impact_window_uses_pre_window_benchmark() -> None:
    bars = pl.DataFrame(
        {
            "bar_number": [0, 1, 2, 3],
            "start_time_ms": [0, 2_000, 4_000, 6_000],
            "end_time_ms": [1_000, 3_000, 5_000, 7_000],
            "start_price": [100.0, 101.0, 102.0, 104.0],
            "end_price": [101.0, 102.0, 104.0, 103.0],
            "gross_volume": [10.0, 20.0, 30.0, 40.0],
            "signed_imbalance": [2.0, 3.0, -1.0, -2.0],
            "trailing_market_volume": [1_000.0, 1_100.0, 1_200.0, 1_300.0],
            "trailing_realised_volatility_bps": [100.0, 110.0, 120.0, 130.0],
        }
    )
    result = build_trailing_impact_observations(
        bars, events_per_bar=2, measurement_events=4
    ).collect()

    assert result["normalised_imbalance"][0] is None
    assert math.isclose(result["signed_imbalance"][1], 5.0)
    assert math.isclose(result["measurement_gross_volume"][1], 30.0)
    assert math.isclose(result["trailing_market_volume"][1], 1_000.0)
    assert math.isclose(result["normalised_imbalance"][1], 0.005)
    expected_return = 10_000.0 * math.log(102.0 / 100.0)
    assert math.isclose(result["measurement_return_bps"][1], expected_return)
    assert math.isclose(
        result["normalised_signed_impact"][1], expected_return / 100.0
    )
    assert result["measurement_side"].to_list() == [None, 1.0, 1.0, -1.0]
    assert result["decision_side"].to_list() == [1.0, 1.0, -1.0, -1.0]
    assert result["side_agreement"].to_list() == [None, True, False, True]
    assert result["is_non_overlapping_measurement"].to_list() == [
        False,
        True,
        False,
        True,
    ]
