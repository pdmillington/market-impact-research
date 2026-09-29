import polars as pl

from alpha_search.screen_features import (
    add_short_bar_screen_features,
    screen_feature_catalog,
)


def test_short_bar_features_use_prior_history() -> None:
    bars = pl.DataFrame(
        {
            "bar_number": [0, 1, 2, 3],
            "end_price": [100.0, 101.0, 103.0, 102.0],
            "signed_imbalance": [2.0, -1.0, 3.0, -2.0],
            "abs_signed_imbalance": [2.0, 1.0, 3.0, 2.0],
            "gross_volume": [10.0, 10.0, 10.0, 10.0],
            "duration_seconds": [2.0, 2.0, 2.0, 2.0],
            "range_bps": [5.0, 5.0, 5.0, 5.0],
            "signed_current_impact_bps": [1.0, 2.0, 3.0, 4.0],
            "normalised_signed_impact": [0.1, 0.2, 0.3, 0.4],
            "trailing_market_volume": [100.0, 100.0, 100.0, 100.0],
            "trailing_realised_volatility_bps": [20.0, 20.0, 20.0, 20.0],
            "flow_rate_btc_per_second": [5.0, 5.0, 5.0, 5.0],
            "event_count": [1_000, 1_000, 1_000, 1_000],
            "fill_count": [1_000, 2_000, 1_500, 1_000],
            "event_price_level_count_sum": [1_000, 1_100, 1_200, 1_300],
            "signed_event_count": [100, -200, 300, -400],
        }
    )
    result = add_short_bar_screen_features(
        bars, events_per_bar=1_000, history_event_counts=(1_000,)
    ).collect()

    assert result["aligned_pretrend_1000_events"][1] is None
    assert result["aligned_pretrend_1000_events"][2] is not None
    expected_flow = 3.0 / 10.0  # current buy side times prior bar's -1/10 is checked below
    assert abs(result["aligned_flow_persistence_1000_events"][2] + expected_flow / 3) < 1e-12
    assert result["fills_per_event"].to_list() == [1.0, 2.0, 1.5, 1.0]


def test_catalog_contains_each_preregistered_family() -> None:
    catalog = dict(screen_feature_catalog())
    assert catalog["impact_residual"] == "fitted_impact"
    assert catalog["aligned_pretrend_4000_events"] == "pretrend"
    assert catalog["aligned_flow_persistence_12000_events"] == "flow_persistence"
