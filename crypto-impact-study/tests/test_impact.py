from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_impact.impact import (
    add_normalised_impact_trajectories,
    add_trailing_impact_normalisation,
    summarise_normalised_impact_trajectory,
)


def test_trailing_normalisation_excludes_current_bar() -> None:
    data = pd.DataFrame(
        {
            "end_time": pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC"),
            "signed_imbalance": [1.0, -2.0, 3.0, -4.0],
            "gross_volume": [10.0, 20.0, 30.0, 40.0],
            "current_return_bps": [2.0, -3.0, 4.0, -5.0],
            "duration_seconds": [10.0, 10.0, 20.0, 20.0],
        }
    )

    result = add_trailing_impact_normalisation(
        data, window="2h", min_periods=1
    )

    assert np.isnan(result.loc[0, "trailing_market_volume"])
    assert np.isclose(result.loc[1, "trailing_market_volume"], 10.0)
    assert np.isclose(result.loc[2, "trailing_market_volume"], 30.0)
    assert np.isclose(result.loc[2, "trailing_realised_volatility_bps"], np.sqrt(13.0))
    assert np.isclose(result.loc[1, "signed_current_impact_bps"], 3.0)
    assert np.isclose(result.loc[1, "normalised_imbalance"], 0.2)
    assert np.isclose(result.loc[1, "flow_rate_btc_per_second"], 2.0)
    assert result.loc[1, "flow_rate_ratio"] > 0


def test_normalised_impact_trajectory_includes_current_and_future_response() -> None:
    data = pd.DataFrame(
        {
            "signed_current_impact_bps": [4.0, 2.0],
            "trailing_realised_volatility_bps": [10.0, 10.0],
            "signed_future_return_1_bps": [-1.0, 1.0],
            "regime": ["high", "high"],
        }
    )

    enriched = add_normalised_impact_trajectories(data, horizons=(1,))
    summary = summarise_normalised_impact_trajectory(
        enriched, group_cols=["regime"], horizons=(0, 1)
    )

    assert np.allclose(enriched["normalised_total_impact_0"], [0.4, 0.2])
    assert np.allclose(enriched["normalised_total_impact_1"], [0.3, 0.3])
    assert np.isclose(summary.loc[summary["horizon"] == 1, "fraction_remaining"].iloc[0], 1.0)
