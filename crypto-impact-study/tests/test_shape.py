from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_impact.binning import bin_impact_shape
from crypto_impact.shape import block_bootstrap_impact_models, block_bootstrap_power_law, compare_impact_models_by_regime, compute_local_loglog_slopes


def test_equal_count_bins_have_percentile_columns() -> None:
    observations = pd.DataFrame({"x": [1, 2, 3, 4], "signed_impact": [0.1, 0.2, 0.3, 0.4]})

    binned = bin_impact_shape(observations, x_col="x", n_bins=2, method="equal_count")

    assert binned.columns.tolist() == [
        "bin_id",
        "mean_x",
        "median_x",
        "mean_impact",
        "median_impact",
        "impact_p10",
        "impact_p25",
        "impact_p75",
        "impact_p90",
        "n_obs",
    ]
    assert binned["n_obs"].tolist() == [2, 2]
    assert np.allclose(binned["mean_x"], [1.5, 3.5])


def test_local_loglog_slopes() -> None:
    binned = pd.DataFrame({"mean_x": [1.0, 4.0, 16.0], "mean_impact": [2.0, 4.0, 8.0]})

    slopes = compute_local_loglog_slopes(binned)

    assert np.allclose(slopes["local_slope"], [0.5, 0.5])


def test_block_bootstrap_power_law_returns_requested_samples() -> None:
    x = np.tile(np.geomspace(1.0, 100.0, 20), 3)
    observations = pd.DataFrame(
        {
            "day": np.repeat(["a", "b", "c"], 20),
            "x": x,
            "impact": 0.2 * np.sqrt(x),
        }
    )

    result = block_bootstrap_power_law(
        observations,
        block_col="day",
        x_col="x",
        impact_col="impact",
        n_bins=6,
        n_bootstrap=5,
        min_obs_per_bin=3,
        random_state=7,
    )

    assert len(result) == 5
    assert np.allclose(result["delta"], 0.5, atol=0.03)


def test_block_bootstrap_impact_models_uses_weighted_level_estimator() -> None:
    x = np.tile(np.geomspace(0.01, 1.0, 30), 3)
    observations = pd.DataFrame(
        {
            "day": np.repeat(["a", "b", "c"], 30),
            "x": x,
            "impact": 0.2 * np.sqrt(x),
        }
    )

    result = block_bootstrap_impact_models(
        observations,
        block_col="day",
        x_col="x",
        impact_col="impact",
        n_bins=6,
        n_bootstrap=4,
        min_obs_per_bin=3,
        random_state=8,
    )

    assert len(result) == 4
    assert np.allclose(result["free_power_delta"], 0.5, atol=0.03)
    assert {"log_minus_power_weighted_rmse", "square_root_weighted_rmse"}.issubset(result.columns)


def test_compare_impact_models_by_regime_uses_common_bins() -> None:
    rows = []
    for regime, scale in [("low", 1.0), ("high", 1.2)]:
        for bin_id, x in enumerate(np.geomspace(0.01, 1.0, 8)):
            rows.append(
                {
                    "regime": regime,
                    "bin_id": bin_id,
                    "mean_x": x,
                    "mean_impact": scale * np.sqrt(x),
                    "standard_error": 0.01,
                    "n_obs": 100,
                }
            )
    curves = pd.DataFrame(rows)

    summary, predictions, common = compare_impact_models_by_regime(
        curves,
        regime_col="regime",
        regimes=["low", "high"],
        min_obs_per_bin=50,
    )

    assert set(summary["regime"]) == {"low", "high"}
    assert set(predictions["regime"]) == {"low", "high"}
    assert common["bin_id"].nunique() == 8
