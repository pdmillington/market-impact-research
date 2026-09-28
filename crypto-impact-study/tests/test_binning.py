from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_impact.binning import bin_by_absolute_imbalance_ratio, bin_impact_by_shared_edges, bin_impact_curve_with_uncertainty, bin_impact_observations, make_log_bins, imbalance_ratio_volume_heatmap


def test_make_log_bins_returns_geometric_edges() -> None:
    values = pd.Series([1.0, 10.0, 100.0])

    bins = make_log_bins(values, n_bins=2)

    assert np.allclose(bins, [1.0, 10.0, 100.0])


def test_bin_impact_observations_defaults_to_equal_count_distribution_summary() -> None:
    observations = pd.DataFrame(
        {
            "abs_imbalance": [1.0, 2.0, 10.0, 20.0],
            "abs_signed_imbalance": [1.0, 2.0, 10.0, 20.0],
            "gross_volume": [10.0, 10.0, 20.0, 20.0],
            "absolute_imbalance_ratio": [0.1, 0.2, 0.5, 1.0],
            "signed_impact": [0.01, 0.03, 0.10, 0.30],
        }
    )

    binned = bin_impact_observations(observations, n_bins=2)

    assert binned["n_obs"].tolist() == [2, 2]
    assert np.allclose(binned["mean_x"], [1.5, 15.0])
    assert np.allclose(binned["mean_impact"], [0.02, 0.20])
    assert {"impact_p10", "impact_p25", "impact_p75", "impact_p90"}.issubset(binned.columns)


def test_bin_by_absolute_imbalance_ratio_computes_requested_fields() -> None:
    observations = pd.DataFrame(
        {
            "abs_signed_imbalance": [1.0, 2.0, 3.0, 4.0],
            "gross_volume": [10.0, 10.0, 10.0, 10.0],
            "absolute_imbalance_ratio": [0.1, 0.2, 0.3, 0.4],
            "signed_impact": [0.01, 0.02, 0.03, 0.04],
        }
    )

    binned = bin_by_absolute_imbalance_ratio(observations, n_bins=2, method="equal_count")

    assert binned["n_obs"].tolist() == [2, 2]
    assert np.allclose(binned["mean_x"], [0.15, 0.35])
    assert np.allclose(binned["median_x"], [0.15, 0.35])
    assert np.allclose(binned["mean_impact"], [0.015, 0.035])
    assert np.allclose(binned["median_impact"], [0.015, 0.035])
    assert np.allclose(binned["mean_absolute_imbalance_ratio"], [0.15, 0.35])


def test_imbalance_ratio_volume_heatmap_computes_cell_means() -> None:
    observations = pd.DataFrame(
        {
            "absolute_imbalance_ratio": [0.1, 0.2, 0.8, 0.9],
            "gross_volume": [10.0, 20.0, 100.0, 200.0],
            "signed_impact": [0.01, 0.02, 0.03, 0.04],
        }
    )

    heatmap = imbalance_ratio_volume_heatmap(observations, imbalance_ratio_bins=2, volume_bins=2)

    assert len(heatmap) == 2
    assert set(heatmap.columns).issuperset({"mean_absolute_imbalance_ratio", "mean_gross_volume", "mean_signed_impact", "n_obs"})


def test_shared_edges_are_used_for_each_regime() -> None:
    observations = pd.DataFrame(
        {
            "regime": ["low"] * 4 + ["high"] * 4,
            "x": [1.0, 2.0, 5.0, 8.0, 1.5, 3.0, 6.0, 9.0],
            "impact": [1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5],
        }
    )

    result, edges = bin_impact_by_shared_edges(
        observations,
        regime_col="regime",
        x_col="x",
        impact_col="impact",
        n_bins=2,
    )

    assert len(edges) == 3
    assert set(result["regime"]) == {"low", "high"}
    assert set(result["bin_id"]) == {0, 1}
    assert {"mean_impact", "standard_error", "n_obs"}.issubset(result.columns)


def test_impact_curve_uncertainty_matches_bin_counts() -> None:
    observations = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0, 4.0], "impact": [1.0, 3.0, 5.0, 7.0]}
    )

    result = bin_impact_curve_with_uncertainty(
        observations, x_col="x", impact_col="impact", n_bins=2
    )

    assert result["n_obs"].tolist() == [2, 2]
    assert np.allclose(result["impact_std"], [np.sqrt(2.0), np.sqrt(2.0)])
    assert np.allclose(result["standard_error"], [1.0, 1.0])
