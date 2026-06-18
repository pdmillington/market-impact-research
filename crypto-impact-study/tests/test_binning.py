from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_impact.binning import bin_by_participation_ratio, bin_impact_observations, make_log_bins


def test_make_log_bins_returns_geometric_edges() -> None:
    values = pd.Series([1.0, 10.0, 100.0])

    bins = make_log_bins(values, n_bins=2)

    assert np.allclose(bins, [1.0, 10.0, 100.0])


def test_bin_impact_observations_defaults_to_equal_count_summary() -> None:
    observations = pd.DataFrame(
        {
            "abs_imbalance": [1.0, 2.0, 10.0, 20.0],
            "abs_signed_imbalance": [1.0, 2.0, 10.0, 20.0],
            "gross_volume": [10.0, 10.0, 20.0, 20.0],
            "participation_ratio": [0.1, 0.2, 0.5, 1.0],
            "signed_impact": [0.01, 0.03, 0.10, 0.30],
        }
    )

    binned = bin_impact_observations(observations, n_bins=2)

    assert binned["n_obs"].tolist() == [2, 2]
    assert np.allclose(binned["mean_x"], [1.5, 15.0])
    assert np.allclose(binned["mean_impact"], [0.02, 0.20])
    assert binned["stderr"].notna().all()
    assert {"ci_lower", "ci_upper"}.issubset(binned.columns)


def test_bin_by_participation_ratio_computes_requested_fields() -> None:
    observations = pd.DataFrame(
        {
            "abs_signed_imbalance": [1.0, 2.0, 3.0, 4.0],
            "gross_volume": [10.0, 10.0, 10.0, 10.0],
            "participation_ratio": [0.1, 0.2, 0.3, 0.4],
            "signed_impact": [0.01, 0.02, 0.03, 0.04],
        }
    )

    binned = bin_by_participation_ratio(observations, n_bins=2, method="equal_count")

    assert binned["n_obs"].tolist() == [2, 2]
    assert np.allclose(binned["mean_x"], [0.15, 0.35])
    assert np.allclose(binned["median_x"], [0.15, 0.35])
    assert np.allclose(binned["mean_impact"], [0.015, 0.035])
    assert np.allclose(binned["mean_participation_ratio"], [0.15, 0.35])
