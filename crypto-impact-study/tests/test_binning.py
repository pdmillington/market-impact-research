from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_impact.binning import bin_impact_observations, make_log_bins


def test_make_log_bins_returns_geometric_edges() -> None:
    values = pd.Series([1.0, 10.0, 100.0])

    bins = make_log_bins(values, n_bins=2)

    assert np.allclose(bins, [1.0, 10.0, 100.0])


def test_bin_impact_observations_computes_summary_stats() -> None:
    observations = pd.DataFrame(
        {
            "abs_imbalance": [1.0, 2.0, 10.0, 20.0],
            "signed_impact": [0.01, 0.03, 0.10, 0.30],
        }
    )

    binned = bin_impact_observations(observations, n_bins=2)

    assert binned["n_obs"].tolist() == [2, 2]
    assert np.allclose(binned["mean_abs_imbalance"], [1.5, 15.0])
    assert np.allclose(binned["mean_signed_impact"], [0.02, 0.20])
    assert binned["standard_error"].notna().all()
