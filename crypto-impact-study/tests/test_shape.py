from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_impact.binning import bin_impact_shape
from crypto_impact.shape import compute_local_loglog_slopes, compute_lowess_curve


def test_equal_count_bins_have_confidence_interval_columns() -> None:
    observations = pd.DataFrame({"x": [1, 2, 3, 4], "signed_impact": [0.1, 0.2, 0.3, 0.4]})

    binned = bin_impact_shape(observations, x_col="x", n_bins=2, method="equal_count")

    assert binned.columns.tolist() == ["bin_id", "mean_x", "median_x", "mean_impact", "median_impact", "stderr", "ci_lower", "ci_upper", "n_obs"]
    assert binned["n_obs"].tolist() == [2, 2]
    assert np.allclose(binned["mean_x"], [1.5, 3.5])


def test_local_loglog_slopes() -> None:
    binned = pd.DataFrame({"mean_x": [1.0, 4.0, 16.0], "mean_impact": [2.0, 4.0, 8.0]})

    slopes = compute_local_loglog_slopes(binned)

    assert np.allclose(slopes["local_slope"], [0.5, 0.5])


def test_lowess_returns_positive_curve() -> None:
    observations = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "signed_impact": [0.1, 0.2, 0.3, 0.4]})

    curve = compute_lowess_curve(observations, x_col="x", frac=0.75)

    assert not curve.empty
    assert (curve["x"] > 0).all()
    assert (curve["lowess_impact"] > 0).all()
