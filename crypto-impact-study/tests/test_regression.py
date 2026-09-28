from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_impact.regression import RegressionFilters, compare_impact_models, fit_power_law, predict_power_law


def test_fit_power_law_recovers_known_exponent() -> None:
    q = np.array([1.0, 4.0, 9.0, 16.0])
    impact = 2.0 * np.sqrt(q)
    binned = pd.DataFrame({"mean_abs_imbalance": q, "mean_signed_impact": impact})

    fit = fit_power_law(binned)

    assert np.isclose(fit.amplitude, 2.0)
    assert np.isclose(fit.delta, 0.5)
    assert fit.n_obs == 4
    assert fit.n_fit_bins == 4
    assert fit.standard_error_delta >= 0


def test_fit_power_law_applies_user_filters() -> None:
    binned = pd.DataFrame(
        {
            "mean_x": [1.0, 2.0, 4.0, 8.0],
            "mean_abs_signed_imbalance": [1.0, 2.0, 4.0, 8.0],
            "mean_absolute_imbalance_ratio": [0.01, 0.02, 0.04, 0.08],
            "mean_impact": [0.001, 0.004, 0.008, 0.016],
            "n_obs": [5, 50, 50, 50],
        }
    )
    filters = RegressionFilters(min_abs_signed_imbalance=2.0, min_mean_impact=0.004, min_obs_per_bin=30)

    fit = fit_power_law(binned, filters=filters)

    assert fit.n_fit_bins == 3
    assert fit.filters == {
        "min_abs_signed_imbalance": 2.0,
        "min_mean_impact": 0.004,
        "min_absolute_imbalance_ratio": None,
        "min_obs_per_bin": 30,
    }


def test_predict_power_law() -> None:
    q = np.array([1.0, 4.0])
    binned = pd.DataFrame({"mean_abs_imbalance": q, "mean_signed_impact": [3.0, 6.0]})
    fit = fit_power_law(binned)

    predictions = predict_power_law(q, fit)

    assert np.allclose(predictions, [3.0, 6.0])


def test_compare_impact_models_recovers_square_root_curve() -> None:
    x = np.geomspace(0.01, 1.0, 12)
    binned = pd.DataFrame(
        {
            "mean_x": x,
            "mean_impact": 2.0 * np.sqrt(x),
            "standard_error": np.full_like(x, 0.01),
            "n_obs": np.full_like(x, 100, dtype=int),
        }
    )

    summary, predictions = compare_impact_models(binned)

    free_power = summary.set_index("model").loc["free_power"]
    assert np.isclose(free_power["delta"], 0.5)
    assert np.isclose(free_power["rmse"], 0.0, atol=1e-8)
    assert set(predictions["model"]) == {
        "square_root", "free_power", "linear", "logarithmic"
    }
