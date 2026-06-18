from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_impact.regression import fit_power_law, predict_power_law


def test_fit_power_law_recovers_known_exponent() -> None:
    q = np.array([1.0, 4.0, 9.0, 16.0])
    impact = 2.0 * np.sqrt(q)
    binned = pd.DataFrame({"mean_abs_imbalance": q, "mean_signed_impact": impact})

    fit = fit_power_law(binned)

    assert np.isclose(fit.amplitude, 2.0)
    assert np.isclose(fit.delta, 0.5)
    assert fit.n_obs == 4


def test_predict_power_law() -> None:
    q = np.array([1.0, 4.0])
    binned = pd.DataFrame({"mean_abs_imbalance": q, "mean_signed_impact": [3.0, 6.0]})
    fit = fit_power_law(binned)

    predictions = predict_power_law(q, fit)

    assert np.allclose(predictions, [3.0, 6.0])
