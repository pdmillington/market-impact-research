from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from alpha_search.impact_surface import (  # noqa: E402
    fit_common_shape_power_surface,
    fit_log_activity_surface,
    fit_power_surface,
    predict_surface,
    surface_cells,
)


def test_power_surface_recovers_distinct_elasticities() -> None:
    rng = np.random.default_rng(7)
    imbalance = np.exp(rng.uniform(math.log(0.01), math.log(0.9), 80_000))
    activity = np.exp(rng.uniform(math.log(1e-4), math.log(2e-2), 80_000))
    impact = 2.5 * imbalance**0.8 * activity**0.4
    cells = surface_cells(
        imbalance,
        activity,
        impact,
        imbalance_bins=12,
        activity_bins=6,
    )
    fit = fit_power_surface(
        cells,
        imbalance_exponents=np.linspace(0.70, 0.90, 41),
        activity_exponents=np.linspace(0.30, 0.50, 41),
    )

    assert math.isclose(fit.imbalance_exponent, 0.8, abs_tol=0.01)
    assert math.isclose(fit.activity_exponent, 0.4, abs_tol=0.01)
    assert math.isclose(fit.collapse_exponent, 0.5, abs_tol=0.02)
    assert fit.cell_rmse < 2e-4


def test_log_activity_surface_predicts_zero_at_zero_imbalance() -> None:
    rng = np.random.default_rng(11)
    imbalance = rng.uniform(0.001, 0.95, 50_000)
    activity = np.exp(rng.uniform(math.log(1e-4), math.log(1e-2), 50_000))
    scale = 0.3
    impact = 1.7 * np.log1p(2.0 * imbalance / scale) * activity**0.6
    cells = surface_cells(imbalance, activity, impact, imbalance_bins=10, activity_bins=5)
    fit = fit_log_activity_surface(
        cells,
        curvatures=np.logspace(-1, 1, 81),
        activity_exponents=np.linspace(0.50, 0.70, 41),
    )
    prediction = predict_surface(
        fit,
        np.array([0.0, 0.2, 0.8]),
        np.array([0.001, 0.001, 0.003]),
    )

    assert prediction[0] == 0.0
    assert np.all(prediction[1:] > 0)
    assert math.isclose(fit.activity_exponent, 0.6, abs_tol=0.03)


def test_surface_grid_can_be_reused_out_of_sample() -> None:
    rng = np.random.default_rng(23)
    training_z = rng.uniform(0.01, 0.9, 10_000)
    training_a = rng.uniform(0.001, 0.02, 10_000)
    training_y = training_z * np.sqrt(training_a)
    training = surface_cells(training_z, training_a, training_y, imbalance_bins=8, activity_bins=4)

    validation = surface_cells(
        training_z * 0.9,
        training_a * 1.1,
        training_y,
        imbalance_bins=8,
        activity_bins=4,
        imbalance_cuts=training.imbalance_cuts,
        activity_cuts=training.activity_cuts,
    )

    assert np.array_equal(validation.imbalance_cuts, training.imbalance_cuts)
    assert np.array_equal(validation.activity_cuts, training.activity_cuts)


def test_common_shape_can_retain_side_specific_amplitudes() -> None:
    rng = np.random.default_rng(31)
    imbalance = np.exp(rng.uniform(math.log(0.01), math.log(0.9), 60_000))
    activity = np.exp(rng.uniform(math.log(1e-4), math.log(2e-2), 60_000))
    base = imbalance**0.9 * activity**0.5
    buy = surface_cells(imbalance, activity, 3.0 * base, imbalance_bins=10, activity_bins=5)
    sell = surface_cells(imbalance, activity, 2.0 * base, imbalance_bins=10, activity_bins=5)
    fit = fit_common_shape_power_surface(
        buy,
        sell,
        imbalance_exponents=np.linspace(0.8, 1.0, 41),
        activity_exponents=np.linspace(0.4, 0.6, 41),
    )

    assert math.isclose(fit.imbalance_exponent, 0.9, abs_tol=0.01)
    assert math.isclose(fit.activity_exponent, 0.5, abs_tol=0.01)
    assert math.isclose(fit.buy_amplitude / fit.sell_amplitude, 1.5, rel_tol=0.01)
