import numpy as np
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from alpha_search.lagged_impact import (
    fit_equal_period_ols,
    lagged_design,
    predict_lagged_correction,
)


def test_lag_impulse_sign_encodes_same_and_opposite_direction() -> None:
    side = np.array([1, 1, -1, -1])
    lag = np.array([0.2, -0.3, 0.4, -0.5])
    design = lagged_design(
        "lag_impact",
        current_side=side,
        lag_impulse=lag,
        previous_log_activity=np.zeros(4),
        log_duration=np.zeros(4),
    )
    np.testing.assert_allclose(design[:, 2], lag)
    np.testing.assert_allclose(design[:, 0], [1, 1, 0, 0])
    np.testing.assert_allclose(design[:, 1], [0, 0, 1, 1])


def test_equal_period_fit_recovers_nested_coefficients() -> None:
    rng = np.random.default_rng(934)
    observations = 600
    side = rng.choice([-1, 1], observations)
    lag = rng.normal(0, 0.3, observations)
    activity = rng.normal(0, 0.8, observations)
    duration = rng.normal(0, 0.5, observations)
    period = np.repeat(np.arange(6), observations // 6)
    expected = np.array([0.01, -0.02, -0.4, 0.25])
    design = lagged_design(
        "lag_activity",
        current_side=side,
        lag_impulse=lag,
        previous_log_activity=activity,
        log_duration=duration,
    )
    response = design @ expected
    fit = fit_equal_period_ols(
        "lag_activity",
        current_side=side,
        lag_impulse=lag,
        previous_log_activity=activity,
        log_duration=duration,
        residual=response,
        period=period,
    )
    np.testing.assert_allclose(fit.coefficients, expected, atol=1e-10)
    predicted = predict_lagged_correction(
        fit,
        current_side=side,
        lag_impulse=lag,
        previous_log_activity=activity,
        log_duration=duration,
    )
    np.testing.assert_allclose(predicted, response, atol=1e-10)


def test_calendar_periods_receive_equal_total_weight() -> None:
    side = np.array([1, -1] + [1, -1] * 100)
    lag = np.ones(202)
    activity = np.zeros(202)
    duration = np.zeros(202)
    period = np.array([0, 0] + [1] * 200)
    response = np.array([1.0, 1.0] + [3.0] * 200)
    fit = fit_equal_period_ols(
        "bias_only",
        current_side=side,
        lag_impulse=lag,
        previous_log_activity=activity,
        log_duration=duration,
        residual=response,
        period=period,
    )
    # Each side receives the equally weighted period mean: (1 + 3) / 2 rather
    # than the bar-weighted 2.98.
    np.testing.assert_allclose(fit.coefficients, [2.0, 2.0], atol=1e-10)
