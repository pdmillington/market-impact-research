import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from alpha_search.pb_scaling import (  # noqa: E402
    fit_pb_scaling,
    pb_shape,
    predict_pb_scaling,
)


def test_pb_shape_has_requested_asymptotic_exponents() -> None:
    small = pb_shape(
        np.array([1e-5, 2e-5]),
        alpha=2.0,
        beta=0.5,
        central_exponent=0.8,
    )
    large = pb_shape(
        np.array([1e4, 2e4]),
        alpha=2.0,
        beta=0.5,
        central_exponent=0.8,
    )
    assert np.isclose(np.log2(small[1] / small[0]), 0.8, atol=1e-5)
    assert np.isclose(np.log2(large[1] / large[0]), 0.3, atol=1e-5)


def test_generalised_scaling_recovers_synthetic_predictions() -> None:
    rng = np.random.default_rng(7)
    n = np.repeat(np.array([200, 500, 1_000, 2_000, 4_000]), 80)
    side = np.tile(np.repeat(np.array([-1, 1]), 40), 5)
    z = rng.uniform(0.02, 0.95, len(n))
    activity = np.exp(rng.uniform(np.log(1e-5), np.log(2e-2), len(n)))
    ratio = n / 1_000
    q = 0.002 * ratio**0.9
    r = np.where(side > 0, 0.018, 0.016) * ratio**0.55
    u = z * activity**0.45 / q
    y = r * pb_shape(u, alpha=1.7, beta=0.55, central_exponent=0.82)

    fit = fit_pb_scaling(
        n,
        side,
        z,
        activity,
        y,
        family="pb_activity_generalised",
        n_starts=3,
    )
    prediction = predict_pb_scaling(fit, n, side, z, activity)
    assert np.sqrt(np.mean((y - prediction) ** 2)) < 1e-5
    assert abs(fit.activity_exponent_gamma - 0.45) < 0.08
    assert abs(fit.central_exponent_p - 0.82) < 0.08


def test_vertical_activity_scaling_recovers_synthetic_predictions() -> None:
    rng = np.random.default_rng(19)
    n = np.repeat(np.array([200, 500, 1_000, 2_000, 4_000]), 120)
    side = np.tile(np.repeat(np.array([-1, 1]), 60), 5)
    ratio = n / 1_000
    activity_scale = 0.002 * ratio**0.95
    activity = activity_scale * np.exp(rng.uniform(-2.0, 2.0, len(n)))
    z = rng.uniform(0.02, 0.95, len(n))
    q = 0.0025 * ratio**0.85
    r = np.where(side > 0, 0.019, 0.016) * ratio**0.58
    gamma = 0.48
    delta = -0.32
    u = z * activity**gamma / q
    y = (
        r
        * (activity / activity_scale) ** delta
        * pb_shape(u, alpha=1.6, beta=0.52, central_exponent=1.0)
    )

    fit = fit_pb_scaling(
        n,
        side,
        z,
        activity,
        y,
        family="pb_activity_vertical",
        n_starts=5,
    )
    prediction = predict_pb_scaling(fit, n, side, z, activity)
    assert np.sqrt(np.mean((y - prediction) ** 2)) < 2e-5
    assert abs(fit.activity_exponent_gamma - gamma) < 0.10
    assert abs(fit.activity_vertical_exponent_delta - delta) < 0.10


def test_activity_dependent_beta_recovers_synthetic_predictions() -> None:
    rng = np.random.default_rng(31)
    n = np.repeat(np.array([200, 500, 1_000, 2_000, 4_000]), 160)
    side = np.tile(np.repeat(np.array([-1, 1]), 80), 5)
    ratio = n / 1_000
    activity_scale = 0.0022 * ratio**0.98
    log_relative_activity = rng.uniform(-2.0, 2.0, len(n))
    activity = activity_scale * np.exp(log_relative_activity)
    z = rng.uniform(0.02, 0.95, len(n))
    q = 0.0024 * ratio**0.88
    r = np.where(side > 0, 0.020, 0.017) * ratio**0.56
    gamma = 0.46
    beta_reference = 0.57
    beta_activity_slope = 0.62
    beta_logit = np.log(beta_reference / (1.0 - beta_reference))
    local_beta = 1.0 / (
        1.0 + np.exp(-(beta_logit + beta_activity_slope * log_relative_activity))
    )
    u = z * activity**gamma / q
    y = r * pb_shape(u, alpha=1.65, beta=local_beta, central_exponent=1.0)

    fit = fit_pb_scaling(
        n,
        side,
        z,
        activity,
        y,
        family="pb_activity_beta",
        n_starts=6,
    )
    prediction = predict_pb_scaling(fit, n, side, z, activity)
    # The fitter estimates the centring activity scale from the finite sample,
    # rather than receiving the data-generating scale directly.
    assert np.sqrt(np.mean((y - prediction) ** 2)) < 6e-5
    assert abs(fit.activity_exponent_gamma - gamma) < 0.10
    assert abs(fit.shape_beta_activity_slope - beta_activity_slope) < 0.12
