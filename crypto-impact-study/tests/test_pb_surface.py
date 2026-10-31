"""Tests for crypto_impact.pb_surface: the default free-scale fit and the power-law (A3g) option."""

from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_impact.pb_collapse import pb_shape
from crypto_impact.pb_surface import fit_activity_surface


def _synthetic_cells(alpha=3.0, beta=0.15, gamma=0.4, q1=0.002, xi=0.25, r1=0.0005, psi=0.6, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for n in (250, 1000, 4000, 16000):
        q_n, r_n = q1 * n ** xi, r1 * n ** psi
        for side in (-1, 1):
            for log_q in np.linspace(np.log(0.005), np.log(0.8), 20):
                for log_a in np.linspace(np.log(n * 1e-6), np.log(n * 1e-5), 10):
                    mean = side * r_n * pb_shape(np.exp(log_q + gamma * log_a) / q_n, alpha, beta)
                    se = 0.02 * r_n
                    rows.append({"n_events": n, "side": side, "mean_logz": log_q, "mean_loga": log_a,
                                 "mean_y": mean + noise * se * rng.normal(), "standard_error": se, "count": 500})
    return pd.DataFrame(rows)


def test_scale_law_recovers_parameters():
    fit = fit_activity_surface(_synthetic_cells(), scale_law=True, n_starts=6)
    law = fit.scale_law
    assert abs(fit.gamma - 0.4) < 0.01
    assert abs(law["xi"] - 0.25) < 0.01 and abs(law["psi"] - 0.6) < 0.01
    assert abs(np.log(law["q1"] / 0.002)) < 0.05 and abs(np.log(law["r1"] / 0.0005)) < 0.05
    assert abs(fit.alpha - 3.0) < 0.1 and abs(fit.beta - 0.15) < 0.01


def test_scale_law_scales_follow_power_laws():
    fit = fit_activity_surface(_synthetic_cells(noise=1.0), scale_law=True, n_starts=4)
    sc = fit.scales
    slope = np.polyfit(np.log(sc["n_events"]), np.log(sc["q_scale"]), 1)[0]
    assert abs(slope - fit.scale_law["xi"]) < 1e-9


def test_default_is_free_scales():
    fit = fit_activity_surface(_synthetic_cells(), n_starts=4)
    assert fit.scale_law is None and len(fit.scales) == 4
    assert abs(fit.gamma - 0.4) < 0.02
