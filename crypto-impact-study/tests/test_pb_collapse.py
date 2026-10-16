from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_impact.pb_collapse import (
    add_trailing_volume_normalisation,
    fit_pb_master_curve,
    independently_binned_signed_response,
    pb_shape,
    scaling_slope,
)


def synthetic_curves(alpha=1.8, beta=0.75, xi=0.95, psi=0.6, seed=3):
    rng = np.random.default_rng(seed)
    frames = []
    for n in (250, 1000, 4000, 16000):
        q_scale, r_scale = 1e-4 * (n / 1000) ** xi, 1e-3 * (n / 1000) ** psi
        x = rng.standard_t(4, 60_000) * q_scale * 2
        y = r_scale * pb_shape(x / q_scale, alpha, beta) + rng.normal(0, r_scale * 0.3, len(x))
        curve = independently_binned_signed_response(pd.DataFrame({"x": x, "log_return": y}), "x", n_bins=40)
        curve.insert(0, "n_events", n)
        frames.append(curve)
    return pd.concat(frames, ignore_index=True)


def test_master_curve_recovers_shape_and_scaling_exponents():
    fit = fit_pb_master_curve(synthetic_curves(), (250, 1000, 4000, 16000), n_starts=4)
    assert abs(fit["alpha"] - 1.8) < 0.3
    assert abs(fit["beta"] - 0.75) < 0.1
    xi = scaling_slope(fit["scales"], "fitted_q_scale", "xi")["ols_exponent"]
    psi = scaling_slope(fit["scales"], "fitted_r_scale", "psi")["ols_exponent"]
    assert abs(xi - 0.95) < 0.05 and abs(psi - 0.6) < 0.05


def test_trailing_volume_excludes_current_block_and_requires_full_window():
    hour = 3_600_000
    blocks = pd.DataFrame({"end_time_ms": np.arange(30) * hour, "gross_volume": np.ones(30), "signed_volume": np.ones(30)})
    out = add_trailing_volume_normalisation(blocks)
    assert out["trailing_volume"].iloc[:24].isna().all()          # first 24 h incomplete
    assert out["trailing_volume"].iloc[24] == 24                   # [t-24h, t): 24 earlier blocks, current excluded
    assert np.isclose(out["pb_signed_volume"].iloc[25], 1 / 24)


def test_activity_surface_recovers_gamma_and_nests_pb():
    from crypto_impact.pb_surface import aggregate_cells, fit_activity_surface, grid_cuts, monthly_cell_stats
    rng = np.random.default_rng(11)
    frames = []
    for n in (250, 1000, 4000):
        q_scale, r_scale = 0.2 * (n / 1000) ** 0.3, 1e-3 * (n / 1000) ** 0.6
        size = 80_000
        z = np.clip(rng.beta(1.2, 5, size), 1e-4, 1)
        a = np.exp(rng.normal(np.log(n / 1000 * 0.01), 0.6, size))
        side = np.where(rng.uniform(size=size) < 0.5, -1, 1)
        y = side * r_scale * pb_shape(z * a ** 0.6 / (q_scale * (n / 1000 * 0.01) ** 0.6), 1.8, 0.75) + rng.normal(0, r_scale * 0.5, size)
        month = rng.integers(0, 12, size)
        cuts = {s: grid_cuts(z[side == s], a[side == s], 12, 6) for s in (-1, 1)}
        stats = monthly_cell_stats(month, side, z, a, y, cuts)
        stats.insert(0, "n_events", n)
        frames.append(stats)
    cells = aggregate_cells(pd.concat(frames), min_obs=50)
    free = fit_activity_surface(cells, n_starts=4)
    pb = fit_activity_surface(cells, gamma=1.0, n_starts=4)
    assert abs(free.gamma - 0.6) < 0.1
    assert pb.gamma == 1.0 and pb.gamma_fixed
    assert free.objective < pb.objective


def test_single_n_surface_model_recovers_gamma_on_cells() -> None:
    import numpy as np
    import pandas as pd
    from crypto_impact import effective_liquidity as el
    rng = np.random.default_rng(5)
    log_q = rng.uniform(np.log(0.01), np.log(0.8), 400)
    log_a = rng.normal(-5.0, 0.8, 400)
    truth = dict(log_Q=-6.0, log_R=np.log(0.2), log_alpha=np.log(2.0), log_beta=np.log(0.6), gamma=0.4)
    cells = pd.DataFrame({"mean_log_q": log_q, "mean_log_a": log_a, "mean_u": np.exp(log_q),
                          "mean_z": log_a + 5.0, "mean_f": 0.0, "standard_error": 1e-3,
                          "activity_bin": 0, "imbalance_bin": np.arange(400) % 20})
    cells["mean_impact"] = el.predict_model("surface_volume", np.array(list(truth.values())), cells)
    fit = el.fit_one_model("surface_volume", cells, {}, n_starts=4)
    assert abs(fit["params"][4] - 0.4) < 1e-3


def test_fixed_parameters_are_held_and_not_counted() -> None:
    import numpy as np
    import pandas as pd
    from crypto_impact import effective_liquidity as el
    rng = np.random.default_rng(6)
    log_q = rng.uniform(np.log(0.01), np.log(0.8), 300)
    log_a = rng.normal(-5.0, 0.8, 300)
    shape = {"log_alpha": np.log(2.0), "log_beta": np.log(0.6)}
    truth = np.array([-6.0, np.log(0.2), shape["log_alpha"], shape["log_beta"], 0.4])
    cells = pd.DataFrame({"mean_log_q": log_q, "mean_log_a": log_a, "mean_u": np.exp(log_q), "mean_z": 0.0,
                          "mean_f": 0.0, "standard_error": 1e-3, "activity_bin": np.arange(300) % 5,
                          "imbalance_bin": np.arange(300) % 20})
    cells["mean_impact"] = el.predict_model("surface_volume", truth, cells)
    fit = el.fit_one_model("surface_volume", cells, {}, n_starts=3, fixed_params=shape)
    assert np.allclose(fit["params"][2:4], [shape["log_alpha"], shape["log_beta"]])
    assert abs(fit["params"][4] - 0.4) < 1e-3
    assert el.fit_summary_row(fit)["parameters"] == 3
