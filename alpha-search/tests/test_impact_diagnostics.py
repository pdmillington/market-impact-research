from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from alpha_search.impact_diagnostics import (  # noqa: E402
    MODEL_ORDER,
    calibration_metrics,
    calibration_rows,
    fit_trimmed_affine_log,
    fit_curve_families,
    frozen_tail_cuts,
    predict_curve,
    tail_summary_rows,
)


def test_curve_families_fit_and_calibrate_monotonic_data() -> None:
    x = np.geomspace(1e-4, 1e-1, 2_000)
    scale = 1e-3
    y = 0.05 * np.log1p(2.0 * x / scale)
    selected = np.ones(len(x), dtype=bool)
    models, cuts = fit_curve_families(
        x,
        y,
        selected,
        raw_intercept=0.0,
        raw_slope=0.05,
        raw_scale=scale / 2.0,
        bins=20,
    )
    assert tuple(model["model"] for model in models) == MODEL_ORDER
    for model in models:
        prediction = predict_curve(model, x)
        assert np.isfinite(prediction).all()
    rows = calibration_rows(x, y, selected, cuts, models)
    metrics = calibration_metrics(rows)
    assert len(rows) == 80
    assert len(metrics) == 4
    fitted = {row["model"]: row for row in metrics}
    assert fitted["binned_log_zero"]["bin_rmse"] < 1e-3


def test_monotonic_diagnostic_never_decreases() -> None:
    x = np.linspace(0.01, 1.0, 1_000)
    y = np.sqrt(x) + 0.02 * np.sin(np.arange(len(x)))
    models, _ = fit_curve_families(
        x,
        y,
        np.ones(len(x), dtype=bool),
        raw_intercept=0.0,
        raw_slope=1.0,
        raw_scale=0.5,
        bins=20,
    )
    monotonic = next(model for model in models if model["model"] == "monotonic_bins")
    grid = np.linspace(x.min(), x.max(), 200)
    assert np.all(np.diff(predict_curve(monotonic, grid)) >= -1e-12)


def test_tail_rows_and_trimmed_fit_use_frozen_training_support() -> None:
    x = np.geomspace(1e-4, 1.0, 2_000)
    y = 0.02 + 0.04 * np.log1p(x / 1e-3)
    returns = 100.0 * y
    selected = np.ones(len(x), dtype=bool)
    cuts = frozen_tail_cuts(x, selected)
    rows = tail_summary_rows(x, y, returns, selected, cuts)
    assert len(rows) == 9
    assert sum(row["observations"] for row in rows) == len(x)
    assert rows[0]["quantile_high"] == 0.01
    assert rows[-1]["quantile_low"] == 0.99

    full = fit_trimmed_affine_log(
        x, y, selected, scale=1e-3, upper_quantile=1.0
    )
    trimmed = fit_trimmed_affine_log(
        x, y, selected, scale=1e-3, upper_quantile=0.95
    )
    assert full["observations"] == len(x)
    assert trimmed["observations"] == 1_900
    assert abs(full["intercept"] - 0.02) < 1e-10
    assert abs(trimmed["slope"] - 0.04) < 1e-10
