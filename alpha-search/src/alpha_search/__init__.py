"""Reusable components for the alpha-search project."""

from .bars import (
    add_bar_horizon_targets,
    add_trailing_impact_features,
    build_fixed_event_bars,
    build_time_bars,
    coarsen_fixed_event_bars,
    marginal_impact_proxy,
)
from .pb_scaling import PBScalingFit, fit_pb_scaling, pb_shape, predict_pb_scaling
from .impact_model import (
    AsymmetricImpactFit,
    SideImpactFit,
    add_asymmetric_impact_features,
    fit_asymmetric_log_impact,
)
from .screen_features import (
    BASE_SCREEN_FEATURES,
    FITTED_SCREEN_FEATURES,
    add_short_bar_screen_features,
    screen_feature_catalog,
)
from .multiscale_impact import (
    build_trailing_impact_observations,
    resolve_measurement_windows,
)
from .impact_diagnostics import (
    MODEL_ORDER,
    calibration_metrics,
    calibration_rows,
    fit_curve_families,
    predict_curve,
)

__all__ = [
    "add_bar_horizon_targets",
    "add_trailing_impact_features",
    "build_fixed_event_bars",
    "build_time_bars",
    "coarsen_fixed_event_bars",
    "marginal_impact_proxy",
    "PBScalingFit",
    "fit_pb_scaling",
    "pb_shape",
    "predict_pb_scaling",
    "AsymmetricImpactFit",
    "SideImpactFit",
    "add_asymmetric_impact_features",
    "fit_asymmetric_log_impact",
    "BASE_SCREEN_FEATURES",
    "FITTED_SCREEN_FEATURES",
    "add_short_bar_screen_features",
    "screen_feature_catalog",
    "build_trailing_impact_observations",
    "resolve_measurement_windows",
    "MODEL_ORDER",
    "calibration_metrics",
    "calibration_rows",
    "fit_curve_families",
    "predict_curve",
]
