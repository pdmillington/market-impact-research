"""Transparent event classifications and response components for decay studies."""

from __future__ import annotations

import numpy as np


SWEEP_CLASS_ORDER = (
    "opposite_endpoint_move",
    "no_endpoint_sweep",
    "sweep_q1",
    "sweep_q2",
    "sweep_q3",
    "sweep_q4",
)


def execution_sweep_bps(
    first_price: np.ndarray,
    last_price: np.ndarray,
    aggressor_sign: np.ndarray,
) -> np.ndarray:
    """Return the side-aligned first-to-last execution-price displacement."""

    first = np.asarray(first_price, dtype=float)
    last = np.asarray(last_price, dtype=float)
    side = np.asarray(aggressor_sign, dtype=float)
    if not (first.shape == last.shape == side.shape):
        raise ValueError("First price, last price and side vectors must align.")
    if np.any(first <= 0) or np.any(last <= 0):
        raise ValueError("Execution prices must be positive.")
    return 10_000.0 * side * np.log(last / first)


def execution_span_bps(
    min_price: np.ndarray,
    max_price: np.ndarray,
) -> np.ndarray:
    """Return the unsigned within-event minimum-to-maximum price span."""

    minimum = np.asarray(min_price, dtype=float)
    maximum = np.asarray(max_price, dtype=float)
    if minimum.shape != maximum.shape:
        raise ValueError("Minimum and maximum price vectors must align.")
    if np.any(minimum <= 0) or np.any(maximum <= 0):
        raise ValueError("Execution prices must be positive.")
    if np.any(maximum < minimum):
        raise ValueError("Maximum execution price cannot be below minimum price.")
    return 10_000.0 * np.log(maximum / minimum)


def sweep_class(
    sweep_bps: np.ndarray,
    positive_cuts_bps: np.ndarray | tuple[float, float, float],
    *,
    zero_tolerance_bps: float = 1e-12,
) -> np.ndarray:
    """Classify events primarily by side-aligned endpoint sweep.

    Positive sweeps are divided using three training-frozen cut points. A
    negative sweep is kept as an explicit reconstruction/path anomaly rather
    than being mixed into economically aggressive events.
    """

    values = np.asarray(sweep_bps, dtype=float)
    cuts = np.asarray(positive_cuts_bps, dtype=float)
    if cuts.shape != (3,):
        raise ValueError("Exactly three positive sweep cut points are required.")
    if not np.all(np.isfinite(cuts)) or np.any(cuts <= 0):
        raise ValueError("Sweep cut points must be finite and positive.")
    if not np.all(np.diff(cuts) > 0):
        raise ValueError("Sweep cut points must be strictly increasing.")
    if zero_tolerance_bps < 0:
        raise ValueError("zero_tolerance_bps cannot be negative.")

    result = np.full(values.shape, "sweep_q4", dtype="U32")
    result[values < -zero_tolerance_bps] = "opposite_endpoint_move"
    result[np.abs(values) <= zero_tolerance_bps] = "no_endpoint_sweep"
    positive = values > zero_tolerance_bps
    result[positive & (values <= cuts[0])] = "sweep_q1"
    result[positive & (values > cuts[0]) & (values <= cuts[1])] = "sweep_q2"
    result[positive & (values > cuts[1]) & (values <= cuts[2])] = "sweep_q3"
    return result


def exclusive_execution_class(
    fill_count: np.ndarray,
    price_level_count: np.ndarray,
) -> np.ndarray:
    """Classify events by fill multiplicity and execution-price depth."""

    fills = np.asarray(fill_count)
    levels = np.asarray(price_level_count)
    if fills.shape != levels.shape:
        raise ValueError("fill_count and price_level_count must align.")
    if np.any(fills < 1) or np.any(levels < 1):
        raise ValueError("Event fill and price-level counts must be positive.")
    result = np.full(fills.shape, "multi_price_10_plus", dtype="U32")
    result[(levels == 1) & (fills == 1)] = "single_fill_single_price"
    result[(levels == 1) & (fills > 1)] = "multi_fill_single_price"
    result[levels == 2] = "multi_price_2"
    result[(levels >= 3) & (levels <= 4)] = "multi_price_3_4"
    result[(levels >= 5) & (levels <= 9)] = "multi_price_5_9"
    return result


def event_response_components(
    previous_price: np.ndarray,
    first_price: np.ndarray,
    last_price: np.ndarray,
    aggressor_sign: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return side-aligned entry, execution-path, and immediate responses."""

    previous = np.asarray(previous_price, dtype=float)
    first = np.asarray(first_price, dtype=float)
    last = np.asarray(last_price, dtype=float)
    side = np.asarray(aggressor_sign, dtype=float)
    if not (previous.shape == first.shape == last.shape == side.shape):
        raise ValueError("Price and side vectors must align.")
    if np.any(previous <= 0) or np.any(first <= 0) or np.any(last <= 0):
        raise ValueError("Prices must be positive.")
    entry = 10_000.0 * side * np.log(first / previous)
    execution = execution_sweep_bps(first, last, side)
    immediate = 10_000.0 * side * np.log(last / previous)
    return entry, execution, immediate


def response_at_horizon(
    previous_price: np.ndarray,
    last_price: np.ndarray,
    future_price: np.ndarray,
    aggressor_sign: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return side-aligned total and post-event responses."""

    previous = np.asarray(previous_price, dtype=float)
    last = np.asarray(last_price, dtype=float)
    future = np.asarray(future_price, dtype=float)
    side = np.asarray(aggressor_sign, dtype=float)
    if not (previous.shape == last.shape == future.shape == side.shape):
        raise ValueError("Price and side vectors must align.")
    if np.any(previous <= 0) or np.any(last <= 0) or np.any(future <= 0):
        raise ValueError("Prices must be positive.")
    total = 10_000.0 * side * np.log(future / previous)
    post = 10_000.0 * side * np.log(future / last)
    return total, post
