"""Leakage-safe helpers for chronological alpha diagnostics."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def normal_p_value(t_stat: float) -> float:
    """Return a two-sided normal approximation p-value for a t statistic."""

    if not math.isfinite(t_stat):
        return math.nan
    return math.erfc(abs(t_stat) / math.sqrt(2.0))


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """Return Benjamini--Hochberg adjusted p-values."""

    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if len(valid) == 0:
        return adjusted
    order = valid[np.argsort(p[valid])]
    ranked = p[order] * len(valid) / np.arange(1, len(valid) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def forward_clock_targets(
    end_time_ms: np.ndarray,
    end_price: np.ndarray,
    horizons_minutes: Sequence[int],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Return close-to-first-subsequent-bar returns at clock-time horizons.

    The price is taken from the first completed bar whose end time is at or
    after ``end_time + horizon``.  The realized elapsed time is returned so
    the approximation error from using bar endpoints remains visible.
    """

    times = np.asarray(end_time_ms, dtype=np.int64)
    prices = np.asarray(end_price, dtype=float)
    if times.ndim != 1 or prices.ndim != 1 or len(times) != len(prices):
        raise ValueError("end_time_ms and end_price must be equally sized vectors.")
    if len(times) and np.any(times[1:] < times[:-1]):
        raise ValueError("end_time_ms must be sorted.")
    if len(times) and (not np.all(np.isfinite(prices)) or np.any(prices <= 0)):
        raise ValueError("end_price must be finite and positive.")

    results: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for value in horizons_minutes:
        horizon = int(value)
        if horizon <= 0:
            raise ValueError("Clock horizons must be positive integer minutes.")
        target_time = times + horizon * 60_000
        index = np.searchsorted(times, target_time, side="left")
        valid = index < len(times)
        future_return = np.full(len(times), np.nan)
        elapsed_minutes = np.full(len(times), np.nan)
        future_return[valid] = 10_000.0 * np.log(
            prices[index[valid]] / prices[valid]
        )
        elapsed_minutes[valid] = (
            times[index[valid]] - times[valid]
        ) / 60_000.0
        results[horizon] = (future_return, elapsed_minutes)
    return results


def frozen_quantiles(
    values: np.ndarray,
    training_mask: np.ndarray,
    groups: np.ndarray,
    *,
    n_quantiles: int = 5,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Assign group-specific quantiles using training observations only."""

    x = np.asarray(values, dtype=float)
    train = np.asarray(training_mask, dtype=bool)
    group_values = np.asarray(groups)
    if not (len(x) == len(train) == len(group_values)):
        raise ValueError("values, training_mask, and groups must align.")
    if n_quantiles < 2:
        raise ValueError("n_quantiles must be at least two.")

    labels = np.zeros(len(x), dtype=np.int8)
    cuts: dict[int, np.ndarray] = {}
    for group in np.unique(group_values):
        selected = train & (group_values == group) & np.isfinite(x)
        if selected.sum() == 0:
            continue
        if selected.sum() < n_quantiles:
            raise ValueError(f"Insufficient training observations for group {group}.")
        group_cuts = np.quantile(
            x[selected], np.linspace(0.0, 1.0, n_quantiles + 1)[1:-1]
        )
        cuts[int(group)] = group_cuts
        applicable = (group_values == group) & np.isfinite(x)
        labels[applicable] = (
            np.digitize(x[applicable], group_cuts, right=True) + 1
        )
    return labels, cuts


def day_block_summary(
    values: np.ndarray,
    day: np.ndarray,
    selected: np.ndarray,
) -> dict[str, float | int]:
    """Summarise a selected signal using equal-weight UTC-day means."""

    y = np.asarray(values, dtype=float)
    blocks = np.asarray(day)
    mask = np.asarray(selected, dtype=bool) & np.isfinite(y)
    if not (len(y) == len(blocks) == len(mask)):
        raise ValueError("values, day, and selected must align.")
    if not mask.any():
        return {
            "observations": 0,
            "days": 0,
            "mean_return_bps": math.nan,
            "day_mean_bps": math.nan,
            "day_standard_error_bps": math.nan,
            "day_t_stat": math.nan,
            "positive_day_share": math.nan,
        }

    unique, inverse = np.unique(blocks[mask], return_inverse=True)
    counts = np.bincount(inverse)
    means = np.bincount(inverse, weights=y[mask]) / counts
    n_days = len(unique)
    standard_error = (
        float(means.std(ddof=1) / math.sqrt(n_days)) if n_days > 1 else math.nan
    )
    day_mean = float(means.mean())
    t_stat = (
        day_mean / standard_error
        if math.isfinite(standard_error) and standard_error > 0
        else math.nan
    )
    return {
        "observations": int(mask.sum()),
        "days": n_days,
        "mean_return_bps": float(y[mask].mean()),
        "day_mean_bps": day_mean,
        "day_standard_error_bps": standard_error,
        "day_t_stat": t_stat,
        "positive_day_share": float(np.mean(means > 0)),
    }


def paired_day_difference(
    values: np.ndarray,
    day: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> dict[str, float | int]:
    """Compare two selections using paired UTC-day mean differences.

    The reported difference is ``mean(first) - mean(second)`` on days where
    both selections occur.
    """

    y = np.asarray(values, dtype=float)
    blocks = np.asarray(day)

    def means(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        chosen = np.asarray(mask, dtype=bool) & np.isfinite(y)
        unique, inverse = np.unique(blocks[chosen], return_inverse=True)
        counts = np.bincount(inverse)
        return unique, np.bincount(inverse, weights=y[chosen]) / counts

    first_days, first_means = means(first)
    second_days, second_means = means(second)
    common, first_index, second_index = np.intersect1d(
        first_days, second_days, assume_unique=True, return_indices=True
    )
    differences = first_means[first_index] - second_means[second_index]
    if len(differences) == 0:
        return {
            "paired_days": 0,
            "day_mean_difference_bps": math.nan,
            "day_standard_error_bps": math.nan,
            "day_t_stat": math.nan,
        }
    standard_error = (
        float(differences.std(ddof=1) / math.sqrt(len(differences)))
        if len(differences) > 1
        else math.nan
    )
    mean = float(differences.mean())
    return {
        "paired_days": int(len(common)),
        "day_mean_difference_bps": mean,
        "day_standard_error_bps": standard_error,
        "day_t_stat": (
            mean / standard_error
            if math.isfinite(standard_error) and standard_error > 0
            else math.nan
        ),
    }
