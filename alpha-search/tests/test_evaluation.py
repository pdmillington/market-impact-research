from __future__ import annotations

import math

import numpy as np

from alpha_search.evaluation import (
    benjamini_hochberg,
    day_block_summary,
    forward_clock_targets,
    frozen_quantiles,
    paired_day_difference,
)


def test_forward_clock_targets_use_first_completed_bar_after_horizon() -> None:
    times = np.array([0, 40_000, 70_000, 130_000], dtype=np.int64)
    prices = np.array([100.0, 101.0, 102.0, 104.0])

    result, elapsed = forward_clock_targets(times, prices, (1,))[1]

    assert math.isclose(result[0], 10_000.0 * math.log(102.0 / 100.0))
    assert math.isclose(elapsed[0], 70.0 / 60.0)
    assert np.isnan(result[-1])


def test_frozen_quantiles_are_group_specific_and_training_only() -> None:
    values = np.array([1, 2, 3, 4, 10, 20, 30, 40, 1_000, -1_000], dtype=float)
    groups = np.array([-1, -1, -1, -1, 1, 1, 1, 1, -1, 1])
    training = np.arange(len(values)) < 8

    labels, cuts = frozen_quantiles(
        values, training, groups, n_quantiles=2
    )

    assert cuts[-1][0] == 2.5
    assert cuts[1][0] == 25.0
    assert labels[-2] == 2
    assert labels[-1] == 1


def test_day_summaries_equal_weight_days_and_pair_common_days() -> None:
    values = np.array([1.0, 3.0, 5.0, -1.0, 1.0])
    day = np.array([1, 1, 2, 1, 2])
    first = np.array([True, True, True, False, False])
    second = np.array([False, False, False, True, True])

    summary = day_block_summary(values, day, first)
    paired = paired_day_difference(values, day, first, second)

    assert summary["observations"] == 3
    assert summary["day_mean_bps"] == 3.5
    assert paired["paired_days"] == 2
    assert paired["day_mean_difference_bps"] == 3.5


def test_benjamini_hochberg_preserves_ordered_adjustment() -> None:
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03, math.nan])

    assert np.allclose(adjusted[:3], [0.03, 0.04, 0.04])
    assert np.isnan(adjusted[3])
