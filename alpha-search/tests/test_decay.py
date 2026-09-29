from __future__ import annotations

import math

import numpy as np

from alpha_search.decay import (
    event_response_components,
    execution_span_bps,
    execution_sweep_bps,
    exclusive_execution_class,
    response_at_horizon,
    sweep_class,
)


def test_execution_class_separates_fill_and_price_multiplicity() -> None:
    result = exclusive_execution_class(
        np.array([1, 3, 2, 4, 8, 12]),
        np.array([1, 1, 2, 4, 8, 12]),
    )

    assert result.tolist() == [
        "single_fill_single_price",
        "multi_fill_single_price",
        "multi_price_2",
        "multi_price_3_4",
        "multi_price_5_9",
        "multi_price_10_plus",
    ]


def test_response_components_add_to_immediate_impact() -> None:
    entry, execution, immediate = event_response_components(
        np.array([100.0]),
        np.array([101.0]),
        np.array([102.0]),
        np.array([1]),
    )
    total, post = response_at_horizon(
        np.array([100.0]),
        np.array([102.0]),
        np.array([101.0]),
        np.array([1]),
    )

    assert math.isclose(entry[0] + execution[0], immediate[0])
    assert math.isclose(total[0] - immediate[0], post[0])


def test_execution_sweep_aligns_buy_and_sell_price_aggression() -> None:
    sweep = execution_sweep_bps(
        np.array([100.0, 100.0, 100.0]),
        np.array([101.0, 99.0, 100.0]),
        np.array([1, -1, 1]),
    )

    assert sweep[0] > 0
    assert sweep[1] > 0
    assert sweep[2] == 0


def test_execution_span_detects_round_trip_path() -> None:
    span = execution_span_bps(
        np.array([99.0, 100.0]),
        np.array([101.0, 100.0]),
    )

    assert span[0] > 0
    assert span[1] == 0


def test_sweep_class_uses_training_frozen_endpoint_cuts() -> None:
    result = sweep_class(
        np.array([-0.1, 0.0, 0.05, 0.2, 0.7, 1.5]),
        (0.1, 0.5, 1.0),
    )

    assert result.tolist() == [
        "opposite_endpoint_move",
        "no_endpoint_sweep",
        "sweep_q1",
        "sweep_q2",
        "sweep_q3",
        "sweep_q4",
    ]
